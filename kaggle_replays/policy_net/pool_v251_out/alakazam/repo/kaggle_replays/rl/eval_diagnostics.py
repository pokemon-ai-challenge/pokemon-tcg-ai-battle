#!/usr/bin/env python3
"""勝率以外の診断を取る（サイド差・先攻後攻・敗因）。

勝率は1試合1ビットしかないので、差を検出するのに数千試合かかる。
この測定の目的は、**同じ試合数でより多くを知る**こと:

1. **最終サイド差**（−6〜+6）。勝敗と同じことを連続値で測っている可能性がある。
   相関が高ければ、勝率の低分散な代理指標として使え、必要試合数が減る。
2. **先攻/後攻の内訳**。現行ルールでは先攻は1ターン目に攻撃できず、**実質2つの別ゲーム**。
   `is_first_player` は状態166次元のうち1ビットでしかなく、モデルが区別できていない疑いがある。
   これまでの全報告に内訳を一度も出していない（交互に回しているので取得は無料）。
3. **試合長**。デッキ切れ負けの兆候。

使い方:
    python eval_diagnostics.py --learner marnie_grimmsnarl_ex \
        --opponents alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex --games 1200
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pools  # noqa: E402
import collect_pool  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learner", required=True)
    ap.add_argument("--learner-weights", default=None)
    ap.add_argument("--opponents", required=True)
    ap.add_argument("--extra-registry", default=None)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.01,
                    help="評価は train_pool.py と同じ 0.01（実質 greedy）。0 はゼロ除算になる")
    args = ap.parse_args()

    if args.extra_registry:
        pools.load_extra_registry(args.extra_registry)

    weights_path, deck_csv = pools.resolve_learner(args.learner)
    if args.learner_weights:
        wp = Path(args.learner_weights)
        weights_path = wp if wp.is_absolute() else pools.WDIR / wp

    from run_league import read_deck_csv_file  # noqa: E402
    deck_l = read_deck_csv_file(deck_csv)
    opponents = pools.build_opponents([t.strip() for t in args.opponents.split(",") if t.strip()])

    print(f"learner={args.learner} weights={weights_path}")
    print(f"opponents={[o[0] for o in opponents]} games={args.games}\n")

    # build_opponents は (name, weights, deck) のタプル列を返す。
    # parallel_collect_pool(weights_path, opponents, deck_l, n_games, seed0, temperature, workers)
    results, _stats = collect_pool.parallel_collect_pool(
        str(weights_path), opponents, deck_l, args.games,
        12345, args.temperature, args.workers,
    )
    ok = [r for r in results if r.get("error") is None and r.get("winner") is not None]
    print(f"有効試合 {len(ok)}/{len(results)}\n")
    if not ok:
        return

    def wr(rs):
        return sum(1 for r in rs if r["reward"] >= 1.0) / len(rs) if rs else float("nan")

    # --- 1. 先攻/後攻 ---
    first = [r for r in ok if r.get("went_first") is True]
    second = [r for r in ok if r.get("went_first") is False]
    print("=== 先攻/後攻の内訳 ===")
    print(f"  全体   {wr(ok):.4f}  (n={len(ok)})")
    if first:
        print(f"  先攻   {wr(first):.4f}  (n={len(first)})")
    if second:
        print(f"  後攻   {wr(second):.4f}  (n={len(second)})")
    if first and second:
        print(f"  差     {(wr(first)-wr(second))*100:+.1f}pt")
    if not first and not second:
        print("  （firstPlayer が取得できなかった）")

    # --- 2. サイド差 ---
    pz = [r for r in ok if r.get("prize_self") is not None]
    if pz:
        diffs = [r["prize_opp"] - r["prize_self"] for r in pz]   # 正 = 自分が優勢
        wins = [1.0 if r["reward"] >= 1.0 else 0.0 for r in pz]
        print("\n=== 最終サイド差（相手の残り − 自分の残り。正が優勢）===")
        print(f"  平均 {st.mean(diffs):+.3f}  SD {st.pstdev(diffs):.3f}")
        if len(set(diffs)) > 1:
            r = (st.mean([d * w for d, w in zip(diffs, wins)]) - st.mean(diffs) * st.mean(wins)) / (
                st.pstdev(diffs) * st.pstdev(wins)) if st.pstdev(wins) > 0 else float("nan")
            print(f"  勝敗との相関 r = {r:.3f}")
            print("  （r が高いほど、サイド差は勝率の低分散な代理指標として使える）")
        # 勝ち試合/負け試合の分布
        w_d = [d for d, w in zip(diffs, wins) if w > 0]
        l_d = [d for d, w in zip(diffs, wins) if w == 0]
        if w_d:
            print(f"  勝ち試合の平均サイド差 {st.mean(w_d):+.3f} (n={len(w_d)})")
        if l_d:
            print(f"  負け試合の平均サイド差 {st.mean(l_d):+.3f} (n={len(l_d)})")

    # --- 3. 試合長 ---
    tn = [r["final_turn"] for r in ok if r.get("final_turn") is not None]
    if tn:
        print(f"\n=== 試合長 ===\n  平均 {st.mean(tn):.1f} ターン  最大 {max(tn)}")

    # --- 3.5 敗因/勝因の内訳(LogType.RESULT の reason) ---
    # 1=サイドを取り切った(取り切られた) 2=山札切れ 3=バトル場にポケモンがいない 4=カード効果
    REASON_LABEL = {
        1: "サイドを取り切った",
        2: "山札切れ",
        3: "ベンチ切れ",
        4: "カード効果",
    }
    er = [r for r in ok if r.get("end_reason") is not None]
    if er:
        wins_er = [r for r in er if r["reward"] >= 1.0]
        losses_er = [r for r in er if r["reward"] < 1.0]

        def _breakdown(rs):
            counts: dict[int, int] = {}
            for r in rs:
                counts[r["end_reason"]] = counts.get(r["end_reason"], 0) + 1
            for reason in sorted(counts):
                label = REASON_LABEL.get(reason, f"不明({reason})")
                n = counts[reason]
                print(f"  {label:<14} {n:>5} 件  {n / len(rs) * 100:5.1f}%")

        print("\n=== 敗因の内訳(負け試合、終局理由) ===")
        if losses_er:
            _breakdown(losses_er)
        else:
            print("  （負け試合で end_reason が取得できなかった）")

        print("\n=== 勝因の内訳(勝ち試合、終局理由) ===")
        if wins_er:
            _breakdown(wins_er)
        else:
            print("  （勝ち試合で end_reason が取得できなかった）")

        print("\n=== 終局理由ごとの平均ターン数 ===")
        for reason in sorted({r["end_reason"] for r in er}):
            label = REASON_LABEL.get(reason, f"不明({reason})")
            rs = [r["final_turn"] for r in er if r["end_reason"] == reason and r.get("final_turn") is not None]
            if rs:
                print(f"  {label:<14} 平均 {st.mean(rs):5.1f} ターン  (n={len(rs)})")
    else:
        print("\n=== 敗因の内訳 ===\n  （end_reason が取得できなかった。collect_parallel.py の対応前の結果の可能性）")

    # --- 4. 相手別 ---
    print("\n=== 相手別 ===")
    for name in sorted({r.get("opponent") for r in ok if r.get("opponent")}):
        rs = [r for r in ok if r.get("opponent") == name]
        f = [r for r in rs if r.get("went_first") is True]
        s = [r for r in rs if r.get("went_first") is False]
        extra = f"  先攻 {wr(f):.3f} / 後攻 {wr(s):.3f}" if f and s else ""
        print(f"  {name:<26} {wr(rs):.4f} (n={len(rs)}){extra}")


if __name__ == "__main__":
    main()
