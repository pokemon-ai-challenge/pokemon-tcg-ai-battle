#!/usr/bin/env python3
"""提出 54959380(full+新デッキ, publicScore 786.7)の実 Kaggle 対戦を分析する読み取り専用診断。

出力:
  1. 相手アーキタイプ別 W-L(4-2 形式)
  2. 敗因タクソノミ(deck_out / bench_wipe / side_race_close / side_race_blowout)
  3. 敗因 × 相手アーキタイプ のクロス集計、試合長・最終サイド差・序盤展開停滞の内訳

分類ロジック・アーキタイプ推定は既存 `_diag_loss_taxonomy.py` の関数(analyze_loss / game_outcome /
load_my_idx、HybridDeckPredictor による事後同定)をそのまま再利用する(追加学習なし・コード変更なし)。
対象エピソードだけ 54959380 に絞る。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import _diag_loss_taxonomy as tax  # noqa: E402
from _common import fetch_episodes  # noqa: E402


def episode_ids_for_ref(ref: str) -> list[int]:
    map_path = _HERE / "_submission_episode_map.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    if ref in mapping and mapping[ref]:
        return [int(e) for e in mapping[ref]]
    ids = [int(e["id"]) for e in fetch_episodes(ref)]
    mapping[ref] = ids
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="54959380")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eids = episode_ids_for_ref(args.ref)
    print(f"=== submission {args.ref} 分析 (episodes={len(eids)}) ===")

    wl = defaultdict(lambda: [0, 0])          # archetype -> [win, loss]
    loss_by_cat = defaultdict(list)
    loss_by_arch = defaultdict(list)
    games = wins = losses = draws = missing = unknown_arch = 0

    for eid in eids:
        path = tax.REPLAYS_DIR / f"episode-{eid}-replay.json"
        if not path.exists():
            missing += 1
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        my_idx = tax.load_my_idx(replay)
        if my_idx is None:
            continue
        outcome = tax.game_outcome(replay, my_idx)
        info = tax.analyze_loss(replay, my_idx)  # category(loss前提) と archetype を得る
        arch = (info or {}).get("opp_archetype", "unknown")
        games += 1
        if outcome == "win":
            wins += 1
            wl[arch][0] += 1
        elif outcome == "loss":
            losses += 1
            wl[arch][1] += 1
            if info is not None:
                loss_by_cat[info["category"]].append(info)
                loss_by_arch[arch].append(info)
        elif outcome == "draw":
            draws += 1

    # 1. 相手アーキタイプ別 W-L
    print(f"\n--- 相手アーキタイプ別 W-L (総合 {wins}-{losses}, draw {draws}, 勝率 "
          f"{wins/(wins+losses):.3f}) ---" if (wins + losses) else "")
    print(f"{'相手アーキタイプ':24s}{'W-L':>8s}{'勝率':>7s}{'試合':>5s}")
    for arch, (w, l) in sorted(wl.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = w + l
        print(f"{arch:24s}{f'{w}-{l}':>8s}{(w/n if n else 0):>7.2f}{n:>5d}")

    # 2. 敗因タクソノミ
    print(f"\n--- 敗因タクソノミ (loss {losses}件) ---")
    for cat, gs in sorted(loss_by_cat.items(), key=lambda kv: -len(kv[1])):
        turns = [g["final_turn"] for g in gs if g["final_turn"] is not None]
        myp = [g["my_final_prize"] for g in gs if g["my_final_prize"] is not None]
        stalled = sum(1 for g in gs if g["detail"].get("development_stalled"))
        tstr = f"avg_turn={statistics.mean(turns):.1f}" if turns else ""
        pstr = f"avg_残サイド={statistics.mean(myp):.1f}" if myp else ""
        extra = f" 序盤展開停滞={stalled}" if any(g["category"].startswith("side_race") for g in gs) else ""
        pct = len(gs) / losses * 100 if losses else 0
        print(f"  {cat:20s} {len(gs):3d}件 ({pct:4.1f}%)  {tstr}  {pstr}{extra}")

    # 3. 敗因 × アーキタイプ
    print(f"\n--- 敗因 × 相手アーキタイプ ---")
    for arch, gs in sorted(loss_by_arch.items(), key=lambda kv: -len(kv[1])):
        cats = defaultdict(int)
        for g in gs:
            cats[g["category"]] += 1
        cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cats.items(), key=lambda kv: -kv[1]))
        print(f"  {arch:22s} 負け{len(gs):2d}件: {cat_str}")

    result = {
        "ref": args.ref, "games": games, "wins": wins, "losses": losses, "draws": draws,
        "missing": missing,
        "winloss_by_archetype": {a: {"win": v[0], "loss": v[1]} for a, v in wl.items()},
        "loss_by_category": {c: len(v) for c, v in loss_by_cat.items()},
        "loss_by_archetype": {a: {"count": len(v), "categories": _cat_counts(v)} for a, v in loss_by_arch.items()},
        "loss_details": [
            {"category": g["category"], "opp_archetype": a, "final_turn": g["final_turn"],
             "my_final_prize": g["my_final_prize"], "development_stalled": g["detail"].get("development_stalled")}
            for a, gs in loss_by_arch.items() for g in gs
        ],
    }
    out = args.out or str(_HERE / f"_analyze_submission_{args.ref}_results.json")
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


def _cat_counts(gs: list) -> dict:
    d = defaultdict(int)
    for g in gs:
        d[g["category"]] += 1
    return dict(d)


if __name__ == "__main__":
    main()
