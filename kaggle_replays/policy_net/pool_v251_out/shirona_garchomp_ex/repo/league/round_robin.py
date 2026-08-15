#!/usr/bin/env python3
"""8アーキタイプ(deck+policy)の総当たり対戦を回し、勝率マトリクスと
メタシェア加重の「対フィールド期待勝率」を出す。

各アーキタイプ = (代表デッキ 01.csv, 専用重み)。alakazam のみ weights=None で
現行 production 重み(policy_weights.json)を使う(= 現提出インカンベント)。
run_league.run_league() を各ペアで呼び、A(=行側)の勝率を記録する。
winrate(j vs i) = 1 - winrate(i vs j)(引き分け無し・エラー除外)で対称に埋める。

出力:
  results/round_robin/<A>__vs__<B>_<GAMES>.json  … 各ペアの生 summary
  results/round_robin/_matrix.json               … 勝率マトリクス + フィールド期待勝率
  (既存ペアJSONは再利用してスキップ)

使い方(デスクトップ側):
  cd C:\\dev\\pokemon-tcg-ai-battle\\league
  python round_robin.py                 # 既定 300 試合/ペア
  python round_robin.py --games 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # league/
_ROOT = _HERE.parent                              # repo root
for p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_league  # noqa: E402

_DECKDIR = "kaggle_replays/meta_analysis/archetype_decks"
_WDIR = "sample_submission/ptcg_ai/learning"

# (name, deck_path(repoルート相対), weights_path or None, meta_share_weight)
# share は deck_labels の実測ラベル件数(全体使用者数の代理)。8アーキ内で再正規化して使う。
ARCHS: list[tuple[str, str, str | None, float]] = [
    ("alakazam",             f"{_DECKDIR}/alakazam/01.csv",             None,                                          2811),
    ("mega_lucario_ex",      f"{_DECKDIR}/mega_lucario_ex/01.csv",      f"{_WDIR}/policy_weights_mega_lucario_ex.json", 1257),
    ("archaludon_ex",        f"{_DECKDIR}/archaludon_ex/01.csv",        f"{_WDIR}/policy_weights_archaludon_ex.json",   1078),
    ("crustle",              f"{_DECKDIR}/crustle/01.csv",              f"{_WDIR}/policy_weights_crustle.json",          737),
    ("dragapult_ex",         f"{_DECKDIR}/dragapult_ex/01.csv",         f"{_WDIR}/policy_weights_dragapult_ex.json",     625),
    ("marnie_grimmsnarl_ex", f"{_DECKDIR}/marnie_grimmsnarl_ex/01.csv", f"{_WDIR}/policy_weights_marnie_grimmsnarl_ex.json", 591),
    ("rocket_mewtwo_ex",     f"{_DECKDIR}/rocket_mewtwo_ex/01.csv",     f"{_WDIR}/policy_weights_rocket_mewtwo_ex.json", 247),
    ("shirona_garchomp_ex",  f"{_DECKDIR}/shirona_garchomp_ex/01.csv",  f"{_WDIR}/policy_weights_shirona_garchomp_ex.json", 181),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--config-base", default="ml_lethal_attackplan_v0only")
    ap.add_argument("--workers", type=int, default=12,
                    help="run_league のプロセス並列数(0以下で自動=CPU数-1)。ml_policy同士は重いので並列必須")
    args = ap.parse_args()
    games = args.games

    outdir = _ROOT / "results" / "round_robin"
    outdir.mkdir(parents=True, exist_ok=True)

    # ml_policy/rule_based が内部で相対参照する deck.csv 等のため、production と同じ cwd に。
    os.chdir(_ROOT / "sample_submission")

    names = [a[0] for a in ARCHS]
    # winrate[i][j] = i(A) が j(B) に勝った率
    winrate: dict[str, dict[str, float]] = {n: {} for n in names}

    pairs = list(itertools.combinations(ARCHS, 2))
    print(f"=== 総当たり {len(names)}アーキ / {len(pairs)}ペア / {games}試合/ペア ===", flush=True)
    t0 = time.time()

    for idx, ((na, da, wa, _), (nb, db, wb, _)) in enumerate(pairs, 1):
        out = outdir / f"{na}__vs__{nb}_{games}.json"
        if out.exists():
            summary = json.load(open(out, encoding="utf-8"))
            print(f"[{idx}/{len(pairs)}] {na} vs {nb}: (既存を再利用)", flush=True)
        else:
            print(f"[{idx}/{len(pairs)}] {na} vs {nb} 実行中...", flush=True)
            summary = run_league.run_league(
                agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
                deck_a_path=da, deck_b_path=db, seed_start=0, progress_every=0,
                weights_a_path=wa, weights_b_path=wb, config_base=args.config_base,
                workers=args.workers,
                log=lambda m: None,
            )
            json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        ov = summary["overall"]
        wr = ov["win_rate"]
        if wr is None:
            print(f"    (有効試合なし: errors={summary['errors']['count']})", flush=True)
            continue
        lo, hi = ov["wilson_95ci"]
        winrate[na][nb] = wr
        winrate[nb][na] = 1.0 - wr
        print(f"    {na} {wr*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] (err={summary['errors']['count']})", flush=True)

    # メタシェア加重の対フィールド期待勝率(8アーキ内で再正規化。自分は除外)。
    share = {a[0]: a[3] for a in ARCHS}
    field: dict[str, float] = {}
    for i in names:
        num = 0.0
        den = 0.0
        for j in names:
            if i == j or j not in winrate[i]:
                continue
            w = share[j]
            num += w * winrate[i][j]
            den += w
        field[i] = num / den if den else float("nan")

    matrix_out = {
        "games_per_pair": games,
        "archetypes": names,
        "share_weight": share,
        "winrate_row_vs_col": winrate,
        "field_expected_winrate": field,
    }
    mpath = outdir / "_matrix.json"
    json.dump(matrix_out, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n=== 対フィールド期待勝率(メタシェア加重, 8アーキ内) ===", flush=True)
    for n, v in sorted(field.items(), key=lambda kv: -kv[1]):
        print(f"  {n:22s} {v*100:5.1f}%", flush=True)
    print(f"\nmatrix -> {mpath}", flush=True)
    print(f"総経過 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
