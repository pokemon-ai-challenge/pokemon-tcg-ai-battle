#!/usr/bin/env python3
"""多様フィールド対戦: alakazam(フーディン=現提出インカンベント)を、config だけ
`abl_5_full`(パイプライン full) と `ml_lethal_attackplan_v0only`(現行本番)で切り替え、
メタ各アーキタイプのデッキ相手に対戦させて「対フィールド期待勝率」を比較する。

狙い: ミラー自己対戦では測れない「実フィールド寄り」の read を local で得る。Kaggle の
publicScore はレーティング収束に時間がかかるため、その判断材料をアイドルなデスクトップで
先に作る。full と v0only で **同じデッキ・同じ重み(構成C)・同じ相手フィールド・同じ seed**
にそろえ、違いを config(意思決定パイプライン有無)だけに絞った controlled 比較にする。

フィールドの定義(deck/専用重み/メタシェア)は `round_robin.ARCHS` をそのまま再利用する。
- 契約者(contender) = alakazam(deck=alakazam/01.csv, weights=None=production policy_weights=構成C)。
- フィールド(opponent) = 他7アーキ(各 deck + 専用重み)。opponent の config は
  `ml_lethal_attackplan_v0only` に固定(現行本番水準の相手)。
- 契約者の config を {full, v0only} で振り、各フィールド相手の勝率とメタシェア加重の
  対フィールド期待勝率を出して full − v0only の差を見る。

run_league / run_match には手を加えず、A/B別config対応(--config-base-a/-b)と
weights/deck 別指定の口をそのまま使う。

使い方(デスクトップ側):
  cd C:\\dev\\pokemon-tcg-ai-battle\\league
  python field_gauntlet.py --games 100 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_league  # noqa: E402
from round_robin import ARCHS  # noqa: E402

CONTENDER_NAME = "alakazam"
OPP_CONFIG = "ml_lethal_attackplan_v0only"
CONTENDER_CONFIGS = {
    "full": "abl_5_full",
    "v0only": "ml_lethal_attackplan_v0only",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="各ペアの試合数(default 100)")
    ap.add_argument("--workers", type=int, default=12, help="run_league の並列プロセス数")
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    outdir = _ROOT / "results" / "field_gauntlet"
    outdir.mkdir(parents=True, exist_ok=True)
    # ml_policy が内部で相対参照する deck.csv 等のため production と同じ cwd に。
    os.chdir(_ROOT / "sample_submission")

    contender = next(a for a in ARCHS if a[0] == CONTENDER_NAME)
    field = [a for a in ARCHS if a[0] != CONTENDER_NAME]
    _, c_deck, c_weights, _ = contender

    # results[cfg_label][field_name] = {"wr":..., "ci":[lo,hi], "share":..., "errors":...}
    results: dict[str, dict[str, dict]] = {k: {} for k in CONTENDER_CONFIGS}

    t0 = time.time()
    for cfg_label, cfg in CONTENDER_CONFIGS.items():
        print(f"\n=== contender=alakazam config={cfg_label} ({cfg}) ===", flush=True)
        for (fn, fd, fw, share) in field:
            out = outdir / f"alakazam_{cfg_label}__vs__{fn}_{args.games}.json"
            if out.exists():
                summary = json.load(open(out, encoding="utf-8"))
                tag = "(既存再利用)"
            else:
                summary = run_league.run_league(
                    agent_a_name="ml_policy", agent_b_name="ml_policy", games=args.games,
                    deck_a_path=c_deck, deck_b_path=fd,
                    weights_a_path=c_weights, weights_b_path=fw,
                    config_base_a=cfg, config_base_b=OPP_CONFIG,
                    seed_start=args.seed_start, progress_every=0,
                    workers=args.workers, log=lambda m: None,
                )
                json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                tag = ""
            ov = summary["overall"]
            wr = ov["win_rate"]
            lo, hi = ov["wilson_95ci"]
            errs = summary["errors"]["count"]
            results[cfg_label][fn] = {"wr": wr, "ci": [lo, hi], "share": share, "errors": errs}
            wr_s = "n/a" if wr is None else f"{wr*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]"
            print(f"  vs {fn:22s} {wr_s} (err={errs}) {tag}", flush=True)

    # メタシェア加重の対フィールド期待勝率。
    def field_expected(cfg_label: str) -> float:
        num = den = 0.0
        for fn, r in results[cfg_label].items():
            if r["wr"] is None:
                continue
            num += r["share"] * r["wr"]
            den += r["share"]
        return num / den if den else float("nan")

    field_wr = {k: field_expected(k) for k in CONTENDER_CONFIGS}

    print("\n=== 対フィールド期待勝率(メタシェア加重) ===", flush=True)
    for k in CONTENDER_CONFIGS:
        print(f"  alakazam[{k:7s}]  {field_wr[k]*100:5.1f}%", flush=True)
    delta = field_wr["full"] - field_wr["v0only"]
    print(f"  delta(full - v0only) = {delta*100:+.1f}pt", flush=True)

    matrix = {
        "games_per_pair": args.games,
        "contender": CONTENDER_NAME,
        "opponent_config": OPP_CONFIG,
        "contender_configs": CONTENDER_CONFIGS,
        "per_field": results,
        "field_expected_winrate": field_wr,
        "delta_full_minus_v0only": delta,
    }
    mpath = outdir / "_gauntlet_matrix.json"
    json.dump(matrix, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nmatrix -> {mpath}\n総経過 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
