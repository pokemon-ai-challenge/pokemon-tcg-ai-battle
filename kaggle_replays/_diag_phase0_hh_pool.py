#!/usr/bin/env python3
"""Phase 0 / H3b: 過去 head-to-head 結果のプール分析(read-only, 使い捨て診断)。

phase0-transfer-gap-diagnosis-implementation-plan.md §2 の実データ裏取り。

league/results/2026-07-23_*.json は「weight 同士の直接 head-to-head」(seed ごとに独立、
A 対 B)。independent binomial なので H3 の検出力表がそのまま適用できる。

本スクリプトは:
1. 各候補の観測勝率と Wilson 95%CI を再集計(c2_b05 は 300+ext600 を seed 重複チェックの上で 900 に合算)。
2. 各結果について「CI が 50% を跨ぐか(=現行ゲートで判別不能か)」を判定。
3. 点推定が真値だと仮定したとき、検出力 0.8 に必要な N を _diag_phase0_power の閉形式で算出。

含意の確認: offline で良く見えた候補が、実戦では「300 試合で判別できない帯」に落ちていたか。

production / weights / config を一切変更しない。既存 JSON の読み取りのみ。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "league" / "results"

# H3 の閉形式検出力を再利用
sys.path.insert(0, str(HERE))
from _diag_phase0_power import required_n, wilson_lower  # noqa: E402

FILES = [
    "2026-07-23_capacity_m32_vs_m64_300.json",
    "2026-07-23_capacity_m32_vs_m128_300.json",
    "2026-07-23_outcome_m32_vs_c1_g000_300.json",
    "2026-07-23_outcome_m32_vs_c1_g025_300.json",
    "2026-07-23_outcome_m32_vs_c1_g050_300.json",
    "2026-07-23_outcome_m32_vs_c2_b05_300.json",
    "2026-07-23_outcome_m32_vs_c2_b10_300.json",
    "2026-07-23_outcome_m32_vs_c2_b20_300.json",
]
POOL_EXTRA = "2026-07-23_outcome_m32_vs_c2_b05_ext600.json"


def load(fname: str) -> dict:
    return json.loads((RESULTS / fname).read_text(encoding="utf-8"))


def seeds_of(d: dict) -> set:
    return {g["seed"] for g in d.get("games", []) if g.get("seed") is not None}


def wilson_ci(successes: int, n: int) -> tuple[float, float]:
    # wilson_lower を流用しつつ上側も出す
    import math
    z = 1.959963984540054
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def main() -> None:
    rows = []
    for fname in FILES:
        d = load(fname)
        cand = d["candidate_name"]
        n = d["games_valid"]
        wins = d["candidate_wins"]
        rows.append({"name": cand, "file": fname, "n": n, "wins": wins})

    # c2_b05 の 300 と ext600 を合算(seed 重複チェック)
    d300 = load("2026-07-23_outcome_m32_vs_c2_b05_300.json")
    d600 = load(POOL_EXTRA)
    overlap = seeds_of(d300) & seeds_of(d600)
    pooled_valid = len(overlap) == 0
    if pooled_valid:
        rows.append({
            "name": "c2_b05_pooled900", "file": "300+ext600",
            "n": d300["games_valid"] + d600["games_valid"],
            "wins": d300["candidate_wins"] + d600["candidate_wins"],
        })

    out_rows = []
    for r in rows:
        n, wins = r["n"], r["wins"]
        wr = wins / n
        lo, hi = wilson_ci(wins, n)
        spans_50 = lo <= 0.5 <= hi
        # 点推定が真値なら検出力0.8に必要なN(50%以下なら到達不能)
        need = required_n(wr, 0.8) if wr > 0.5 else None
        out_rows.append({
            **r, "win_rate": round(wr, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "ci_spans_50pct": spans_50,
            "passes_gate_lower>50": lo > 0.5,
            "required_n_if_true_for_power0.8": need,
        })

    results = {
        "note": "weight-vs-weight direct head-to-head (independent binomial). "
                "candidate_win_rate = 候補が baseline(m32=production) に勝った率。",
        "c2_b05_pool_seed_overlap": len(overlap),
        "c2_b05_pooled_valid": pooled_valid,
        "rows": out_rows,
    }
    outp = HERE / "_diag_phase0_hh_pool_results.json"
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== 過去 head-to-head(候補 vs m32=production)の実測 ==")
    print(f"{'候補':<18} {'N':>4} {'勝率':>7} {'Wilson95%CI':>18} {'50%跨ぎ':>7} {'ゲート':>6} {'真値なら要N(pow.8)':>16}")
    print("-" * 86)
    for r in out_rows:
        ci = f"[{r['wilson95'][0]*100:.1f},{r['wilson95'][1]*100:.1f}]"
        gate = "通過" if r["passes_gate_lower>50"] else "不通過"
        span = "跨ぐ" if r["ci_spans_50pct"] else "跨がない"
        need = r["required_n_if_true_for_power0.8"]
        need_s = f"{need}" if need is not None else "≤50%(不能)"
        print(f"{r['name']:<18} {r['n']:>4} {r['win_rate']*100:>6.2f}% {ci:>18} {span:>7} {gate:>6} {need_s:>16}")
    print(f"\n-> {outp.name} に保存")

    n_pass = sum(1 for r in out_rows if r["passes_gate_lower>50"])
    n_span = sum(1 for r in out_rows if r["ci_spans_50pct"])
    print(f"\n要約: {len(out_rows)}件中 ゲート通過 {n_pass}件 / CIが50%を跨ぐ(判別不能) {n_span}件")


if __name__ == "__main__":
    main()
