#!/usr/bin/env python3
"""Phase 0 / H3: 評価ゲートの検出力(read-only, 使い捨て診断)。

phase0-transfer-gap-diagnosis-implementation-plan.md §2。

現行の採用ゲートは「head-to-head N 試合で Wilson 95%CI 下限 > 0.5」。
このゲートが、offline で見えた小さな優位(+0.4〜0.5pt = 真勝率 50.4〜50.5%)を
どのくらいの確率で拾えるのか(検出力)を測る。

もし 300 試合スクリーニングでこれらが検出不能なら、これまでの「転移しなかった」の一部は
実際には「測れていなかった」であり、評価計画自体の見直しが必要になる。

手法: モンテカルロではなく閉形式。
- 各 N について「Wilson 95%CI 下限 > 0.5」となる最小成功数 s* を厳密に求める。
- 真勝率 true_p での S ~ Binomial(N, true_p) について P(S >= s*) を正規近似
  (連続性補正つき)で評価 = 検出力。
production / weights / config を一切変更しない。数式のみ。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

Z = 1.959963984540054  # 95%


def wilson_lower(successes: int, n: int, z: float = Z) -> float:
    if n == 0:
        return 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def wilson_halfwidth(n: int, p: float = 0.5, z: float = Z) -> float:
    if n == 0:
        return 0.5
    s = round(p * n)
    phat = s / n
    z2 = z * z
    denom = 1.0 + z2 / n
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    return margin / denom


def min_successes_to_pass(n: int, z: float = Z) -> int:
    """Wilson 95%CI 下限 > 0.5 を満たす最小成功数 s*(単調なので線形/二分探索で厳密)。"""
    lo, hi = n // 2, n
    # 上限が通過しないことは基本ないが安全側に。
    if wilson_lower(hi, n, z) <= 0.5:
        return n + 1  # 到達不能
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lower(mid, n, z) > 0.5:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def detection_power(true_p: float, n: int) -> float:
    """真勝率 true_p で N 試合したとき『Wilson 95%CI 下限 > 0.5』となる確率。

    S ~ Binomial(N, true_p) を正規近似(連続性補正)して P(S >= s*) を返す。
    """
    s_star = min_successes_to_pass(n)
    if s_star > n:
        return 0.0
    mean = n * true_p
    var = n * true_p * (1.0 - true_p)
    if var <= 0:
        return 1.0 if mean >= s_star else 0.0
    sd = math.sqrt(var)
    # P(S >= s_star) 連続性補正: P(S >= s_star) ≈ 1 - Phi((s_star - 0.5 - mean)/sd)
    z = (s_star - 0.5 - mean) / sd
    return 1.0 - _normal_cdf(z)


def required_n(true_p: float, target_power: float = 0.8, n_hi: int = 1_000_000) -> int | None:
    if true_p <= 0.5:
        return None
    if detection_power(true_p, n_hi) < target_power:
        return None
    lo, hi = 10, n_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if detection_power(true_p, mid) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    true_ps = [0.505, 0.51, 0.52, 0.53, 0.55, 0.58, 0.60]
    ns = [300, 600, 900, 2000, 5000]

    power_matrix = {
        f"{tp:.3f}": {str(n): round(detection_power(tp, n), 4) for n in ns}
        for tp in true_ps
    }
    req = {f"{tp:.3f}": required_n(tp, 0.8) for tp in true_ps}
    s_stars = {str(n): min_successes_to_pass(n) for n in ns}
    halfwidths = {str(n): round(wilson_halfwidth(n, 0.5), 4) for n in ns}

    results = {
        "gate": "Wilson 95% CI lower bound > 0.5",
        "method": "closed-form normal approximation to binomial (continuity-corrected)",
        "min_successes_to_pass_by_N": s_stars,
        "min_winrate_to_pass_by_N": {str(n): round(s_stars[str(n)] / n, 4) for n in ns},
        "power_matrix": power_matrix,
        "required_n_for_power_0.8": req,
        "wilson_halfwidth_at_p0.5": halfwidths,
    }

    out = Path(__file__).resolve().parent / "_diag_phase0_power_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== 採用ゲート: Wilson 95%CI 下限 > 0.5 ==")
    print("各 N で通過に必要な勝率(= s*/N):")
    for n in ns:
        print(f"  N={n:>5}: s*={s_stars[str(n)]:>5}  ->  勝率 {s_stars[str(n)]/n*100:.2f}% 以上が必要")
    print()
    print("== 検出力(真勝率 true_p でゲート通過する確率) ==")
    header = "true_p \\ N | " + " | ".join(f"{n:>6}" for n in ns)
    print(header)
    print("-" * len(header))
    for tp in true_ps:
        cells = " | ".join(f"{power_matrix[f'{tp:.3f}'][str(n)]:>6.3f}" for n in ns)
        print(f"  {tp:.3f}   | {cells}")
    print()
    print("== 検出力 0.8 に必要な試合数 ==")
    for tp in true_ps:
        r = req[f"{tp:.3f}"]
        print(f"  true_p={tp:.3f}: N = {r if r is not None else '到達不能'}")
    print()
    print("== p=0.5 での Wilson 半幅(測定分解能) ==")
    for n in ns:
        print(f"  N={n:>5}: ±{halfwidths[str(n)]*100:.2f}pt")
    print(f"\n-> {out.name} に保存")


if __name__ == "__main__":
    main()
