"""defect#7(評価指標を勝率ptからElo差へ)の検証。train_v3.elo_diff の性質を確認する。

roadmap-2026-08-05.md §6: 「勝率から 173.7 × logit(p)」が採否判断の主指標として既に定義されている
(design-transformer-representation-2026-08-08.md / requirements-kamitsuorochi-2026-08-12.md も同じ式を
引用)。実装(train_v3.elo_diff)がこの式と一致し、かつ p=0/1 の境界で壊れないことを確認する。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from train_v3 import elo_diff  # noqa: E402


def main():
    # ------------------------------------------------------------------
    # 検証1: p=0.5 -> 0
    # ------------------------------------------------------------------
    v = elo_diff(50, 100)
    print(f"elo_diff(50,100) = {v}")
    assert abs(v) < 1e-9, v
    print("検証1 PASS: p=0.5 -> 0")

    # ------------------------------------------------------------------
    # 検証2: 式が 173.7 * logit(p) と一致する(継続性補正のない中間の p で)
    # ------------------------------------------------------------------
    for w, n in [(60, 100), (30, 100), (75, 200), (1, 3)]:
        p = w / n
        expected = 173.7 * math.log(p / (1 - p))
        got = elo_diff(w, n)
        print(f"elo_diff({w},{n}) = {got:.4f}  (173.7*logit(p) = {expected:.4f})")
        assert abs(got - expected) < 1e-6, (got, expected)
    print("検証2 PASS: 式が 173.7 * logit(p) と一致")

    # ------------------------------------------------------------------
    # 検証3: 単調増加(pが大きいほどEloが高い)
    # ------------------------------------------------------------------
    vals = [elo_diff(w, 100) for w in range(1, 100)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), "単調増加でない"
    print("検証3 PASS: wins に対して単調増加")

    # ------------------------------------------------------------------
    # 検証4: p=0 (w=0) と p=1 (w=n) で NaN/inf/crash しない(境界の継続性補正)
    # ------------------------------------------------------------------
    for w, n in [(0, 10), (10, 10), (0, 1), (1, 1), (0, 200), (200, 200)]:
        v = elo_diff(w, n)
        print(f"elo_diff({w},{n}) = {v}")
        assert math.isfinite(v), f"elo_diff({w},{n}) = {v} は有限でない"
    # w=0 と w=n は符号が反転した同じ大きさ(対称)になるはず
    assert abs(elo_diff(0, 10) + elo_diff(10, 10)) < 1e-9
    print("検証4 PASS: p=0/1 の境界でも有限値を返す(NaN/inf なし)、符号対称")

    # ------------------------------------------------------------------
    # 検証5: n<=0 は NaN(未定義な入力の明示化)
    # ------------------------------------------------------------------
    v = elo_diff(0, 0)
    print(f"elo_diff(0,0) = {v}")
    assert math.isnan(v)
    print("検証5 PASS: n<=0 は NaN")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
