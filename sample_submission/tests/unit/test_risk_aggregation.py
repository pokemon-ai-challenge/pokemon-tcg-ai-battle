"""``ptcg_ai.search.risk_aggregation.aggregate()`` の固定値検証（EXP-A44 P1）。

要件書 §5.4 / §11 の回帰テスト計画に対応する
（``test_risk_aggregation.py``: 4mode の数値検証、N=1・全値同一・alpha境界のエッジ）。
"""

import math

import pytest

from ptcg_ai.search import risk_aggregation as ra


def test_mean_basic():
    assert ra.aggregate([1.0, 2.0, 3.0], mode="mean") == pytest.approx(2.0)


def test_mean_std_basic():
    # pstdev([1,2,3]) == sqrt(2/3)
    expected = 2.0 - 1.0 * math.sqrt(2.0 / 3.0)
    assert ra.aggregate([1.0, 2.0, 3.0], mode="mean_std", beta=1.0) == pytest.approx(expected)


def test_mean_std_zero_beta_equals_mean():
    values = [0.2, 0.8, 0.5]
    assert ra.aggregate(values, mode="mean_std", beta=0.0) == pytest.approx(ra.aggregate(values, mode="mean"))


def test_cvar_basic():
    # N=3, alpha=0.3 -> ceil(0.9)=1 件 -> 最小値のみ。
    assert ra.aggregate([1.0, 2.0, 3.0], mode="cvar", alpha=0.3) == pytest.approx(1.0)


def test_cvar_alpha_selects_expected_tail_count():
    # N=10, alpha=0.3 -> ceil(3.0)=3 件 -> 下位3件の平均。
    values = [float(i) for i in range(1, 11)]  # 1..10
    expected = (1.0 + 2.0 + 3.0) / 3.0
    assert ra.aggregate(values, mode="cvar", alpha=0.3) == pytest.approx(expected)


def test_cvar_alpha_boundary_minimum_one_sample():
    # N=10, alpha=0.01 -> ceil(0.1)=1 件（最低1件保証）。
    values = [float(i) for i in range(1, 11)]
    assert ra.aggregate(values, mode="cvar", alpha=0.01) == pytest.approx(1.0)


def test_cvar_alpha_boundary_all_samples():
    # alpha=1.0 -> 全件平均 == mean。
    values = [1.0, 5.0, 9.0]
    assert ra.aggregate(values, mode="cvar", alpha=1.0) == pytest.approx(ra.aggregate(values, mode="mean"))


def test_n_equals_one_all_modes_return_the_single_value():
    for mode in ra.MODES:
        kwargs = {"beta": 1.0, "alpha": 0.3, "prize_diff": 0}
        assert ra.aggregate([0.7], mode=mode, **kwargs) == pytest.approx(0.7)


def test_all_values_identical():
    values = [0.6, 0.6, 0.6, 0.6]
    for mode in ("mean", "mean_std", "cvar"):
        assert ra.aggregate(values, mode=mode) == pytest.approx(0.6)


def test_adaptive_advantage_uses_cvar():
    values = [0.1, 0.5, 0.9]
    result = ra.aggregate(values, mode="adaptive", alpha=0.34, prize_diff=-3)
    assert result == pytest.approx(ra.aggregate(values, mode="cvar", alpha=0.34))


def test_adaptive_behind_uses_upper_tail():
    values = [0.1, 0.5, 0.9]
    result = ra.aggregate(values, mode="adaptive", alpha=0.34, prize_diff=3)
    # ceil(0.34*3)=2 -> 上位2件(0.9, 0.5)の平均。
    assert result == pytest.approx((0.9 + 0.5) / 2.0)


def test_adaptive_even_blends_mean_and_cvar():
    values = [0.1, 0.5, 0.9]
    result = ra.aggregate(values, mode="adaptive", alpha=0.34, prize_diff=0)
    expected = 0.5 * ra.aggregate(values, mode="mean") + 0.5 * ra.aggregate(values, mode="cvar", alpha=0.34)
    assert result == pytest.approx(expected)


def test_adaptive_prize_diff_none_treated_as_even():
    values = [0.1, 0.5, 0.9]
    result = ra.aggregate(values, mode="adaptive", alpha=0.34, prize_diff=None)
    expected = ra.aggregate(values, mode="adaptive", alpha=0.34, prize_diff=0)
    assert result == pytest.approx(expected)


def test_empty_values_returns_zero_safely():
    for mode in ra.MODES:
        assert ra.aggregate([], mode=mode) == 0.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        ra.aggregate([0.1, 0.2], mode="not_a_mode")
