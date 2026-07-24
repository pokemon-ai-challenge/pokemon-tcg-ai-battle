from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import math

import pytest

from ptcg_ai.hidden_information.zone_math import (
    expected_in_prize,
    prob_in_prize,
    prob_in_prize_exact,
)


def test_prob_in_prize_exact_matches_hand_calculation():
    # M=10, n=2, k=3: 対象カード3枚のうち、サイド2枚にちょうど j 枚落ちている確率。
    # P(j=0) = C(3,0)*C(7,2)/C(10,2) = 1*21/45 = 21/45
    # P(j=1) = C(3,1)*C(7,1)/C(10,2) = 3*7/45 = 21/45
    # P(j=2) = C(3,2)*C(7,0)/C(10,2) = 3*1/45 = 3/45
    assert prob_in_prize_exact(10, 2, 3, 0) == pytest.approx(21 / 45)
    assert prob_in_prize_exact(10, 2, 3, 1) == pytest.approx(21 / 45)
    assert prob_in_prize_exact(10, 2, 3, 2) == pytest.approx(3 / 45)
    # ありえない j は 0.0 を返す(未確認プールの3枚しかないのに3枚超はあり得ない)。
    assert prob_in_prize_exact(10, 2, 3, 3) == 0.0
    assert prob_in_prize_exact(10, 2, 3, -1) == 0.0


def test_prob_in_prize_exact_distribution_sums_to_one():
    # 観点1: 既知の小さい (M, n, k) で全域の合計が1.0になる(分布として整合)。
    for pool_size, prize_size, target_count in [(10, 2, 3), (60, 6, 4), (7, 6, 1), (5, 0, 2), (1, 1, 1)]:
        total = sum(
            prob_in_prize_exact(pool_size, prize_size, target_count, j)
            for j in range(0, min(target_count, prize_size) + 1)
        )
        assert total == pytest.approx(1.0), (pool_size, prize_size, target_count, total)


def test_prob_in_prize_at_least_zero_equals_one():
    # at_least=0 は全域の累積なので、必ず1.0(観点1の別表現)。
    assert prob_in_prize(60, 6, 4, at_least=0) == pytest.approx(1.0)
    assert prob_in_prize(10, 2, 3, at_least=0) == pytest.approx(1.0)


def test_prob_in_prize_at_least_one_matches_complement():
    # P(at least 1) = 1 - P(exactly 0)
    pool_size, prize_size, target_count = 60, 6, 4
    expected = 1.0 - prob_in_prize_exact(pool_size, prize_size, target_count, 0)
    assert prob_in_prize(pool_size, prize_size, target_count, at_least=1) == pytest.approx(expected)


def test_prob_in_prize_at_least_beyond_target_count_is_zero():
    # target_count 枚しか無いカードが、それを超える枚数サイドに落ちていることはあり得ない。
    assert prob_in_prize(60, 6, 4, at_least=5) == 0.0


def test_prob_in_prize_simple_case_matches_ratio():
    # 1枚だけの対象カードが「特定の1山」に落ちている周辺確率は n/M に単純化できる(design.md §4.2)。
    pool_size, prize_size = 56, 6
    assert prob_in_prize(pool_size, prize_size, target_count=1, at_least=1) == pytest.approx(
        prize_size / pool_size
    )


def test_prob_in_prize_exact_handles_zero_prize_and_zero_pool():
    # prize_size=0: サイドに1枚も無いのでどんな j>=1 も0、j=0のみ1.0。
    assert prob_in_prize_exact(60, 0, 4, 0) == pytest.approx(1.0)
    assert prob_in_prize_exact(60, 0, 4, 1) == 0.0
    # pool_size=0, prize_size=0, target_count=0: 空のプール同士、j=0の1通りしかない。
    assert prob_in_prize_exact(0, 0, 0, 0) == pytest.approx(1.0)


def test_expected_in_prize_matches_formula():
    assert expected_in_prize(60, 6, 4) == pytest.approx(4 * 6 / 60)
    assert expected_in_prize(10, 2, 3) == pytest.approx(3 * 2 / 10)


def test_expected_in_prize_zero_pool_is_zero():
    assert expected_in_prize(0, 0, 0) == 0.0


def test_expected_in_prize_matches_weighted_sum_of_exact():
    # 期待値の定義どおり Σ j * P(j) と一致することを確認する(prob_in_prize_exactとの整合)。
    pool_size, prize_size, target_count = 60, 6, 4
    weighted_sum = sum(
        j * prob_in_prize_exact(pool_size, prize_size, target_count, j)
        for j in range(0, min(target_count, prize_size) + 1)
    )
    assert weighted_sum == pytest.approx(expected_in_prize(pool_size, prize_size, target_count))
