"""select_teacher.py の統計計算部分の単体テスト(D1・評価方法の分離)。cgエンジン不要。"""

from __future__ import annotations

import select_teacher as st


def test_winrate_diff_ci_symmetric_when_equal():
    lo, hi = st.winrate_diff_ci95(500, 1000, 500, 1000)
    assert abs((lo + hi) / 2) < 1e-9  # 差の点推定は0


def test_winrate_diff_ci_zero_valid_returns_nan():
    lo, hi = st.winrate_diff_ci95(0, 0, 500, 1000)
    assert lo != lo and hi != hi  # NaN


def test_winrate_diff_ci_positive_when_student_better():
    lo, hi = st.winrate_diff_ci95(700, 1000, 500, 1000)
    assert lo > 0
    assert hi > lo


def test_compare_pool_results_uses_margin_not_absolute_threshold():
    """点推定の差だけでなく、95%CIの下限で判定する(サンプリング誤差込み)。"""
    teacher_per = [{"id": "a", "wins": 500, "valid": 1000, "winrate": 0.5}]
    # diff=0(教師と同等)なら、n=1000で95%CI下限は-5%を明確に上回るはず。
    student_per_equal = [{"id": "a", "wins": 500, "valid": 1000, "winrate": 0.5}]
    result = st.compare_pool_results(teacher_per, student_per_equal, (500, 1000), (500, 1000))
    assert result["margin"] == -0.05
    opp = result["per_opponent"][0]
    assert abs(opp["diff"]) < 1e-9
    assert opp["passes_margin"] is True

    # diff=-15%は明確にマージンを割る。
    student_per_bad = [{"id": "a", "wins": 350, "valid": 1000, "winrate": 0.35}]
    result_bad = st.compare_pool_results(teacher_per, student_per_bad, (500, 1000), (350, 1000))
    assert result_bad["per_opponent"][0]["passes_margin"] is False


def test_compare_pool_results_skips_unknown_opponents():
    teacher_per = [{"id": "a", "wins": 500, "valid": 1000, "winrate": 0.5}]
    student_per = [{"id": "b", "wins": 500, "valid": 1000, "winrate": 0.5}]
    result = st.compare_pool_results(teacher_per, student_per, (500, 1000), (500, 1000))
    assert result["per_opponent"] == []
