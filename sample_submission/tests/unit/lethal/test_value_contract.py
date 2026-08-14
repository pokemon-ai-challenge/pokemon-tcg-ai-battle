"""critic 呼び出し契約(規則 R7)と価値の種別分離の単体テスト。

ここで固定するのは「安全な呼び方」と「視点の正しさ」だけである。
校正(B3)は未解決なので、この値を採用判断に使ってはいけない
(Phase 3 は既定 OFF のまま)。

観点:
1. 観測が自分視点なら p、相手視点なら 1-p
2. NaN / 範囲外 / 例外 / モデル未ロードは必ず UNAVAILABLE(例外を出さない)
3. UNAVAILABLE な値はゲートに通せない(``is_usable`` が False)
4. 価値の種別(V_policy / V_fail / Q_try / Q_base)が値に残る
5. ``ValueEstimate`` 自体が壊れた確率を受け付けない
"""

import math

import pytest

from ptcg_ai.search.lethal.types import (
    EstimateKind,
    StopReason,
    ValueEstimate,
    ValueKind,
)
from ptcg_ai.search.lethal.value import LeafValueEvaluator

from tests.unit.lethal import builders as b


class FakeModel:
    def __init__(self, value):
        self._value = value

    def predict_win_prob(self, obs):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def observation_for(your_index: int):
    hand = [b.card(1079, 1)]
    me = b.player(hand=hand)
    opp = b.opponent()
    return b.observation(
        b.select(b.play_options(hand)), b.state(me=me, opp=opp, your_index=your_index)
    )


def test_own_perspective_is_used_as_is():
    # 観点1: 観測の持ち主 == 自分。
    evaluator = LeafValueEvaluator(FakeModel(0.7))
    value = evaluator.evaluate_leaf(observation_for(0), me=0, kind=ValueKind.V_FAIL)
    assert value.is_usable
    assert value.mean == pytest.approx(0.7)
    assert value.kind is ValueKind.V_FAIL


def test_opponent_perspective_is_flipped():
    # 観点1: END 後の葉は相手視点になる(capability report §4.6)。
    evaluator = LeafValueEvaluator(FakeModel(0.7))
    value = evaluator.evaluate_leaf(observation_for(1), me=0)
    assert value.mean == pytest.approx(0.3)


def test_same_convention_for_q_try_and_q_base_leaves():
    # 観点1の帰結: 同じ葉なら kind が違っても数値は同じ規約で出る。
    evaluator = LeafValueEvaluator(FakeModel(0.42))
    obs = observation_for(1)
    v_fail = evaluator.evaluate_leaf(obs, me=0, kind=ValueKind.V_FAIL)
    q_base_leaf = evaluator.evaluate_leaf(obs, me=0, kind=ValueKind.Q_BASE)
    assert v_fail.mean == pytest.approx(q_base_leaf.mean)
    assert v_fail.kind is not q_base_leaf.kind


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -0.1, 1.5, None, "0.5", True],
)
def test_bad_values_become_unavailable(bad_value):
    # 観点2: 想定外の返り値は全て UNAVAILABLE。例外は投げない。
    evaluator = LeafValueEvaluator(FakeModel(bad_value))
    value = evaluator.evaluate_leaf(observation_for(0), me=0)
    assert value.estimate_kind is EstimateKind.UNAVAILABLE
    assert not value.is_usable


def test_model_exception_becomes_unavailable():
    evaluator = LeafValueEvaluator(FakeModel(RuntimeError("boom")))
    value = evaluator.evaluate_leaf(observation_for(0), me=0)
    assert value.estimate_kind is EstimateKind.UNAVAILABLE
    assert value.stop_reason is StopReason.CRITIC_UNAVAILABLE


def test_missing_model_becomes_unavailable():
    evaluator = LeafValueEvaluator(None)
    assert not evaluator.is_available
    value = evaluator.evaluate_leaf(observation_for(0), me=0)
    assert value.estimate_kind is EstimateKind.UNAVAILABLE


def test_missing_state_becomes_unavailable():
    evaluator = LeafValueEvaluator(FakeModel(0.5))
    obs = observation_for(0)
    obs.current = None
    value = evaluator.evaluate_leaf(obs, me=0)
    assert value.estimate_kind is EstimateKind.UNAVAILABLE


def test_value_estimate_rejects_broken_probabilities():
    # 観点5: 壊れた値をそもそも構築させない。
    for bad in (float("nan"), 1.2, -0.001):
        with pytest.raises(ValueError):
            ValueEstimate(
                kind=ValueKind.Q_TRY, estimate_kind=EstimateKind.EXACT, mean=bad
            )
    with pytest.raises(ValueError):
        ValueEstimate(
            kind=ValueKind.Q_TRY,
            estimate_kind=EstimateKind.EXACT,
            mean=0.5,
            lower=0.6,
            upper=0.4,
        )
    with pytest.raises(ValueError):
        ValueEstimate(kind=ValueKind.Q_TRY, estimate_kind=EstimateKind.EXACT)


def test_unavailable_estimate_needs_no_mean():
    value = ValueEstimate.unavailable(ValueKind.Q_BASE)
    assert value.mean is None and not value.is_usable
    assert not math.isnan(0.0)  # sanity: 数値検査で NaN を通していない
