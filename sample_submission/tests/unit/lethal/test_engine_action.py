"""抽象アクション ↔ ``EngineAction`` の対応(Step 1-3 指示 7)。

観点:
1. 選択インデックス列から意味(カードID・対象)へ解決できる
2. 合法性(個数・重複・範囲)を実行直前に再検査する
3. 状態が変わっていたら(意味が一致しなければ)実行しない
4. 意味へ解決できない選択肢は、同一観測が保証されたときだけ実行する
5. 別観測への引き直しは、一意に決まらなければ None(勝手に解決しない)
"""

import pytest

from ptcg_ai.search.lethal.action import (
    EngineAction,
    describe,
    is_legal_selection,
    reindex,
    validate,
)

from tests.unit.lethal import builders as b


def observation_with(hand_ids):
    return b.simple_observation(hand_ids)


def test_describe_resolves_hand_indices_to_card_ids():
    obs = observation_with([1079, 1086, 1182])
    action = describe([1], obs.select, obs.current, me=0)
    assert action is not None and action.resolved
    assert action.selection == (1,)
    # 意味表現の中にカードIDが入っている(位置ではなくカードで照合できる)。
    assert 1086 in action.descriptors[0]


@pytest.mark.parametrize(
    "selection",
    [[], [0, 0], [5], [-1], "0", [True]],
)
def test_illegal_selections_are_rejected(selection):
    obs = observation_with([1079, 1086])
    assert not is_legal_selection(selection, obs.select)
    assert describe(selection, obs.select, obs.current, me=0) is None


def test_validate_accepts_the_same_observation():
    obs = observation_with([1079, 1086])
    action = describe([0], obs.select, obs.current, me=0)
    assert validate(action, obs.select, obs.current, me=0) == [0]


def test_validate_rejects_a_changed_state():
    """状態不一致(同じ index が別のカードを指す)なら実行しない。"""
    searched = observation_with([1079, 1086])
    action = describe([0], searched.select, searched.current, me=0)
    changed = observation_with([1182, 1086])  # index 0 が別のカードになった
    assert validate(action, changed.select, changed.current, me=0) is None


def test_validate_requires_same_observation_for_unresolved_options():
    # deck listing が無いのに DECK を指す選択肢 = 意味へ解決できない。
    hand = [b.card(1079, 1)]
    me = b.player(hand=hand)
    opp = b.opponent()
    options = [b.Option(type=b.OptionType.CARD, area=b.AreaType.DECK, index=0, playerIndex=0)]
    obs = b.observation(
        b.select(options, context=b.SelectContext.TO_HAND), b.state(me=me, opp=opp)
    )
    action = describe([0], obs.select, obs.current, me=0)
    assert action is not None and not action.resolved
    assert validate(action, obs.select, obs.current, me=0) is None
    assert validate(action, obs.select, obs.current, me=0, same_observation=True) == [0]


def test_reindex_maps_semantics_to_a_new_index():
    searched = observation_with([1079, 1086])
    action = describe([0], searched.select, searched.current, me=0)  # 1079 を出す
    reordered = observation_with([1086, 1079])                        # 手札の並びが逆
    assert reindex(action, reordered.select, reordered.current, me=0) == [1]


def test_reindex_refuses_when_ambiguous():
    """同じ意味の選択肢が複数ある場合は解決しない(UNKNOWN へ送る)。"""
    searched = observation_with([1079, 1086])
    action = describe([0], searched.select, searched.current, me=0)
    duplicated = observation_with([1079, 1079])
    assert reindex(action, duplicated.select, duplicated.current, me=0) is None


def test_reindex_refuses_when_missing():
    searched = observation_with([1079, 1086])
    action = describe([0], searched.select, searched.current, me=0)
    without = observation_with([1086, 1182])
    assert reindex(action, without.select, without.current, me=0) is None


def test_multi_select_actions_keep_all_indices():
    obs = b.simple_observation([1079, 1086, 1182])
    obs.select.minCount = 2
    obs.select.maxCount = 2
    action = describe([0, 2], obs.select, obs.current, me=0)
    assert action is not None
    assert action.selection == (0, 2)
    assert len(action.descriptors) == 2
    assert validate(action, obs.select, obs.current, me=0) == [0, 2]


def test_engine_action_is_immutable():
    action = EngineAction((0,), ((1,),), True)
    with pytest.raises(Exception):
        action.selection = (1,)  # type: ignore[misc]
