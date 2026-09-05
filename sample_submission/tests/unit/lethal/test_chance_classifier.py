"""C/M/S 分類が「能力ベース」であることの単体テスト(規則 R3)。

最重要の観点は次の2つ:

- **SHUFFLE ログが無いだけでは C(完全列挙可能)にしない**。
  再生一致検査(R3)を通していない事象は ``UNCLASSIFIED`` になる。
- **SHUFFLE ログは S へ倒す方向にだけ使う**。再生が決定的でも、
  経路にシャッフルがあれば供給順序で outcome を指定できないので C にしない。

加えて、未解決ブロッカー(B4 相手選択 / B8 多コイン / B9 サイド取得)を
「実装する」のではなく「検出して UNSUPPORTED_EFFECT へ送る」ことを確認する。
"""

from cg.api import SelectContext

from ptcg_ai.search.lethal.chance import classify_pending_chance, detect_unsupported_events
from ptcg_ai.search.lethal.engine import DRAW, PRIZE, StepEvents
from ptcg_ai.search.lethal.types import ChanceClass, StopReason

from tests.unit.lethal import builders as b


def coin_select():
    return b.select([], context=SelectContext.COIN_HEAD, min_count=1, max_count=1)


def main_select():
    return b.select([], context=SelectContext.MAIN)


def test_unverified_event_is_never_controlled():
    # 中核: ログ上シャッフルが無くても、R3 未検査なら C と呼ばない。
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=False,
        deck_order_controlled=True,
        shuffle_on_path=False,
        replay_deterministic=None,
    )
    assert result.chance_class is ChanceClass.UNCLASSIFIED
    assert result.stop_reason is StopReason.REPLAY_NOT_VERIFIED
    assert not result.is_enumerable


def test_replay_verified_event_is_controlled():
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=False,
        deck_order_controlled=True,
        shuffle_on_path=False,
        replay_deterministic=True,
    )
    assert result.chance_class is ChanceClass.CONTROLLED_SUPPLY_ORDER
    assert result.is_enumerable
    assert "replay_verified" in result.evidence


def test_replay_mismatch_is_engine_random():
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=False,
        deck_order_controlled=True,
        shuffle_on_path=False,
        replay_deterministic=False,
    )
    assert result.chance_class is ChanceClass.ENGINE_RANDOM
    assert result.stop_reason is StopReason.REPLAY_MISMATCH
    assert not result.is_enumerable


def test_shuffle_forces_engine_random_even_when_replay_is_deterministic():
    # SHUFFLE は「S へ倒す」方向にだけ効く。再生が決定的でも C にしない。
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=False,
        deck_order_controlled=True,
        shuffle_on_path=True,
        replay_deterministic=True,
    )
    assert result.chance_class is ChanceClass.ENGINE_RANDOM
    assert result.stop_reason is StopReason.SHUFFLE_ENCOUNTERED
    assert "shuffle_on_path" in result.evidence


def test_deck_revealed_at_root_is_engine_random():
    # B1: 実デッキが使われている局面では供給順序が効かない。
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=False,
        deck_order_controlled=False,
        shuffle_on_path=False,
        replay_deterministic=True,
    )
    assert result.chance_class is ChanceClass.ENGINE_RANDOM
    assert result.stop_reason is StopReason.DECK_REVEALED_AT_ROOT


def test_coin_head_select_is_explicit_choice():
    result = classify_pending_chance(
        select=coin_select(),
        manual_coin=True,
        deck_order_controlled=True,
        shuffle_on_path=True,      # シャッフル後でもコインは指定できる
        replay_deterministic=None,  # 再生検査は不要
    )
    assert result.chance_class is ChanceClass.CONTROLLED_EXPLICIT_CHOICE
    assert result.is_enumerable


def test_coin_select_without_manual_coin_is_not_controllable():
    result = classify_pending_chance(
        select=coin_select(),
        manual_coin=False,
        deck_order_controlled=True,
        shuffle_on_path=False,
        replay_deterministic=None,
    )
    assert result.chance_class is not ChanceClass.CONTROLLED_EXPLICIT_CHOICE


def test_opponent_choice_during_our_turn_is_unsupported():
    # B4: 実装せず、検出して止める。
    result = classify_pending_chance(
        select=main_select(),
        manual_coin=True,
        deck_order_controlled=True,
        shuffle_on_path=False,
        replay_deterministic=True,
        acting_is_me=False,
        turn_ended=False,
    )
    assert result.chance_class is ChanceClass.OPPONENT_CHOICE
    assert result.stop_reason is StopReason.UNSUPPORTED_EFFECT
    assert not result.is_enumerable


def test_multi_coin_per_select_is_detected():
    # B8: 1つの COIN_HEAD 選択で2枚以上のコインが出たら列挙不能とする。
    reason, tags = detect_unsupported_events(
        StepEvents(sequence=("coin", "coin"), coin_heads=(True, True)),
        answered_coin_select=True,
    )
    assert reason is StopReason.UNSUPPORTED_EFFECT
    assert "multi_coin_per_select" in tags


def test_single_coin_per_select_is_fine():
    reason, tags = detect_unsupported_events(
        StepEvents(sequence=("coin",), coin_heads=(True,)), answered_coin_select=True
    )
    assert reason is None and tags == ()


def test_prize_take_is_detected_as_unverified():
    # B9: サイド取得の可制御性は未検証なので、確定探索には通さない。
    reason, tags = detect_unsupported_events(StepEvents(sequence=(PRIZE,)))
    assert reason is StopReason.UNSUPPORTED_EFFECT
    assert "prize_taken_control_unverified" in tags


def test_plain_draw_is_supported():
    reason, tags = detect_unsupported_events(StepEvents(sequence=(DRAW,), drawn=(1079,)))
    assert reason is None and tags == ()
