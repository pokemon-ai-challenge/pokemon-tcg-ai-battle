from cg.api import Observation, SelectContext

from src.decision.card_move.bench_field import choose_bench_or_field_action
from src.decision.card_move.discard import choose_discard_action as choose_discard_action_impl
from src.decision.card_move.hand_like import choose_hand_like_action
from src.decision.card_move.hidden_zone import choose_hidden_zone_action
from src.decision.card_move.not_move_or_look import (
    choose_not_move_or_look_action as choose_not_move_or_look_action_impl,
)
from src.decision.fallback import choose_random_legal_action


def choose_to_bench_or_field_action(obs: Observation) -> list[int]:
    """自分の場に出す候補を専用評価へ渡す。"""
    return choose_bench_or_field_action(obs)


def choose_to_hand_like_action(obs: Observation) -> list[int]:
    """TO_HAND の対象選択を専用ロジックへ渡す。"""
    return choose_hand_like_action(obs)


def choose_discard_action(obs: Observation) -> list[int]:
    """DISCARD はまだ安全側のフォールバックを使う。"""
    return choose_discard_action_impl(obs)


def choose_not_move_or_look_action(obs: Observation) -> list[int]:
    """NOT_MOVE / LOOK の選択を専用ロジックへ渡す。"""
    return choose_not_move_or_look_action_impl(obs)


def choose_card_move_action(obs: Observation) -> list[int]:
    """card move 系の context を対応する処理へ振り分ける。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during card move decisions.")

    context = obs.select.context

    if context in {
        SelectContext.TO_BENCH,
        SelectContext.TO_FIELD,
    }:
        return choose_to_bench_or_field_action(obs)

    if context == SelectContext.TO_HAND:
        return choose_to_hand_like_action(obs)

    if context in {
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_PRIZE,
    }:
        return choose_hidden_zone_action(obs)

    if context == SelectContext.DISCARD:
        return choose_discard_action(obs)

    if context in {
        SelectContext.NOT_MOVE,
        SelectContext.LOOK,
    }:
        return choose_not_move_or_look_action(obs)

    return choose_random_legal_action(obs)
