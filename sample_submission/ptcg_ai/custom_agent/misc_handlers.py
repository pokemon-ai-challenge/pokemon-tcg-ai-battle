"""ユーザー仕様に明記が無い SelectContext 向けの単純なデフォルト処理。

なるべく単純で説明可能なヒューリスティックに留める（判断材料が乏しい場面の安全な既定値）。
"""

from __future__ import annotations

from cg.api import Observation, OptionType, SelectContext, SpecialConditionType

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import board_features
from ptcg_ai.custom_agent import constants, items, matchup, supporters
from ptcg_ai.rule_based.card_move import common

_DISCARD_LIKE_CONTEXTS = (
    SelectContext.DISCARD,
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    SelectContext.DISCARD_ENERGY,
)

_DAMAGE_CONTEXTS = (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY)
_SUPPORT_CONTEXTS = (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER)

_AFFECT_SPECIAL_CONDITION_PRIORITY: dict[SpecialConditionType, int] = {
    SpecialConditionType.PARALYZE: 4,
    SpecialConditionType.SLEEP: 3,
    SpecialConditionType.CONFUSE: 2,
    SpecialConditionType.POISON: 1,
    SpecialConditionType.BURN: 0,
}

_DEFAULT_YES_CONTEXTS = (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT, SelectContext.COIN_HEAD)


def _fallback_search_order(select, state) -> list[int]:
    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in constants.SEARCH_FALLBACK_PRIORITY:
            return float(len(constants.SEARCH_FALLBACK_PRIORITY) - constants.SEARCH_FALLBACK_PRIORITY.index(card_id))
        return 0.0

    return common.pick_top(select, score)


def handle_hand_like(obs: Observation) -> list[int]:
    """TO_HAND / TO_DECK / TO_DECK_BOTTOM。まずグッズ/サポート固有のロジックに委譲する。"""
    state = obs.current
    select = obs.select

    item_target = items.choose_target(select, state)
    if item_target is not None:
        return item_target

    effect = select.effect
    if effect is not None:
        if effect.id == constants.TOUKO:
            return supporters.choose_touko_target(select, state)
        if effect.id == constants.HIKARI:
            return supporters.choose_hikari_target(select, state)
        if effect.id == constants.LANAS_AID:
            return supporters.choose_lanas_aid_target(select, state)

    if select.context == SelectContext.TO_HAND:
        return _fallback_search_order(select, state)
    # TO_DECK / TO_DECK_BOTTOM: 優先度の低いものから戻す。
    order = constants.SEARCH_FALLBACK_PRIORITY

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in order:
            return float(order.index(card_id))
        return float(len(order))

    return common.pick_top(select, score)


def handle_bench_field(obs: Observation) -> list[int]:
    """TO_BENCH / TO_FIELD（SETUP系以外）。まずグッズ固有のロジック(なかよしポフィン等)に委譲する。"""
    state = obs.current
    select = obs.select
    item_target = items.choose_target(select, state)
    if item_target is not None:
        return item_target

    # 対アーキタイプ加点(matchup)はベンチに出す場面(TO_BENCH)にのみ適用する
    # (setup.py 参照: シェイミがバトル場に出す判断に流用されないようにするため)。
    allow_matchup_boost = select.context == SelectContext.TO_BENCH

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        base = 0.0
        if card_id in constants.POKEMON_PRIORITY:
            base = float(len(constants.POKEMON_PRIORITY) - constants.POKEMON_PRIORITY.index(card_id))
        if allow_matchup_boost and card_id is not None:
            base += matchup.card_priority_boost(card_id)
        return base

    return common.pick_top(select, score)


def handle_discard_like(obs: Observation) -> list[int]:
    state = obs.current

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        return 0.0 if card_id in constants.PROTECTED_CARD_IDS else 1.0

    return common.pick_top(obs.select, score)


def handle_to_prize_or_look(obs: Observation) -> list[int]:
    state = obs.current
    select = obs.select
    if select.context == SelectContext.TO_PRIZE:
        return common.pick_top(
            select, lambda option: 0.0 if common.resolve_card_id(option, state) in constants.PROTECTED_CARD_IDS else 1.0
        )
    return _fallback_search_order(select, state)


def handle_not_move_or_effect_target(obs: Observation) -> list[int]:
    return common.pick_top(obs.select, lambda option: 0.0)


def handle_damage_target(obs: Observation) -> list[int]:
    context = obs.select.context
    if context in _DAMAGE_CONTEXTS:
        return _choose_damage_target(obs)
    if context in _SUPPORT_CONTEXTS:
        return _choose_support_target(obs)
    return fallback.safe_choice(obs)


def _choose_damage_target(obs: Observation) -> list[int]:
    state = obs.current

    def score(option) -> float:
        pokemon = common.resolve_pokemon(option, state)
        if pokemon is None:
            return float("-inf")
        remaining_hp_ratio = pokemon.hp / pokemon.maxHp if pokemon.maxHp else 1.0
        return board_features.attacker_score(pokemon) + (1.0 - remaining_hp_ratio)

    return common.pick_top(obs.select, score)


def _choose_support_target(obs: Observation) -> list[int]:
    state = obs.current

    def score(option) -> float:
        pokemon = common.resolve_pokemon(option, state)
        if pokemon is None:
            return float("-inf")
        damage_taken = max(0, pokemon.maxHp - pokemon.hp)
        if damage_taken == 0:
            return float("-inf")
        return board_features.attacker_score(pokemon) * 10.0 + damage_taken

    return common.pick_top(obs.select, score)


def handle_count(obs: Observation) -> list[int]:
    """DRAW_COUNT / DAMAGE_COUNTER_COUNT / REMOVE_DAMAGE_COUNTER_COUNT: 常に最大値を選ぶ。"""
    best_index = None
    best_number = None
    for i, option in enumerate(obs.select.option):
        if option.number is None:
            continue
        if best_number is None or option.number > best_number:
            best_number = option.number
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]


def handle_special_condition(obs: Observation) -> list[int]:
    if obs.select.context == SelectContext.AFFECT_SPECIAL_CONDITION:
        best_index = None
        best_priority = -1
        for i, option in enumerate(obs.select.option):
            if option.specialConditionType is None:
                continue
            priority = _AFFECT_SPECIAL_CONDITION_PRIORITY.get(option.specialConditionType, 0)
            if priority > best_priority:
                best_priority = priority
                best_index = i
        if best_index is None:
            return fallback.safe_choice(obs)
        return [best_index]
    return fallback.safe_choice(obs)


def handle_effect_choice(obs: Observation) -> list[int]:
    """SKILL_ORDER / ACTIVATE / FIRST_EFFECT / COIN_HEAD。"""
    if obs.select.context in _DEFAULT_YES_CONTEXTS:
        for i, option in enumerate(obs.select.option):
            if option.type == OptionType.YES:
                return [i]
        return fallback.safe_choice(obs)
    return fallback.safe_choice(obs)


def handle_yes_no_fallback(obs: Observation) -> list[int]:
    return fallback.safe_choice(obs)
