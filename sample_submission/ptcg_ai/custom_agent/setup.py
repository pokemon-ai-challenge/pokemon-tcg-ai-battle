"""対戦開始時の初期配置・先攻後攻・マリガンの選択。"""

from __future__ import annotations

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import constants, matchup
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import card_cache

_PREFER_GOING_FIRST = True


def handle(obs: Observation) -> list[int]:
    context = obs.select.context
    if context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        # 対アーキタイプ加点(matchup)はベンチに出す場面(SETUP_BENCH_POKEMON)にのみ適用する。
        # シェイミのようなベンチ運用前提のポケモンへの加点が、誤ってバトル場に出す判断に
        # 流用されないようにするための線引き。
        allow_matchup_boost = context == SelectContext.SETUP_BENCH_POKEMON
        return _choose_by_priority(obs, allow_matchup_boost)
    if context == SelectContext.IS_FIRST:
        return _choose_yes_no(obs, prefer_yes=_PREFER_GOING_FIRST)
    if context == SelectContext.MULLIGAN:
        return _choose_yes_no(obs, prefer_yes=not _has_basic_pokemon_in_hand(obs))
    return fallback.safe_choice(obs)


def _choose_by_priority(obs: Observation, allow_matchup_boost: bool) -> list[int]:
    state = obs.current

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        base = 0.0
        if card_id in constants.POKEMON_PRIORITY:
            base = float(len(constants.POKEMON_PRIORITY) - constants.POKEMON_PRIORITY.index(card_id))
        if allow_matchup_boost and card_id is not None:
            base += matchup.card_priority_boost(card_id)
        return base

    return common.pick_top(obs.select, score)


def _choose_yes_no(obs: Observation, prefer_yes: bool) -> list[int]:
    target_type = OptionType.YES if prefer_yes else OptionType.NO
    for i, option in enumerate(obs.select.option):
        if option.type == target_type:
            return [i]
    return fallback.safe_choice(obs)


def _has_basic_pokemon_in_hand(obs: Observation) -> bool:
    player = obs.current.players[obs.current.yourIndex]
    hand = player.hand or []
    return any(card_cache.get_card(card.id).basic for card in hand)
