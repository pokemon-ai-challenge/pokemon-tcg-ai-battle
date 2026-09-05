"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON, IS_FIRST, MULLIGAN

対戦開始時の初期配置・先攻後攻・マリガン（引き直し）に関する選択。
バトル場/ベンチに出す Basic ポケモンの優先順位は
knowledge.profile_registry.get_deck_plan().opening_priority を参照する。
"""

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.rule_based.card_move import bench_field
from ptcg_ai.shared import card_cache

# 先攻には「相手より先に展開・攻撃できる」というテンポ上の利点があるため、既定では先攻を選ぶ。
_PREFER_GOING_FIRST = True


def handle(obs: Observation) -> list[int]:
    """SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON / IS_FIRST / MULLIGAN を振り分けて処理する。"""
    context = obs.select.context
    if context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        # 初期配置の優先順位も opening_priority を使う（手札/山札からの展開という点で同じ判断）。
        return bench_field.choose(obs.select, obs.current)
    if context == SelectContext.IS_FIRST:
        return _choose_yes_no(obs, prefer_yes=_PREFER_GOING_FIRST)
    if context == SelectContext.MULLIGAN:
        # 手札に Basic ポケモンが無ければ引き直す。
        return _choose_yes_no(obs, prefer_yes=not _has_basic_pokemon_in_hand(obs))
    return fallback.safe_choice(obs)


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
