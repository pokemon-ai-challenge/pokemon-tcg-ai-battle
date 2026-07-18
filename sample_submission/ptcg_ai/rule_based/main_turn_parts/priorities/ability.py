"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: ability（特性の使用）

特性の使いどころを提案する。特性は Attack とは別物（cg.api の CardData.skills に対応）なので、
knowledge.profile_registry.get_pokemon_profile(card_id) が返す PokemonProfile の
has_ability / ability_category / ability_priority を参照する。
"""

from cg.api import Observation, OptionType

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry


def propose(obs: Observation) -> ActionProposal | None:
    """特性使用の行動を1つ提案する。該当行動が無ければ None。"""
    best_index: int | None = None
    best_priority = 0.0
    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ABILITY:
            continue
        # ABILITY の Option は cardId を持たない（cg/api.py: area/index のみ）ため、
        # resolve_card_id で場のポケモンの card_id を引く。
        card_id = common.resolve_card_id(option, obs.current)
        if card_id is None:
            continue
        profile = profile_registry.get_pokemon_profile(card_id)
        if profile is None or not profile.has_ability:
            continue
        if profile.ability_priority > best_priority:
            best_priority = profile.ability_priority
            best_index = i

    if best_index is None:
        return None
    return ActionProposal(category="ability", select=[best_index], score=best_priority, reason="use ability")
