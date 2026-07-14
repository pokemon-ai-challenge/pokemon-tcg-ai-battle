"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: ability（特性の使用）

特性の使いどころを提案する。特性は Attack とは別物（cg.api の CardData.skills に対応）なので、
knowledge.profile_registry.get_pokemon_profile(card_id) が返す PokemonProfile の
has_ability / ability_category / ability_priority を参照する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry


def propose(obs: Observation) -> ActionProposal | None:
    """特性使用の行動を1つ提案する。該当行動が無ければ None。"""
    raise NotImplementedError
