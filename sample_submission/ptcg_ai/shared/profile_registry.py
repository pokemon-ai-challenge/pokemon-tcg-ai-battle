"""クラスタ⑤ カード知識アクセス（引き方）／担当B

プロファイルを引くための共通関数。中身は decks.active（担当Aのデータ）を参照するだけで、
判断ロジックはここには持たない。担当Bの他モジュールは、カードIDが必要な場面で
必ずこの関数群を経由する（ptcg_ai/action_selection・ptcg_ai/rule_based 配下から decks/ を直接 import しない）。
"""

from decks.new_deck.deck_plan import DeckPlan
from ptcg_ai.shared.profile_types import (
    AttackProfile,
    EnergyProfile,
    ItemProfile,
    PokemonProfile,
    StadiumProfile,
    SupporterProfile,
    ToolProfile,
)


def get_deck_plan() -> DeckPlan:
    """decks.active.deck_plan.PLAN を返す。DeckPlan（方針データ）への唯一の参照経路。

    ptcg_ai/action_selection・ptcg_ai/rule_based 配下がデッキ方針（opening_priority, protected_card_ids など）を必要とする場合、
    decks.active を直接 import せず、必ずこの関数を経由する。
    """
    raise NotImplementedError


def get_pokemon_profile(card_id: int) -> PokemonProfile | None:
    """decks.active.pokemon_profiles から card_id のプロファイルを引く。"""
    raise NotImplementedError


def get_attack_profile(attack_id: int) -> AttackProfile | None:
    """decks.active.attack_profiles から attack_id のプロファイルを引く。"""
    raise NotImplementedError


def get_item_profile(card_id: int) -> ItemProfile | None:
    raise NotImplementedError


def get_supporter_profile(card_id: int) -> SupporterProfile | None:
    raise NotImplementedError


def get_tool_profile(card_id: int) -> ToolProfile | None:
    raise NotImplementedError


def get_stadium_profile(card_id: int) -> StadiumProfile | None:
    raise NotImplementedError


def get_energy_profile(card_id: int) -> EnergyProfile | None:
    raise NotImplementedError
