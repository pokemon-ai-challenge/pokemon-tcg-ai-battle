"""クラスタ⑤ カード知識アクセス（引き方）／担当B

プロファイルを引くための共通関数。中身は decks.active（担当Aのデータ）を参照するだけで、
判断ロジックはここには持たない。担当Bの他モジュールは、カードIDが必要な場面で
必ずこの関数群を経由する（ptcg_ai/action_selection・ptcg_ai/rule_based 配下から decks/ を直接 import しない）。
"""

from decks import active
from ptcg_ai.shared.profile_types import (
    AttackProfile,
    DeckPlan,
    EnergyProfile,
    ItemProfile,
    PokemonProfile,
    StadiumProfile,
    SupporterProfile,
    ToolProfile,
)

_deck_plan_cache: DeckPlan | None = None


def get_deck_plan() -> DeckPlan:
    """decks.active.deck_plan（担当Aのデータ）から DeckPlan を組み立てて返す。

    担当Aの decks/new_deck/deck_plan.py は、優先度に「なぜそうするか」の理由メモを
    添えた独自のデータクラス（AttackerPlan/PriorityEntry/PrizeStageWinCondition など）で
    書かれている。担当Bの判断ロジックが必要とするのは card_id の列/集合だけなので、
    ここで DeckPlan（ptcg_ai.shared.profile_types）の形に正規化してから返す。
    デッキは対戦中に変わらないため、初回のみ変換して以降はキャッシュを使い回す。

    ptcg_ai/action_selection・ptcg_ai/rule_based 配下がデッキ方針
    （opening_priority, protected_card_ids など）を必要とする場合、
    decks.active を直接 import せず、必ずこの関数を経由する。
    """
    global _deck_plan_cache
    if _deck_plan_cache is None:
        _deck_plan_cache = _build_deck_plan()
    return _deck_plan_cache


def _build_deck_plan() -> DeckPlan:
    plan = active.deck_plan
    return DeckPlan(
        main_attacker_ids=[plan.MAIN_ATTACKER.card_id],
        sub_attacker_ids=[entry.card_id for entry in plan.SUB_ATTACKERS],
        opening_priority=[entry.card_id for entry in plan.OPENING_BENCH_PRIORITY],
        evolution_priority=[entry.card_id for entry in plan.EVOLUTION_PRIORITY],
        energy_priority=[entry.card_id for entry in plan.ENERGY_TARGET_PRIORITY],
        search_priority=[entry.card_id for entry in plan.SEARCH_PRIORITY],
        protected_card_ids={entry.card_id for entry in plan.PROTECT_CARDS},
        # win_condition_by_prize: 担当Aのデータは "6〜4枚（序盤）" のような自由記述の
        # サイド枚数レンジで持っており、DeckPlan の dict[int, str]（サイド枚数->文字列）
        # へは機械的に変換できない。現状どの判断ロジックもこのフィールドを参照していないため
        # 空のままにしておく（必要になれば plan.WIN_CONDITIONS_BY_PRIZE を直接使う専用の
        # アクセサを別途用意する）。
        win_condition_by_prize={},
        # matchup_plans: 対アーキタイプ戦略。担当Aがまだ書いていないデッキでは
        # plan.MATCHUP_PLANS 自体が無いこともあるため getattr で欠損時は空dictにする。
        matchup_plans=getattr(plan, "MATCHUP_PLANS", {}),
    )


def reset_deck_plan_cache() -> None:
    """テスト用: DeckPlan のキャッシュを破棄する（通常の対戦では不要）。"""
    global _deck_plan_cache
    _deck_plan_cache = None


def get_pokemon_profile(card_id: int) -> PokemonProfile | None:
    """decks.active.pokemon_profiles から card_id のプロファイルを引く。"""
    return active.pokemon_profiles.PROFILES.get(card_id)


def get_attack_profile(attack_id: int) -> AttackProfile | None:
    """decks.active.attack_profiles から attack_id のプロファイルを引く。"""
    return active.attack_profiles.PROFILES.get(attack_id)


def get_item_profile(card_id: int) -> ItemProfile | None:
    return active.item_profiles.PROFILES.get(card_id)


def get_supporter_profile(card_id: int) -> SupporterProfile | None:
    return active.supporter_profiles.PROFILES.get(card_id)


def get_tool_profile(card_id: int) -> ToolProfile | None:
    return active.tool_profiles.PROFILES.get(card_id)


def get_stadium_profile(card_id: int) -> StadiumProfile | None:
    return active.stadium_profiles.PROFILES.get(card_id)


def get_energy_profile(card_id: int) -> EnergyProfile | None:
    return active.energy_profiles.PROFILES.get(card_id)
