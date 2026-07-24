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
    EnergyPriorityRule,
    ItemProfile,
    PokemonProfile,
    SearchPriorityRule,
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
        search_priority_rules=list(getattr(plan, "SEARCH_PRIORITY_RULES", [])),
        protected_card_ids={entry.card_id for entry in plan.PROTECT_CARDS},
        # win_condition_by_prize: 担当Aのデータは "6〜4枚（序盤）" のような自由記述の
        # サイド枚数レンジで持っており、DeckPlan の dict[int, str]（サイド枚数->文字列）
        # へは機械的に変換できない。現状どの判断ロジックもこのフィールドを参照していないため
        # 空のままにしておく（必要になれば plan.WIN_CONDITIONS_BY_PRIZE を直接使う専用の
        # アクセサを別途用意する）。
        win_condition_by_prize={},
        energy_recycle_target_ids=frozenset(getattr(plan, "ENERGY_RECYCLE_TARGET_CARD_IDS", frozenset())),
        energy_recycle_card_id=getattr(plan, "ENERGY_RECYCLE_CARD_ID", None),
        energy_recycle_backup_item_id=getattr(plan, "ENERGY_RECYCLE_BACKUP_ITEM_ID", None),
        reserved_bench_line_ids=frozenset(getattr(plan, "RESERVED_BENCH_LINE_IDS", frozenset())),
        reserved_bench_slots=getattr(plan, "RESERVED_BENCH_SLOTS_FOR_DRAW_ENGINE", 0),
        reserved_bench_basic_id=getattr(plan, "RESERVED_BENCH_BASIC_ID", None),
    )


def reset_deck_plan_cache() -> None:
    """テスト用: DeckPlan のキャッシュを破棄する（通常の対戦では不要）。"""
    global _deck_plan_cache
    _deck_plan_cache = None


def get_opponent_effect_lock_energy_ids() -> frozenset[int]:
    """decks.active.deck_plan.OPPONENT_EFFECT_LOCK_ENERGY_IDS を返す。

    相手の「ワザの効果を無効化する特殊エネルギー」（ミストエネルギー等）の card_id 集合。
    改造ハンマーの破壊対象選択で優先的に壊すために使う。デッキが未定義なら空集合。
    """
    return frozenset(getattr(active.deck_plan, "OPPONENT_EFFECT_LOCK_ENERGY_IDS", frozenset()))


def get_ko_replacement_priority() -> list[int]:
    """decks.active.deck_plan.KO_REPLACEMENT_PRIORITY の card_id 列を返す。

    同じ card_id が複数回登場しうる（例: 同じポケモンでもエネルギー充足状況で
    優先度を分けたい場合）。呼び出し側でその重複をどう解決するかを判断する。
    デッキがこのデータを持たない場合は空リスト。
    """
    entries = getattr(active.deck_plan, "KO_REPLACEMENT_PRIORITY", [])
    return [entry.card_id for entry in entries]


def get_energy_required_count(card_id: int) -> int | None:
    """decks.active.deck_plan.ENERGY_REQUIRED_COUNT から、そのポケモンが攻撃に必要とする
    エネルギー総数の上限を引く（無ければ None = 上限不明）。
    """
    table: dict[int, int] = getattr(active.deck_plan, "ENERGY_REQUIRED_COUNT", {})
    return table.get(card_id)


def get_energy_card_priority_rules() -> list[EnergyPriorityRule]:
    """decks.active.deck_plan.ENERGY_CARD_PRIORITY_RULES をそのまま返す。

    各ルールの condition(EnergyCardContext) -> bool を先頭から順に試し、
    最初に True になったルールの order（card_id の優先順）を使う。
    """
    return list(getattr(active.deck_plan, "ENERGY_CARD_PRIORITY_RULES", []))


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
