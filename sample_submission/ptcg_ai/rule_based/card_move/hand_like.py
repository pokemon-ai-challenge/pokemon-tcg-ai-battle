"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_HAND, TO_DECK, TO_DECK_BOTTOM

手札に加える、山札（上/下）に戻すカードを選ぶ。サーチで最初に探すべきカードの
優先順位は knowledge.profile_registry.get_deck_plan().search_priority を参照する。
盤面状態に応じて優先順位を変えたい場合（例: トウコの進化ポケモン選択）は
search_priority_rules（条件付き、search_priority より優先）を使う。
"""

from cg.api import SelectContext, SelectData, State

from ptcg_ai.board_evaluation import usage_context
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def _resolve_search_order(state: State) -> list[int]:
    """search_priority_rules の最初に条件を満たしたルールの order を先頭に、
    search_priority（無条件）を後ろに連結した優先順を返す。
    """
    plan = profile_registry.get_deck_plan()
    ctx = usage_context.build_usage_context(state, state.yourIndex)
    for rule in plan.search_priority_rules:
        if rule.condition(ctx):
            return rule.order + [card_id for card_id in plan.search_priority if card_id not in rule.order]
    return plan.search_priority


def choose(select: SelectData, state: State) -> list[int]:
    """TO_HAND / TO_DECK / TO_DECK_BOTTOM の選択肢インデックスを返す。"""
    order = _resolve_search_order(state)

    def priority_rank(option) -> int:
        card_id = common.resolve_card_id(option, state)
        if card_id in order:
            return order.index(card_id)
        return len(order)

    if select.context == SelectContext.TO_HAND:
        # 優先度の高い（rankが小さい）カードを優先して手札に加える。
        return common.pick_top(select, lambda option: -priority_rank(option))

    # TO_DECK / TO_DECK_BOTTOM: 優先度の低いカードから山札に戻す。
    return common.pick_top(select, lambda option: priority_rank(option))
