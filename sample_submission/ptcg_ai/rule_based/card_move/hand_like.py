"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_HAND, TO_DECK, TO_DECK_BOTTOM

手札に加える、山札（上/下）に戻すカードを選ぶ。サーチで最初に探すべきカードの
優先順位は knowledge.profile_registry.get_deck_plan().search_priority を参照する。
"""

from cg.api import SelectContext, SelectData, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """TO_HAND / TO_DECK / TO_DECK_BOTTOM の選択肢インデックスを返す。"""
    plan = profile_registry.get_deck_plan()

    def priority_rank(option) -> int:
        card_id = common.resolve_card_id(option, state)
        if card_id in plan.search_priority:
            return plan.search_priority.index(card_id)
        return len(plan.search_priority)

    if select.context == SelectContext.TO_HAND:
        # 優先度の高い（rankが小さい）カードを優先して手札に加える。
        return common.pick_top(select, lambda option: -priority_rank(option))

    # TO_DECK / TO_DECK_BOTTOM: 優先度の低いカードから山札に戻す。
    return common.pick_top(select, lambda option: priority_rank(option))
