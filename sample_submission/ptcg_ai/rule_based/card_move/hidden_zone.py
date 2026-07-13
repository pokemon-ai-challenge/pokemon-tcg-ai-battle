"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: LOOK, TO_PRIZE

山札の上からのぞき見（LOOK）や、サイドに送るカード（TO_PRIZE）の選択。
サイドに送る際は「守りたいカード」(knowledge.profile_registry.get_deck_plan().protected_card_ids)
を優先的に避ける。
"""

from cg.api import SelectContext, SelectData, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """LOOK / TO_PRIZE の選択肢インデックスを返す。"""
    plan = profile_registry.get_deck_plan()

    if select.context == SelectContext.TO_PRIZE:
        # 守りたいカード（protected_card_ids）はサイドに送る候補から避ける。
        return common.pick_top(
            select, lambda option: 0 if common.resolve_card_id(option, state) in plan.protected_card_ids else 1
        )

    # LOOK: 優先して見ておきたい（search_priorityが高い）カードを選ぶ。
    def priority_rank(option) -> int:
        card_id = common.resolve_card_id(option, state)
        if card_id in plan.search_priority:
            return plan.search_priority.index(card_id)
        return len(plan.search_priority)

    return common.pick_top(select, lambda option: -priority_rank(option))
