"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: LOOK, TO_PRIZE

山札の上からのぞき見（LOOK）や、サイドに送るカード（TO_PRIZE）の選択。
サイドに送る際は「守りたいカード」(knowledge.profile_registry.get_deck_plan().protected_card_ids)
を優先的に避ける。
"""

from cg.api import SelectData, State

from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """LOOK / TO_PRIZE の選択肢インデックスを返す。"""
    raise NotImplementedError
