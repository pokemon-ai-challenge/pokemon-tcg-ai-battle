"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_HAND, TO_DECK, TO_DECK_BOTTOM

手札に加える、山札（上/下）に戻すカードを選ぶ。サーチで最初に探すべきカードの
優先順位は knowledge.profile_registry.get_deck_plan().search_priority を参照する。
"""

from cg.api import SelectData, State

from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """TO_HAND / TO_DECK / TO_DECK_BOTTOM の選択肢インデックスを返す。"""
    raise NotImplementedError
