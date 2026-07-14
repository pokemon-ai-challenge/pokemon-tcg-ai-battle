"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_BENCH, TO_FIELD, TO_ACTIVE

手札や山札からベンチ/バトル場に出すカードを選ぶ。優先順位は
knowledge.profile_registry.get_deck_plan() の opening_priority / evolution_priority を参照する。
"""

from cg.api import SelectData, State

from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """ベンチ/バトル場に出すポケモンの選択肢インデックスを返す。"""
    raise NotImplementedError
