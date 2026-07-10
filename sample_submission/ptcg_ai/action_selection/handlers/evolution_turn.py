"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: EVOLVE, EVOLVES_FROM, EVOLVES_TO, DEVOLVE, MORE_DEVOLVE

進化・退化に関する選択。どのポケモンをどの順で進化させるかの優先順位は
knowledge.profile_registry.get_deck_plan().evolution_priority を参照する。
"""

from cg.api import Observation

from ptcg_ai.shared import profile_registry


def handle(obs: Observation) -> list[int]:
    """EVOLVE 系の選択肢から進化元・進化先の組み合わせを決めて返す。"""
    raise NotImplementedError
