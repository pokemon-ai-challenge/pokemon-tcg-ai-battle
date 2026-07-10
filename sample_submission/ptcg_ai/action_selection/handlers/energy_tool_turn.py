"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: ATTACH_FROM, ATTACH_TO, DETACH_FROM, DISCARD_ENERGY_CARD,
DISCARD_TOOL_CARD, SWITCH_ENERGY_CARD, DISCARD_ENERGY, TO_HAND_ENERGY,
TO_DECK_ENERGY, SWITCH_ENERGY

エネルギー・ポケモンのどうぐの付け外し・入れ替えに関する選択。
どのポケモンに付けるべきかの評価は decision.evaluation.energy_requirements /
decision.main_turn_parts.energy_eval を参照する。
"""

from cg.api import Observation

from ptcg_ai.board_evaluation import energy_requirements


def handle(obs: Observation) -> list[int]:
    """ATTACH/DETACH/DISCARD/SWITCH のエネルギー・どうぐ系選択肢を処理する。"""
    raise NotImplementedError
