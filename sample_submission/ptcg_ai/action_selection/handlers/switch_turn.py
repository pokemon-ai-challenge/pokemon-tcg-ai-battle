"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SWITCH, TO_ACTIVE

バトル場のポケモンを入れ替える場面（にげる後の交代、バトル場が空いたときの繰り出しなど）。
どのポケモンに交代するかの評価は decision.evaluation.switch_eval に委譲する。
"""

from cg.api import Observation

from ptcg_ai.board_evaluation import switch_eval


def handle(obs: Observation) -> list[int]:
    """SWITCH / TO_ACTIVE の選択肢から、switch_eval のスコアが最も高い候補を選ぶ。"""
    raise NotImplementedError
