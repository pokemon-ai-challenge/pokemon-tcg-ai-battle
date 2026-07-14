"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: ATTACK, DISABLE_ATTACK

使用するワザを選ぶ場面。ダメージ計算・エネルギー充足判定は
decision.evaluation.attack_features / energy_requirements に委譲する。
リーサル（相手を倒しきれるか）判定を最優先することが多い。
"""

from cg.api import Observation

from ptcg_ai.board_evaluation import attack_features


def handle(obs: Observation) -> list[int]:
    """ATTACK / DISABLE_ATTACK の選択肢からワザを1つ選んで返す。"""
    raise NotImplementedError
