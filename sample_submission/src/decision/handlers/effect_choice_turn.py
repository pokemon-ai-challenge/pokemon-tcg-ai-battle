from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_effect_choice_action(obs: Observation) -> list[int]:
    """効果の順番や追加効果の選択を処理する。"""
    # TODO:
    # - SKILL_ORDER で、情報が増える順や失敗しにくい順に処理を並べる
    # - DISABLE_ATTACK で、相手の次ターンの最大打点や主要プランを止めるワザを選ぶ
    return choose_random_legal_action(obs)
