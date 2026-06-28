from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_yes_no_action(obs: Observation) -> list[int]:
    """Yes / No 系の分岐を処理する。"""
    # TODO:
    # - IS_FIRST で、デッキ速度や先攻後攻の相性を見て選ぶ
    # - MULLIGAN で、初手品質と事故率を見て引き直しを判断する
    # - ACTIVATE / FIRST_EFFECT / COIN_HEAD で、期待値が高い側を選ぶ
    return choose_random_legal_action(obs)
