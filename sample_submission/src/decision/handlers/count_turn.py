from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_count_action(obs: Observation) -> list[int]:
    """枚数や個数などの数値選択を処理する。"""
    # TODO:
    # - DRAW_COUNT で、山札切れや手札枚数を見ながら適切な枚数を選ぶ
    # - DAMAGE_COUNTER_COUNT で、必要十分な打点になる個数を選ぶ
    # - REMOVE_DAMAGE_COUNTER_COUNT で、きぜつ回避や主力維持を優先して個数を選ぶ
    return choose_random_legal_action(obs)
