from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_evolution_action(obs: Observation) -> list[int]:
    """進化・退化の対象を選ぶ。"""
    # TODO:
    # - EVOLVES_FROM / EVOLVES_TO / EVOLVE で、最も価値の高い進化ラインを選ぶ
    # - DEVOLVE / MORE_DEVOLVE で、相手への妨害価値や自分の再利用価値を見て選ぶ
    # - 進化後に攻撃可能か、耐久が上がるか、特性が有効かも判断材料に入れる
    return choose_random_legal_action(obs)
