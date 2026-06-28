from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_energy_tool_action(obs: Observation) -> list[int]:
    """エネルギーやどうぐの付け替え・回収・トラッシュ先を選ぶ。"""
    # TODO:
    # - ATTACH_FROM / ATTACH_TO で、主力アタッカーに必要なエネルギーを優先して集める
    # - DETACH_FROM / DISCARD_ENERGY / DISCARD_TOOL_CARD で、失っても影響が小さい札を選ぶ
    # - TO_HAND_ENERGY / TO_DECK_ENERGY / SWITCH_ENERGY 系で、次ターンの攻撃準備が進むように選ぶ
    return choose_random_legal_action(obs)
