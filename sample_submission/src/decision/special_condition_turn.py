from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_special_condition_action(obs: Observation) -> list[int]:
    """状態異常の付与先や回復対象を選ぶ。"""
    # TODO:
    # - AFFECT_SPECIAL_CONDITION で、相手の行動を最も止めやすい状態異常を選ぶ
    # - RECOVER_SPECIAL_CONDITION で、自分の主力が動けるようになる回復を優先する
    return choose_random_legal_action(obs)
