from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_action(obs: Observation) -> list[int]:
    """通常ターンの処理をここから呼び分ける。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during turn decisions.")
    
    return choose_random_legal_action(obs)
