from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_attack_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """攻撃できるなら高めの重みで候補にする。"""
    if buckets.attack:
        return MainActionProposal(
            action=[buckets.attack[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["attack"],
            label="attack",
        )
    return None
