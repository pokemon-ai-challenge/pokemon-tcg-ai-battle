from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_ability_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """特性を使う候補を返す。"""
    if buckets.ability:
        return MainActionProposal(
            action=[buckets.ability[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["ability"],
            label="ability",
        )
    return None
