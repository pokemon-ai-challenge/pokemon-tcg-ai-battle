from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_retreat_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """まだ簡易条件だが、必要そうならにげる候補を返す。"""
    if not should_retreat(obs, buckets):
        return None

    return MainActionProposal(
        action=[buckets.retreat[0]],
        score=MAIN_ACTION_BASE_WEIGHTS["retreat"],
        label="retreat",
    )


def should_retreat(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> bool:
    """にげる候補を出すかどうかの簡易条件。"""
    if obs.current is None or not buckets.retreat:
        return False
    if buckets.attack:
        return False
    return True
