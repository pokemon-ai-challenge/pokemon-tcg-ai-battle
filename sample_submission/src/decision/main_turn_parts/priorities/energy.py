from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_energy_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """このターンまだ貼っていないなら、エネルギー貼りを候補にする。"""
    if obs.current is None:
        return None
    if obs.current.energyAttached or not buckets.attach:
        return None

    score = MAIN_ACTION_BASE_WEIGHTS["energy"]
    if buckets.attack:
        score += 6
    return MainActionProposal(
        action=[buckets.attach[0]],
        score=score,
        label="energy",
    )
