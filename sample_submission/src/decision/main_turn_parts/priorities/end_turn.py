from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_end_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """ほかに強い候補がないときのためにターン終了を候補にする。"""
    if buckets.end:
        return MainActionProposal(
            action=[buckets.end[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["end"],
            label="end",
        )
    return None
