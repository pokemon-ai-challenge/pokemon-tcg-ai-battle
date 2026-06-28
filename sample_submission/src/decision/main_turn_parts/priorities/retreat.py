from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.decision.evaluation.switch_eval import choose_best_retreat_option


def propose_retreat_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """にげた後の前衛が明確に改善する場合だけ retreat を提案する。"""
    if obs.current is None or not buckets.retreat:
        return None
    if buckets.attack:
        return None

    # retreat 候補が複数あっても、ここでは最善の 1 手だけ返す。
    best_retreat = choose_best_retreat_option(obs, buckets.retreat)
    if not best_retreat.should_offer:
        return None

    return MainActionProposal(
        action=[best_retreat.option_index],
        score=MAIN_ACTION_BASE_WEIGHTS["retreat"],
        label="retreat",
    )
