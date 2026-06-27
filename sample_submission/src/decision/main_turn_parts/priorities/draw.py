from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_draw_or_search_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """手札が細いときは、手札補充やサーチを優先候補にする。"""
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    hand_count = obs.current.players[your_index].handCount
    if hand_count > 5:
        return None

    if buckets.supporter_play:
        score = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"] + (6 - hand_count) * 3
        return MainActionProposal(
            action=[buckets.supporter_play[0]],
            score=score,
            label="draw_or_search_supporter",
        )
    if buckets.item_play:
        score = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"] + (6 - hand_count)
        return MainActionProposal(
            action=[buckets.item_play[0]],
            score=score,
            label="draw_or_search_item",
        )
    return None
