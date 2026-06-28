from cg.api import AreaType, Observation, SelectContext

from .common import analyze_card_move_option
from .hidden_zone import (
    _card_data_lookup,
    _select_best_indices,
    _self_keep_score,
)
from .to_hand_eval import score_opponent_gain_value
from ..evaluation.attack_features import build_attack_lookup
from ..fallback import choose_random_legal_action


def choose_not_move_or_look_action(obs: Observation) -> list[int]:
    """Handle NOT_MOVE and LOOK without bloating the router."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during card move decisions.")

    if obs.select.context == SelectContext.NOT_MOVE:
        return choose_not_move_action(obs)
    if obs.select.context == SelectContext.LOOK:
        return choose_look_action(obs)
    raise ValueError("choose_not_move_or_look_action only supports NOT_MOVE and LOOK.")


def choose_not_move_action(obs: Observation) -> list[int]:
    """Select the cards that should stay in place."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during NOT_MOVE decisions.")

    views = [
        analyze_card_move_option(obs, option_index)
        for option_index in range(len(obs.select.option))
    ]
    views = [view for view in views if view is not None]
    if not views:
        return choose_random_legal_action(obs)

    scored_views: list[tuple[float, int]] = []
    card_data_by_id = _card_data_lookup()
    for view in views:
        score = _score_not_move_candidate(obs, view, views, card_data_by_id)
        scored_views.append((score, view.option_index))

    scored_views.sort(reverse=True)
    return _select_best_indices(
        scored_views,
        min_count=obs.select.minCount,
        max_count=obs.select.maxCount,
    )


def choose_look_action(obs: Observation) -> list[int]:
    """Use a deterministic prefix so LOOK stays easy to debug."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during LOOK decisions.")

    option_count = len(obs.select.option)
    if option_count == 0:
        return [] if obs.select.minCount == 0 else choose_random_legal_action(obs)

    target_count = min(obs.select.maxCount, option_count)
    return list(range(target_count))


def _score_not_move_candidate(
    obs: Observation,
    view,
    all_views: list,
    card_data_by_id: dict,
) -> float:
    owner_is_self = _infer_owner_is_self(obs, view)
    if owner_is_self is True:
        return _self_keep_score(obs, view, all_views, card_data_by_id)
    if owner_is_self is False:
        return _score_opponent_not_move_candidate(obs, view, all_views, card_data_by_id)
    return 0.0


def _score_opponent_not_move_candidate(
    obs: Observation,
    view,
    all_views: list,
    card_data_by_id: dict,
) -> float:
    """For NOT_MOVE, leave the opponent with the least useful card when possible."""
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 0.0
    return -score_opponent_gain_value(
        obs,
        card_data,
        card_data_by_id,
        build_attack_lookup(),
    )


def _infer_owner_is_self(obs: Observation, view) -> bool | None:
    if view.owner_is_self is not None:
        return view.owner_is_self
    if obs.select is None or obs.current is None:
        return None

    option = obs.select.option[view.option_index]
    if option.playerIndex is not None:
        return option.playerIndex == obs.current.yourIndex

    if view.area in {
        AreaType.HAND,
        AreaType.DECK,
        AreaType.DISCARD,
        AreaType.PRIZE,
        AreaType.LOOKING,
    }:
        return True

    return None
