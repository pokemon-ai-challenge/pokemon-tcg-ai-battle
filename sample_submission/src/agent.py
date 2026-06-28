from collections.abc import Callable

from cg.api import Observation, to_observation_class

from src.decision.router import choose_action


DeckLoader = Callable[[], list[int]]


def choose_action_from_observation_dict(
    obs_dict: dict,
    deck_loader: DeckLoader,
) -> list[int]:
    """Bridge the submission entrypoint and the internal decision modules."""
    obs: Observation = to_observation_class(obs_dict)
    # When obs.select is None, this is the initial deck-selection phase, so
    # we return the prepared 60-card deck instead of a turn action.
    # 初回デッキ選択
    if obs.select is None:
        return deck_loader()
    # 通常ターン
    return choose_action(obs)
