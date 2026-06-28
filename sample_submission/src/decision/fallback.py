import random

from cg.api import Observation


def choose_random_legal_action(obs: Observation) -> list[int]:
    """Keep the current starter behavior as a safe fallback."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during turn decisions.")
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)
