from cg.api import Observation, to_observation_class
from ptcg_ai.core.agent import agent as _core_agent
from ptcg_ai.core.agent import read_deck_csv

__all__ = ["agent", "read_deck_csv"]


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    return _core_agent(obs)
