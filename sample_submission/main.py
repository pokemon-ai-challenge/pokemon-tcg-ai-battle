import os
import sys

# On Kaggle, main.py is executed via exec() from /kaggle_simulations/agent/
# (no __file__), so make sure the agent directory is importable.
for _path in ("/kaggle_simulations/agent", os.getcwd()):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class
from ptcg_ai.action_selection.selector import select_action
from ptcg_ai.core.config import load_config

_CONFIG = load_config()
_DECK: list[int] | None = None

def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def _deck() -> list[int]:
    global _DECK
    if _DECK is None:
        _DECK = read_deck_csv()
    return _DECK

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return list(_deck())

    return select_action(obs, _deck(), _CONFIG)
