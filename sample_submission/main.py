import sys
import os
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Kaggle は exec() で実行するため __file__ が未定義になる
    sys.path.insert(0, '/kaggle_simulations/agent')

from cg.api import Observation, to_observation_class
from src.decision.router import choose_action
from src.decision.search_policy import choose_action_with_search
from src.knowledge.opponent_model import get_opponent_model


def read_deck_csv() -> list[int]:
    """deck.csv からカード ID リストを読み込む。"""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def validate_choice(obs: Observation, chosen: list[int]) -> None:
    """返すインデックスが選択ルールを満たしているか確認する。"""
    select = obs.select
    if not (select.minCount <= len(chosen) <= select.maxCount):
        raise ValueError(
            f"Action length must be between {select.minCount} and {select.maxCount}, got {len(chosen)}."
        )
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate select elements are not allowed.")
    if not all(0 <= i < len(select.option) for i in chosen):
        raise ValueError("Each selected index must be within the option range.")


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    get_opponent_model().update(obs)
    chosen = choose_action(obs)  # Phase 5 完了後に choose_action_with_search に戻す
    validate_choice(obs, chosen)
    return chosen
