"""Integration tests for the lethal search (issue #57).

Runs the real engine (``cg``): checks that the full agent plays legal
moves with lethal search enabled, and that when the search claims a
lethal during a random-vs-random game, following it actually wins the
game within that turn.

Run from sample_submission/:
    python -m pytest tests/integration/test_lethal_search.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import Observation, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent, read_deck_csv
from ptcg_ai.action_selection import selector
from ptcg_ai.core.config import load_config
from ptcg_ai.search import lethal_simple

LETHAL_CONFIG = {
    "enabled": True,
    "max_remaining_prizes": 2,  # Phase 2 (#58); also covers the Phase 1 case
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000,
    "verify_shuffles": 1,
}


def _validate(obs: Observation, action: list[int]) -> None:
    assert isinstance(action, list)
    if obs.select is None:
        assert len(action) == 60
        return
    assert obs.select.minCount <= len(action) <= obs.select.maxCount
    assert len(action) == len(set(action))
    assert all(0 <= i < len(obs.select.option) for i in action)


def test_config_file_loads():
    config = load_config("rule_lethal")
    lethal = config["lethal_search"]
    assert lethal["enabled"] is True
    assert lethal["module"] == "lethal_simple"
    assert lethal["max_remaining_prizes"] == 2


def test_agent_plays_full_game_without_errors():
    """main.agent (lethal search enabled) vs random: every action legal."""
    random.seed(1)
    deck = read_deck_csv()

    def random_agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        return random.sample(range(len(obs.select.option)), obs.select.maxCount)

    obs_dict, start = battle_start(list(deck), list(deck))
    assert start.errorType == 0
    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                assert obs.current.result in (0, 1)
                break
            acting = obs.current.yourIndex if obs.current is not None else 0
            action = agent(obs_dict) if acting == 0 else random_agent(obs_dict)
            _validate(obs, action)
            obs_dict = battle_select(action)
            steps += 1
            assert steps < 5000, "game did not finish"
    finally:
        battle_finish()


def _play_probe_game(seed: int) -> str:
    """Random-vs-random game; both sides probe for a lethal at <=1 prize.

    Once the search returns a line for a player, that player follows it.
    Returns "lethal_won" if the game then ends in that player's victory
    within the same turn, "no_lethal" if the search never fired, and
    raises AssertionError if a claimed lethal fails to win in the turn.
    """
    rng = random.Random(seed)
    deck = read_deck_csv()
    obs_dict, start = battle_start(list(deck), list(deck))
    assert start.errorType == 0
    fired: tuple[int, int] | None = None  # (player_index, turn)
    try:
        while True:
            obs = to_observation_class(obs_dict)
            state = obs.current
            if state is not None and state.result != -1:
                if fired is not None:
                    player, turn = fired
                    assert state.result == player, (
                        f"lethal claimed by player {player} on turn {turn} "
                        f"but player {state.result} won"
                    )
                    return "lethal_won"
                return "no_lethal"

            if fired is not None and state is not None:
                player, turn = fired
                assert state.turn == turn, (
                    f"lethal claimed by player {player} on turn {turn} "
                    f"but the game reached turn {state.turn}"
                )

            action = None
            if (
                obs.select is not None
                and state is not None
                and len(state.players[state.yourIndex].prize)
                <= LETHAL_CONFIG["max_remaining_prizes"]
            ):
                context = {
                    "observation": obs,
                    "full_deck": list(deck),
                    "config": LETHAL_CONFIG,
                    "rng": rng,
                }
                action = lethal_simple.search(state, obs.select.option, context)
                if action is not None and fired is None:
                    fired = (state.yourIndex, state.turn)

            if action is None:
                action = rng.sample(
                    range(len(obs.select.option)), obs.select.maxCount
                )
            _validate(obs, action)
            obs_dict = battle_select(action)
    finally:
        battle_finish()


def test_claimed_lethal_wins_within_the_turn():
    """When the search fires in a real game, the line must win that turn."""
    # The engine shuffles decks with its own RNG, so which game produces
    # a lethal spot is not reproducible; sample until one fires.
    # Empirically ~1/6 of random-vs-random games fire even at the
    # 1-prize threshold, so 30 games make a false skip very unlikely.
    lethal_simple.reset_stats()
    outcomes = []
    for seed in range(30):
        outcomes.append(_play_probe_game(seed))
        if outcomes.count("lethal_won") >= 2:
            break
    stats = lethal_simple.get_stats()
    if "lethal_won" not in outcomes:
        pytest.skip("no lethal situation was sampled in 30 games")
    assert stats["found"] >= 1
    assert stats["searches"] >= stats["found"]
