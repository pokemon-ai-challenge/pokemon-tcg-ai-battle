"""全10アーキタイプ vs 自エージェントのベンチマーク。

Usage:
    python run_benchmark.py --games 30 --my-deck deck.csv
"""
import argparse
import random
import sys
from collections import Counter

from cg.api import Observation, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent, read_deck_csv
from src.knowledge.meta_decks import DECK_RECIPES


def _build_deck_from_recipe(arch: str) -> list[int]:
    recipe = DECK_RECIPES.get(arch)
    if not recipe:
        return []
    deck = []
    for card_id, count in recipe:
        deck.extend([card_id] * count)
    return deck[:60]


def random_agent_with_deck(deck: list[int]):
    def _agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        return random.sample(range(len(obs.select.option)), obs.select.maxCount)
    return _agent


def play_one_game(player0, player1, deck0, deck1) -> int:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start errorType={start_data.errorType}")
    steps = 0
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                return obs.current.result
            acting_player = obs.current.yourIndex if obs.current is not None else 0
            action = player0(obs_dict) if acting_player == 0 else player1(obs_dict)
            obs_dict = battle_select(action)
            steps += 1
    finally:
        battle_finish()


def run_benchmark(my_deck: list[int], games: int) -> dict[str, dict]:
    archs = list(DECK_RECIPES.keys())
    results: dict[str, dict] = {}
    for arch in archs:
        opp_deck = _build_deck_from_recipe(arch)
        if len(opp_deck) < 60:
            print(f"  {arch}: deck too short ({len(opp_deck)}), skipping")
            continue
        opp_agent = random_agent_with_deck(opp_deck)
        wins = 0
        for g in range(games):
            result = play_one_game(agent, opp_agent, my_deck, opp_deck)
            if result == 0:
                wins += 1
        pct = wins / games * 100
        results[arch] = {"wins": wins, "games": games, "pct": pct}
        print(f"  {arch}: {wins}/{games} = {pct:.0f}%")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    args = parser.parse_args()

    my_deck = read_deck_csv()
    print(f"Benchmark: {args.games} games per arch\n")
    results = run_benchmark(my_deck, args.games)

    total_wins = sum(r["wins"] for r in results.values())
    total_games = sum(r["games"] for r in results.values())
    avg = total_wins / total_games * 100
    print(f"\nAverage: {total_wins}/{total_games} = {avg:.0f}%")


if __name__ == "__main__":
    main()
