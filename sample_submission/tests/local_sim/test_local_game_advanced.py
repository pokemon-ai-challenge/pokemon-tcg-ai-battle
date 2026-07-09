import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable


SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))


from cg.api import Observation, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent, read_deck_csv


AgentFn = Callable[[dict], list[int]]


def random_agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    # Pick any legal action uniformly at random as a simple baseline opponent.
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def validate_action(obs: Observation, action: list[int]) -> None:
    if obs.select is None:
        if len(action) != 60:
            raise ValueError(f"Deck selection must return 60 cards, got {len(action)}.")
        return

    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise TypeError("agent() must return list[int].")
    if not (obs.select.minCount <= len(action) <= obs.select.maxCount):
        raise ValueError(
            f"Action length must be between {obs.select.minCount} and "
            f"{obs.select.maxCount}, got {len(action)}."
        )
    if len(action) != len(set(action)):
        raise ValueError(f"Action must not contain duplicates: {action}")

    option_count = len(obs.select.option)
    for index in action:
        if not 0 <= index < option_count:
            raise IndexError(f"Action index out of range: {index} (options={option_count})")


def play_one_game(
    player0: AgentFn,
    player1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    verbose: bool,
) -> int:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")

    steps = 0
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                if verbose:
                    print(f"finished: result={obs.current.result}, steps={steps}")
                return obs.current.result

            # The simulator tells us which side must act next.
            acting_player = obs.current.yourIndex if obs.current is not None else 0
            acting_agent = player0 if acting_player == 0 else player1
            action = acting_agent(obs_dict)
            validate_action(obs, action)

            if verbose and obs.select is not None:
                print(
                    f"turn={obs.current.turn if obs.current is not None else '?'} "
                    f"player={acting_player} "
                    f"context={obs.select.context} "
                    f"action={action}"
                )

            obs_dict = battle_select(action)
            steps += 1
    finally:
        battle_finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Pokemon TCG AI Battle simulation."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to run.",
    )
    parser.add_argument(
        "--opponent",
        choices=("self", "random"),
        default="self",
        help="Opponent policy. 'self' uses main.agent for both players.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each decision before sending it to the simulator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    # Swap only the opponent policy; player0 is always the implementation in main.py.
    player1 = agent if args.opponent == "self" else random_agent

    results = Counter()
    for game_index in range(1, args.games + 1):
        result = play_one_game(agent, player1, deck0, deck1, args.verbose)
        results[result] += 1
        print(f"game {game_index}/{args.games}: result={result}")

    # result=0 means player0 won, result=1 means player1 won.
    print("summary:", dict(sorted(results.items())))


if __name__ == "__main__":
    main()
