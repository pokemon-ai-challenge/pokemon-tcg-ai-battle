from pathlib import Path
import sys


SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))


from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent, read_deck_csv


def main() -> None:
    # Load two decks for local play. Here both players use the same deck.csv.
    my_deck = read_deck_csv()
    opp_deck = read_deck_csv()

    # Start a battle in the local simulator.
    obs_dict, start = battle_start(my_deck, opp_deck)
    print("errorType:", start.errorType)

    try:
        while True:
            # Convert the raw dict from the simulator into the Observation dataclass.
            obs = to_observation_class(obs_dict)

            # When result is no longer -1, the game has finished.
            if obs.current is not None and obs.current.result != -1:
                print("result:", obs.current.result)
                break

            # Ask your agent in main.py to choose the next action.
            action = agent(obs_dict)

            if obs.select is not None:
                # Basic safety checks for the action returned by agent().
                assert obs.select.minCount <= len(action) <= obs.select.maxCount
                assert len(action) == len(set(action))
                assert all(0 <= i < len(obs.select.option) for i in action)

            # Send the selected action to the simulator and get the next observation.
            obs_dict = battle_select(action)
    finally:
        # Always finish the battle to release simulator resources.
        battle_finish()


if __name__ == "__main__":
    main()
