"""Task 2-2: Tier-S deck benchmark.

Tests our agent (main.py) against a random agent using each Tier-S deck recipe.
Outputs a win-rate report per archetype.

Usage:
    cd sample_submission
    python src/tests/tier_s_benchmark.py --games 30
    python src/tests/tier_s_benchmark.py --games 10 --decks terapagos_ex raging_bolt_ex
"""
import sys
import os
import argparse
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cg.api import Observation, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent
from src.knowledge.meta_decks import DECK_CATALOG, tier_s_decks

# --my-deck で指定できるデッキ設定: {短縮名: (csvファイル名, DeckPlan変数名)}
MY_DECK_CONFIGS: dict[str, tuple[str, str]] = {
    "maries":    ("deck_maries_obstagoon.csv", "MARIES_OBSTAGOON_PLAN"),
    "lucario":   ("deck_lucario.csv",           "LUCARIO_PLAN"),
    "hydrapple": ("deck_hydrapple.csv",          "HYDRAPPLE_PLAN"),
}


def _load_deck_csv(csv_file: str) -> list[int]:
    """指定したデッキCSVをカードIDリストとして読み込む（CWD=sample_submission/前提）。"""
    path = csv_file
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", csv_file)
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


def _expand_recipe(recipe: list[tuple[int, int]]) -> list[int]:
    """Expand (card_id, count) tuples to a flat 60-card list."""
    result = []
    for card_id, count in recipe:
        result.extend([card_id] * count)
    return result


def _make_random_agent_with_deck(deck_ids: list[int]):
    """Return a random agent that returns the given deck on initial selection."""
    def _agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return deck_ids
        return random.sample(range(len(obs.select.option)), obs.select.maxCount)
    return _agent


def play_one_game(
    player0_agent,
    player1_agent,
    deck0: list[int],
    deck1: list[int],
) -> int:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed errorType={start_data.errorType}")
    steps = 0
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                return obs.current.result
            acting_player = obs.current.yourIndex if obs.current is not None else 0
            acting_agent = player0_agent if acting_player == 0 else player1_agent
            action = acting_agent(obs_dict)
            obs_dict = battle_select(action)
            steps += 1
    finally:
        battle_finish()


def run_benchmark(
    deck_names: list[str],
    games_per_deck: int,
    our_deck: list[int] | None = None,
) -> dict[str, dict]:
    """Run games against each named Tier-S deck. Returns per-deck results."""
    if our_deck is None:
        from main import read_deck_csv
        our_deck = read_deck_csv()

    all_results: dict[str, dict] = {}
    for deck_name in deck_names:
        meta = DECK_CATALOG[deck_name]
        opp_deck = _expand_recipe(meta.recipe)
        opp_agent = _make_random_agent_with_deck(opp_deck)

        counts: Counter = Counter()
        print(f"\n[{deck_name}] tier={meta.tier} arch={meta.archetype}")
        for i in range(1, games_per_deck + 1):
            try:
                result = play_one_game(agent, opp_agent, our_deck, opp_deck)
                counts[result] += 1
                wins = counts[0]
                print(f"  game {i:>3}/{games_per_deck}: result={result}  "
                      f"running_win_rate={wins/i:.0%}")
            except Exception as e:
                print(f"  game {i} ERROR: {e}")
                counts[-1] += 1

        wins = counts[0]
        losses = counts[1]
        errors = counts[-1]
        played = wins + losses
        win_rate = wins / played if played else 0.0
        all_results[deck_name] = {
            "tier": meta.tier,
            "archetype": meta.archetype,
            "wins": wins,
            "losses": losses,
            "errors": errors,
            "win_rate": win_rate,
        }
        print(f"  => {wins}W/{losses}L  win_rate={win_rate:.0%}")

    return all_results


def print_report(results: dict[str, dict], games_per_deck: int) -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT - Lucario vs Tier-S/A Random Agents")
    print(f"Games per deck: {games_per_deck}")
    print("=" * 60)
    header = f"{'Deck':<25} {'Tier':<6} {'Arch':<20} {'W':>4} {'L':>4} {'WR':>6}"
    print(header)
    print("-" * 60)
    for deck_name, r in sorted(results.items(), key=lambda x: -x[1]["win_rate"]):
        wr = f"{r['win_rate']:.0%}"
        print(f"{deck_name:<25} {r['tier']:<6} {r['archetype']:<20} "
              f"{r['wins']:>4} {r['losses']:>4} {wr:>6}")
    print("=" * 60)

    # Aggregate win rate across all Tier-S decks
    s_decks = {k: v for k, v in results.items() if v["tier"] == "S"}
    if s_decks:
        total_wins = sum(v["wins"] for v in s_decks.values())
        total_played = sum(v["wins"] + v["losses"] for v in s_decks.values())
        overall_wr = total_wins / total_played if total_played else 0.0
        print(f"Overall vs Tier-S: {total_wins}/{total_played} = {overall_wr:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-S deck benchmark")
    parser.add_argument("--games", type=int, default=30, help="Games per deck")
    parser.add_argument(
        "--decks",
        nargs="*",
        default=None,
        help="Opponent deck names to test (default: all Tier-S). "
             f"Available: {', '.join(DECK_CATALOG.keys())}",
    )
    parser.add_argument(
        "--my-deck",
        type=str,
        default=None,
        choices=list(MY_DECK_CONFIGS.keys()),
        help="My deck to use without touching deck.csv / deck_plan.py. "
             f"Choices: {', '.join(MY_DECK_CONFIGS.keys())}",
    )
    args = parser.parse_args()

    deck_names = args.decks if args.decks else tier_s_decks()
    invalid = [d for d in deck_names if d not in DECK_CATALOG]
    if invalid:
        print(f"Unknown deck names: {invalid}")
        sys.exit(1)

    our_deck = None
    if args.my_deck:
        import src.knowledge.deck_plan as dp
        csv_file, plan_name = MY_DECK_CONFIGS[args.my_deck]
        plan = getattr(dp, plan_name)
        dp.set_active_plan(plan)
        our_deck = _load_deck_csv(csv_file)
        print(f"[my-deck] {args.my_deck} ({plan.name}) / {csv_file}")

    results = run_benchmark(deck_names, args.games, our_deck)
    print_report(results, args.games)


if __name__ == "__main__":
    main()
