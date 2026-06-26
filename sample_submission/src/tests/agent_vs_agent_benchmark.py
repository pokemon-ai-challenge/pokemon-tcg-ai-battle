"""エージェント間ベンチマーク: 現行エージェント vs 別実装エージェント。

デッキを指定して両者を対戦させ勝率を計測する。
main_minimal_snapshot.py を相手エージェントとして使うことで
ランダムより意味のある比較ができる。

Usage:
    cd sample_submission
    python src/tests/agent_vs_agent_benchmark.py --games 30
    python src/tests/agent_vs_agent_benchmark.py --games 10 --our-deck maries --opp-deck lucario
"""
import sys
import os
import argparse
import importlib.util
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cg.api import Observation, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import agent as current_agent
from src.knowledge.meta_decks import DECK_CATALOG


DECK_CONFIGS: dict[str, tuple[str, str]] = {
    "maries":    ("deck_maries_obstagoon.csv",  "MARIES_OBSTAGOON_PLAN"),
    "lucario":   ("deck_lucario.csv",            "LUCARIO_PLAN"),
    "hydrapple": ("deck_hydrapple.csv",          "HYDRAPPLE_PLAN"),
    "dragapult": ("deck_dragapult.csv",          "DRAGAPULT_PLAN"),
    "raging":    ("deck_raging_bolt.csv",        "RAGING_BOLT_PLAN"),
}


def _load_deck(csv_name: str) -> list[int]:
    path = csv_name
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", csv_name)
    with open(path) as f:
        return [int(l) for l in f.read().strip().split("\n")[:60]]


def _load_snapshot_agent(snapshot_path: str):
    """main_minimal_snapshot.py などを動的にロードしてエージェント関数を返す。"""
    spec = importlib.util.spec_from_file_location("snapshot_agent", snapshot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def play_one_game(agent0, agent1, deck0: list[int], deck1: list[int]) -> int:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start errorType={start_data.errorType}")
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                return obs.current.result
            acting = obs.current.yourIndex if obs.current is not None else 0
            action = (agent0 if acting == 0 else agent1)(obs_dict)
            obs_dict = battle_select(action)
    finally:
        battle_finish()


def run_vs_snapshot(
    our_deck: list[int],
    opp_deck: list[int],
    snapshot_agent,
    games: int,
) -> dict:
    counts: Counter = Counter()
    for i in range(1, games + 1):
        try:
            # 半分は先攻、半分は後攻で公平に
            if i % 2 == 1:
                result = play_one_game(current_agent, snapshot_agent, our_deck, opp_deck)
                our_result = result  # 0=current win, 1=snapshot win
            else:
                result = play_one_game(snapshot_agent, current_agent, opp_deck, our_deck)
                our_result = 1 - result  # 後攻なので反転
            counts[our_result] += 1
            wins = counts[0]
            print(f"  game {i:>3}/{games}: our_result={our_result}  "
                  f"running_win_rate={wins/i:.0%}")
        except Exception as e:
            print(f"  game {i} ERROR: {e}")
            counts[-1] += 1

    wins = counts[0]
    losses = counts[1]
    played = wins + losses
    return {"wins": wins, "losses": losses, "win_rate": wins / played if played else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--our-deck", choices=list(DECK_CONFIGS), default="maries")
    parser.add_argument("--opp-deck", choices=list(DECK_CONFIGS), default="maries")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Path to snapshot agent file (default: main_minimal_snapshot.py)",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    snapshot_path = args.snapshot or os.path.join(here, "..", "..", "main_minimal_snapshot.py")
    if not os.path.exists(snapshot_path):
        print(f"Snapshot not found: {snapshot_path}")
        sys.exit(1)

    our_csv, our_plan_name = DECK_CONFIGS[args.our_deck]
    opp_csv, _ = DECK_CONFIGS[args.opp_deck]

    import src.knowledge.deck_plan as dp
    dp.set_active_plan(getattr(dp, our_plan_name))

    our_deck = _load_deck(our_csv)
    opp_deck = _load_deck(opp_csv)
    snapshot_agent = _load_snapshot_agent(snapshot_path)

    print(f"Current agent ({args.our_deck}) vs Snapshot ({os.path.basename(snapshot_path)}) ({args.opp_deck})")
    print(f"先攻/後攻を交互に切り替えて {args.games} 試合")
    r = run_vs_snapshot(our_deck, opp_deck, snapshot_agent, args.games)

    print("\n" + "=" * 50)
    print(f"Current agent wins: {r['wins']}/{r['wins']+r['losses']} = {r['win_rate']:.0%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
