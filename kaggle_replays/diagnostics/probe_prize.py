import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SS = ROOT / "sample_submission"
for p in (str(ROOT), str(SS), str(ROOT / "league")):
    if p not in sys.path:
        sys.path.insert(0, p)
import os
os.chdir(SS)

from cg.api import to_observation_class, LogType
from cg.game import battle_start, battle_select, battle_finish
from run_league import read_deck_csv_file, build_agent

deck_a = read_deck_csv_file("kaggle_replays/meta_analysis/archetype_decks/crustle/06.csv")
deck_b = read_deck_csv_file(None)  # default deck.csv

agent_a = build_agent("rule_based", None, "abl_5_full_wallguard")
agent_b = build_agent("rule_based", None, "abl_5_full_wallguard")

obs_dict, start = battle_start(deck_a, deck_b)
agents = {0: agent_a, 1: agent_b}
steps = 0
try:
    while steps < 400:
        obs = to_observation_class(obs_dict)
        for log in obs.logs:
            if log.type == LogType.RESULT:
                print("RESULT", log.result, log.reason)
        if obs.current is None:
            print("current None"); break
        p0 = obs.current.players[0]
        p1 = obs.current.players[1]
        if steps % 5 == 0 or obs.current.result != -1:
            print(f"step={steps} turn={obs.current.turn} prize0_len={len(p0.prize)} prize1_len={len(p1.prize)} "
                  f"discard0={len(p0.discard)} discard1={len(p1.discard)}")
        if obs.current.result != -1:
            print("FINAL", obs.current.result)
            break
        tp = obs.current.yourIndex
        action = agents[tp](obs)
        obs_dict = battle_select(action)
        steps += 1
finally:
    battle_finish()
