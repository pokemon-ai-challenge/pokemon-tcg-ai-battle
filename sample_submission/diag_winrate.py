# -*- coding: utf-8 -*-
"""対ランダム勝率の高速検証（提出非対象）。探索時間を短縮して多試合回す。"""
import sys, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter

import main
from main import agent, read_deck_csv
from cg.api import Observation, to_observation_class
from cg.game import battle_start, battle_select, battle_finish

main.TIME_BUDGET_SEC = 0.5   # 高速検証用

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
deck0 = read_deck_csv()
deck1 = read_deck_csv()
results = Counter()

for g in range(GAMES):
    obs_dict, start = battle_start(deck0, deck1)
    if start.errorType != 0:
        raise RuntimeError(f"errorType={start.errorType}")
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                results[obs.current.result] += 1
                break
            acting = obs.current.yourIndex if obs.current is not None else 0
            action = agent(obs_dict) if acting == 0 else random_agent(obs_dict)
            obs_dict = battle_select(action)
    finally:
        battle_finish()
    print(f"game {g+1}/{GAMES}: result={results}")

wins = results.get(0, 0)
print(f"\n=== 勝率: {wins}/{GAMES} = {100*wins/GAMES:.0f}% (player0=AI, player1=random) ===")
