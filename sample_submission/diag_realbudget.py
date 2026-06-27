# -*- coding: utf-8 -*-
"""
本番の動的時間予算（TIME_BUDGET_SEC を上書きしない）で 1 ゲーム実行し、
player0(AI) が実際に消費した総時間が GAME_TIME_CAP 以内かを検証する。
"""
import sys, time, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import main
from main import read_deck_csv
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)

deck0 = read_deck_csv()
deck1 = read_deck_csv()
obs_dict, start = battle_start(deck0, deck1)
assert start.errorType == 0

p0_time = 0.0
p0_calls = 0
first_budget = None
wall0 = time.time()

steps = 0
while steps < 5000:
    obs = to_observation_class(obs_dict)
    if obs.current is not None and obs.current.result != -1:
        print("result:", obs.current.result)
        break
    acting = obs.current.yourIndex if obs.current is not None else 0
    if acting == 0:
        t0 = time.time()
        action = main.agent(obs_dict)
        dt = time.time() - t0
        p0_time += dt
        p0_calls += 1
        if first_budget is None and dt > 0.2:
            first_budget = dt
    else:
        action = random_agent(obs_dict)
    obs_dict = battle_select(action)
    steps += 1

battle_finish()
wall = time.time() - wall0

print(f"=== 本番予算 1 ゲーム実測 ===")
print(f"player0 探索総時間 : {p0_time:.1f} 秒  (GAME_TIME_CAP={main.GAME_TIME_CAP})")
print(f"  main._GAME_TIME_USED 内部値 : {main._GAME_TIME_USED:.1f} 秒")
print(f"最初の本格探索の所要 : {first_budget:.1f} 秒 (期待 ~10s)" if first_budget else "n/a")
print(f"player0 呼び出し回数 : {p0_calls}")
print(f"ゲーム全体の実時間   : {wall:.1f} 秒")
print("判定:", "OK (上限内)" if p0_time <= main.GAME_TIME_CAP else "NG (上限超過!)")
