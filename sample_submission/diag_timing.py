# -*- coding: utf-8 -*-
"""
1ゲームで player0(AI) が消費する探索時間・回数を計測する（提出非対象）。

目的:
  1手あたりの探索時間を増やしたとき、ゲーム全体(600秒)の上限に
  収まるかを見積もるための実測データを取る。
"""
import sys, time, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import main
from main import read_deck_csv
from cg.api import Observation, to_observation_class, OptionType
from cg.game import battle_start, battle_select, battle_finish

# 計測用の探索時間（この値で 1 手にかける）
PER_MOVE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
main.TIME_BUDGET_SEC = PER_MOVE

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)

deck0 = read_deck_csv()
deck1 = read_deck_csv()

obs_dict, start = battle_start(deck0, deck1)
assert start.errorType == 0

p0_total_time = 0.0      # player0 が agent() に費やした総時間
p0_calls = 0             # player0 の agent() 呼び出し回数
p0_expensive = 0         # うち、候補が複数あって実際に探索した回数
p0_max_single = 0.0      # 1 手の最大消費時間

steps = 0
while steps < 5000:
    obs = to_observation_class(obs_dict)
    if obs.current is not None and obs.current.result != -1:
        break
    acting = obs.current.yourIndex if obs.current is not None else 0

    if acting == 0:
        # 候補数を事前に把握（複数なら探索が走る＝コストがかかる）
        n_opt = len(obs.select.option) if obs.select is not None else 0
        t0 = time.time()
        action = main.agent(obs_dict)
        dt = time.time() - t0
        p0_total_time += dt
        p0_calls += 1
        if dt > 0.2:           # 0.2 秒以上かかった = 実際に探索した
            p0_expensive += 1
        p0_max_single = max(p0_max_single, dt)
    else:
        action = random_agent(obs_dict)

    obs_dict = battle_select(action)
    steps += 1

battle_finish()

print(f"=== 探索時間計測 (PER_MOVE={PER_MOVE}s, 対ランダム) ===")
print(f"player0 agent() 総呼び出し回数 : {p0_calls}")
print(f"  うち探索が走った回数(>0.2s) : {p0_expensive}")
print(f"player0 探索に使った総時間     : {p0_total_time:.1f} 秒")
print(f"1手の最大消費時間              : {p0_max_single:.2f} 秒")
print(f"ゲーム全体ステップ数           : {steps}")
print()
# 倍々にした場合の単純外挿（探索回数 × 1手あたり時間）
for mult in (1, 2, 3):
    est = p0_expensive * PER_MOVE * mult
    print(f"  1手 {PER_MOVE*mult:.1f}s に増やすと総探索時間 ≒ {est:.0f} 秒"
          f"  {'(600秒以内 OK)' if est < 540 else '(危険: 600秒超過の恐れ)'}")
