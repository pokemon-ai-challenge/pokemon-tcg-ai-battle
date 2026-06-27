# -*- coding: utf-8 -*-
"""特性・ダメカン使用の計測診断スクリプト（提出には含めない）。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from cg.api import (Observation, to_observation_class, SelectContext,
                    OptionType, LogType)
from cg.game import battle_start, battle_select, battle_finish
import main
from main import agent, read_deck_csv

# 診断を高速化するため探索時間を短縮
main.TIME_BUDGET_SEC = 0.4

deck0 = read_deck_csv()
deck1 = read_deck_csv()

ctx_counter = Counter()
ability_used = 0
dmg_counter_ctx = 0
ability_log = 0

obs_dict, start = battle_start(deck0, deck1)
print("errorType:", start.errorType)

steps = 0
while steps < 5000:
    obs = to_observation_class(obs_dict)
    if obs.current is not None and obs.current.result != -1:
        print("result:", obs.current.result)
        break

    # ログから特性・効果の発動を観測
    if obs.logs:
        for entry in obs.logs:
            pass  # logs are generic; counted separately below

    if obs.select is not None:
        ctx = obs.select.context
        ctx_counter[int(ctx)] += 1
        if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            dmg_counter_ctx += 1
        # MAIN フェーズで ABILITY オプションが存在し、それが選ばれたか
        action = agent(obs_dict)
        if ctx == SelectContext.MAIN:
            ability_idxs = [i for i, o in enumerate(obs.select.option)
                            if o.type == OptionType.ABILITY]
            if ability_idxs and any(a in ability_idxs for a in action):
                ability_used += 1
    else:
        action = agent(obs_dict)

    obs_dict = battle_select(action)
    steps += 1

battle_finish()

print()
print("=== 診断結果 ===")
print(f"総ステップ数: {steps}")
print(f"MAIN で特性(ABILITY)を選んだ回数: {ability_used}")
print(f"ダメカン配置コンテキスト出現回数: {dmg_counter_ctx}")
print(f"出現したコンテキスト種別: {dict(sorted(ctx_counter.items()))}")
