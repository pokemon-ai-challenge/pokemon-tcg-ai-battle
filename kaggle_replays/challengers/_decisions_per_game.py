"""Phase E helper — 1 ゲーム/エージェントあたりの ISMCTS 適用 decision 数分布を実測。

policy argmax の高速 self-play(search 無し)で、各手番プレイヤー視点の MAIN maxCount==1 且つ option>=2
(=ISMCTS が発火しうる decision)を game×player 別に数える。Phase E の decisions/game を得る。
使用: python _decisions_per_game.py [n_games=30]
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import to_observation_class, SelectType  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402

_CFG = A.load_config()


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def main(n_games=30):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB); match_context.reset()
    deck = [int(x) for x in (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")[:60]]
    model = mpa._get_model(_CFG)
    per_agent = []   # 各 (game, player) の ISMCTS 適用 decision 数
    game_total = []  # 各 game の総 turn 数(参考)
    t0 = time.perf_counter(); played = 0
    for g in range(n_games):
        match_context.reset()
        od, sd = battle_start(deck, deck)
        if sd.errorType != 0:
            continue
        counts = {0: 0, 1: 0}; turns = 0
        try:
            for _ in range(600):
                obs = to_observation_class(od); cur = obs.current
                if cur is None or cur.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                turns = max(turns, int(getattr(cur, "turn", 0)))
                try:
                    match_context.update(obs)
                except Exception:
                    pass
                if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                    counts[cur.yourIndex] = counts.get(cur.yourIndex, 0) + 1
                factory = mpa._model_hidden_state_factory(obs, _CFG)
                if sel.maxCount == 1:
                    idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                    action = [idx if idx is not None else 0]
                else:
                    action = list(range(sel.minCount))
                od = battle_select(action)
        finally:
            battle_finish()
        per_agent.append(counts[0]); per_agent.append(counts[1]); game_total.append(turns); played += 1

    if not per_agent:
        print("失敗"); return
    print(f"=== Phase E: decisions/game(ISMCTS 適用 MAIN,maxCount==1,option>=2)===")
    print(f"  games={played}  per-agent samples={len(per_agent)}  [{time.perf_counter()-t0:.0f}s]")
    print(f"  decisions/game/agent: mean={statistics.mean(per_agent):.1f} median={int(statistics.median(per_agent))} "
          f"P95={_pct(per_agent,0.95)} max={max(per_agent)}")
    print(f"  game turns: mean={statistics.mean(game_total):.0f} max={max(game_total)}")
    out = _HERE / "_decisions_per_game_results.json"
    out.write_text(json.dumps({
        "games": played, "per_agent_decisions": per_agent,
        "mean": statistics.mean(per_agent), "median": statistics.median(per_agent),
        "p95": _pct(per_agent, 0.95), "max": max(per_agent),
        "turns_mean": statistics.mean(game_total), "turns_max": max(game_total),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
