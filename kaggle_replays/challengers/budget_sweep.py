"""ISMCTS v1 budget sweep(Phase G)。固定 state 集合で iterations 別に latency/depth/branching/
action-change を計測。**勝率は見ない**(budget を development 勝率で選ばない)。Reference Pool v2 不使用。

`python budget_sweep.py`
"""
from __future__ import annotations

import os
import random
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
import ismcts  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402

_CFG = A.load_config()


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


_BUDGETS = (8, 16, 32, 64, 128)


def main(n_states=12, max_steps=400):
    """1 ゲームを進め、MAIN maxCount==1 の各 state で **その場で(信念が現行のうちに)** budget sweep を計測。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
    model = mpa._get_model(_CFG)
    # per-budget accumulators
    acc = {b: {"lat": [], "depth": [], "mdepth": [], "nodes": [], "changed": 0} for b in _BUDGETS}
    branchings = []
    n_done = 0
    try:
        for _ in range(max_steps):
            if n_done >= n_states:
                break
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None or cur.result != -1:
                break
            sel = obs.select
            if sel is None:
                break
            try:
                match_context.update(obs)
            except Exception:
                pass
            factory = mpa._model_hidden_state_factory(obs, _CFG)
            if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                det = A._determinize_factory(obs, _CFG)
                ok = False
                try:
                    ok = det() is not None
                except Exception:
                    ok = False
                if ok:
                    pri = ismcts._priors(model, obs, sel, ismcts._legal_actions(sel))
                    top1 = max(pri, key=pri.get)
                    branchings.append(len(sel.option))
                    for b in _BUDGETS:   # ★ 信念が現行のうちに計測(stale 回避)
                        st = ismcts.SearchStats(); t0 = time.perf_counter()
                        a = ismcts.search(obs, {"iterations": b, "world_pool_size": 8, "c_puct": 1.4,
                                                "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60},
                                          model, A._get_evaluator(_CFG), det,
                                          time.perf_counter() + 30, random.Random(0), st)
                        acc[b]["lat"].append((time.perf_counter() - t0) * 1000)
                        acc[b]["depth"].append(st.max_depth); acc[b]["mdepth"].append(st.mean_depth)
                        acc[b]["nodes"].append(st.nodes)
                        if a is not None and tuple(a) != top1:
                            acc[b]["changed"] += 1
                    n_done += 1
            # advance
            if sel.maxCount == 1:
                idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                action = [idx if idx is not None else 0]
            else:
                action = list(range(sel.minCount))
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    if not branchings:
        print("MAIN state 収集失敗"); return
    print(f"measured {n_done} MAIN states  branching: min={min(branchings)} "
          f"median={int(statistics.median(branchings))} max={max(branchings)} mean={statistics.mean(branchings):.1f}")
    print("\n=== budget sweep(勝率非依存: latency/depth/change)===")
    for b in _BUDGETS:
        d = acc[b]; lat = sorted(d["lat"])
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0
        print(f"  iters={b:3d}: latency p50={statistics.median(d['lat']):.0f}ms p95={p95:.0f}ms "
              f"max_depth med={int(statistics.median(d['depth']))} mean_depth={statistics.mean(d['mdepth']):.2f} "
              f"nodes med={int(statistics.median(d['nodes']))} change={d['changed']}/{n_done}")
    print("\n注: Kaggle 予算 ~1350ms/select。change=Policy top1 から ISMCTS が変えた state 数。")


if __name__ == "__main__":
    main()
