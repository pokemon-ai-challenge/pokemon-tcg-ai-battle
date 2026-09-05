"""ISMCTS v2.11 Phase A/C/E/F — Batched H4 scorer の equivalence + microbenchmark。

実 rollout state で PolicyModel(pure-Python)vs BatchedPolicyModel(numpy)を比較:
  E: score max/mean abs error、top1 agreement(100% 必須)、tie-sensitive(gap<1e-6)別集計、feature 同一。
  F: ms/call(old vs batched)、options/call 分布、speedup。
H4 weights=training/rollout_student_h4.json(不変)。read-only。`python batched_equiv_profile.py [n_states]`
"""
from __future__ import annotations

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
from cg import api as cg_api  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.search import pipeline as _pl  # noqa: E402
from batched_policy import BatchedPolicyModel  # noqa: E402

_CFG = A.load_config()
_H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_REPEAT = 30


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _time(fn, repeat):
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0


def main(n_states=200, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    model = mpa._get_model(_CFG)                 # tree/rollout advance 用(Original)
    h4_old = PolicyModel(weights_path=_H4)       # pure-Python H4
    h4_new = BatchedPolicyModel(weights_path=_H4)  # numpy batched H4
    print(f"  h4 old ready={h4_old.is_ready} new ready={h4_new.is_ready} np_ready={h4_new._np_ready}")

    max_err = 0.0; sum_err = 0.0; n_scores = 0
    top1_total = 0; top1_agree = 0
    tie_total = 0; tie_agree = 0
    lat_old = []; lat_new = []; nopts = []
    n_done = 0
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
    try:
        for _ in range(max_steps):
            if n_done >= n_states:
                break
            obs = to_observation_class(obs_dict); cur = obs.current
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
            # rollout 内 state で scoring を比較(MAIN root から H4 rollout を数手進める)
            if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                det = A._determinize_factory(obs, _CFG)
                world = None
                try:
                    world = det()
                except Exception:
                    world = None
                if world is not None:
                    try:
                        ncg = cg_api.search_begin(obs, world["your_deck"], world["your_prize"],
                                                  world["opponent_deck"], world["opponent_prize"],
                                                  world["opponent_hand"], world["opponent_active"])
                        for _step in range(8):
                            o = ncg.observation; s = o.current
                            if s is None or s.result != -1 or o.select is None or not o.select.option:
                                break
                            if o.select.maxCount == 1 and len(o.select.option) >= 2:
                                so = h4_old.score_options_from_state(s, o.select)
                                sn = h4_new.score_options_from_state(s, o.select)
                                if so and sn and len(so) == len(sn):
                                    errs = [abs(a - b) for a, b in zip(so, sn)]
                                    max_err = max(max_err, max(errs)); sum_err += sum(errs); n_scores += len(errs)
                                    a_old = max(range(len(so)), key=lambda i: so[i])
                                    a_new = max(range(len(sn)), key=lambda i: sn[i])
                                    top1_total += 1; top1_agree += int(a_old == a_new)
                                    ss = sorted(so, reverse=True)
                                    if len(ss) >= 2 and (ss[0] - ss[1]) < 1e-6:
                                        tie_total += 1; tie_agree += int(a_old == a_new)
                                    nopts.append(len(so))
                                    lat_old.append(_time(lambda: h4_old.score_options_from_state(s, o.select), _REPEAT))
                                    lat_new.append(_time(lambda: h4_new.score_options_from_state(s, o.select), _REPEAT))
                                    n_done += 1
                            selc = _pl._greedy_selection(h4_old, o)
                            if not selc:
                                break
                            try:
                                ncg = cg_api.search_step(ncg.searchId, selc)
                            except ValueError:
                                break
                    finally:
                        try: cg_api.search_release(ncg.searchId)
                        except Exception: pass
                        try: cg_api.search_end()
                        except Exception: pass
            idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
            obs_dict = battle_select([idx if idx is not None else 0])
    finally:
        battle_finish()

    print(f"\n=== Phase E: numerical / behavioral equivalence(n_scoring_calls={top1_total})===")
    print(f"  score max abs error = {max_err:.2e}  mean abs error = {sum_err/max(n_scores,1):.2e}  (freeze bar <= 1e-6)")
    print(f"  **top1 agreement = {top1_agree}/{top1_total} = {top1_agree/max(top1_total,1):.4f}** (100% 必須)")
    print(f"  tie-sensitive(gap<1e-6): {tie_agree}/{tie_total} agree")
    print(f"\n=== Phase A/F: options/call 分布 + microbenchmark ===")
    if nopts:
        so = sorted(nopts)
        print(f"  options/call: mean={statistics.mean(nopts):.2f} median={int(statistics.median(nopts))} "
              f"P90={so[int(len(so)*0.9)]} max={max(nopts)}")
    lo = statistics.mean(lat_old); ln = statistics.mean(lat_new)
    print(f"  old(pure-Python) ms/call: mean={lo:.4f} median={statistics.median(lat_old):.4f}")
    print(f"  new(batched np)  ms/call: mean={ln:.4f} median={statistics.median(lat_new):.4f}")
    print(f"  **speedup = {lo/max(ln,1e-9):.2f}x  latency reduction = {(1-ln/lo)*100:.1f}%** (bar: >=20%)")
    ok = (top1_agree == top1_total) and (max_err <= 1e-6)
    print(f"\n  equivalence PASS = {ok}  |  perf bar(>=20% reduction)= {(1-ln/lo)>=0.20}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n_states=n)
