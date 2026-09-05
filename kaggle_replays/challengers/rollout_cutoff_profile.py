"""ISMCTS v2.3 Phase E/F/G — H0/H1/H2/FULL の profiling + fixed-iteration diagnostic(勝率非依存)。

固定 MAIN state で各 cutoff variant を比較:
  Phase E/F: iters=32 の Policy calls/iteration(own/opp)・latency・speedup vs FULL。
  Phase G: 16/32/64 iterations の latency/depth/nodes/selected + FULL・policy top1 との action agreement。
  leaf-state diagnostic: 各 variant が評価した handcrafted leaf 値の分布(何を見ているか)。
Reference Pool v2 不使用。`python rollout_cutoff_profile.py [n_states]`
"""
from __future__ import annotations

import collections
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
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402

_CFG = A.load_config()
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60}
_ITERS = (16, 32, 64)
_VARIANTS = {   # name -> (leaf_mode, rollout_cutoff)
    "H0_node":   ("node", None),
    "H1_one":    ("rollout", "one_handoff"),
    "H2_two":    ("rollout", "two_handoff"),
    "FULL":      ("rollout", None),
}


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


class _CountModel:
    def __init__(self, inner):
        self._inner = inner; self.calls = 0
    def score_options_from_state(self, s, sel):
        self.calls += 1; return self._inner.score_options_from_state(s, sel)
    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountEval:
    def __init__(self, inner):
        self._inner = inner; self.outputs = []
    def evaluate(self, state, me):
        p = self._inner.evaluate(state, me); self.outputs.append(p); return p


def _cfg(variant, iters):
    leaf_mode, cutoff = _VARIANTS[variant]
    c = dict(_COMMON); c["iterations"] = iters; c["leaf_mode"] = leaf_mode
    if cutoff is not None:
        c["rollout_cutoff"] = cutoff
    return c


def main(n_states=12, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
    model = mpa._get_model(_CFG)
    hand_ev = HandcraftedEvaluator()

    prof = {v: {"lat": [], "pol": []} for v in _VARIANTS}         # @32
    sweep = {v: {b: {"lat": [], "depth": [], "nodes": []} for b in _ITERS} for v in _VARIANTS}
    agree_full = {v: {b: 0 for b in _ITERS} for v in _VARIANTS}   # v の選択が FULL と一致
    agree_pol = {v: {b: 0 for b in _ITERS} for v in _VARIANTS}
    leaf_out = {v: [] for v in _VARIANTS}
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
                    branchings.append(len(sel.option))
                    pri = ismcts._priors(model, obs, sel, ismcts._legal_actions(sel))
                    top1 = max(pri, key=pri.get)

                    # Phase F: policy calls / latency @32
                    for v in _VARIANTS:
                        cm = _CountModel(model)
                        st = ismcts.SearchStats(); t0 = time.perf_counter()
                        ismcts.search(obs, _cfg(v, 32), cm, hand_ev, det, time.perf_counter() + 60,
                                      random.Random(0), st)
                        prof[v]["lat"].append((time.perf_counter() - t0) * 1000); prof[v]["pol"].append(cm.calls)

                    # Phase G: fixed-iteration + agreement
                    full_sel = {}
                    sel_by = {v: {} for v in _VARIANTS}
                    for v in _VARIANTS:
                        for b in _ITERS:
                            ce = _CountEval(hand_ev)
                            st = ismcts.SearchStats(); t0 = time.perf_counter()
                            a = ismcts.search(obs, _cfg(v, b), model, ce, det, time.perf_counter() + 60,
                                              random.Random(0), st)
                            sweep[v][b]["lat"].append((time.perf_counter() - t0) * 1000)
                            sweep[v][b]["depth"].append(st.max_depth); sweep[v][b]["nodes"].append(st.nodes)
                            sel_by[v][b] = tuple(a) if a else None
                            if b == _ITERS[-1]:
                                leaf_out[v].extend(ce.outputs)
                    for b in _ITERS:
                        full_sel[b] = sel_by["FULL"][b]
                        for v in _VARIANTS:
                            if sel_by[v][b] is not None and sel_by[v][b] == full_sel[b]:
                                agree_full[v][b] += 1
                            if sel_by[v][b] is not None and sel_by[v][b] == top1:
                                agree_pol[v][b] += 1
                    n_done += 1

            if sel.maxCount == 1:
                idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                action = [idx if idx is not None else 0]
            else:
                action = list(range(sel.minCount))
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    if n_done == 0:
        print("MAIN state 収集失敗"); return
    print(f"measured {n_done} MAIN states  branching med={int(statistics.median(branchings))}\n")

    print("=== Phase E/F: Policy calls / iteration + latency @32 + speedup vs FULL ===")
    full_lat = statistics.mean(prof["FULL"]["lat"])
    full_pol = statistics.mean(prof["FULL"]["pol"]) / 32
    print(f"  {'variant':9s} {'policy/iter':>12s} {'latency@32':>12s} {'speedup vs FULL':>16s} {'policy cut':>11s}")
    for v in _VARIANTS:
        pol = statistics.mean(prof[v]["pol"]) / 32; lat = statistics.mean(prof[v]["lat"])
        cut = (1 - pol / full_pol) * 100 if full_pol else 0
        print(f"  {v:9s} {pol:12.2f} {lat:10.0f}ms {full_lat/max(lat,1e-9):14.1f}x {cut:9.0f}%")

    print("\n=== Phase G: fixed-iteration latency / depth ===")
    print(f"  {'variant':9s} " + " ".join(f"|it{b:<3d}" for b in _ITERS))
    for v in _VARIANTS:
        cells = [f"{statistics.mean(sweep[v][b]['lat']):5.0f}ms d{int(statistics.median(sweep[v][b]['depth']))}" for b in _ITERS]
        print(f"  {v:9s} " + " ".join(f"|{c}" for c in cells))

    print("\n=== action agreement(of {}）：vs FULL / vs policy top1 ===".format(n_done))
    for b in _ITERS:
        print(f"  iters={b}: " + "  ".join(f"{v}={agree_full[v][b]}/{agree_pol[v][b]}" for v in _VARIANTS))

    print("\n=== leaf-state diagnostic: handcrafted leaf 値分布(iters=64)===")
    for v in _VARIANTS:
        o = leaf_out[v]
        if o:
            print(f"  {v:9s} n={len(o):5d} mean={statistics.mean(o):.3f} stdev={statistics.pstdev(o):.3f} "
                  f"min={min(o):.3f} max={max(o):.3f}")
    print("\n注: 勝率非依存。online strength は Phase H(equal-wall-time H2H)で判定。offline agreement だけで採用しない(v2.2 教訓)。")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n_states=n)
