"""ISMCTS v2.2 Phase E/F/G(offline value sanity + profiling + fixed-iteration)。

固定 MAIN state 集合(1ゲームを champion で進めながら、信念が現行のうちに inline 捕捉)に対し:
  Phase E  offline value sanity: 各 state で A=handcrafted-at-node / B=value-at-node /
           C=rollout-backed(v1 の leaf 推定)を比較(|A-C|,|B-C|,相関)+ 実 ISMCTS leaf の
           value 予測ヒストグラム(saturation/OOD)。
  Phase F  profiling: v1(rollout)/ v2.1(handcrafted-node)/ v2.2(value-node)を iters=PROFILE_ITERS で
           走らせ、policy forward 数 / value forward 数 / latency / ms-per-forward を計数。
  Phase G  fixed-iteration sweep 8/16/32/64/128: latency/depth/nodes/selected + v1・policy top1 一致率。

**勝率は測らない**(Reference Pool v2 不使用)。読み取り専用・git 未追跡。
`python value_leaf_profile.py [n_states]`
"""
from __future__ import annotations

import math
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
from cg import api as cg_api  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator, ValueModelEvaluator  # noqa: E402

_CFG = A.load_config()
_BUDGETS = (8, 16, 32, 64, 128)
PROFILE_ITERS = 32

_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40,
           "opponent_depth": 1, "max_depth": 60}
_VARIANTS = {
    "v1_rollout":  {"leaf_mode": "rollout", "kind": "handcrafted"},
    "v2_1_hand":   {"leaf_mode": "node",    "kind": "handcrafted"},
    "v2_2_value":  {"leaf_mode": "node",    "kind": "value"},
}


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


class _CountModel:
    """score_options_from_state の呼び出し回数を数える proxy(policy forward = rollout の逐次評価)。"""
    def __init__(self, inner):
        self._inner = inner; self.calls = 0
    def score_options_from_state(self, s, sel):
        self.calls += 1
        return self._inner.score_options_from_state(s, sel)
    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountEval:
    """evaluate 呼び出し回数と、返した p(value/handcrafted の 0..1)を記録する proxy。"""
    def __init__(self, inner):
        self._inner = inner; self.calls = 0; self.outputs = []
    def evaluate(self, state, me):
        self.calls += 1
        p = self._inner.evaluate(state, me)
        self.outputs.append(p)
        return p


def _make_eval(kind):
    return HandcraftedEvaluator() if kind == "handcrafted" else ValueModelEvaluator()


def _corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs)); dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _rollout_backed_value(obs, model, det, me):
    """v1 の leaf 推定(この state を root とし policy-greedy rollout → handcrafted、opponent_depth=1)を [0,1] で。"""
    world = det()
    if world is None:
        return None
    try:
        node_cg = cg_api.search_begin(obs, world["your_deck"], world["your_prize"], world["opponent_deck"],
                                      world["opponent_prize"], world["opponent_hand"], world["opponent_active"])
    except Exception:
        return None
    try:
        v = ismcts._rollout(node_cg, me, HandcraftedEvaluator(), model,
                            {"leaf_mode": "rollout", "opponent_depth": 1, "max_rollout_steps": 40},
                            time.perf_counter() + 20)
    finally:
        try:
            cg_api.search_release(node_cg.searchId)
        except Exception:
            pass
        try:
            cg_api.search_end()
        except Exception:
            pass
    return (v + 1.0) / 2.0   # [-1,1] -> [0,1]


def main(n_states=14, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
    model = mpa._get_model(_CFG)
    hand_ev, val_ev = HandcraftedEvaluator(), ValueModelEvaluator()

    # Phase E accumulators
    e_A, e_B, e_C = [], [], []
    # Phase F accumulators (per variant): latency, policy_calls, value_calls
    prof = {v: {"lat": [], "pol": [], "val": []} for v in _VARIANTS}
    # Phase G accumulators (per variant, per budget)
    sweep = {v: {b: {"lat": [], "depth": [], "nodes": []} for b in _BUDGETS} for v in _VARIANTS}
    agree_v1 = {b: 0 for b in _BUDGETS}          # v2.2 の選択が v1 と一致
    agree_pol = {b: 0 for b in _BUDGETS}         # v2.2 の選択が policy top1 と一致
    v22_leaf_outputs = []                        # 実 ISMCTS leaf の value 予測(OOD ヒストグラム)
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
                    me = cur.yourIndex
                    branchings.append(len(sel.option))
                    pri = ismcts._priors(model, obs, sel, ismcts._legal_actions(sel))
                    top1 = max(pri, key=pri.get)

                    # --- Phase E: A/B/C 比較 ---
                    a_val = hand_ev.evaluate(cur, me)
                    b_val = val_ev.evaluate(cur, me)
                    c_val = _rollout_backed_value(obs, model, det, me)
                    if c_val is not None:
                        e_A.append(a_val); e_B.append(b_val); e_C.append(c_val)

                    # --- Phase F: profiling @ PROFILE_ITERS ---
                    for name, spec in _VARIANTS.items():
                        cm = _CountModel(model); ce = _CountEval(_make_eval(spec["kind"]))
                        cfg = dict(_COMMON); cfg["iterations"] = PROFILE_ITERS; cfg["leaf_mode"] = spec["leaf_mode"]
                        st = ismcts.SearchStats(); t0 = time.perf_counter()
                        ismcts.search(obs, cfg, cm, ce, det, time.perf_counter() + 60, random.Random(0), st)
                        prof[name]["lat"].append((time.perf_counter() - t0) * 1000)
                        prof[name]["pol"].append(cm.calls); prof[name]["val"].append(ce.calls)

                    # --- Phase G: fixed-iteration sweep ---
                    v1_sel_by_b, v22_sel_by_b = {}, {}
                    for name, spec in _VARIANTS.items():
                        for b in _BUDGETS:
                            ce = _CountEval(_make_eval(spec["kind"]))
                            cfg = dict(_COMMON); cfg["iterations"] = b; cfg["leaf_mode"] = spec["leaf_mode"]
                            st = ismcts.SearchStats(); t0 = time.perf_counter()
                            a = ismcts.search(obs, cfg, model, ce, det, time.perf_counter() + 60,
                                              random.Random(0), st)
                            sweep[name][b]["lat"].append((time.perf_counter() - t0) * 1000)
                            sweep[name][b]["depth"].append(st.max_depth); sweep[name][b]["nodes"].append(st.nodes)
                            if name == "v1_rollout":
                                v1_sel_by_b[b] = tuple(a) if a else None
                            if name == "v2_2_value":
                                v22_sel_by_b[b] = tuple(a) if a else None
                                if b == _BUDGETS[-1]:
                                    v22_leaf_outputs.extend(ce.outputs)
                    for b in _BUDGETS:
                        if v22_sel_by_b.get(b) is not None and v22_sel_by_b[b] == v1_sel_by_b.get(b):
                            agree_v1[b] += 1
                        if v22_sel_by_b.get(b) is not None and v22_sel_by_b[b] == top1:
                            agree_pol[b] += 1
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
    print(f"measured {n_done} MAIN states  branching med="
          f"{int(statistics.median(branchings))} (min {min(branchings)} max {max(branchings)})\n")

    # ---- Phase E ----
    print("=== Phase E: offline value sanity (A=handcrafted-node / B=value-node / C=rollout-backed=v1) ===")
    if e_C:
        dAC = [abs(a - c) for a, c in zip(e_A, e_C)]
        dBC = [abs(b - c) for b, c in zip(e_B, e_C)]
        print(f"  n={len(e_C)}  mean A={statistics.mean(e_A):.3f} B={statistics.mean(e_B):.3f} C={statistics.mean(e_C):.3f}")
        print(f"  mean|A-C|={statistics.mean(dAC):.3f}  mean|B-C|={statistics.mean(dBC):.3f}  "
              f"(B が C に近ければ value が rollout 推定を近似)")
        print(f"  corr(A,C)={_corr(e_A, e_C):.3f}  corr(B,C)={_corr(e_B, e_C):.3f}")
    print("\n=== Phase E3: 実 ISMCTS leaf の value 予測ヒストグラム (v2.2 hc128, saturation/OOD) ===")
    if v22_leaf_outputs:
        buckets = [0] * 10
        for p in v22_leaf_outputs:
            buckets[min(9, int(p * 10))] += 1
        tot = len(v22_leaf_outputs)
        print(f"  n_leaf_evals={tot}  min={min(v22_leaf_outputs):.3f} max={max(v22_leaf_outputs):.3f} "
              f"mean={statistics.mean(v22_leaf_outputs):.3f} stdev={statistics.pstdev(v22_leaf_outputs):.3f}")
        print("  histogram [0.0..1.0]: " + " ".join(f"{c*100//tot:2d}%" for c in buckets))

    # ---- Phase F ----
    print(f"\n=== Phase F: profiling @ iters={PROFILE_ITERS} (policy/value forwards, latency) ===")
    for name in _VARIANTS:
        d = prof[name]
        lat = statistics.mean(d["lat"]); pol = statistics.mean(d["pol"]); val = statistics.mean(d["val"])
        print(f"  {name:11s}: latency {lat:7.1f}ms  policy_calls {pol:6.1f}  value_calls {val:5.1f}")
    # value ms/forward 推定: (v2.2 latency - v2.1 latency) / value_calls
    v21_lat = statistics.mean(prof["v2_1_hand"]["lat"]); v22_lat = statistics.mean(prof["v2_2_value"]["lat"])
    v22_val = statistics.mean(prof["v2_2_value"]["val"])
    v1_lat = statistics.mean(prof["v1_rollout"]["lat"]); v1_pol = statistics.mean(prof["v1_rollout"]["pol"])
    if v22_val:
        print(f"  → value ms/forward ≈ (v2.2 {v22_lat:.0f} - v2.1 {v21_lat:.0f}) / {v22_val:.0f} "
              f"= {(v22_lat - v21_lat) / v22_val:.3f} ms")
    if v1_pol:
        print(f"  → policy ms/forward ≈ v1 {v1_lat:.0f}ms / {v1_pol:.0f} calls = {v1_lat / v1_pol:.3f} ms "
              f"(v1 iteration の 90% が rollout の policy)")
    print(f"  → v2.2/v1 speedup ≈ {v1_lat / v22_lat:.1f}x   v2.2/v2.1 cost ≈ {v22_lat / max(v21_lat,1e-9):.2f}x")

    # ---- Phase G ----
    print("\n=== Phase G: fixed-iteration sweep (latency / max_depth / nodes) ===")
    print(f"  {'variant':11s} " + " ".join(f"|iter{b:<3d}" for b in _BUDGETS))
    for name in _VARIANTS:
        cells = []
        for b in _BUDGETS:
            s = sweep[name][b]
            cells.append(f"{statistics.mean(s['lat']):5.0f}ms d{int(statistics.median(s['depth']))}")
        print(f"  {name:11s} " + " ".join(f"|{c:9s}" for c in cells))
    print("\n  v2.2 selected-action agreement:")
    print(f"    {'budget':>7} {'vs v1':>10} {'vs policy top1':>16}")
    for b in _BUDGETS:
        print(f"    {b:>7} {agree_v1[b]:>4}/{n_done:<5} {agree_pol[b]:>8}/{n_done}")
    print("\n注: 勝率非依存。value ms/forward が policy rollout(v1) 総コストを大幅に下回れば H2H の前提(H1)成立。")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    main(n_states=n)
