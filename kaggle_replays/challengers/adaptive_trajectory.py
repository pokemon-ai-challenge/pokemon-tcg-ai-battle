"""ISMCTS v2.12 Phase A/C/D — fixed-1350ms trajectory 収集 + checkpoint 別 final-action agreement + stop-rule replay。

v2.11 batched H4 を budget_ms=1350 + log_trajectory で走らせ、各 root の checkpoint(16,32,48,..)での
top1/share/gap と final action を記録。checkpoint 別 final-action 一致率を出し、stop rule 候補を offline replay
(early-stop 率・false-stop 率・iterations 節約)。実 search は再実行せず trajectory から評価。
`python adaptive_trajectory.py [n_roots]`
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import to_observation_class, SelectType  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts, ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
from batched_policy import BatchedPolicyModel  # noqa: E402

_CFG = A.load_config()
_H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60,
           "iterations": 100000, "log_trajectory": True, "early_stop": {"n_min": 16, "check_every": 16}}
_CHECKPOINTS = (32, 48, 64, 96, 128, 160)


def _load_deck():
    return [int(x) for x in (_SUB / "deck.csv").read_text().split("\n")[:60]]


def _replay_rule(traj, final_action, n_min, k, S, G):
    """trajectory を rule で replay。返り: (fired, stop_iter, stop_action)。fired 無ければ (False, final_iter, final)。"""
    hist = []
    for (it, t1, share, gap) in traj:
        if it < n_min:
            hist.append(t1); continue
        hist.append(t1)
        stable = len(hist) >= k and len(set(hist[-k:])) == 1
        if stable and share >= S and gap >= G:
            return True, it, t1
    return False, (traj[-1][0] if traj else 0), final_action


def main(n_roots=30, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB); match_context.reset()
    deck = _load_deck()
    model = mpa._get_model(_CFG); ev = HandcraftedEvaluator()
    h4 = BatchedPolicyModel(weights_path=_H4)
    trajs = []      # (trajectory, final_action, final_iter, n_options)
    done = 0; game = 0
    while done < n_roots and game < 30:
        game += 1
        match_context.reset()
        od, sd = battle_start(deck, deck)
        if sd.errorType != 0:
            continue
        try:
            for _ in range(max_steps):
                if done >= n_roots:
                    break
                obs = to_observation_class(od); cur = obs.current
                if cur is None or cur.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                try: match_context.update(obs)
                except Exception: pass
                if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                    det = A._determinize_factory(obs, _CFG)
                    ok = False
                    try: ok = det() is not None
                    except Exception: ok = False
                    if ok:
                        st = ismcts.SearchStats()
                        a = ismcts.search(obs, dict(_COMMON), model, ev, det, time.perf_counter() + 1.35,
                                          random.Random(0), st, rollout_policy=h4)
                        if a is not None and st.trajectory:
                            trajs.append((st.trajectory, tuple(a), st.iterations, len(sel.option)))
                            done += 1
                factory = mpa._model_hidden_state_factory(obs, _CFG)
                if sel.maxCount == 1:
                    idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                    action = [idx if idx is not None else 0]
                else:
                    action = list(range(sel.minCount))
                od = battle_select(action)
        finally:
            battle_finish()

    if not trajs:
        print("収集失敗"); return
    fin_iters = [fi for (_t, _a, fi, _n) in trajs]
    print(f"=== Phase A: fixed-1350ms trajectory({len(trajs)} roots)===")
    print(f"  final iterations: mean={statistics.mean(fin_iters):.0f} median={int(statistics.median(fin_iters))} "
          f"P90={sorted(fin_iters)[int(len(fin_iters)*0.9)]}")
    print(f"  checkpoint 別 final-action 一致率:")
    for cp in _CHECKPOINTS:
        ags = []
        for (traj, fa, fi, _n) in trajs:
            cand = [t1 for (it, t1, sh, gp) in traj if it <= cp]
            if cand:
                ags.append(int(cand[-1] == fa))
        if ags:
            print(f"    ~{cp:>4} iter: agreement {statistics.mean(ags):.3f} (n={len(ags)})")

    print(f"\n=== Phase C/D: stop-rule replay(target final-action agreement>=0.98)===")
    print(f"  {'rule':22s} {'stop%':>7} {'agree':>7} {'false-stop':>11} {'mean_iters':>11} {'iters saved':>12}")
    rules = {
        "Conservative(64,3,.7,.4)": (64, 3, 0.7, 0.4),
        "Balanced(48,3,.6,.3)":     (48, 3, 0.6, 0.3),
        "Balanced2(48,2,.65,.35)":  (48, 2, 0.65, 0.35),
        "Aggressive(32,2,.55,.25)": (32, 2, 0.55, 0.25),
    }
    for name, (nmin, k, S, G) in rules.items():
        fired_ct = 0; agree = 0; false_stop = 0; iters = []; saved = []
        for (traj, fa, fi, _n) in trajs:
            fired, si, sa = _replay_rule(traj, fa, nmin, k, S, G)
            iters.append(si if fired else fi)
            if fired:
                fired_ct += 1; agree += int(sa == fa); false_stop += int(sa != fa); saved.append(fi - si)
        n = len(trajs)
        ag = agree / fired_ct if fired_ct else float("nan")
        print(f"  {name:22s} {fired_ct/n*100:>6.0f}% {ag:>7.3f} {false_stop/n*100:>10.1f}% "
              f"{statistics.mean(iters):>11.0f} {statistics.mean(saved) if saved else 0:>12.0f}")
    print("\n注: agree=early-stop した root の fixed-final 一致率(>=0.98 目標)。false-stop=stop したが final と違う割合。")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
