"""ISMCTS v2.4 Phase G/J + cg correctness — student の end-to-end latency + rollout 組込 profiling + 等価性。

1) latency(Phase G): teacher + 各 student で score_options_from_state の end-to-end ms/call(固定 rollout states)。
2) profiling(Phase J): 実 ISMCTS search(rollout_policy=teacher/student)で ms/iteration・iterations@1350ms・policy calls。
3) cg 等価性(F9/F10): rollout_policy=PolicyModel(teacher 重み) は rollout_policy=None(v1)と同一 action(identity)。
   rollout_policy=student は rollout でのみ呼ばれ、tree prior は teacher(F10 isolation)。
`python student_integration_profile.py [n_states]`  student JSON は training/rollout_student_h{4,8,16}.json。
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
_TRAIN = _ROOT / "kaggle_replays" / "training"
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
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402

_CFG = A.load_config()
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60}
_STUDENTS = {"H4": _TRAIN / "rollout_student_h4.json", "H8": _TRAIN / "rollout_student_h8.json",
             "H16": _TRAIN / "rollout_student_h16.json"}
_REPEAT = 20


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _time_call(model, state, select, repeat):
    t0 = time.perf_counter()
    for _ in range(repeat):
        model.score_options_from_state(state, select)
    return (time.perf_counter() - t0) / repeat * 1000.0


def main(n_states=14, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    teacher = mpa._get_model(_CFG)
    hand_ev = HandcraftedEvaluator()
    students = {k: PolicyModel(weights_path=v) for k, v in _STUDENTS.items() if v.exists()}
    for k, m in students.items():
        print(f"  loaded student {k}: is_ready={m.is_ready}")
    teacher_copy = PolicyModel(weights_path=_SUB / "ptcg_ai" / "learning" / "policy_weights.json")

    lat = {"teacher": []}
    for k in students:
        lat[k] = []
    iters = {"teacher": [], **{k: [] for k in students}}
    itlat = {"teacher": [], **{k: [] for k in students}}
    f9_mismatch = 0; f9_total = 0
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
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
                    # (1) latency: teacher + students on this MAIN state
                    lat["teacher"].append(_time_call(teacher, cur, sel, _REPEAT))
                    for k, m in students.items():
                        lat[k].append(_time_call(m, cur, sel, _REPEAT))
                    # (2) profiling @1350ms: teacher rollout vs each student rollout
                    for name, rp in [("teacher", None)] + [(k, students[k]) for k in students]:
                        st = ismcts.SearchStats()
                        cfg = dict(_COMMON); cfg["iterations"] = 100000
                        dl = time.perf_counter() + 1.35
                        ismcts.search(obs, cfg, teacher, hand_ev, det, dl, random.Random(0), st,
                                      rollout_policy=rp)
                        iters[name].append(st.iterations)
                        itlat[name].append(st.elapsed_ms / max(1, st.iterations))
                    # (3) F9 identity: teacher_copy as rollout_policy == None(v1)
                    a_v1 = ismcts.search(obs, {**_COMMON, "iterations": 24}, teacher, hand_ev, det,
                                         time.perf_counter() + 30, random.Random(0), ismcts.SearchStats(),
                                         rollout_policy=None)
                    a_id = ismcts.search(obs, {**_COMMON, "iterations": 24}, teacher, hand_ev, det,
                                         time.perf_counter() + 30, random.Random(0), ismcts.SearchStats(),
                                         rollout_policy=teacher_copy)
                    f9_total += 1
                    if tuple(a_v1 or ()) != tuple(a_id or ()):
                        f9_mismatch += 1
                    n_done += 1
            if sel.maxCount == 1:
                idx = teacher.select_option(obs, factory, time.perf_counter() + 0.05)
                action = [idx if idx is not None else 0]
            else:
                action = list(range(sel.minCount))
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    print(f"\nprofiled {n_done} MAIN states\n")
    print("=== Phase G: end-to-end ms / policy call ===")
    tl = statistics.mean(lat["teacher"])
    print(f"  {'model':8s} {'ms/call':>9s} {'speedup':>9s}")
    print(f"  {'teacher':8s} {tl:9.3f} {1.0:8.1f}x")
    for k in students:
        ml = statistics.mean(lat[k])
        print(f"  {k:8s} {ml:9.3f} {tl/max(ml,1e-9):8.1f}x")
    print("\n=== Phase J: rollout 組込 profiling @1350ms(same wall-time)===")
    print(f"  {'model':8s} {'iters@1.35s':>12s} {'ms/iter':>9s} {'iters speedup':>14s}")
    ti = statistics.mean(iters["teacher"])
    for name in ["teacher"] + list(students.keys()):
        it = statistics.mean(iters[name]); il = statistics.mean(itlat[name])
        print(f"  {name:8s} {it:12.1f} {il:9.1f} {it/max(ti,1e-9):12.2f}x")
    print(f"\n=== F9 identity(teacher-copy as rollout_policy == v1)===")
    print(f"  mismatch {f9_mismatch}/{f9_total}  " +
          ("(全一致=rollout_policy 機構は v1 と等価)" if f9_mismatch == 0 else "← 要調査"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    main(n_states=n)
