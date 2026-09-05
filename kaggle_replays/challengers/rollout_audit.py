"""ISMCTS v2.3 Phase A — v1 rollout semantics audit(read-only)。

固定 MAIN state 集合で、v1 FULL rollout の実挙動を実測する:
  - Policy forward 内訳: prior/tree(NODE mode の call 数)vs rollout(FULL-NODE)、own(ref手番)vs opp。
  - turn handoff 回数 / rollout 長(steps)/ 終了理由 の分布。
  - semantic boundary(handoff)定義の裏取り。
instrumented replica の policy-call 数を実 ismcts._rollout(counting model)と突合して忠実性を検証。
ismcts.py は一切変更しない。`python rollout_audit.py [n_states]`
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
from cg import api as cg_api  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
from ptcg_ai.search import pipeline as _pl  # noqa: E402

_CFG = A.load_config()
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60}


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


class _CountModel:
    """score_options_from_state の呼び出し回数 + yourIndex 分布を記録する proxy。"""
    def __init__(self, inner):
        self._inner = inner; self.calls = 0; self.by_actor = collections.Counter()
    def score_options_from_state(self, s, sel):
        self.calls += 1
        try:
            self.by_actor[s.yourIndex] += 1
        except Exception:
            pass
        return self._inner.score_options_from_state(s, sel)
    def __getattr__(self, name):
        return getattr(self._inner, name)


def _instrumented_full_rollout(node_cg, ref_player, model, deadline):
    """ismcts._rollout の FULL 経路を逐語 replica + instrument。返り: (reason, steps, handoffs, own, opp, actor_seq)。"""
    opponent_depth = 1; max_steps = 40
    o0 = node_cg.observation
    prev_actor = o0.current.yourIndex if o0.current is not None else ref_player
    opp_turns = 0; steps = 0; handoffs = 0; own = 0; opp = 0
    actor_seq = [prev_actor]
    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            return ("deadline", steps, handoffs, own, opp, actor_seq)
        o = node_cg.observation; s = o.current
        if s is None:
            return ("none", steps, handoffs, own, opp, actor_seq)
        if s.result != -1:
            return ("terminal", steps, handoffs, own, opp, actor_seq)
        actor = s.yourIndex
        if actor != prev_actor:
            handoffs += 1; actor_seq.append(actor)
        if prev_actor == ref_player and actor != ref_player:
            opp_turns += 1
        if actor == ref_player and prev_actor != ref_player and opp_turns >= opponent_depth:
            return ("handoff_return", steps, handoffs, own, opp, actor_seq)   # v1 leaf 評価点
        if o.select is None or not o.select.option:
            return ("noselect", steps, handoffs, own, opp, actor_seq)
        selection = _pl._greedy_selection(model, o)
        if actor == ref_player: own += 1
        else: opp += 1
        if not selection:
            return ("nosel", steps, handoffs, own, opp, actor_seq)
        try:
            node_cg = cg_api.search_step(node_cg.searchId, selection)
        except ValueError:
            return ("valueerror", steps, handoffs, own, opp, actor_seq)
        prev_actor = actor
        steps += 1
    return ("maxsteps", steps, handoffs, own, opp, actor_seq)


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

    node_calls, full_calls = [], []          # search iters=32 の policy 総 calls(NODE/FULL)
    full_by_actor = collections.Counter()
    r_steps, r_handoffs, r_own, r_opp = [], [], [], []
    r_reasons = collections.Counter()
    fidelity = []                             # (replica policy calls, real rollout policy calls)
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

                    # --- A1: search iters=32 の policy 総 calls(NODE vs FULL、own/opp) ---
                    cmN = _CountModel(model)
                    cfgN = dict(_COMMON); cfgN["iterations"] = 32; cfgN["leaf_mode"] = "node"
                    ismcts.search(obs, cfgN, cmN, hand_ev, det, time.perf_counter() + 60, random.Random(0),
                                  ismcts.SearchStats())
                    node_calls.append(cmN.calls)

                    cmF = _CountModel(model)
                    cfgF = dict(_COMMON); cfgF["iterations"] = 32; cfgF["leaf_mode"] = "rollout"
                    ismcts.search(obs, cfgF, cmF, hand_ev, det, time.perf_counter() + 60, random.Random(0),
                                  ismcts.SearchStats())
                    full_calls.append(cmF.calls); full_by_actor.update(cmF.by_actor)

                    # --- A2: 単発 rollout の handoff / steps / own-opp(determinize→search_begin→instrumented replica)---
                    world = det()
                    if world is not None:
                        try:
                            ncg = cg_api.search_begin(obs, world["your_deck"], world["your_prize"],
                                                      world["opponent_deck"], world["opponent_prize"],
                                                      world["opponent_hand"], world["opponent_active"])
                            reason, steps, handoffs, own, opp, seq = _instrumented_full_rollout(
                                ncg, me, model, time.perf_counter() + 20)
                            r_steps.append(steps); r_handoffs.append(handoffs)
                            r_own.append(own); r_opp.append(opp); r_reasons[reason] += 1
                        finally:
                            try: cg_api.search_release(ncg.searchId)
                            except Exception: pass
                            try: cg_api.search_end()
                            except Exception: pass
                        # 忠実性: 同 world で実 ismcts._rollout を counting model で回し policy calls を突合
                        world2 = world
                        try:
                            ncg2 = cg_api.search_begin(obs, world2["your_deck"], world2["your_prize"],
                                                       world2["opponent_deck"], world2["opponent_prize"],
                                                       world2["opponent_hand"], world2["opponent_active"])
                            cm2 = _CountModel(model)
                            ismcts._rollout(ncg2, me, hand_ev, cm2,
                                            {"leaf_mode": "rollout", "opponent_depth": 1, "max_rollout_steps": 40},
                                            time.perf_counter() + 20)
                            fidelity.append((own + opp, cm2.calls))
                        finally:
                            try: cg_api.search_release(ncg2.searchId)
                            except Exception: pass
                            try: cg_api.search_end()
                            except Exception: pass
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

    print("=== A1: Policy forwards / iteration(search iters=32)===")
    nc = statistics.mean(node_calls) / 32; fc = statistics.mean(full_calls) / 32
    print(f"  NODE mode(prior/tree のみ): {nc:.2f} calls/iter")
    print(f"  FULL mode(prior + rollout): {fc:.2f} calls/iter")
    print(f"  → rollout 由来 ≈ {fc - nc:.2f} calls/iter(全体の {100*(fc-nc)/fc:.0f}%)")
    tot_actor = sum(full_by_actor.values()) or 1
    print(f"  FULL policy calls actor 分布: " +
          ", ".join(f"idx{k}={v} ({100*v//tot_actor}%)" for k, v in sorted(full_by_actor.items())))

    print("\n=== A2: 単発 v1 FULL rollout の semantic 長 ===")
    print(f"  steps(=policy calls)/rollout: mean={statistics.mean(r_steps):.2f} "
          f"median={int(statistics.median(r_steps))} max={max(r_steps)}")
    print(f"  turn handoffs/rollout: mean={statistics.mean(r_handoffs):.2f} "
          f"dist={dict(collections.Counter(r_handoffs))}")
    print(f"  own(ref手番) calls: mean={statistics.mean(r_own):.2f}   opp calls: mean={statistics.mean(r_opp):.2f}")
    print(f"  termination reason: {dict(r_reasons)}")

    print("\n=== 忠実性検証(replica steps == 実 ismcts._rollout policy calls)===")
    mism = [(a, b) for a, b in fidelity if a != b]
    print(f"  n={len(fidelity)}  一致={len(fidelity)-len(mism)}/{len(fidelity)}  " +
          ("(全一致=replica は実挙動を忠実再現)" if not mism else f"不一致例={mism[:5]}"))

    print("\n=== semantic boundary 定義(handoff = actor transition)===")
    print("  H0=0 handoffs(node=v2.1) / H1=1st handoff で leaf / H2=2nd handoff で leaf / FULL=v1(opp_turns 基準)")
    print(f"  実測: FULL は平均 {statistics.mean(r_handoffs):.1f} handoffs/rollout。H1<H2≲FULL の cost 順序が期待される。")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n_states=n)
