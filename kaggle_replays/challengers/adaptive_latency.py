"""ISMCTS v2.12 Phase G/H — adaptive(early-stop)vs fixed の iterations/latency 実測 + early-stop 率。"""
from __future__ import annotations
import os, random, statistics, sys, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _ROOT = _HERE.parents[1]; _SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path: sys.path.insert(0, p)
from cg.api import to_observation_class, SelectType  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts, ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
from batched_policy import BatchedPolicyModel  # noqa: E402

_CFG = A.load_config(); _H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_BASE = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60, "iterations": 100000}
_ES = {"enabled": True, "n_min": 64, "check_every": 16, "k_stable": 3, "share": 0.7, "gap": 0.4}


def main(n=60):
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB); deck = [int(x) for x in (_SUB / "deck.csv").read_text().split("\n")[:60]]
    model = mpa._get_model(_CFG); ev = HandcraftedEvaluator(); h4 = BatchedPolicyModel(weights_path=_H4)
    fx_it = []; fx_ms = []; ad_it = []; ad_ms = []; stopped = 0; done = 0; game = 0
    while done < n and game < 30:
        game += 1; match_context.reset()
        od, sd = battle_start(deck, deck)
        if sd.errorType != 0: continue
        try:
            for _ in range(400):
                if done >= n: break
                obs = to_observation_class(od); cur = obs.current
                if cur is None or cur.result != -1: break
                sel = obs.select
                if sel is None: break
                try: match_context.update(obs)
                except Exception: pass
                if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                    det = A._determinize_factory(obs, _CFG)
                    ok = False
                    try: ok = det() is not None
                    except Exception: ok = False
                    if ok:
                        stf = ismcts.SearchStats(); t0 = time.perf_counter()
                        ismcts.search(obs, dict(_BASE), model, ev, det, time.perf_counter()+1.35, random.Random(0), stf, rollout_policy=h4)
                        fx_it.append(stf.iterations); fx_ms.append((time.perf_counter()-t0)*1000)
                        sta = ismcts.SearchStats(); t0 = time.perf_counter()
                        ismcts.search(obs, {**_BASE, "early_stop": _ES}, model, ev, det, time.perf_counter()+1.35, random.Random(0), sta, rollout_policy=h4)
                        ad_it.append(sta.iterations); ad_ms.append((time.perf_counter()-t0)*1000); stopped += int(sta.early_stopped)
                        done += 1
                factory = mpa._model_hidden_state_factory(obs, _CFG)
                if sel.maxCount == 1:
                    idx = model.select_option(obs, factory, time.perf_counter()+0.05); action = [idx if idx is not None else 0]
                else: action = list(range(sel.minCount))
                od = battle_select(action)
        finally:
            battle_finish()
    def p95(x): return sorted(x)[int(len(x)*0.95)] if x else 0
    print(f"=== Phase G/H: adaptive vs fixed(n={done} roots)===")
    print(f"  early-stop rate = {stopped}/{done} = {stopped/max(done,1):.1%}")
    print(f"  iterations: fixed mean={statistics.mean(fx_it):.0f} P95={p95(fx_it)} | adaptive mean={statistics.mean(ad_it):.0f} P95={p95(ad_it)}")
    print(f"  latency ms: fixed mean={statistics.mean(fx_ms):.0f} P95={p95(fx_ms):.0f} | adaptive mean={statistics.mean(ad_ms):.0f} P95={p95(ad_ms):.0f}")
    print(f"  → iterations reduction = {(1-statistics.mean(ad_it)/statistics.mean(fx_it))*100:.0f}%  latency reduction = {(1-statistics.mean(ad_ms)/statistics.mean(fx_ms))*100:.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
