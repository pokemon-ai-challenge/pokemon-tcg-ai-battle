"""ISMCTS v2.11 Phase G/H — fixed-iter 探索等価 + iterations@1350ms(v2.4 H4 old vs v2.11 batched)。"""
from __future__ import annotations
import os, random, statistics, sys, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _ROOT = _HERE.parents[1]; _SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path: sys.path.insert(0, p)
from cg.api import to_observation_class, SelectType  # noqa: E402
import ismcts, ismcts_v1_agent as A  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from batched_policy import BatchedPolicyModel  # noqa: E402

_CFG = A.load_config(); _H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60}


def main(n=12):
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB); match_context.reset()
    deck = [int(x) for x in (_SUB / "deck.csv").read_text().split("\n")[:60]]
    model = mpa._get_model(_CFG); ev = HandcraftedEvaluator()
    old = PolicyModel(weights_path=_H4); new = BatchedPolicyModel(weights_path=_H4)
    od, sd = battle_start(deck, deck)
    fixeq = 0; fixtot = 0; it_old = []; it_new = []; done = 0
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
                # Phase G: fixed 128 iters、同一 world pool(同一 det seq は同 RNG で再現)→ old/new 同一 action
                cfg = {**_COMMON, "iterations": 128}
                a_old = ismcts.search(obs, cfg, model, ev, det, time.perf_counter()+60, random.Random(0), ismcts.SearchStats(), rollout_policy=old)
                a_new = ismcts.search(obs, cfg, model, ev, det, time.perf_counter()+60, random.Random(0), ismcts.SearchStats(), rollout_policy=new)
                fixtot += 1; fixeq += int(tuple(a_old or ()) == tuple(a_new or ()))
                # Phase H: iterations@1350ms
                for pol, acc in ((old, it_old), (new, it_new)):
                    st = ismcts.SearchStats()
                    ismcts.search(obs, {**_COMMON, "iterations": 100000}, model, ev, det, time.perf_counter()+1.35, random.Random(0), st, rollout_policy=pol)
                    acc.append(st.iterations)
                done += 1
        factory = mpa._model_hidden_state_factory(obs, _CFG)
        idx = model.select_option(obs, factory, time.perf_counter()+0.05)
        od = battle_select([idx if idx is not None else 0])
    battle_finish()
    print(f"=== Phase G: fixed-128-iter old vs batched action agreement = {fixeq}/{fixtot} "
          f"({'同一 world で完全一致' if fixeq==fixtot else '不一致=要調査'}) ===")
    print(f"=== Phase H: iterations@1350ms ===")
    print(f"  v2.4 H4 old   : mean={statistics.mean(it_old):.1f} median={int(statistics.median(it_old))}")
    print(f"  v2.11 batched : mean={statistics.mean(it_new):.1f} median={int(statistics.median(it_new))}")
    print(f"  → iterations increase = {(statistics.mean(it_new)/max(statistics.mean(it_old),1e-9)-1)*100:+.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
