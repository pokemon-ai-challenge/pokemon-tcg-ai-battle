"""F9 mismatch が determinize() の確率的世界プール由来かを確認: None vs None の mismatch 率。"""
import os, sys, random, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
from cg.api import to_observation_class, SelectType
from cg.game import battle_start, battle_select, battle_finish
import ismcts, ismcts_v1_agent as A
from ptcg_ai.ml_policy import ml_policy_agent as mpa
from ptcg_ai.hidden_information import match_context
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator

CFG = A.load_config()
COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60, "iterations": 24}
os.chdir(_SUB)
deck = [int(x) for x in (_SUB / "deck.csv").read_text().split("\n")[:60]]
teacher = mpa._get_model(CFG); ev = HandcraftedEvaluator(); match_context.reset()
od, sd = battle_start(deck, deck); mm = 0; tot = 0
for _ in range(300):
    obs = to_observation_class(od); cur = obs.current
    if cur is None or cur.result != -1: break
    sel = obs.select
    if sel is None: break
    try: match_context.update(obs)
    except Exception: pass
    if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2 and tot < 14:
        det = A._determinize_factory(obs, CFG)
        try: ok = det() is not None
        except Exception: ok = False
        if ok:
            a1 = ismcts.search(obs, dict(COMMON), teacher, ev, det, time.perf_counter()+30, random.Random(0), ismcts.SearchStats(), rollout_policy=None)
            a2 = ismcts.search(obs, dict(COMMON), teacher, ev, det, time.perf_counter()+30, random.Random(0), ismcts.SearchStats(), rollout_policy=None)
            tot += 1
            if tuple(a1 or ()) != tuple(a2 or ()): mm += 1
    fac = mpa._model_hidden_state_factory(obs, CFG)
    if sel.maxCount == 1:
        i = teacher.select_option(obs, fac, time.perf_counter()+0.05); act = [i if i is not None else 0]
    else: act = list(range(sel.minCount))
    od = battle_select(act)
battle_finish()
print(f"None-vs-None mismatch {mm}/{tot}  (>0 なら determinize RNG 由来 = teacher-copy 3/14 も同起源、機構は正常)")
