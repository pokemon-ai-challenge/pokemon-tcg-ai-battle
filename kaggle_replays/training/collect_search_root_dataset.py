"""ISMCTS v2.5 Phase C/D — Search-to-Root distillation dataset(Teacher=v2.4 H4 ISMCTS)。

alakazam agent が Teacher Search(H4 rollout, root/tree prior=Original 735dd38a, handcrafted leaf, FULL horizon,
**固定 128 iterations**)で各 root MAIN 判断を行う trajectory を生成し、各 root で:
  h(239次元標準化入力/legal option、v2.4 と同一)/ Original Policy score / Search visit N(a) / Search Q(a) /
  CHANGED(Original top1 != Search top1)/ confidence(top1 share・top1-2 gap・visit entropy)/ meta
を記録。forced(option 1個)は除外(改善 signal なし)。game 単位 split。

leakage: 記録は Original Policy と同一の公開情報のみ(determinize は belief)。Reference Pool v2 不使用。
`python collect_search_root_dataset.py [n_games] [out.npz] [teacher_iters]`  (pilot: n_games 小、save は out 指定時)
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_ROOT / "kaggle_replays" / "challengers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import to_observation_class, SelectType  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
import random  # noqa: E402

_CFG = A.load_config()
_H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1, "max_depth": 60}


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


_ARCH_DIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"


def _archetype_deck(name):
    """archetype_decks/<name>/01.csv を 60枚 deck として読む(開発資産、Reference Pool v2 とは無関係)。"""
    p = _ARCH_DIR / name / "01.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _build_h(model, state, select):
    sf = encoder.encode_state_from_state(state)
    orows = encoder.encode_options_from_state(state, select)
    cids = encoder.encode_option_card_ids(state, select)
    if not orows or len(orows) != len(select.option):
        return None
    sm, ss = model._state_mean, model._state_std
    om, os_ = model._option_mean, model._option_std
    st = [(sf[i] - sm[i]) / ss[i] if ss[i] else 0.0 for i in range(len(sf))]
    return [st + [(o[i] - om[i]) / os_[i] if os_[i] else 0.0 for i in range(len(o))] + model._card_embedding(c)
            for o, c in zip(orows, cids)]


def _entropy(ps):
    return -sum(p * math.log(p) for p in ps if p > 0)


def main(n_games=2, out=None, teacher_iters=128, opponents="mirror"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    deck = _load_deck()
    model = mpa._get_model(_CFG)                       # Original Policy 735dd38a(tree prior)
    h4 = PolicyModel(weights_path=_H4)                 # frozen rollout student
    hand_ev = HandcraftedEvaluator()
    tcfg = dict(_COMMON); tcfg["iterations"] = int(teacher_iters)
    schedule = [o.strip() for o in opponents.split(",") if o.strip()]   # "mirror" or archetype 名の rotation
    opp_counts = Counter()

    store = {k: [] for k in ("X", "visit", "oscore", "q", "group", "game", "turn", "nopt",
                             "changed", "top1share", "gap", "entropy", "opp")}
    gctr = 0
    forced = 0; total_roots = 0
    changed_ct = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        opp_name = schedule[g % len(schedule)]
        opp_deck = deck if opp_name == "mirror" else _archetype_deck(opp_name)
        our_side = None if opp_name == "mirror" else 0     # mirror=両者記録, else player0(alakazam)のみ
        opp_id = 0 if opp_name == "mirror" else (list(sorted(set(schedule) - {"mirror"})).index(opp_name) + 1)
        opp_counts[opp_name] += 1
        match_context.reset()
        obs_dict, sd = battle_start(deck, opp_deck)
        if sd.errorType != 0:
            continue
        for _ in range(400):
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
            action = None
            record_side = (our_side is None) or (cur.yourIndex == our_side)   # opponent turn は記録しない
            if record_side and sel.type == SelectType.MAIN and sel.maxCount == 1 and bool(sel.option):
                if len(sel.option) == 1:
                    forced += 1
                else:
                    det = A._determinize_factory(obs, _CFG)
                    ok = False
                    try:
                        ok = det() is not None
                    except Exception:
                        ok = False
                    if ok:
                        st = ismcts.SearchStats()
                        best = ismcts.search(obs, tcfg, model, hand_ev, det,
                                             time.perf_counter() + 60, random.Random(0), st, rollout_policy=h4)
                        rc = st.root_children_full
                        hs = _build_h(model, cur, sel)
                        oscores = model.score_options_from_state(cur, sel)
                        if best is not None and rc and hs and oscores and len(oscores) == len(sel.option):
                            n = len(sel.option)
                            visits = [0] * n
                            qv = [0.0] * n
                            for (a, vis, q, pr) in rc:
                                idx = a[0]
                                if 0 <= idx < n:
                                    visits[idx] = vis; qv[idx] = q
                            tot = sum(visits)
                            if tot > 0:
                                total_roots += 1
                                pv = [v / tot for v in visits]
                                svd = sorted(pv, reverse=True)
                                top1share = svd[0]; gap = svd[0] - (svd[1] if len(svd) > 1 else 0.0)
                                ent = _entropy(pv)
                                otop1 = max(range(n), key=lambda i: oscores[i])
                                stop1 = max(range(n), key=lambda i: visits[i])
                                changed = int(otop1 != stop1)
                                changed_ct += changed
                                gid = gctr; gctr += 1
                                for i in range(n):
                                    store["X"].append(np.asarray(hs[i], dtype=np.float32))
                                    store["visit"].append(np.int32(visits[i]))
                                    store["oscore"].append(np.float32(oscores[i]))
                                    store["q"].append(np.float32(qv[i]))
                                    store["group"].append(gid); store["game"].append(g)
                                    store["turn"].append(int(getattr(cur, "turn", -1))); store["nopt"].append(n)
                                    store["changed"].append(changed)
                                    store["top1share"].append(np.float32(top1share))
                                    store["gap"].append(np.float32(gap)); store["entropy"].append(np.float32(ent))
                                    store["opp"].append(opp_id)
                                action = list(best)
            if action is None:
                if sel.maxCount == 1:
                    idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                    action = [idx if idx is not None else 0]
                else:
                    action = list(range(sel.minCount))
            obs_dict = battle_select(action)
        battle_finish()
        print(f"  game {g+1}/{n_games}  roots={total_roots} changed={changed_ct} "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---- pilot report ----
    print(f"\n=== Pilot report(Teacher=H4 @{teacher_iters} iters)===")
    print(f"  usable roots = {total_roots}  forced(除外) = {forced}")
    if total_roots == 0:
        print("  収集失敗"); return
    print(f"  CHANGED(Search top1 != Original top1) = {changed_ct}/{total_roots} = {changed_ct/total_roots:.1%}")
    ent = np.asarray([store["entropy"][j] for j in range(len(store["entropy"]))], dtype=np.float32)
    grp_first = {}
    for j, gid in enumerate(store["group"]):
        grp_first.setdefault(gid, j)
    firsts = list(grp_first.values())
    t1 = np.asarray([store["top1share"][j] for j in firsts]); gp = np.asarray([store["gap"][j] for j in firsts])
    ch = np.asarray([store["changed"][j] for j in firsts]); np_ = np.asarray([store["nopt"][j] for j in firsts])
    en = np.asarray([store["entropy"][j] for j in firsts])
    hi = (t1 >= 0.5) & (gp >= 0.2)
    print(f"  top1 visit share: mean={t1.mean():.3f}  top1-2 gap: mean={gp.mean():.3f}  visit entropy: mean={en.mean():.3f}")
    print(f"  high-confidence roots(share>=0.5 & gap>=0.2)= {int(hi.sum())} ({hi.mean():.1%})")
    print(f"  high-confidence CHANGED = {int((hi & (ch == 1)).sum())}")
    print(f"  legal action count: min={int(np_.min())} median={int(np.median(np_))} max={int(np_.max())}")
    print(f"  opponent games: {dict(opp_counts)}")

    if out:
        X = np.stack(store["X"]).astype(np.float32)
        rng = np.random.default_rng(0)
        game = np.asarray(store["game"], dtype=np.int64)
        ngame = int(game.max()) + 1
        perm = rng.permutation(ngame)
        tr = set(perm[:int(0.70*ngame)].tolist()); va = set(perm[int(0.70*ngame):int(0.85*ngame)].tolist())
        split = np.array([0 if int(gg) in tr else (1 if int(gg) in va else 2) for gg in game], dtype=np.int8)
        np.savez_compressed(_HERE / out, X=X,
                            visit=np.asarray(store["visit"], dtype=np.int32),
                            oscore=np.asarray(store["oscore"], dtype=np.float32),
                            q=np.asarray(store["q"], dtype=np.float32),
                            group=np.asarray(store["group"], dtype=np.int64), game=game,
                            turn=np.asarray(store["turn"], dtype=np.int32),
                            nopt=np.asarray(store["nopt"], dtype=np.int32),
                            changed=np.asarray(store["changed"], dtype=np.int8),
                            top1share=np.asarray(store["top1share"], dtype=np.float32),
                            gap=np.asarray(store["gap"], dtype=np.float32),
                            entropy=np.asarray(store["entropy"], dtype=np.float32),
                            opp=np.asarray(store["opp"], dtype=np.int32), split=split)
        print(f"\nsaved {_HERE / out}  option-rows={len(X)} states={total_roots} games={ngame} "
              f"split rows tr/va/te={(split==0).sum()}/{(split==1).sum()}/{(split==2).sum()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n_games", type=int, nargs="?", default=2)
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--iters", type=int, default=128)
    ap.add_argument("--opponents", default="mirror")
    a = ap.parse_args()
    main(n_games=a.n_games, out=a.out, teacher_iters=a.iters, opponents=a.opponents)
