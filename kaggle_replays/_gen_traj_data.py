"""Phase19.8: trajectory divergence audit dataset。

各 root で:
  1. Policy top-k 候補に H1 Block A / B(各 M=8、独立 seed)を生成 -> stable pair 判定
  2. 凍結 TD1_N2499 で候補を採点 -> TD1 の選好
  3. stable かつ mean margin >= 0.10 の pair から、TD1 が誤る pair と正解する pair を
     最大 1 つずつ選ぶ(Core Failure 2.0 / Matched Correct)
  4. 選んだ pair について **独立 seed の Block C**(§11)で trajectory を再生成し、
     C1..C4 の observable snapshot を保存
  5. 一部で same-action null(同じ行動を2つの独立 seed 群で)を生成

checkpoint(engine から検出できる actor 遷移で定義。§13 のラベルとの対応を明記):
  C1  = 候補行動の直後
  C1b = 自分の**現ターン**終了(me->opp 遷移 1 回目)
  C2  = 相手ターン終了 / 自分の次ターン開始(opp->me 遷移)   <- §13 C2 と C3 は engine 上同一瞬間
  C3  = 自分の次ターンで最初の行動を打った直後
  C4  = 自分の次ターン終了(me->opp 遷移 2 回目)= H1 endpoint

hidden 情報(相手手札等)は snapshot に含めない(§14 observable のみ)。
root は再生できない前提で、その場で必要な観測を全部保存する(§46)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "value_net"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

import _entity_extract as EX  # noqa: E402
import train_lh as L  # noqa: E402
import train_transition as TT  # noqa: E402
from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
_FROZEN = _HERE / "value_net" / "frozen"

TOP_K = 8
CKPTS = ("C1", "C1b", "C2", "C3", "C4")
GROUPS: list[dict] = []
STATS = Counter()
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "first": True}
QUOTA = None
MODELS: dict = {}
EVALS: dict = {}
OPTS = {"m": 8, "max_steps": 300, "null_rate": 0.25, "tag": "x"}


def load_td1(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TT.PoolArm(ck["option_dim"], mode="current")
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], np.float32),
            "std": np.asarray(ck["std"], np.float32)}


def snapshot(st, me, ev):
    """observable snapshot(hidden 情報は入れない)。"""
    if st is None:
        return None
    try:
        e = EX.extract(st, me)
        s166 = encoder.encode_state_from_state(st)
        term = int(st.result)
        return {"entity": e,
                "state166": [round(float(x), 5) for x in s166],
                "value": round(float(ev.evaluate(st, me)), 6),
                "terminal": term if term != -1 else None,
                "outcome": (1.0 if term == me else (0.0 if term == 1 - me else 0.5))
                if term != -1 else None,
                "turn": int(getattr(st, "turn", 0) or 0),
                "actor": int(getattr(st, "yourIndex", -1))}
    except Exception:                              # noqa: BLE001
        return None


def run_traj(obs, me, action, seed, policy_model, want_ckpt):
    """1 本の継続。C1..C4 の snapshot と action log を記録し、H1 endpoint 値を返す。"""
    ev_v = EVALS["value"]
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(seed))
        root = P._begin(obs, hs)
    except Exception:                              # noqa: BLE001
        STATS["begin_fail"] += 1
        return None
    child = None
    try:
        try:
            child = P.cg_api.search_step(root.searchId, [action])
        except ValueError:
            STATS["illegal"] += 1
            return None
        node = child
        ck = {}
        alog = []
        if want_ckpt:
            ck["C1"] = snapshot(node.observation.current, me, ev_v)
        me_to_opp = 0
        opp_to_me = 0
        own_acts_after = 0
        prev = me
        steps = 0
        h1 = None
        while steps < OPTS["max_steps"]:
            o = node.observation
            s = o.current
            if s is None:
                break
            if s.result != -1:
                h1 = 1.0 if s.result == me else (0.0 if s.result == 1 - me else 0.5)
                if want_ckpt:
                    for k in CKPTS:
                        ck.setdefault(k, snapshot(s, me, ev_v))
                break
            actor = s.yourIndex
            if prev == me and actor != me:
                me_to_opp += 1
                if me_to_opp == 1 and want_ckpt:
                    ck["C1b"] = snapshot(s, me, ev_v)
                if me_to_opp == 2:
                    h1 = float(ev_v.evaluate(s, me))
                    if want_ckpt:
                        ck["C4"] = snapshot(s, me, ev_v)
                    break
            if prev != me and actor == me:
                opp_to_me += 1
                if opp_to_me == 1 and want_ckpt:
                    ck["C2"] = snapshot(s, me, ev_v)
            if o.select is None or not o.select.option:
                break
            sel = P._greedy_selection(policy_model, o)
            if not sel:
                break
            if want_ckpt and len(alog) < 24:
                try:
                    op = o.select.option[sel[0]]
                    cid = encoder.encode_option_card_ids(s, o.select)
                    alog.append({"actor": actor,
                                 "type": int(getattr(op.type, "value", op.type)),
                                 "card": int(cid[sel[0]]) if cid else -1,
                                 "turn": int(getattr(s, "turn", 0) or 0),
                                 "n_legal": len(o.select.option)})
                except Exception:                  # noqa: BLE001
                    pass
            try:
                node = P.cg_api.search_step(node.searchId, sel)
            except ValueError:
                break
            if actor == me and me_to_opp >= 1 and opp_to_me >= 1:
                own_acts_after += 1
                if own_acts_after == 1 and want_ckpt:
                    ck["C3"] = snapshot(node.observation.current, me, ev_v)
            prev = actor
            steps += 1
        if h1 is None:
            s = node.observation.current
            h1 = float(ev_v.evaluate(s, me)) if s is not None else None
        return {"h1": h1, "ckpt": ck if want_ckpt else None,
                "actions": alog, "steps": steps}
    finally:
        for sid in (child.searchId if child is not None else None, root.searchId):
            if sid is not None:
                try:
                    P.cg_api.search_release(sid)
                except Exception:                  # noqa: BLE001
                    pass
        try:
            P.cg_api.search_end()
        except Exception:                          # noqa: BLE001
            pass


def block(obs, me, cand, policy_model, seeds, want_ckpt=False):
    out = {}
    for i in cand:
        rs = [run_traj(obs, me, i, s, policy_model, want_ckpt) for s in seeds]
        rs = [r for r in rs if r is not None and r["h1"] is not None]
        if not rs:
            return None
        out[i] = rs
    return out


def _probe(obs, policy_model):
    select, state = obs.select, obs.current
    me = state.yourIndex
    try:
        ps = policy_model.score_options(obs, None, None)
    except Exception:                              # noqa: BLE001
        return
    if not ps or len(ps) != len(select.option):
        return
    ranked = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)
    cand = ranked[:TOP_K]
    if len(cand) < 2:
        return
    t = int(getattr(state, "turn", 0) or 0)
    if not QUOTA.accept(t, len(cand), _ctx["opp"], _ctx["in_game"]):
        return
    try:
        sf = encoder.encode_state_from_state(state)
        orow = encoder.encode_options_from_state(state, select)
        cids = encoder.encode_option_card_ids(state, select)
        ent = EX.extract(state, me)
    except Exception:                              # noqa: BLE001
        return
    orows = [[round(float(x), 6) for x in orow[i]] for i in cand]
    cid = [int(cids[i]) if cids[i] is not None else -1 for i in cand]
    otp = [int(getattr(select.option[i].type, "value", select.option[i].type)) for i in cand]

    base = (hash((_ctx["game"], _ctx["in_game"], t)) & 0x3FFFFF) * 1000
    sa = [base + m for m in range(OPTS["m"])]
    sb = [base + 300000 + m for m in range(OPTS["m"])]
    blkA = block(obs, me, cand, policy_model, sa)
    if blkA is None:
        STATS["blockA_fail"] += 1
        return
    blkB = block(obs, me, cand, policy_model, sb)
    if blkB is None:
        STATS["blockB_fail"] += 1
        return
    ha = {i: statistics.mean([r["h1"] for r in blkA[i]]) for i in cand}
    hb = {i: statistics.mean([r["h1"] for r in blkB[i]]) for i in cand}

    # 凍結 TD1 の選好
    g_tmp = {"state_feat": [float(x) for x in sf], "entity": ent,
             "candidates": [{"option_feat": orows[n], "action_card_id": cid[n],
                             "option_type": otp[n]} for n in range(len(cand))]}
    g_tmp["_tok"] = __import__("entity_tokens").tokenize(ent)
    g_tmp["_rel"] = __import__("entity_tokens").relation_matrix(g_tmp["_tok"])
    g_tmp["_yA"] = [0.0] * len(cand)
    td1 = L.score_group(MODELS["TD1"]["model"], g_tmp,
                        MODELS["TD1"]["mean"], MODELS["TD1"]["std"], "TD1")

    # stable pair 抽出
    cands_pairs = []
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            i, j = cand[a], cand[b]
            da, db = ha[i] - ha[j], hb[i] - hb[j]
            if da == 0 or db == 0 or da * db <= 0:
                continue                            # stable でない
            mm = (abs(da) + abs(db)) / 2
            if mm < 0.10:
                continue
            ok = (td1[a] - td1[b]) * da > 0
            cands_pairs.append({"a": a, "b": b, "i": i, "j": j, "margin": mm,
                                "td1_correct": bool(ok),
                                "winner_idx": a if da > 0 else b})
    if not cands_pairs:
        STATS["no_stable_pair"] += 1
        return
    wrong = sorted([p for p in cands_pairs if not p["td1_correct"]],
                   key=lambda p: -p["margin"])
    right = sorted([p for p in cands_pairs if p["td1_correct"]], key=lambda p: -p["margin"])
    picked = []
    if wrong:
        picked.append(("core_failure", wrong[0]))
    if right:
        # margin をなるべく近い相手に合わせる(§9 matching)
        if wrong:
            right.sort(key=lambda p: abs(p["margin"] - wrong[0]["margin"]))
        picked.append(("matched_correct", right[0]))
    if not picked:
        return

    # ---- 独立 Block C(seed family を完全分離)----
    sc = [base + 700000 + m for m in range(OPTS["m"])]
    recs = []
    for kind, pr in picked:
        c = {}
        for key, idx in (("A", pr["i"]), ("B", pr["j"])):
            rs = [run_traj(obs, me, idx, s, policy_model, True) for s in sc]
            rs = [r for r in rs if r is not None]
            if len(rs) < OPTS["m"] // 2:
                c = None
                break
            c[key] = rs
        if c is None:
            STATS["blockC_fail"] += 1
            continue
        recs.append({"kind": kind, "pair": pr, "blockC": c})
    if not recs:
        return

    null = None
    if random.random() < OPTS["null_rate"]:
        sn = [base + 900000 + m for m in range(OPTS["m"])]
        idx = picked[0][1]["i"]
        r1 = [r for r in (run_traj(obs, me, idx, s, policy_model, True) for s in sc)
              if r is not None]
        r2 = [r for r in (run_traj(obs, me, idx, s, policy_model, True) for s in sn)
              if r is not None]
        if r1 and r2:
            null = {"action": idx, "X": r1, "Y": r2}

    GROUPS.append({
        "group_id": "tj{}_{}".format(_ctx["game"], _ctx["in_game"]),
        "game": _ctx["game"], "turn": t, "turn_band": turn_band(t),
        "cand_band": cand_band(len(cand)), "arch": _ctx["opp"],
        "me_first": _ctx["first"], "me": me,
        "state_feat": [round(float(x), 6) for x in sf], "entity": ent,
        "candidates": [{"option_index": cand[n], "option_feat": orows[n],
                        "action_card_id": cid[n], "option_type": otp[n],
                        "policy_rank": n, "td1_score": round(td1[n], 6),
                        "h1_a": round(ha[cand[n]], 6), "h1_b": round(hb[cand[n]], 6)}
                       for n in range(len(cand))],
        "pairs": recs, "null": null, "seeds": {"A": sa, "B": sb, "C": sc}})
    QUOTA.commit(t, len(cand), _ctx["opp"])
    _ctx["in_game"] += 1
    STATS["groups"] += 1
    if len(GROUPS) % 5 == 0:
        _flush()


def _flush():
    out = _HERE / "_tj_{}.jsonl.gz".format(OPTS["tag"])
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    tmp.replace(out)


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        if (_ctx["recording"] and obs.current is not None
                and obs.current.yourIndex == _ctx["me"] and obs.select is not None
                and obs.select.option and obs.select.type == SelectType.MAIN
                and obs.select.maxCount == 1 and len(obs.select.option) >= 2
                and QUOTA.total < QUOTA.target):
            try:
                _probe(obs, ml_policy_agent._get_model(config))
            except Exception as exc:               # noqa: BLE001
                STATS["probe_err"] += 1
                print("[err]", exc, file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=160)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--per-game-cap", type=int, default=2)
    ap.add_argument("--null-rate", type=float, default=0.25)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=1300000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="w0")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(m=args.m, null_rate=args.null_rate, tag=args.tag)
    torch.set_num_threads(1)
    L.HSTAR[0] = "H1"

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap, cand_soft_cap=0.45)
    EVALS["value"] = leaf_eval_module.build_evaluator({"kind": "value"})
    MODELS["TD1"] = load_td1(_FROZEN / "TD1_N2499.pt")
    _install()

    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
    climb = agents.make_ml_policy_agent(cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for gi in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch, first=p0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        if gi % 3 == 0:
            print("  [w{}] game#{} groups={}/{} {}min".format(
                args.worker_id, g, QUOTA.total, args.target,
                int((time.perf_counter() - t0) / 60)), file=sys.stderr, flush=True)

    _flush()
    print(json.dumps({"tag": args.tag, "n_groups": len(GROUPS), "stats": dict(STATS),
                      "core_failure": sum(1 for r in GROUPS for p in r["pairs"]
                                          if p["kind"] == "core_failure"),
                      "matched_correct": sum(1 for r in GROUPS for p in r["pairs"]
                                             if p["kind"] == "matched_correct"),
                      "null_cases": sum(1 for r in GROUPS if r["null"]),
                      "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
                      "arch": dict(Counter(r["arch"] for r in GROUPS))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
