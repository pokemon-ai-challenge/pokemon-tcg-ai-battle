"""Phase18: Q0-expanded vs EntityQ_EPool の独立長期評価(L1 / L2)。EXPLORATORY。

Phase15 の正式判定(Gate B FAIL)は変更しない。設計は Phase13 と同一の paired branch。

**教師一致度は測らない**(§4)。測るのは
    「Q0 の top-1 を打った未来」vs「S3 の top-1 を打った未来」。

paired branch(§7): 同じ root・同じ決定化から、**最初の1手だけ**を変え、
その後は両 branch とも**同一の continuation policy**(climb Policy 貪欲)で進める。

CRN(§8): 決定化は `random.Random(seed)` を両 branch で共有するので完全一致する。
engine 内部 RNG は API から制御できないため共有できない。noise floor は
null-control(同じ行動を2回)で実測する。

教師との独立性(§14):
  * horizon が長い  : teacher = 自分の次ターン**開始**時点 / L1 = 自分の次ターン**終了**時点
  * L2 は terminal outcome(評価器を通さない)
  * L1 の評価器が違う: teacher = handcrafted / L1 = 凍結学習 Value
  * continuation が違う: teacher = 候補ごとの短い rollout / ここは 1 本の長い継続

root observation は保存できない(Phase11D で確認済み)ため、branch 評価は
**局面に到達したその場で**実行する。
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import time
from collections import Counter
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

from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
import _entity_extract as EX  # noqa: E402
import delta_features as DF  # noqa: E402
from action_q import ActionQNet  # noqa: E402
from train_delta_q import DeltaQNet  # noqa: E402
import entity_tokens as ET  # noqa: E402
import train_transition as TT  # noqa: E402
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
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

MAX_CANDS = 8
TEACHER_NB, TEACHER_K, TEACHER_TIE = 4, 6, 0.005   # Phase11D/12 と同一仕様
GAME_OFFSET = 400000            # 20000/30000/40000 は既使用
GROUPS: list[dict] = []
STATS = Counter()
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "first": True}
QUOTA = None
OPTS = {"n_cont": 4, "max_steps": 600, "null_control": 0.0}
MODELS: dict = {}
EVALS: dict = {}


# ---------------- モデル ----------------

def load_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ActionQNet(ck["state_dim"], ck["option_dim"],
                   use_cards=False, use_action_card=ck.get("use_action_card", True))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], dtype=np.float32),
            "std": np.asarray(ck["std"], dtype=np.float32)}


def load_delta_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = DeltaQNet(ck["state_dim"], ck["option_dim"], delta_dim=ck["delta_dim"],
                  hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], dtype=np.float32),
            "std": np.asarray(ck["std"], dtype=np.float32),
            "delta_dim": ck["delta_dim"]}


def load_entity_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TT.PoolArm(ck["option_dim"], mode="transition")
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], dtype=np.float32),
            "std": np.asarray(ck["std"], dtype=np.float32)}


@torch.no_grad()
def eq_scores(b, orows, cids, ent_before, ent_afters):
    """EntityQ_EPool の推論。before/after を raw entity から token 化して渡す。"""
    tb = ET.tokenize(ent_before)
    pb = TT._pack([tb], [ET.relation_matrix(tb)])
    at, ar = [], []
    for ae in ent_afters:
        t = ET.tokenize(ae) if ae is not None else tb
        at.append(t)
        ar.append(ET.relation_matrix(t) if ae is not None else ET.relation_matrix(tb))
    pa = TT._pack(at, ar)
    C = len(orows)
    batch = {"opt": torch.from_numpy(np.asarray(orows, dtype=np.float32)[None, ...]),
             "card": torch.tensor([[c if 0 < c < ET.VOCAB else 0 for c in cids]],
                                  dtype=torch.long),
             "mask": torch.ones(1, C, dtype=torch.bool),
             "before": pb,
             "after": {k: v.view(1, C, *v.shape[1:]) for k, v in pa.items()}}
    return b["model"](batch)[0].tolist()


@torch.no_grad()
def dq_scores(b, sf, orows, cids, otypes, deltas):
    st = torch.from_numpy(((np.asarray(sf, dtype=np.float32) - b["mean"])
                           / b["std"])[None, :])
    C = len(orows)
    batch = {"state": st,
             "opt": torch.from_numpy(np.asarray(orows, dtype=np.float32)[None, ...]),
             "card": torch.tensor([[c if 0 < c < 2048 else 0 for c in cids]],
                                  dtype=torch.long),
             "mask": torch.ones(1, C, dtype=torch.bool),
             "delta": torch.tensor([deltas], dtype=torch.float32)}
    return b["model"](batch)[0].tolist()


def _candidate_deltas(obs, me, cand, seed):
    """候補ごとに 1step 進めて delta23 を作る(DeltaQ の推論に必要な入力)。

    1 determinization のみ(Phase15 D1)。seed 固定なので後から再現できる(§12)。
    """
    before = EX.summarize(obs.current, me)
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(seed))
        root = P._begin(obs, hs)
    except Exception:                              # noqa: BLE001
        STATS["delta_begin_fail"] += 1
        return before, None, None, None
    afters, ents = {}, {}
    try:
        for i in cand:
            try:
                ch = P.cg_api.search_step(root.searchId, [i])
            except ValueError:
                afters[i] = None
                continue
            try:
                stt = ch.observation.current
                afters[i] = EX.summarize(stt, me) if stt is not None else None
                ents[i] = EX.extract(stt, me) if stt is not None else None
            finally:
                try:
                    P.cg_api.search_release(ch.searchId)
                except Exception:                  # noqa: BLE001
                    pass
    finally:
        try:
            P.cg_api.search_release(root.searchId)
        except Exception:                          # noqa: BLE001
            pass
        try:
            P.cg_api.search_end()
        except Exception:                          # noqa: BLE001
            pass
    return (before, afters, [DF.single(before, afters.get(i)) for i in cand],
            [ents.get(i) for i in cand])


@torch.no_grad()
def q_scores(bundle, sf, orows, cids, otypes):
    st = torch.from_numpy(((np.asarray(sf, dtype=np.float32) - bundle["mean"])
                           / bundle["std"])[None, :])
    C = len(orows)
    of = torch.from_numpy(np.asarray(orows, dtype=np.float32)[None, ...])
    ci = torch.tensor([[max(0, c + 1) if c >= 0 else 0 for c in cids]], dtype=torch.long)
    ot = torch.tensor([[min(t, 63) for t in otypes]], dtype=torch.long)
    mk = torch.ones(1, C, dtype=torch.bool)
    z = torch.zeros((1, 1), dtype=torch.long)
    q, _ = bundle["model"](st, [z, z, z], of, ci, ot, mk)
    return q[0].tolist()


# ---------------- branch 実行 ----------------

def _continue(node, me, l1_eval, max_steps, policy_model):
    """1 本の継続。L1 地平(自分の次ターン終了)の値と、terminal 到達結果を返す。

    L1 地平に達する前に決着したら L1 も terminal 値にする。
    """
    prev_actor = me
    phase = "await_opp"
    l1 = None
    l1_info = None
    steps = 0
    term = None
    last_state = None
    while steps < max_steps:
        obs = node.observation
        st = obs.current
        if st is None:
            break
        last_state = st
        if st.result != -1:
            term = int(st.result)
            if l1 is None:
                l1 = 1.0 if st.result == me else 0.0
                l1_info = _snapshot(st, me)
            break
        actor = st.yourIndex
        if l1 is None:
            if phase == "await_opp" and actor != me:
                phase = "await_me"
            elif phase == "await_me" and actor == me:
                phase = "await_leave_me"
            elif phase == "await_leave_me" and actor != me:
                l1 = float(l1_eval.evaluate(st, me))
                l1_info = _snapshot(st, me)
        if obs.select is None or not obs.select.option:
            break
        sel = P._greedy_selection(policy_model, obs)
        if not sel:
            break
        try:
            node = P.cg_api.search_step(node.searchId, sel)
        except ValueError:
            break
        prev_actor = actor
        steps += 1
    if l1 is None and last_state is not None:      # 地平未到達 -> 最終盤面で代用(要フラグ)
        l1 = float(l1_eval.evaluate(last_state, me))
        l1_info = _snapshot(last_state, me)
        l1_info["horizon_reached"] = False
    elif l1_info is not None:
        l1_info.setdefault("horizon_reached", True)
    return {"l1": l1, "l1_info": l1_info, "terminal": term, "steps": steps,
            "l2_state": _snapshot(last_state, me) if last_state is not None else None}


def _snapshot(st, me):
    try:
        mine, opp = st.players[me], st.players[1 - me]
        return {"prize_me": len(mine.prize) if mine.prize is not None else None,
                "prize_opp": len(opp.prize) if opp.prize is not None else None,
                "deck_me": int(getattr(mine, "deckCount", -1)),
                "deck_opp": int(getattr(opp, "deckCount", -1)),
                "turn": int(getattr(st, "turn", 0) or 0)}
    except Exception:                              # noqa: BLE001
        return {}


def _det_hash(hs):
    """決定化(隠れ情報)の同一性を確認するためのハッシュ。§8 CRN 実現率の実測に使う。"""
    try:
        import hashlib
        key = repr(sorted((k, str(v)) for k, v in hs.items()))
        return hashlib.md5(key.encode()).hexdigest()[:12]
    except Exception:                              # noqa: BLE001
        return None


def _one_branch(obs, me, action, seed, l1_eval, policy_model):
    """決定化を seed で固定して 1 branch 実行(CRN: 同 seed なら同じ隠れ情報)。"""
    rng = random.Random(seed)
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=rng)
    except Exception:                              # noqa: BLE001
        STATS["determinize_fail"] += 1
        return None
    if hs is None:
        STATS["determinize_none"] += 1
        return None
    dh = _det_hash(hs)
    try:
        root = P._begin(obs, hs)
    except Exception:                              # noqa: BLE001
        STATS["begin_fail"] += 1
        return None
    child = None
    try:
        try:
            child = P.cg_api.search_step(root.searchId, [action])
        except ValueError:
            STATS["illegal_forced_action"] += 1
            return None
        r = _continue(child, me, l1_eval, OPTS["max_steps"], policy_model)
        if r is not None:
            r["det_hash"] = dh
        return r
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


def _teacher_pair(obs, me, a_q0, a_s3, policy_model):
    """Q0手 と S3手 の**2候補だけ**について、Phase11D/12 と同一仕様の教師を取る。

    4 block x K=6、handcrafted leaf、opponent_depth=1、max_rollout_steps=40。
    L1/L2 の判定には使わない(§4)。後解析(§18/§19)で「教師支持が長期評価へ転移するか」
    を見るためだけに保存する。
    """
    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    cfg = {"opponent_depth": 1, "max_rollout_steps": 40}
    blocks = []
    for _b in range(TEACHER_NB):
        vals = {a_q0: [], a_s3: []}
        for _k in range(TEACHER_K):
            try:
                hs = search_adapter.to_search_begin_kwargs(
                    match_context.get_own_state(me),
                    match_context.get_opponent_state(me), obs)
            except Exception:                      # noqa: BLE001
                continue
            if hs is None:
                continue
            try:
                root = P._begin(obs, hs)
            except Exception:                      # noqa: BLE001
                continue
            try:
                dl = time.perf_counter() + 20.0
                for a in (a_q0, a_s3):
                    try:
                        ch = P.cg_api.search_step(root.searchId, [a])
                    except ValueError:
                        continue
                    try:
                        v = P._rollout_and_eval(ch, me, cfg, dl, ev, policy_model)
                        if v is not None:
                            vals[a].append(v)
                    finally:
                        try:
                            P.cg_api.search_release(ch.searchId)
                        except Exception:          # noqa: BLE001
                            pass
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:                  # noqa: BLE001
                    pass
        if min(len(vals[a_q0]), len(vals[a_s3])) < TEACHER_K:
            try:
                P.cg_api.search_end()
            except Exception:                      # noqa: BLE001
                pass
            return None                            # K 未達 block があれば教師なし
        blocks.append([statistics.mean(vals[a_q0][:TEACHER_K]),
                       statistics.mean(vals[a_s3][:TEACHER_K])])
    try:
        P.cg_api.search_end()
    except Exception:                              # noqa: BLE001
        pass
    # vote: Q0手 が S3手 より良ければ 1(= teacher supports Q0)
    diffs = [b[0] - b[1] for b in blocks]
    votes = [0.5 if abs(d) < TEACHER_TIE else (1.0 if d > 0 else 0.0) for d in diffs]
    support_q0 = statistics.mean(votes)
    signs = [0 if abs(d) < TEACHER_TIE else (1 if d > 0 else -1) for d in diffs]
    nz = [x for x in signs if x != 0]
    cls = ("always_tie" if not nz else
           "stable" if max(nz.count(1), nz.count(-1)) == TEACHER_NB else
           "mostly" if max(nz.count(1), nz.count(-1)) == TEACHER_NB - 1 else "unstable")
    return {"support_q0": support_q0,
            "confidence": round(2 * abs(support_q0 - 0.5), 4),
            "mean_margin": round(statistics.mean(diffs), 6),
            "cls": cls, "blocks": [[round(x, 6) for x in b] for b in blocks]}


def t_pre(state):
    return int(getattr(state, "turn", 0) or 0)


def _probe(obs, policy_model):
    select, state = obs.select, obs.current
    me = state.yourIndex
    try:
        pscores = policy_model.score_options(obs, None, None)
    except Exception:                              # noqa: BLE001
        return None
    if not pscores or len(pscores) != len(select.option):
        return None
    ranked = sorted(range(len(pscores)), key=lambda i: pscores[i], reverse=True)
    cand = ranked[:MAX_CANDS]
    if len(cand) < 2:
        return None
    try:
        sf = encoder.encode_state_from_state(state)
        orow = encoder.encode_options_from_state(state, select)
        cids = encoder.encode_option_card_ids(state, select)
    except Exception:                              # noqa: BLE001
        return None
    orows = [orow[i] for i in cand]
    cid = [int(cids[i]) if cids[i] is not None else -1 for i in cand]
    otp = [int(getattr(select.option[i].type, "value", select.option[i].type)) for i in cand]

    dseed = (hash((_ctx["game"], _ctx["in_game"], t_pre(state))) & 0x7FFFFFFF)
    before, afters, dl, ent_afters = _candidate_deltas(obs, me, cand, dseed)
    if dl is None:
        return None
    # delta replay sanity: 同じ seed でもう一度作って一致するか
    _b2, _a2, dl2, _e2 = _candidate_deltas(obs, me, cand, dseed)
    STATS["delta_replay_checked"] += 1
    if dl2 is None or any(abs(x - y) > 1e-6 for r1, r2 in zip(dl, dl2)
                          for x, y in zip(r1, r2)):
        STATS["delta_replay_mismatch"] += 1

    qs = {"Q0": q_scores(MODELS["Q0"], sf, orows, cid, otp),
          "S3": eq_scores(MODELS["EQ"], orows, cid, EX.extract(state, me), ent_afters)}
    a_q0 = cand[max(range(len(cand)), key=lambda n: qs["Q0"][n])]
    a_s3 = cand[max(range(len(cand)), key=lambda n: qs["S3"][n])]
    a_pol = cand[0]
    STATS["observed"] += 1
    if a_q0 == a_s3:
        STATS["agree"] += 1
        return None
    STATS["disagree"] += 1

    t = int(getattr(state, "turn", 0) or 0)
    if not QUOTA.accept(t, len(cand), _ctx["opp"], _ctx["in_game"]):
        return None

    n0 = cand.index(a_q0)
    n3 = cand.index(a_s3)
    base = (hash((_ctx["game"], _ctx["in_game"])) & 0xFFFFFF) * 1000
    runs = {"Q0": [], "S3": [], "NULL": []}
    for c in range(OPTS["n_cont"]):
        seed = base + c
        for tag, act in (("Q0", a_q0), ("S3", a_s3)):
            r = _one_branch(obs, me, act, seed, EVALS["l1"], policy_model)
            if r is None:
                return None                        # 片方でも欠けたら pair にならない
            runs[tag].append(r)
    if OPTS["null_control"] > 0 and random.random() < OPTS["null_control"]:
        for c in range(OPTS["n_cont"]):            # Q0 と同一行動・同一 seed をもう一度
            r = _one_branch(obs, me, a_q0, base + c, EVALS["l1"], policy_model)
            if r is not None:
                runs["NULL"].append(r)

    def agg(rs):
        l1 = [x["l1"] for x in rs if x["l1"] is not None]
        term = [x["terminal"] for x in rs]
        out = {"l1_mean": statistics.mean(l1) if l1 else None,
               "l1_values": [round(v, 6) for v in l1],
               "terminal_rate": sum(1 for x in term if x is not None) / len(rs),
               "steps_mean": statistics.mean([x["steps"] for x in rs]),
               "horizon_rate": statistics.mean(
                   [1.0 if (x["l1_info"] or {}).get("horizon_reached") else 0.0 for x in rs]),
               }
        oc = []
        for x in rs:
            if x["terminal"] is None:
                continue
            oc.append(1.0 if x["terminal"] == me else (0.0 if x["terminal"] == 1 - me else 0.5))
        out["l2_outcomes"] = oc
        out["l2_mean"] = statistics.mean(oc) if oc else None
        pd = [(x["l2_state"] or {}).get("prize_opp", 0) - (x["l2_state"] or {}).get("prize_me", 0)
              for x in rs if x["l2_state"]]
        out["prize_diff_mean"] = statistics.mean(pd) if pd else None
        return out

    same_det = sum(1 for a, b in zip(runs["Q0"], runs["S3"])
                   if a.get("det_hash") and a["det_hash"] == b.get("det_hash"))
    STATS["det_pairs"] += len(runs["Q0"])
    STATS["det_shared"] += same_det
    ident = sum(1 for a, b in zip(runs["Q0"], runs.get("NULL", []))
                if a["l1"] == b["l1"] and a["terminal"] == b["terminal"])
    if runs.get("NULL"):
        STATS["null_pairs"] += len(runs["NULL"])
        STATS["null_identical"] += ident

    teacher = _teacher_pair(obs, me, a_q0, a_s3, policy_model)
    if teacher is None:
        STATS["teacher_fail"] += 1

    rec = {
        "teacher": teacher,
        "group_id": "l{}_{}".format(_ctx["game"], _ctx["in_game"]),
        "game": _ctx["game"], "turn": t, "turn_band": turn_band(t),
        "cand_band": cand_band(len(cand)), "arch": _ctx["opp"],
        "me_first": bool(_ctx["first"]), "n_cands": len(cand),
        "action_type_q0": otp[n0], "action_type_s3": otp[n3],
        "q0_action": a_q0, "s3_action": a_s3, "policy_action": a_pol,
        "q0_margin": round(sorted(qs["Q0"], reverse=True)[0]
                           - sorted(qs["Q0"], reverse=True)[1], 6),
        "s3_margin": round(sorted(qs["S3"], reverse=True)[0]
                           - sorted(qs["S3"], reverse=True)[1], 6),
        "q0_scores": [round(v, 6) for v in qs["Q0"]],
        "s3_scores": [round(v, 6) for v in qs["S3"]],
        "delta_seed": dseed,
        "before": [round(x, 4) for x in before],
        "cand_deltas": [[round(x, 6) for x in r] for r in dl],
        "after_q0": ([round(x, 4) for x in afters[a_q0]]
                     if afters.get(a_q0) is not None else None),
        "after_delta": ([round(x, 4) for x in afters[a_s3]]
                        if afters.get(a_s3) is not None else None),
        "entity_before": EX.extract(state, me),
        "delta_q0": [round(x, 6) for x in dl[n0]],
        "delta_dq": [round(x, 6) for x in dl[n3]],
        "delta_distance": round(DF.pair_distance(dl[n0], dl[n3]), 6),
        "delta_identical": bool(DF.pair_distance(dl[n0], dl[n3]) < 1e-9),
        "Q0": agg(runs["Q0"]), "S3": agg(runs["S3"]),
    }
    if runs["NULL"]:
        rec["NULL"] = agg(runs["NULL"])
    return rec


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.MAIN and sel.maxCount == 1
                and len(sel.option) >= 2 and QUOTA.total < QUOTA.target):
            try:
                r = _probe(obs, ml_policy_agent._get_model(config))
                if r:
                    GROUPS.append(r)
                    QUOTA.commit(obs.current.turn or 0, r["n_cands"], _ctx["opp"])
                    _ctx["in_game"] += 1
            except Exception as exc:               # noqa: BLE001
                STATS["probe_err"] += 1
                print("[probe-err] {}".format(exc), file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--per-game-cap", type=int, default=4)
    ap.add_argument("--n-cont", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--null-control", type=float, default=0.0)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=600000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="l1l2")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(n_cont=args.n_cont, max_steps=args.max_steps,
                null_control=args.null_control)

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap, cand_soft_cap=0.45)
    MODELS["Q0"] = load_q(_FROZEN / "Q0-expanded.pt")
    MODELS["EQ"] = load_entity_q(_FROZEN / "EntityQ_EPool.pt")
    EVALS["l1"] = leaf_eval_module.build_evaluator({"kind": "value"})   # 教師と別の凍結評価器
    _install()

    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
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
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch,
                    first=p0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        if gi % 5 == 0:
            print("  [w{}] game#{} groups={}/{} obs={} disagree={} {}min".format(
                args.worker_id, g, QUOTA.total, args.target, STATS["observed"],
                STATS["disagree"], int((time.perf_counter() - t0) / 60)),
                file=sys.stderr, flush=True)

    out = _HERE / "_p18_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "tag": args.tag, "n_groups": len(GROUPS), "stats": dict(STATS),
        "disagreement_rate": round(STATS["disagree"] / max(1, STATS["observed"]), 4),
        "crn_determinization_shared": round(STATS["det_shared"] / max(1, STATS["det_pairs"]), 4),
        "crn_trajectory_identical": round(STATS["null_identical"] / max(1, STATS["null_pairs"]), 4),
        "n_cont": OPTS["n_cont"], "max_steps": OPTS["max_steps"],
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "turn_band": dict(Counter(r["turn_band"] for r in GROUPS)),
        "arch": dict(Counter(r["arch"] for r in GROUPS)),
        "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
