"""Phase19.5: Multi-Horizon bootstrapped target dataset。

1本の trajectory から H0/H1/H2/H4/HT を同時に記録する(§8)。horizon ごとに
別々の simulation を回さない。H_k = 「自分の k 回目の次ターン終了時点」で、
そこに到達する前に決着したら terminal outcome をそのまま使う(§10)。

各候補を1手だけ強制し、**同一 continuation policy** で terminal まで進めた勝敗の平均を
target にする(§5)。候補ごとに continuation policy を変えない。

  y_LH(s,a) = mean_{m<M} outcome(force a, continuation(seed_m))

CRN(§7): continuation index m の determinization seed を**候補間で共有**する。
audit subset では **独立な第2ブロック**を生成し、
  - §14/§15 target reliability
  - §16 null control(同じ候補を2回評価した差)
の両方をそこから取る。

短期 teacher(4block x K=6)も保存するが、**Long-Horizon Q の target には使わない**(§13)。
情報リーク禁止: 相手手札・山札順・未来は保存しない。
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
import entity_tokens as ET  # noqa: E402
import train_transition as TT  # noqa: E402
from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from action_q import ActionQNet  # noqa: E402
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
NB, K, TIE = 4, 6, 0.005        # 短期 teacher(Phase11D/12 と同一)
HIST_N = 8

GROUPS: list[dict] = []
STATS = Counter()
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "first": True,
        "hist": deque(maxlen=HIST_N), "rows": []}
QUOTA = None
MODELS: dict = {}
EVALS: dict = {}
OPTS = {"m": 8, "max_steps": 300, "audit_every": 4, "tag": "x"}


def load_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ActionQNet(ck["state_dim"], ck["option_dim"], use_cards=False,
                   use_action_card=ck.get("use_action_card", True))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], np.float32),
            "std": np.asarray(ck["std"], np.float32)}


def load_entity_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TT.PoolArm(ck["option_dim"], mode="transition")
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], np.float32),
            "std": np.asarray(ck["std"], np.float32)}


@torch.no_grad()
def q0_scores(b, sf, orows, cids, otypes):
    st = torch.from_numpy(((np.asarray(sf, np.float32) - b["mean"]) / b["std"])[None, :])
    C = len(orows)
    q, _ = b["model"](st, [torch.zeros((1, 1), dtype=torch.long)] * 3,
                      torch.from_numpy(np.asarray(orows, np.float32)[None, ...]),
                      torch.tensor([[c if 0 < c < 2048 else 0 for c in cids]], dtype=torch.long),
                      torch.tensor([[min(t, 63) for t in otypes]], dtype=torch.long),
                      torch.ones(1, C, dtype=torch.bool))
    return q[0].tolist()


@torch.no_grad()
def eq_scores(b, orows, cids, ent_before, ent_afters):
    tb = ET.tokenize(ent_before)
    pb = TT._pack([tb], [ET.relation_matrix(tb)])
    at, ar = [], []
    for ae in ent_afters:
        t = ET.tokenize(ae) if ae is not None else tb
        at.append(t)
        ar.append(ET.relation_matrix(t) if ae is not None else ET.relation_matrix(tb))
    pa = TT._pack(at, ar)
    C = len(orows)
    batch = {"opt": torch.from_numpy(np.asarray(orows, np.float32)[None, ...]),
             "card": torch.tensor([[c if 0 < c < ET.VOCAB else 0 for c in cids]],
                                  dtype=torch.long),
             "mask": torch.ones(1, C, dtype=torch.bool), "before": pb,
             "after": {k: v.view(1, C, *v.shape[1:]) for k, v in pa.items()}}
    return b["model"](batch)[0].tolist()


HORIZONS = (1, 2, 4)          # H1/H2/H4 = 自分の 1/2/4 回目の「次ターン終了」


def _continue_to_terminal(node, me, policy_model, max_steps, value_eval=None):
    """同一 continuation policy(climb Policy 貪欲)で terminal まで。

    途中で H0/H1/H2/H4 の checkpoint を記録する(§9)。
    own_turn_ends は me -> 相手 の遷移回数。forced action は**現在のターン中**に
    起きるので、own_turn_ends=1 は現ターン終了、=2 が「自分の次ターン終了」= H1。
    Phase18 L1 と同じ意味になるよう k+1 で対応付ける。
    """
    steps, term = 0, None
    last = None
    hz = {}
    st0 = node.observation.current
    if value_eval is not None and st0 is not None:
        hz["H0"] = (1.0 if st0.result == me else 0.0) if st0.result != -1             else float(value_eval.evaluate(st0, me))
    own_turn_ends = 0
    prev_actor = me
    while steps < max_steps:
        obs = node.observation
        st = obs.current
        if st is None:
            break
        last = st
        if st.result != -1:
            term = int(st.result)
            break
        actor = st.yourIndex
        if prev_actor == me and actor != me:
            own_turn_ends += 1
            k = own_turn_ends - 1          # k 回目の「次ターン終了」
            if k in HORIZONS and value_eval is not None:
                hz["H%d" % k] = float(value_eval.evaluate(st, me))
            if OPTS.get("stop_h") and k >= OPTS["stop_h"]:
                break                      # H* まで来たら打ち切る
        prev_actor = actor
        if obs.select is None or not obs.select.option:
            break
        sel = P._greedy_selection(policy_model, obs)
        if not sel:
            break
        try:
            node = P.cg_api.search_step(node.searchId, sel)
        except ValueError:
            break
        steps += 1
    outcome = None
    if term is not None:
        outcome = 1.0 if term == me else (0.0 if term == 1 - me else 0.5)
    # §10 horizon 到達前に決着したら、その先の horizon は terminal outcome を使う
    for k in HORIZONS:
        key = "H%d" % k
        if key not in hz:
            hz[key] = outcome if outcome is not None else hz.get("H0")
    if "H0" not in hz:
        hz["H0"] = outcome
    reason = None
    if last is not None:
        try:
            mine, opp = last.players[me], last.players[1 - me]
            if int(getattr(mine, "deckCount", 1)) == 0:
                reason = "deckout_self"
            elif int(getattr(opp, "deckCount", 1)) == 0:
                reason = "deckout_opp"
            elif not (mine.active or []):
                reason = "no_pokemon_self"
            elif not (opp.active or []):
                reason = "no_pokemon_opp"
            else:
                reason = "prize"
        except Exception:                          # noqa: BLE001
            pass
    return {"outcome": outcome, "steps": steps, "reason": reason, "hz": hz}


def _rollout_candidate(obs, me, action, seed, policy_model):
    """1候補 x 1 continuation。決定化 seed 固定(候補間で共有 = CRN)。"""
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
        return _continue_to_terminal(child, me, policy_model, OPTS["max_steps"],
                                     value_eval=EVALS.get("value"))
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


def _long_horizon_block(obs, me, cand, policy_model, seeds):
    """候補 x len(seeds) の terminal 継続。seeds は候補間で共有(§7)。"""
    out = {}
    for i in cand:
        rs = []
        for s in seeds:
            r = _rollout_candidate(obs, me, i, s, policy_model)
            # stop_h 指定時は terminal に到達しないので outcome は None。
            # その場合は目的の horizon 値があれば採用する。
            if r is None:
                continue
            need = "H%d" % OPTS["stop_h"] if OPTS.get("stop_h") else None
            if (r["outcome"] is not None) or (need and r["hz"].get(need) is not None):
                rs.append(r)
        if not rs:
            return None
        out[i] = rs
    return out


def _short_teacher(obs, me, cand, policy_model):
    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    cfg = {"opponent_depth": 1, "max_rollout_steps": 40}
    blocks = []
    for _b in range(NB):
        vals = {i: [] for i in cand}
        for _k in range(K):
            try:
                hs = search_adapter.to_search_begin_kwargs(
                    match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
                root = P._begin(obs, hs)
            except Exception:                      # noqa: BLE001
                continue
            try:
                dl = time.perf_counter() + 30.0
                for i in cand:
                    try:
                        ch = P.cg_api.search_step(root.searchId, [i])
                    except ValueError:
                        continue
                    try:
                        v = P._rollout_and_eval(ch, me, cfg, dl, ev, policy_model)
                        if v is not None:
                            vals[i].append(v)
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
        if min((len(vals[i]) for i in cand), default=0) < K:
            try:
                P.cg_api.search_end()
            except Exception:                      # noqa: BLE001
                pass
            return None
        blocks.append([statistics.mean(vals[i][:K]) for i in cand])
    try:
        P.cg_api.search_end()
    except Exception:                              # noqa: BLE001
        pass
    return blocks


def _flush():
    out = _HERE / "_mh_{}.jsonl.gz".format(OPTS["tag"])
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    tmp.replace(out)


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
    cand = ranked[:TOP_K]                          # §9 Policy top-k(8未満なら全候補)
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
        STATS["encode_fail"] += 1
        return
    orows = [[round(float(x), 6) for x in orow[i]] for i in cand]
    cid = [int(cids[i]) if cids[i] is not None else -1 for i in cand]
    otp = [int(getattr(select.option[i].type, "value", select.option[i].type)) for i in cand]

    base = (hash((_ctx["game"], _ctx["in_game"], t)) & 0x3FFFFF) * 1000
    seeds_a = [base + m for m in range(OPTS["m"])]
    blk_a = _long_horizon_block(obs, me, cand, policy_model, seeds_a)
    if blk_a is None:
        STATS["lh_fail"] += 1
        return
    # audit subset: 独立な第2ブロック(§14 reliability + §16 null control)
    is_audit = (OPTS["audit_every"] > 0 and
                (_ctx["in_game"] + len(GROUPS)) % OPTS["audit_every"] == 0)
    blk_b = None
    if is_audit:
        seeds_b = [base + 500000 + m for m in range(OPTS["m"])]
        blk_b = _long_horizon_block(obs, me, cand, policy_model, seeds_b)
        if blk_b is None:
            STATS["audit_fail"] += 1

    short = _short_teacher(obs, me, cand, policy_model) if OPTS.get("short", 1) else None
    if OPTS.get("short", 1) and short is None:
        STATS["short_fail"] += 1

    # after entity(EntityQ 推論に必要)
    ent_afters = []
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(base + 999))
        root = P._begin(obs, hs)
        for i in cand:
            try:
                ch = P.cg_api.search_step(root.searchId, [i])
            except ValueError:
                ent_afters.append(None)
                continue
            try:
                stt = ch.observation.current
                ent_afters.append(EX.extract(stt, me) if stt is not None else None)
            finally:
                try:
                    P.cg_api.search_release(ch.searchId)
                except Exception:                  # noqa: BLE001
                    pass
        P.cg_api.search_release(root.searchId)
        P.cg_api.search_end()
    except Exception:                              # noqa: BLE001
        ent_afters = [None] * len(cand)

    q0 = q0_scores(MODELS["Q0"], sf, orows, cid, otp)
    eq = eq_scores(MODELS["EQ"], orows, cid, ent, ent_afters)

    def agg(rs):
        oc = [x["outcome"] for x in rs if x["outcome"] is not None]
        hz = {}
        for k in ("H0",) + tuple("H%d" % h for h in HORIZONS):
            v = [x["hz"].get(k) for x in rs if x["hz"].get(k) is not None]
            if v:
                hz[k] = {"mean": round(statistics.mean(v), 6),
                         "var": round(statistics.pvariance(v), 6) if len(v) > 1 else 0.0,
                         "values": [round(z, 6) for z in v]}
        return {"outcomes": oc, "hz": hz,
                "mean": round(statistics.mean(oc), 6) if oc else None,
                "var": (round(statistics.pvariance(oc), 6) if len(oc) > 1 else 0.0),
                "len_mean": round(statistics.mean([x["steps"] for x in rs]), 2),
                "reasons": dict(Counter(x["reason"] for x in rs))}

    rec = {
        "group_id": "lh{}_{}".format(_ctx["game"], _ctx["in_game"]),
        "game": _ctx["game"], "turn": t, "turn_band": turn_band(t),
        "cand_band": cand_band(len(cand)), "arch": _ctx["opp"],
        "me_first": _ctx["first"], "n_legal": len(select.option),
        "state_feat": [round(float(x), 6) for x in sf],
        "entity": ent, "history": list(_ctx["hist"]),
        "m": OPTS["m"], "seeds_a": seeds_a, "is_audit": bool(blk_b is not None),
        "candidates": [{
            "option_index": cand[n], "option_feat": orows[n], "action_card_id": cid[n],
            "option_type": otp[n], "policy_score": round(float(ps[cand[n]]), 6),
            "policy_rank": n,
            "q0_score": round(q0[n], 6), "entityq_score": round(eq[n], 6),
            "short_blocks": ([round(short[b][n], 6) for b in range(NB)]
                             if short else None),
            "lh_a": agg(blk_a[cand[n]]),
            "lh_b": agg(blk_b[cand[n]]) if blk_b else None,
            "after_entity": ent_afters[n],
        } for n in range(len(cand))],
    }
    GROUPS.append(rec)
    QUOTA.commit(t, len(cand), _ctx["opp"])
    _ctx["in_game"] += 1
    _ctx["rows"].append(rec)
    STATS["groups"] += 1
    if len(GROUPS) % 10 == 0:
        _flush()


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        rec = (_ctx["recording"] and obs.current is not None
               and obs.current.yourIndex == _ctx["me"])
        sel = obs.select
        if (rec and sel is not None and sel.option and sel.type == SelectType.MAIN
                and sel.maxCount == 1 and len(sel.option) >= 2
                and QUOTA.total < QUOTA.target):
            try:
                _probe(obs, ml_policy_agent._get_model(config))
            except Exception as exc:               # noqa: BLE001
                STATS["probe_err"] += 1
                print("[err]", exc, file=sys.stderr)
        out = orig(obs, config=config)
        if rec and sel is not None and sel.option and out:
            try:
                i = out[0]
                o = sel.option[i]
                cd = encoder.encode_option_card_ids(obs.current, sel)
                _ctx["hist"].append({"type": int(getattr(o.type, "value", o.type)),
                                     "sel_type": int(getattr(sel.type, "value", sel.type)),
                                     "card_id": int(cd[i]) if cd and cd[i] is not None else -1,
                                     "turn": int(getattr(obs.current, "turn", 0) or 0)})
            except Exception:                      # noqa: BLE001
                pass
        return out

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--audit-every", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--per-game-cap", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--stop-h", type=int, default=0)
    ap.add_argument("--short", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=800000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="w0")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(m=args.m, max_steps=args.max_steps, audit_every=args.audit_every,
                tag=args.tag, stop_h=args.stop_h, short=args.short)
    torch.set_num_threads(1)

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap,
                            cand_soft_cap=0.45)
    EVALS["value"] = leaf_eval_module.build_evaluator({"kind": "value"})
    MODELS["Q0"] = load_q(_FROZEN / "Q0-expanded.pt")
    MODELS["EQ"] = load_entity_q(_FROZEN / "EntityQ_EPool.pt")
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
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch,
                    first=p0, rows=[])
        _ctx["hist"].clear()
        res = (runner.play_game(climb, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        won = None
        try:
            w = getattr(res, "winner", None)
            if w is not None and int(w) != -1:
                won = 1.0 if int(w) == _ctx["me"] else 0.0
        except Exception:                          # noqa: BLE001
            pass
        for r in _ctx["rows"]:
            r["game_outcome"] = won
        if gi % 3 == 0:
            print("  [w{}] game#{} groups={}/{} {}min".format(
                args.worker_id, g, QUOTA.total, args.target,
                int((time.perf_counter() - t0) / 60)), file=sys.stderr, flush=True)

    _flush()
    nc = sum(len(r["candidates"]) for r in GROUPS)
    print(json.dumps({
        "tag": args.tag, "n_groups": len(GROUPS), "n_candidates": nc,
        "m": OPTS["m"], "audit_groups": sum(1 for r in GROUPS if r["is_audit"]),
        "total_continuations": nc * OPTS["m"] + sum(
            len(r["candidates"]) * OPTS["m"] for r in GROUPS if r["is_audit"]),
        "stats": dict(STATS), "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "sec_per_group": round((time.perf_counter() - t0) / max(1, len(GROUPS)), 1),
        "turn_band": dict(Counter(r["turn_band"] for r in GROUPS)),
        "arch": dict(Counter(r["arch"] for r in GROUPS))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
