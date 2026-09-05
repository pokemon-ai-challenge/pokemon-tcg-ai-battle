"""Phase19.14 Stage B/C/D: CARD 候補の terminal 価値差(forced-action + terminal continuation)。

production(abl_5_full + climb)を回し、`SelectType.CARD` の decision で

    候補 a_i を **1回だけ** 強制 -> 以降は Frozen production Policy -> terminal

を候補ごとに M 本ずつ実行する(§13/§14)。CARD chain の後続は oracle 操作しない。

対象は C2_genuine と C3_unobservable(デッキサーチ等、候補の識別情報が観測に無い)。
C3 を含めるのは「価値差はあるが構造的に到達不能」かどうかを分けて測るため。

M は group_id の hash で **事前に** 32 / 8 を割り当てる(§24 の optional stopping 回避)。
M=32 の root では C1/C1b/C4 の Frozen Value も記録する(§43、secondary)。
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from cg.api import AreaType, SelectContext, SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
PILE_AREAS = {int(AreaType.HAND), int(AreaType.DECK), int(AreaType.DISCARD),
              int(AreaType.PRIZE), int(AreaType.LOOKING)}

GROUPS: list[dict] = []
STATS: Counter = Counter()
_ctx: dict = {"game": 0, "arch": "?", "me": 0, "first": True, "rec": False,
              "idx": 0, "taken": 0, "chain": 0, "chain_pos": 0, "prev_main": None}
OPTS = {"m": 8, "m_big": 32, "big_pct": 40, "max_steps": 300, "max_cands": 8,
        "only_cls": ("C2_genuine", "C3_unobservable")}
QUOTA: StratifiedQuota | None = None
EV = {"value": None}


def opt_key(o, state):
    cid = ENC._resolve_card_id(o, state)
    a = int(o.area) if o.area is not None else -1
    if a in PILE_AREAS:
        return (int(o.type), a, cid)
    return (int(o.type), a, o.index, o.playerIndex, o.toolIndex, o.energyIndex, cid)


def classify(sel, state):
    n = len(sel.option or [])
    ids = [ENC._resolve_card_id(o, state) for o in (sel.option or [])]
    if n <= 1 or sel.minCount >= n:
        return "C0_trivial", n, ids
    if all(i is None for i in ids):
        return "C3_unobservable", n, ids
    eff = len({opt_key(o, state) for o in sel.option})
    return ("C1_dup_equivalent" if eff <= 1 else "C2_genuine"), eff, ids


def _forced_run(obs, me, action, seed, policy_model, want_ckpt):
    """1手だけ強制 -> Policy 貪欲 -> terminal。want_ckpt なら C1/C1b/C4 の Value も返す。"""
    ev = EV["value"]
    ck = {}
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(seed))
        root = P._begin(obs, hs)
    except Exception:                                          # noqa: BLE001
        STATS["begin_fail"] += 1
        return None
    child = None
    try:
        try:
            child = P.cg_api.search_step(root.searchId, [action])
        except ValueError:
            STATS["illegal_force"] += 1
            return None
        node = child
        if want_ckpt:
            s0 = node.observation.current
            if s0 is not None:
                ck["C1"] = float(ev.evaluate(s0, me))
        prev, steps = me, 0
        me_to_opp = opp_to_me = 0
        while steps < OPTS["max_steps"]:
            o = node.observation
            s = o.current
            if s is None:
                break
            if s.result != -1:
                r = int(s.result)
                return {"j": 1.0 if r == me else (0.0 if r == 1 - me else 0.5),
                        "terminal": True, "steps": steps, "ckpt": ck}
            actor = s.yourIndex
            if want_ckpt:
                if prev == me and actor != me:
                    me_to_opp += 1
                    if me_to_opp == 1:
                        ck["C1b"] = float(ev.evaluate(s, me))
                    elif me_to_opp == 2:
                        ck["C4"] = float(ev.evaluate(s, me))
                if prev != me and actor == me:
                    opp_to_me += 1
            if o.select is None or not o.select.option:
                break
            sel = P._greedy_selection(policy_model, o)
            if not sel:
                break
            try:
                node = P.cg_api.search_step(node.searchId, sel)
            except ValueError:
                break
            prev = actor
            steps += 1
        STATS["no_terminal"] += 1
        return {"j": None, "terminal": False, "steps": steps, "ckpt": ck}
    finally:
        for sid in (child.searchId if child is not None else None, root.searchId):
            if sid is not None:
                try:
                    P.cg_api.search_release(sid)
                except Exception:                              # noqa: BLE001
                    pass
        try:
            P.cg_api.search_end()
        except Exception:                                      # noqa: BLE001
            pass


def audit(obs, me, sel, cls, ids, policy_model, cfg_full, gid, seed):
    # 実効候補(C2 は重複除去、C3 は識別不能なので全候補をそのまま)
    if cls == "C2_genuine":
        seen, cand = {}, []
        for i, o in enumerate(sel.option):
            k = opt_key(o, obs.current)
            if k not in seen:
                seen[k] = i
                cand.append(i)
    else:
        cand = list(range(len(sel.option)))
    capped = len(cand) > OPTS["max_cands"]
    if capped:
        cand = cand[:OPTS["max_cands"]]
    if len(cand) < 2:
        return None
    big = int(hashlib.sha1(gid.encode()).hexdigest(), 16) % 100 < OPTS["big_pct"]
    m = OPTS["m_big"] if big else OPTS["m"]
    probs = None
    try:
        sc = policy_model.score_options(
            obs, MA._model_hidden_state_factory(obs, cfg_full),
            time.perf_counter() + 5.0)
        if sc and len(sc) == len(sel.option):
            probs = P._softmax(sc)
    except Exception:                                          # noqa: BLE001
        STATS["policy_fail"] += 1
    rows = []
    for i in cand:
        runs = []
        for k in range(m):
            r = _forced_run(obs, me, i, seed + 1000 * k, policy_model, big)
            if r is not None:
                runs.append(r)
        js = [r["j"] for r in runs if r["j"] is not None]
        if len(js) < max(4, m // 4):
            return None
        rows.append({"idx": i, "card_id": ids[i] if i < len(ids) else None,
                     "policy_prob": round(probs[i], 6) if probs else None,
                     "j": js,
                     "terminal_rate": round(sum(1 for r in runs if r["terminal"])
                                            / len(runs), 3),
                     "ckpt": [r["ckpt"] for r in runs] if big else None})
    return {"cands": rows, "M": m, "big": big, "capped": capped,
            "n_legal_options": len(sel.option), "n_effective": len(cand)}


def make_probe(inner, cfg_full, policy_model):
    def probe(obs):
        sel = obs.select
        if sel is None or not _ctx["rec"]:
            return inner(obs)
        stt = obs.current
        if sel.type == SelectType.MAIN:
            _ctx["chain"] += 1
            _ctx["chain_pos"] = 0
        act = inner(obs)
        _ctx["idx"] += 1
        if sel.type == SelectType.MAIN and act:
            o = sel.option[act[0]] if act[0] < len(sel.option) else None
            _ctx["prev_main"] = {"type": int(o.type),
                                 "cardId": ENC._resolve_card_id(o, stt)} if o else None
        if sel.type != SelectType.CARD or stt is None or stt.result != -1:
            return act
        _ctx["chain_pos"] += 1
        # 単一選択(min<=1<=max)に限定。複数選択は forced 意味論が別問題。
        if not (sel.minCount <= 1 <= sel.maxCount):
            STATS["skip_multiselect"] += 1
            return act
        cls, eff, ids = classify(sel, stt)
        if cls not in OPTS["only_cls"]:
            return act
        turn = int(getattr(stt, "turn", 0))
        tb, cb = turn_band(turn), cand_band(len(sel.option))
        if QUOTA is None or not QUOTA.accept(turn, len(sel.option), _ctx["arch"],
                                             _ctx["taken"]):
            return act
        me = stt.yourIndex
        gid = "g{}_{}".format(_ctx["game"], _ctx["idx"])
        seed = 6_000_000 + _ctx["game"] * 997 + _ctx["idx"]
        t0 = time.perf_counter()
        try:
            res = audit(obs, me, sel, cls, ids, policy_model, cfg_full, gid, seed)
        except Exception as exc:                               # noqa: BLE001
            STATS["audit_err_" + type(exc).__name__] += 1
            return act
        if res is None:
            STATS["audit_none"] += 1
            return act
        res.update({"group_id": gid, "game": _ctx["game"], "arch": _ctx["arch"],
                    "turn": turn, "turn_band": tb, "cand_band": cb, "cls": cls,
                    "context": int(sel.context) if sel.context is not None else -1,
                    "context_name": SelectContext(int(sel.context)).name
                    if sel.context is not None
                    and int(sel.context) in SelectContext._value2member_map_
                    else str(sel.context),
                    "me_first": _ctx["first"], "chain_pos": _ctx["chain_pos"],
                    "prev_main": _ctx["prev_main"],
                    "production_selected": act[0] if act else None,
                    "audit_ms": round((time.perf_counter() - t0) * 1000, 1)})
        GROUPS.append(res)
        QUOTA.commit(turn, len(sel.option), _ctx["arch"])
        _ctx["taken"] += 1
        STATS[cls] += 1
        STATS["M32" if res["big"] else "M8"] += 1
        return act
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--per-game-cap", type=int, default=3)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--m-big", type=int, default=32)
    ap.add_argument("--big-pct", type=int, default=40)
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=4200000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--only-cls", default="C2_genuine,C3_unobservable")
    ap.add_argument("--tag", default="p0")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(m=args.m, m_big=args.m_big, big_pct=args.big_pct,
                max_cands=args.max_cands,
                only_cls=tuple(x for x in args.only_cls.split(",") if x))

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap,
                            cand_soft_cap=0.50)
    EV["value"] = leaf_eval_module.build_evaluator({"kind": "value"})

    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    policy_model = MA._get_model({"policy_weights_path": CLIMB})
    probe = make_probe(climb, cfg, policy_model)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for gi in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("abl_5_full")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, arch=arch, me=0 if p0 else 1, first=p0, rec=True,
                    idx=0, taken=0, chain=0, chain_pos=0, prev_main=None)
        (runner.play_game(probe, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, probe, deck_o, deck_c))
        _ctx["rec"] = False
        if gi % 3 == 0:
            print("  [w{}] game#{} groups={}/{} C2={} C3={} {}min".format(
                args.worker_id, g, QUOTA.total, args.target, STATS["C2_genuine"],
                STATS["C3_unobservable"], int((time.perf_counter() - t0) / 60)),
                file=sys.stderr, flush=True)

    out = _HERE / "_p1914q_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(json.dumps({"tag": args.tag, "n": len(GROUPS), "stats": dict(STATS)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
