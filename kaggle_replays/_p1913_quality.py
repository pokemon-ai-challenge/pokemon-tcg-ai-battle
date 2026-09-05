"""Phase19.13 Stage B/C/D: leaf ranking の terminal 妥当性 と shortcut の changed-action 品質。

production(abl_5_full + climb)を回しながら、到達した decision root でそのまま監査する
(root observation は保存できないため in situ。Phase11D 以降と同じ制約)。

root の種類で分岐:

  * **searched root**(production が pipeline を実走)
      production と同じ手順で top-k 候補 × N 決定化のロールアウトを再現し、
      **葉に着いた時点で** handcrafted スコアを記録 -> そこから **さらに terminal まで**
      同じ Policy 貪欲で続ける。
        - 葉単位の (handcrafted, terminal) 対 -> §14 absolute validity
        - 候補単位の (mean handcrafted, mean terminal) -> §15/§16 within-root ranking
      handcrafted は葉で、outcome は葉より**後**の軌跡から来るので、
      「同じ rollout の結果で自分を採点する」ことにはならない(§12)。

  * **shortcut root**(top1_shortcut_prob で探索が省略された)
      shortcut が選んだ手 と full-search が選ぶ手 を比較し、**食い違ったときだけ**
      両者を forced-action で paired 評価(M=8, terminal)。§27/§33。

注意(報告に必ず書く): terminal outcome は **決定化された世界**の中の結果であり、
真の相手デッキではない。候補間比較は同一決定化で paired なので順位付けの評価には使えるが、
絶対勝率としては読まない。
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import statistics as st
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
from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402


def _hand_terms(state, me):
    """HandcraftedEvaluator が実際に見ている4差分(§21 error taxonomy 用)。"""
    from ptcg_ai.search.leaf_eval import _board_strength, _bench_count, _energy_on_board
    mine, opp = state.players[me], state.players[1 - me]
    return {"prize": len(opp.prize or []) - len(mine.prize or []),
            "board": round(_board_strength(mine) - _board_strength(opp), 4),
            "energy": _energy_on_board(mine) - _energy_on_board(opp),
            "bench": _bench_count(mine) - _bench_count(opp),
            "hand": (getattr(mine, "handCount", 0) or 0) - (getattr(opp, "handCount", 0) or 0),
            "deck_me": int(getattr(mine, "deckCount", -1)),
            "deck_opp": int(getattr(opp, "deckCount", -1))}

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

GROUPS: list[dict] = []
STATS: Counter = Counter()
_ctx: dict = {"game": 0, "arch": "?", "me": 0, "first": True, "rec": False, "idx": 0, "taken": 0}
OPTS = {"n_det": 8, "max_steps": 300, "m_changed": 8}
QUOTA: StratifiedQuota | None = None
EV = {"hand": None}


def _begin(obs, seed):
    me = obs.current.yourIndex
    hs = search_adapter.to_search_begin_kwargs(
        match_context.get_own_state(me), match_context.get_opponent_state(me),
        obs, rng=random.Random(seed))
    return P._begin(obs, hs)


def _rollout_to_leaf(node, me, cfg, policy_model):
    """production の `_rollout_and_eval` と同じ停止条件で葉ノードまで進め、そのノードを返す。"""
    opponent_depth = max(1, int(cfg["opponent_depth"]))
    max_steps = int(cfg["max_rollout_steps"])
    prev, opp_turns = me, 0
    for _ in range(max_steps):
        o = node.observation
        s = o.current
        if s is None:
            return None
        if s.result != -1:
            return node
        actor = s.yourIndex
        if prev == me and actor != me:
            opp_turns += 1
        if actor == me and prev != me and opp_turns >= opponent_depth:
            return node
        if o.select is None or not o.select.option:
            return node
        sel = P._greedy_selection(policy_model, o)
        if not sel:
            return node
        try:
            node = P.cg_api.search_step(node.searchId, sel)
        except ValueError:
            return node
        prev = actor
    return node


def _to_terminal(node, me, policy_model, max_steps):
    """葉から先を Policy 貪欲で terminal まで進める。決着しなければ None。"""
    steps = 0
    while steps < max_steps:
        o = node.observation
        s = o.current
        if s is None:
            return None, steps
        if s.result != -1:
            r = int(s.result)
            return (1.0 if r == me else (0.0 if r == 1 - me else 0.5)), steps
        if o.select is None or not o.select.option:
            return None, steps
        sel = P._greedy_selection(policy_model, o)
        if not sel:
            return None, steps
        try:
            node = P.cg_api.search_step(node.searchId, sel)
        except ValueError:
            return None, steps
        steps += 1
    return None, steps


def _policy_view(obs, cfg_full, policy_model):
    """production と同じ Policy スコア -> probs / ranked / candidates。"""
    deadline = time.perf_counter() + 10.0
    factory = MA._model_hidden_state_factory(obs, cfg_full)
    scores = policy_model.score_options(obs, factory, deadline)
    if not scores or len(scores) != len(obs.select.option):
        return None
    probs = P._softmax(scores)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return probs, ranked


def audit_searched(obs, me, pcfg, cfg_full, policy_model, seed):
    """§10-§18: 葉 handcrafted と、その葉から先の terminal を対で集める。"""
    pv = _policy_view(obs, cfg_full, policy_model)
    if pv is None:
        return None
    probs, ranked = pv
    cand = P._select_candidate_indices(obs.select, ranked, pcfg, probs)
    if len(cand) < 2:
        return None
    ev = EV["hand"]
    leaves: list[dict] = []
    per_cand: dict[int, dict] = {i: {"h": [], "j": []} for i in cand}
    for d in range(OPTS["n_det"]):
        try:
            root = _begin(obs, seed + 1000 * d)
        except Exception:                                  # noqa: BLE001
            STATS["begin_fail"] += 1
            continue
        try:
            for i in cand:
                try:
                    child = P.cg_api.search_step(root.searchId, [i])
                except ValueError:
                    continue
                try:
                    leaf = _rollout_to_leaf(child, me, pcfg, policy_model)
                    if leaf is None:
                        continue
                    ls = leaf.observation.current
                    if ls is None:
                        continue
                    h = float(ev.evaluate(ls, me))
                    term = int(ls.result) if ls.result != -1 else None
                    j, steps = ((1.0 if term == me else (0.0 if term == 1 - me else 0.5)), 0) \
                        if term is not None else _to_terminal(leaf, me, policy_model,
                                                              OPTS["max_steps"])
                    leaves.append({"cand": i, "det": d, "h": round(h, 6),
                                   "j": j, "leaf_terminal": term is not None,
                                   "steps": steps, "terms": _hand_terms(ls, me)})
                    per_cand[i]["h"].append(h)
                    if j is not None:
                        per_cand[i]["j"].append(j)
                finally:
                    try:
                        P.cg_api.search_release(child.searchId)
                    except Exception:                      # noqa: BLE001
                        pass
        finally:
            try:
                P.cg_api.search_release(root.searchId)
            except Exception:                              # noqa: BLE001
                pass
            try:
                P.cg_api.search_end()
            except Exception:                              # noqa: BLE001
                pass
    cands = []
    for i in cand:
        h, j = per_cand[i]["h"], per_cand[i]["j"]
        if len(h) < 2 or len(j) < 4:
            continue
        cands.append({"idx": i, "mean_h": round(st.mean(h), 6), "n_h": len(h),
                      "mean_j": round(st.mean(j), 6), "n_j": len(j),
                      "policy_prob": round(probs[i], 6)})
    if len(cands) < 2:
        return None
    return {"kind": "searched", "cands": cands, "leaves": leaves,
            "policy_top1": ranked[0], "top1_prob": round(max(probs), 6)}


def _forced_terminal(obs, me, action, policy_model, seeds):
    """1手だけ強制 -> 以降 Policy 貪欲 -> terminal。seeds ごとに1本。"""
    out = []
    for sd in seeds:
        try:
            root = _begin(obs, sd)
        except Exception:                                  # noqa: BLE001
            continue
        child = None
        try:
            try:
                child = P.cg_api.search_step(root.searchId, [action])
            except ValueError:
                continue
            j, _ = _to_terminal(child, me, policy_model, OPTS["max_steps"])
            if j is not None:
                out.append(j)
        finally:
            for sid in (child.searchId if child is not None else None, root.searchId):
                if sid is not None:
                    try:
                        P.cg_api.search_release(sid)
                    except Exception:                      # noqa: BLE001
                        pass
            try:
                P.cg_api.search_end()
            except Exception:                              # noqa: BLE001
                pass
    return out


def audit_shortcut(obs, me, cfg_full, policy_model, seed, prod_action):
    """§32-§35: shortcut が省略した探索を回し、手が変わるときだけ paired 評価。"""
    cf = copy.deepcopy(cfg_full)
    cf["pipeline"]["top1_shortcut_prob"] = 1.01
    t = time.perf_counter()
    try:
        alt = MA._try_pipeline(obs, config=cf)
    except Exception:                                      # noqa: BLE001
        alt = None
    cf_ms = (time.perf_counter() - t) * 1000
    if alt is None:
        STATS["cf_none"] += 1
        return None
    disagree = alt[0] != prod_action
    row = {"kind": "shortcut", "prod_action": prod_action, "cf_action": alt[0],
           "disagree": disagree, "cf_ms": round(cf_ms, 1)}
    if disagree:
        seeds = [seed + 7000 + k for k in range(OPTS["m_changed"])]
        row["j_prod"] = _forced_terminal(obs, me, prod_action, policy_model, seeds)
        row["j_cf"] = _forced_terminal(obs, me, alt[0], policy_model, seeds)
    return row


def make_probe(inner, cfg_full, policy_model):
    pcfg = {**P.DEFAULTS, **cfg_full["pipeline"]}

    def probe(obs):
        act = inner(obs)
        if obs.select is None or not _ctx["rec"] or QUOTA is None:
            return act
        _ctx["idx"] += 1
        sel, stt = obs.select, obs.current
        if (sel.type != SelectType.MAIN or sel.maxCount != 1 or not sel.option
                or stt is None or stt.result != -1):
            return act
        turn = int(getattr(stt, "turn", 0))
        tb, cb = turn_band(turn), cand_band(len(sel.option))
        if not QUOTA.accept(turn, len(sel.option), _ctx["arch"], _ctx["taken"]):
            return act
        me = stt.yourIndex
        seed = 5_000_000 + _ctx["game"] * 997 + _ctx["idx"]
        t0 = time.perf_counter()
        try:
            pv = _policy_view(obs, cfg_full, policy_model)
            if pv is None:
                return act
            probs, ranked = pv
            is_shortcut = probs[ranked[0]] >= float(pcfg["top1_shortcut_prob"])
            if is_shortcut:
                res = audit_shortcut(obs, me, cfg_full, policy_model, seed, ranked[0])
            else:
                res = audit_searched(obs, me, pcfg, cfg_full, policy_model, seed)
        except Exception as exc:                            # noqa: BLE001
            STATS["audit_err_" + type(exc).__name__] += 1
            return act
        if res is None:
            return act
        res.update({"group_id": "g{}_{}".format(_ctx["game"], _ctx["idx"]),
                    "game": _ctx["game"], "arch": _ctx["arch"], "turn": turn,
                    "turn_band": tb, "cand_band": cb, "n_opt": len(sel.option),
                    "me_first": _ctx["first"], "audit_ms": round(
                        (time.perf_counter() - t0) * 1000, 1)})
        GROUPS.append(res)
        QUOTA.commit(turn, len(sel.option), _ctx["arch"])
        _ctx["taken"] += 1
        STATS[res["kind"]] += 1
        return act
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--per-game-cap", type=int, default=4)
    ap.add_argument("--n-det", type=int, default=8)
    ap.add_argument("--m-changed", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=3200000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="q0")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(n_det=args.n_det, max_steps=args.max_steps, m_changed=args.m_changed)

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap, cand_soft_cap=0.45)
    EV["hand"] = leaf_eval_module.build_evaluator({"kind": "handcrafted"})

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
        _ctx.update(game=g, arch=arch, me=0 if p0 else 1, first=p0, rec=True, idx=0, taken=0)
        (runner.play_game(probe, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, probe, deck_o, deck_c))
        _ctx["rec"] = False
        if gi % 3 == 0:
            print("  [w{}] game#{} groups={}/{} searched={} shortcut={} {}min".format(
                args.worker_id, g, QUOTA.total, args.target, STATS["searched"],
                STATS["shortcut"], int((time.perf_counter() - t0) / 60)),
                file=sys.stderr, flush=True)

    out = _HERE / "_p1913q_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(json.dumps({"tag": args.tag, "n": len(GROUPS), "stats": dict(STATS)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
