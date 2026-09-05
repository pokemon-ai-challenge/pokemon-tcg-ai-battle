"""Phase19.13 Stage A/D/E: production decision census + shortcut/tie 監査(端末評価なし)。

production(abl_5_full + climb 重み)をそのまま回し、**1 decision ごとに**どの経路が
Action を決めたかを分類する。production コードは変更せず、モジュール関数を wrap して
観測するだけ(config も本番値のまま)。

counterfactual(§24)は **オフラインでだけ** 実行する:
  * shortcut が発火した decision について、`top1_shortcut_prob` を無効化した config で
    もう一度 pipeline を回し、選ぶ手が変わるかを見る。本番の返り値は差し替えない。

出力は 1 decision 1 行の JSONL。terminal 評価は別スクリプト(_p1913_quality.py)。
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

ROWS: list[dict] = []
STATS: Counter = Counter()
_cur: dict = {}
_ctx: dict = {"game": 0, "arch": "?", "me": 0, "first": True, "rec": False, "idx": 0}
OPTS = {"counterfactual": 1}


def _install() -> None:
    o_lethal = MA._try_lethal
    o_pipe = MA._try_pipeline
    o_sci = P._select_candidate_indices
    o_ec = P._evaluate_candidate
    o_sm = P._softmax

    def try_lethal(obs, config=None):
        t = time.perf_counter()
        r = o_lethal(obs, config=config)
        _cur["lethal_ms"] = (time.perf_counter() - t) * 1000
        _cur["lethal_hit"] = r is not None
        return r

    def try_pipe(obs, config=None):
        t = time.perf_counter()
        r = o_pipe(obs, config=config)
        _cur["pipe_ms"] = (time.perf_counter() - t) * 1000
        _cur["pipe_hit"] = r is not None
        _cur["pipe_action"] = r[0] if r else None
        return r

    def sci(select, ranked, config, probs=None):
        r = o_sci(select, ranked, config, probs)
        _cur["searched"] = True
        _cur["cands"] = list(r)
        return r

    def ec(root, candidate, me, config, deadline, evaluator, policy_model):
        s = o_ec(root, candidate, me, config, deadline, evaluator, policy_model)
        if s is not None:
            _cur.setdefault("agg", {}).setdefault(candidate[0], []).append(float(s))
        return s

    def sm(scores):
        p = o_sm(scores)
        if _cur.get("probs") is None:      # search 冒頭の1回だけ拾う
            _cur["probs"] = list(p)
        return p

    MA._try_lethal, MA._try_pipeline = try_lethal, try_pipe
    P._select_candidate_indices, P._evaluate_candidate, P._softmax = sci, ec, sm


def _classify(obs, act, prod_cfg) -> dict:
    sel = obs.select
    st = obs.current
    n_opt = len(sel.option) if sel.option else 0
    eligible = (sel.type == SelectType.MAIN and sel.maxCount == 1 and n_opt > 0)
    probs = _cur.get("probs")
    top1p = max(probs) if probs else None
    top2 = sorted(probs, reverse=True)[1] if probs and len(probs) > 1 else None
    agg = _cur.get("agg") or {}
    scored = {i: sum(v) / len(v) for i, v in agg.items() if v}
    gap = best_idx = second = None
    tie_changed = None
    if len(scored) >= 1:
        order = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        best_idx, best = order[0]
        if len(order) > 1:
            second = order[1][1]
            gap = best - second
        # tie_eps によって「平均最大」以外が選ばれたか
        tie_changed = (_cur.get("pipe_action") is not None
                       and _cur["pipe_action"] != best_idx)

    if _cur.get("lethal_hit"):
        path = "D0_lethal"
    elif _cur.get("pipe_hit"):
        path = "D1_searched" if _cur.get("searched") else "D2_shortcut"
    elif not eligible:
        path = "D3_ineligible"
    else:
        path = "D4_fallback"

    row = {
        "game": _ctx["game"], "arch": _ctx["arch"], "me": _ctx["me"], "first": _ctx["first"],
        "idx": _ctx["idx"], "turn": int(getattr(st, "turn", -1)),
        "select_type": str(sel.type), "n_opt": n_opt, "eligible": eligible,
        "path": path, "action": act[0] if act else None,
        "ms": _cur.get("total_ms"), "lethal_ms": _cur.get("lethal_ms"),
        "pipe_ms": _cur.get("pipe_ms"),
        "policy_top1_prob": top1p,
        "policy_margin": (top1p - top2) if (top1p is not None and top2 is not None) else None,
        "n_cands": len(_cur.get("cands") or []),
        "cand_scores": {str(k): round(v, 6) for k, v in scored.items()} or None,
        "gap": round(gap, 6) if gap is not None else None,
        "best_mean_idx": best_idx, "tie_changed": tie_changed,
    }
    # C-A / C-B: 技術的に同じ pipeline を回せたか
    if path == "D3_ineligible":
        row["coverage_class"] = "C-B_unsupported"
    elif path in ("D2_shortcut", "D4_fallback"):
        row["coverage_class"] = "C-A_searchable_skipped"
    else:
        row["coverage_class"] = None

    # §24 counterfactual: shortcut を外して回すとどうなるか(オフライン診断のみ)
    if OPTS["counterfactual"] and path == "D2_shortcut":
        cf = copy.deepcopy(prod_cfg)
        cf["pipeline"]["top1_shortcut_prob"] = 1.01   # 決して発火しない
        keep = dict(_cur)
        _cur.clear()
        t = time.perf_counter()
        try:
            alt = MA._try_pipeline(obs, config=cf)
        except Exception:
            alt = None
        row["cf_ms"] = (time.perf_counter() - t) * 1000
        row["cf_action"] = alt[0] if alt else None
        row["cf_disagree"] = (alt is not None and act and alt[0] != act[0])
        agg2 = _cur.get("agg") or {}
        sc2 = {i: sum(v) / len(v) for i, v in agg2.items() if v}
        row["cf_cand_scores"] = {str(k): round(v, 6) for k, v in sc2.items()} or None
        _cur.clear()
        _cur.update(keep)
    return row


def make_probe(inner, prod_cfg):
    def probe(obs):
        if obs.select is None or not _ctx["rec"]:
            return inner(obs)
        _cur.clear()
        t = time.perf_counter()
        act = inner(obs)
        _cur["total_ms"] = (time.perf_counter() - t) * 1000
        _ctx["idx"] += 1
        try:
            row = _classify(obs, act, prod_cfg)
            ROWS.append(row)
            STATS[row["path"]] += 1
        except Exception as exc:  # noqa: BLE001 - 監査が試合を壊してはいけない
            STATS["classify_error"] += 1
            STATS["err_" + type(exc).__name__] += 1
        return act
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=3100000)
    ap.add_argument("--counterfactual", type=int, default=1)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="c0")
    args = ap.parse_args()
    OPTS["counterfactual"] = args.counterfactual
    _install()

    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    probe = make_probe(climb, cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for gi in range(args.games):
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("abl_5_full")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, arch=arch, me=0 if p0 else 1, first=p0, rec=True, idx=0)
        res = (runner.play_game(probe, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, probe, deck_o, deck_c))
        _ctx["rec"] = False
        STATS["games"] += 1
        if getattr(res, "winner", None) is not None:
            STATS["win" if res.winner == _ctx["me"] else "loss"] += 1
        if gi % 5 == 0:
            print("  [w{}] game#{} decisions={} {}min".format(
                args.worker_id, g, len(ROWS), int((time.perf_counter() - t0) / 60)),
                file=sys.stderr, flush=True)

    out = _HERE / "_p1913_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in ROWS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(json.dumps({"tag": args.tag, "rows": len(ROWS), "stats": dict(STATS)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
