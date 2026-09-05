"""探索が時間予算で切られているかの測定(climb v1.5 診断 / 測定忠実度)。

`abl_5_full` は num_determinizations=8 / top_k=4 = 1手あたり最大32葉評価を要求するが、
`time_limit_ms` を超えると途中で打ち切られる。**実際に何世界・何候補ぶん評価できているか**を測る。

同時に、ローカル harness と Kaggle 本番の予算差も検証する:
  - ローカル `runner.play_game` は `obs.select is None`(デッキ選択)経路を通らないため
    `ml_policy_agent._match_start_perf` が None のままで、動的予算が効かず固定 time_limit_ms になる。
  - Kaggle 本番は select=None を受けるので動的予算(total_ms/assumed_total_selects ≒ 1350ms)になる。
つまりローカル計測は本番より**短い予算**で探索している可能性がある。--time-limit-ms で両方を測る。

出力: _diag_search_budget_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

SEARCHES: list[dict] = []
_cur = {"active": False, "worlds": 0, "evals": 0, "cands": 0, "t0": 0.0}


def _install() -> None:
    orig_search = P.search
    orig_begin = P._begin
    orig_eval = P._evaluate_candidate
    orig_cand = P._select_candidate_indices

    def search(state, legal_actions, context):
        _cur.update(active=True, worlds=0, evals=0, cands=0, t0=time.perf_counter())
        try:
            return orig_search(state, legal_actions, context)
        finally:
            if _cur["cands"]:
                SEARCHES.append({
                    "worlds": _cur["worlds"],
                    "cands": _cur["cands"],
                    "evals": _cur["evals"],
                    "ms": (time.perf_counter() - _cur["t0"]) * 1000.0,
                    "requested_evals": _cur["worlds"] * _cur["cands"],
                })
            _cur["active"] = False

    def begin(obs, hidden_state):
        out = orig_begin(obs, hidden_state)
        if _cur["active"]:
            _cur["worlds"] += 1
        return out

    def evaluate_candidate(root, candidate, me, config, deadline, evaluator, policy_model):
        out = orig_eval(root, candidate, me, config, deadline, evaluator, policy_model)
        if _cur["active"] and out is not None:
            _cur["evals"] += 1
        return out

    def cand(select, ranked, config, probs=None):
        out = orig_cand(select, ranked, config, probs)
        if _cur["active"]:
            _cur["cands"] = len(out)
        return out

    P.search = search
    P._begin = begin
    P._evaluate_candidate = evaluate_candidate
    P._select_candidate_indices = cand
    # ml_policy_agent は import 時に `from ptcg_ai.search import pipeline` で束縛済みなので
    # モジュールオブジェクト経由のパッチで届く(同一オブジェクト)。


def _pct(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--time-limit-ms", type=float, default=None,
                    help="pipeline.time_limit_ms を上書き(未指定なら config の 400)")
    ap.add_argument("--determinizations", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _install()
    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB_WEIGHTS
    if args.time_limit_ms is not None:
        cfg["pipeline"]["time_limit_ms"] = args.time_limit_ms
        cfg["pipeline"].pop("time_budget", None)  # 固定予算で測る
    if args.determinizations is not None:
        cfg["pipeline"]["num_determinizations"] = args.determinizations
    climb = agents.make_ml_policy_agent(cfg)

    cfg_o = agents.load_config_copy("abl_5_full")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)

    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    for g in range(args.games):
        if g % 2 == 0:
            runner.play_game(climb, opp, deck_c, deck_o)
        else:
            runner.play_game(opp, climb, deck_o, deck_c)
        print(f"  game {g+1}/{args.games} searches={len(SEARCHES)}", file=sys.stderr)

    req_worlds = cfg["pipeline"]["num_determinizations"]
    worlds = [s["worlds"] for s in SEARCHES]
    ms = [s["ms"] for s in SEARCHES]
    completion = [s["evals"] / (req_worlds * s["cands"]) for s in SEARCHES if s["cands"]]

    out = {
        "config_time_limit_ms": cfg["pipeline"]["time_limit_ms"],
        "config_num_determinizations": req_worlds,
        "dynamic_budget_active": ml_policy_agent._match_start_perf is not None,
        "n_searches": len(SEARCHES),
        "worlds_completed": {
            "mean": round(statistics.mean(worlds), 3) if worlds else None,
            "median": statistics.median(worlds) if worlds else None,
            "min": min(worlds) if worlds else None,
            "max": max(worlds) if worlds else None,
            "frac_reaching_requested": round(
                sum(1 for w in worlds if w >= req_worlds) / len(worlds), 4) if worlds else None,
        },
        "eval_completion_ratio": {
            "mean": round(statistics.mean(completion), 4) if completion else None,
            "p10": round(_pct(completion, 0.10), 4),
            "p50": round(_pct(completion, 0.50), 4),
        },
        "search_ms": {
            "mean": round(statistics.mean(ms), 2) if ms else None,
            "p50": round(_pct(ms, 0.5), 2), "p90": round(_pct(ms, 0.9), 2),
            "p99": round(_pct(ms, 0.99), 2), "max": round(max(ms), 2) if ms else None,
        },
        "frac_searches_hitting_deadline": round(
            sum(1 for s in SEARCHES if s["ms"] >= cfg["pipeline"]["time_limit_ms"] * 0.95)
            / len(SEARCHES), 4) if SEARCHES else None,
    }
    dest = args.out or str(_HERE / f"_diag_search_budget_{int(cfg['pipeline']['time_limit_ms'])}ms.json")
    Path(dest).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
