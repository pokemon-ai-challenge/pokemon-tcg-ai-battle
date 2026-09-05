"""Phase5A: n_options-gated dynamic top-k の**候補回収能力**診断(実ゲームA/Bの前段)。

「requested top-k が増えた」だけでは回収成功ではない。400ms の本番予算内で
**実際に評価を開始し、完了できたか**まで見る。各 MAIN 意思決定で:

  1. 参照評価(広く深く: 16候補 × 12決定化 × 8秒)で "診断上の最良候補" を決める
  2. その候補が **固定 top-4 の外**だったかを見る
  3. n_options ゲートの候補集合に**入るか**(candidate recall)
  4. さらに **本番と同じ 400ms / det=8** で回したとき、その候補が
     started / completed / fully-evaluated まで到達するか(completed recall)

参照評価と本番予算シミュレーションは同じ評価器(leaf α=0.5)・同じ決定化生成を使い、
**候補数と予算だけ**を変える。これにより「候補生成の問題」と「予算不足の問題」を分離する。

出力: _diag_topk_recall_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

# A/B と同じ leaf(α=0.5)。α を動かさないこと自体が Phase5 の前提。
LEAF_CFG = {"kind": "blend", "alpha": 0.5}
PROD_DETS = 8            # 決定化数は変更しない(Phase5 の固定条件)
PROD_BUDGET_MS = 400.0   # 既定=従来のローカル固定。--budget-mode で切替える。

# 予算モード。正式な再診断に使うのは production_dynamic のみ(他は原因分析用)。
BUDGET_MODES = ("fixed_400", "fixed_1350", "fixed_2000", "production_dynamic")
REF_CANDS, REF_WORLDS, REF_MS = 16, 12, 8000.0

ROWS: list[dict] = []
# `n_selects`/`start` は production と同じ意味の per-agent カウンタ
# (production は 1プロセス1エージェントなので `_selects_seen` はその陣営のみを数える。
#  ローカルは両陣営が module global を共有してしまうため、ここでは自前で持つ)。
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0,
        "start": 0.0, "n_selects": 0}
OPTS = {"every": 4, "max_per_game": 6, "min_options": 5}
_BUDGET = {"mode": "fixed_400", "time_budget": None, "base_ms": 400.0}


def _current_budget_ms() -> float:
    """この select に production が割り当てるはずの予算(ms)を返す。"""
    mode = _BUDGET["mode"]
    if mode == "fixed_400":
        return 400.0
    if mode == "fixed_1350":
        return 1350.0
    if mode == "fixed_2000":
        return 2000.0
    tb = _BUDGET["time_budget"]
    if not tb:
        return _BUDGET["base_ms"]
    elapsed_ms = (time.perf_counter() - _ctx["start"]) * 1000.0
    total_ms = float(tb["total_ms"])
    min_ms = float(tb.get("min_ms", 50))
    max_ms = float(tb.get("max_ms", 2000))
    assumed_total = int(tb.get("assumed_total_selects", 400))
    remaining_ms = total_ms - elapsed_ms
    if remaining_ms <= 0:
        return min_ms
    remaining_selects = max(1, assumed_total - _ctx["n_selects"])
    return max(min_ms, min(max_ms, remaining_ms / remaining_selects))


def _budgeted_eval(obs, model, cand: list[int], num_worlds: int, time_ms: float,
                   evaluator) -> dict:
    """候補集合を「決定化ループ + 予算」で評価し、候補ごとの到達段階を返す。

    pipeline.search と同じ順序(世界ごとに全候補を回す)なので、予算切れの影響も同じ形で出る。
    """
    state = obs.current
    me = state.yourIndex
    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    cfg = {"opponent_depth": 1, "max_rollout_steps": 40}
    t0 = time.perf_counter()
    deadline = t0 + time_ms / 1000.0

    started: set[int] = set()
    vals: dict[int, list[float]] = {i: [] for i in cand}
    worlds_built = 0
    timed_out = False
    try:
        for _ in range(num_worlds):
            if time.perf_counter() > deadline:
                timed_out = True
                break
            hs = factory()
            if hs is None:
                continue
            try:
                root = P._begin(obs, hs)
            except Exception:  # noqa: BLE001
                continue
            worlds_built += 1
            try:
                for i in cand:
                    if time.perf_counter() > deadline:
                        timed_out = True
                        break
                    started.add(i)
                    v = P._evaluate_candidate(root, [i], me, cfg, deadline, evaluator, model)
                    if v is not None:
                        vals[i].append(v)
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            P.cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass

    completed = {i for i, v in vals.items() if v}
    fully = {i for i, v in vals.items() if len(v) >= num_worlds}
    means = {i: statistics.mean(v) for i, v in vals.items() if v}
    return {
        "requested": list(cand), "started": started, "completed": completed,
        "fully": fully, "means": means, "worlds_built": worlds_built,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0, "timed_out": timed_out,
    }


def _probe(obs, model) -> dict | None:
    select = obs.select
    try:
        scores = model.score_options(obs, None, None)
    except Exception:  # noqa: BLE001
        return None
    if not scores or len(scores) != len(select.option):
        return None
    n_options = len(select.option)
    ranked = sorted(range(n_options), key=lambda i: scores[i], reverse=True)
    probs = P._softmax(scores)
    evaluator = leaf_eval_module.build_evaluator(LEAF_CFG)

    # 1. 参照評価(広く深く)→ 診断上の最良候補
    ref = _budgeted_eval(obs, model, ranked[:REF_CANDS], REF_WORLDS, REF_MS, evaluator)
    if not ref["means"]:
        return None
    ref_best = max(ref["means"], key=lambda i: ref["means"][i])
    ref_rank = ranked.index(ref_best) + 1

    # 2/3. 固定top4 と n_options ゲートの候補集合
    cfg_fixed = {"top_k": 4}
    cfg_dyn = {"top_k": 4, "dynamic_top_k": {
        "enabled": True, "mode": "n_options",
        "thresholds": [{"max_options": 7, "top_k": 4},
                       {"max_options": 11, "top_k": 8},
                       {"max_options": None, "top_k": 12}]}}
    k_fixed = P._resolve_top_k(cfg_fixed, probs, ranked)
    k_dyn = P._resolve_top_k(cfg_dyn, probs, ranked)
    set_fixed, set_dyn = ranked[:k_fixed], ranked[:k_dyn]

    # 4. 本番予算(400ms / det=8)での到達段階を両方で測る
    budget_ms = _current_budget_ms()
    prod_fixed = _budgeted_eval(obs, model, set_fixed, PROD_DETS, budget_ms, evaluator)
    prod_dyn = _budgeted_eval(obs, model, set_dyn, PROD_DETS, budget_ms, evaluator)

    def _sel(pr):
        return (max(pr["means"], key=lambda i: pr["means"][i]) if pr["means"] else None)

    return {
        "game": _ctx["game"], "turn": int(getattr(obs.current, "turn", 0) or 0),
        "n_options": n_options,
        "budget_ms": round(budget_ms, 1),
        "ref_best_index": ref_best, "ref_best_policy_rank": ref_rank,
        "ref_best_advantage": round(
            ref["means"][ref_best] - ref["means"].get(ranked[0], ref["means"][ref_best]), 5),
        "outside_fixed_top4": ref_rank > k_fixed,
        "k_fixed": k_fixed, "k_dyn": k_dyn,
        "recalled_into_candidate_set": ref_best in set_dyn,
        "recalled_started": ref_best in prod_dyn["started"],
        "recalled_completed": ref_best in prod_dyn["completed"],
        "recalled_fully": ref_best in prod_dyn["fully"],
        "dyn_selected_index": _sel(prod_dyn),
        "fixed_selected_index": _sel(prod_fixed),
        "fixed": {
            "requested": len(prod_fixed["requested"]), "started": len(prod_fixed["started"]),
            "completed": len(prod_fixed["completed"]), "fully": len(prod_fixed["fully"]),
            "worlds_built": prod_fixed["worlds_built"],
            "elapsed_ms": round(prod_fixed["elapsed_ms"], 1),
            "timed_out": prod_fixed["timed_out"],
        },
        "dyn": {
            "requested": len(prod_dyn["requested"]), "started": len(prod_dyn["started"]),
            "completed": len(prod_dyn["completed"]), "fully": len(prod_dyn["fully"]),
            "worlds_built": prod_dyn["worlds_built"],
            "elapsed_ms": round(prod_dyn["elapsed_ms"], 1),
            "timed_out": prod_dyn["timed_out"],
        },
    }


def _install() -> None:
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        is_mine = obs.current.yourIndex == _ctx["me"]
        if is_mine:
            _ctx["n_selects"] += 1        # production の _selects_seen と同じ数え方
        if (is_mine and sel is not None
                and sel.type == SelectType.MAIN and sel.maxCount == 1
                and sel.option and len(sel.option) >= OPTS["min_options"]):
            _ctx["seen"] += 1
            if (_ctx["seen"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    row = _probe(obs, ml_policy_agent._get_model(config))
                    if row is not None:
                        ROWS.append(row)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--every", type=int, default=OPTS["every"])
    ap.add_argument("--max-per-game", type=int, default=OPTS["max_per_game"])
    ap.add_argument("--budget-mode", default="fixed_400", choices=list(BUDGET_MODES))
    ap.add_argument("--out", default="_diag_topk_recall_results.json")
    args = ap.parse_args()
    OPTS.update(every=args.every, max_per_game=args.max_per_game)

    _install()
    cfg = agents.load_config_copy("climb_v15_leaf_a05")
    cfg["policy_weights_path"] = CLIMB
    _BUDGET["mode"] = args.budget_mode
    _BUDGET["time_budget"] = cfg["pipeline"].get("time_budget")
    _BUDGET["base_ms"] = float(cfg["pipeline"].get("time_limit_ms", 400))
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("climb_baseline")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    t0 = time.perf_counter()
    for g in range(args.games):
        p0 = (g % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, probed=0,
                    start=time.perf_counter(), n_selects=0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        print(f"  game {g+1}/{args.games} probes={_ctx['probed']} total={len(ROWS)}",
              file=sys.stderr, flush=True)

    n = len(ROWS)
    outside = [r for r in ROWS if r["outside_fixed_top4"]]
    rec_set = [r for r in outside if r["recalled_into_candidate_set"]]
    rec_started = [r for r in outside if r["recalled_started"]]
    rec_done = [r for r in outside if r["recalled_completed"]]
    rec_full = [r for r in outside if r["recalled_fully"]]
    chosen = [r for r in rec_done if r["dyn_selected_index"] == r["ref_best_index"]]

    def band(r):
        no = r["n_options"]
        return "1-4" if no <= 4 else "5-7" if no <= 7 else "8-11" if no <= 11 else "12+"

    bands: dict[str, list] = {}
    for r in ROWS:
        bands.setdefault(band(r), []).append(r)

    out = {
        "note": ("requested/started/completed/fully を区別。400ms・det=8 の本番予算で"
                 "実際に評価完了できたかを重視する。"),
        "settings": {"leaf": LEAF_CFG, "prod_dets": PROD_DETS,
                     "budget_mode": args.budget_mode,
                     "mean_budget_ms": (round(statistics.mean(
                         [r["budget_ms"] for r in ROWS]), 1) if ROWS else None),
                     "ref": {"cands": REF_CANDS, "worlds": REF_WORLDS, "ms": REF_MS},
                     **OPTS},
        "games": args.games, "opponent": args.opponent,
        "n_probes": n, "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "recall": {
            "fixed_top4_outside_count": len(outside),
            "recalled_into_candidate_set": len(rec_set),
            "candidate_recall": round(len(rec_set) / len(outside), 4) if outside else None,
            "started": len(rec_started),
            "completed": len(rec_done),
            "completed_recall": round(len(rec_done) / len(outside), 4) if outside else None,
            "fully_evaluated": len(rec_full),
            "fully_recall": round(len(rec_full) / len(outside), 4) if outside else None,
            "selected_as_final": len(chosen),
            "mean_policy_rank_of_recalled": round(
                statistics.mean([r["ref_best_policy_rank"] for r in rec_set]), 2) if rec_set else None,
            "mean_leaf_advantage_of_recalled": round(
                statistics.mean([r["ref_best_advantage"] for r in rec_set]), 5) if rec_set else None,
        },
        "by_n_options_band": {
            b: {
                "n": len(rs),
                "mean_k_fixed": round(statistics.mean([r["k_fixed"] for r in rs]), 2),
                "mean_k_dyn": round(statistics.mean([r["k_dyn"] for r in rs]), 2),
                "fixed_started": round(statistics.mean([r["fixed"]["started"] for r in rs]), 2),
                "fixed_completed": round(statistics.mean([r["fixed"]["completed"] for r in rs]), 2),
                "fixed_fully": round(statistics.mean([r["fixed"]["fully"] for r in rs]), 2),
                "dyn_started": round(statistics.mean([r["dyn"]["started"] for r in rs]), 2),
                "dyn_completed": round(statistics.mean([r["dyn"]["completed"] for r in rs]), 2),
                "dyn_fully": round(statistics.mean([r["dyn"]["fully"] for r in rs]), 2),
                "fixed_timeout_rate": round(
                    sum(1 for r in rs if r["fixed"]["timed_out"]) / len(rs), 4),
                "dyn_timeout_rate": round(
                    sum(1 for r in rs if r["dyn"]["timed_out"]) / len(rs), 4),
                "fixed_ms": round(statistics.mean([r["fixed"]["elapsed_ms"] for r in rs]), 1),
                "dyn_ms": round(statistics.mean([r["dyn"]["elapsed_ms"] for r in rs]), 1),
                "outside_top4_rate": round(
                    sum(1 for r in rs if r["outside_fixed_top4"]) / len(rs), 4),
            }
            for b, rs in sorted(bands.items())
        },
        "overall_budget": {
            "fixed_timeout_rate": round(
                sum(1 for r in ROWS if r["fixed"]["timed_out"]) / n, 4) if n else None,
            "dyn_timeout_rate": round(
                sum(1 for r in ROWS if r["dyn"]["timed_out"]) / n, 4) if n else None,
            "fixed_mean_completed": round(
                statistics.mean([r["fixed"]["completed"] for r in ROWS]), 2) if n else None,
            "dyn_mean_completed": round(
                statistics.mean([r["dyn"]["completed"] for r in ROWS]), 2) if n else None,
            "fixed_mean_worlds": round(
                statistics.mean([r["fixed"]["worlds_built"] for r in ROWS]), 2) if n else None,
            "dyn_mean_worlds": round(
                statistics.mean([r["dyn"]["worlds_built"] for r in ROWS]), 2) if n else None,
            "action_changed_rate": round(sum(
                1 for r in ROWS
                if r["dyn_selected_index"] is not None
                and r["dyn_selected_index"] != r["fixed_selected_index"]) / n, 4) if n else None,
        },
        "rows": ROWS,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
