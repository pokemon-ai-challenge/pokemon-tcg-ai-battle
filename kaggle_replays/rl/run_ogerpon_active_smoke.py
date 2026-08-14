"""design.md Phase4初期活性化: strict thresholdでの能動ゲート(`shadow_only: false`)を
20-50試合程度の実self-playで安全性だけ確認するsmokeスクリプト。

Stage3(kaggle_replays/rl/collect_ogerpon_counterfactuals.py)の実測で ``main_attach``
のみが30状態以上(supported_trigger)に到達しているため、``active_triggers`` は
``["main_attach"]`` に限定する(``promote``/``retreat`` は shadow logging のみ継続)。

安全性の受入基準(勝率比較はまだ行わない。それはStage7のpaired A/Bの役目):

- engine error: 0
- illegal action: 0
- integrity violation(final_action変化・RNG状態変化): 0
- shadow推論例外: 0(発生しても必ずbaselineへフォールバックすることは既に単体テスト・
  `_ogerpon_q_shadow`の安全網で保証済みだが、実測でも0件であることを確認する)
- n_active_overrides > 0(ゲートが実際に機能していることの確認。0件なら
  active_triggers/thresholdsが厳しすぎて実質何もしていない可能性がある)
- 非Ogerponデッキへの影響: このスモークはOgerponデッキ側でのみ実行するため直接は
  測らない(構造的保証は単体テスト`test_ogerpon_q_shadow_skips_non_ogerpon_deck`等で
  別途確認済み)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import atomic_write_json, resolve_path  # noqa: E402
from eval_agent_field import build_field, make_tasks  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}

# !!! 実験専用設定。sample_submission/configs/ 配下の本番configには絶対にコピーしないこと !!!
# `shadow_only: False` はこのスクリプト(オフラインsmoke test)内でのみ使う。本番提出パス
# (`abl_5_full` 等)には `ogerpon_q_critic` キー自体が存在せず、既定で `shadow_only=True`
# (=Q-criticは一切行動へ介入しない)。2026-08-14時点でheld-out strict override件数が
# 7件(<30必須)のため、active_approved=falseのまま(weights側のmeta.active_gateを参照)。
ACTIVE_Q_CONFIG = {
    "shadow_only": False,
    "active_triggers": ["main_attach"],
    "supported_triggers": ["main_attach"],
    "decision_thresholds": {
        "strict_override_threshold": 0.02, "near_tie_enabled": False,
        "near_tie_epsilon": 0.015, "loop_threshold": 0.45, "uncertainty_coef": 1.0,
    },
}


def _init(weights_path: str, ml_config_name: str, workdir: str, opponents):
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    config["ogerpon_q_critic"] = dict(ACTIVE_Q_CONFIG)
    _W["config"] = config
    _W["agent_mod"] = ml_policy_agent
    models = {}
    for _a, wp, _d in opponents:
        if wp not in models:
            models[wp] = PolicyModel(wp)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _opponent_action(obs, opp_pm):
    sel = obs.select
    if sel is None or not sel.option:
        return []
    if sel.maxCount == 1:
        oi = opp_pm.select_option(obs)
        return [oi if oi is not None else 0]
    scores = opp_pm.score_options(obs)
    n = len(sel.option)
    c = max(sel.minCount, min(sel.maxCount, n))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:c] if scores else list(range(c))


def _is_valid_action(action, select) -> bool:
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def _play_one_game(deck0, deck1, learner_index: int, opp_pm, config, seed: int) -> dict:
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.hidden_information import match_context

    agent_mod = _W["agent_mod"]
    match_context.reset()
    agent_mod.OGERPON_Q_SHADOW_LOG = []
    agent_mod.OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS = []
    agent_mod.OGERPON_Q_SHADOW_EXCEPTIONS = []
    random.seed(seed)

    n_decisions = 0
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}", "n_decisions": 0, "result": None,
               "shadow_log": [], "integrity_violations": [], "exceptions": []}

    n = 0
    err = None
    result = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"
                break
            if cur.result != -1:
                result = cur.result
                break
            if n >= MAX_STEPS:
                err = "max_steps"
                break

            if cur.yourIndex == learner_index and sel is not None:
                n_decisions += 1
                action = agent_mod.agent(obs, config)
                if not _is_valid_action(action, sel):
                    err = f"illegal_action: {action!r}"
                    break
            else:
                action = _opponent_action(obs, opp_pm)

            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    win = None if result is None else int(result == learner_index)
    return {
        "error": err, "n_decisions": n_decisions, "win": win,
        "shadow_log": list(agent_mod.OGERPON_Q_SHADOW_LOG),
        "integrity_violations": list(agent_mod.OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS),
        "exceptions": list(agent_mod.OGERPON_Q_SHADOW_EXCEPTIONS),
    }


def _process_task(task):
    from matchup_common import read_deck

    learner_index, seed, opp_idx = task
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    result = _play_one_game(deck0, deck1, learner_index, opp_pm, _W["config"], seed)
    n_active_overrides = sum(1 for rec in result["shadow_log"] if rec.get("active_override_applied"))
    outcome_counts: dict[str, int] = {}
    for rec in result["shadow_log"]:
        outcome_counts[rec["outcome"]] = outcome_counts.get(rec["outcome"], 0) + 1
    evaluated_fields = ("trigger_kind", "p_ex", "p_single", "delta", "delta_std", "lcb_delta",
                       "p_loop_complete", "required_ko_gain", "would_override",
                       "active_override_applied", "is_lethal_baseline")
    evaluated_records = [
        {k: rec.get(k) for k in evaluated_fields}
        for rec in result["shadow_log"] if rec.get("outcome") == "evaluated"
    ]
    return {
        "seed": seed, "learner_index": learner_index, "opponent_archetype": arch,
        "error": result["error"], "win": result["win"], "n_decisions": result["n_decisions"],
        "n_shadow_log_entries": len(result["shadow_log"]),
        "outcome_counts": outcome_counts,
        "n_active_overrides": n_active_overrides,
        "evaluated_records": evaluated_records,
        "n_integrity_violations": len(result["integrity_violations"]),
        "integrity_violations": result["integrity_violations"],
        "n_exceptions": len(result["exceptions"]),
        "exceptions": result["exceptions"][:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ml-config", default="abl_5_full")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13571113)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from matchup_common import read_deck

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)

    workdir = tempfile.mkdtemp(prefix="ogerq_active_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    t0 = time.time()
    results = []
    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        for res in pool.imap_unordered(_process_task, tasks):
            results.append(res)

    n_games = len(results)
    n_errors = sum(1 for r in results if r["error"])
    n_illegal = sum(1 for r in results if r["error"] and "illegal_action" in str(r["error"]))
    n_integrity_violations_total = sum(r["n_integrity_violations"] for r in results)
    n_exceptions_total = sum(r["n_exceptions"] for r in results)
    n_active_overrides_total = sum(r["n_active_overrides"] for r in results)
    n_decisions_total = sum(r["n_decisions"] for r in results)
    wins = [r["win"] for r in results if r["win"] is not None]

    outcome_counts_total: dict[str, int] = {}
    for r in results:
        for k, v in r["outcome_counts"].items():
            outcome_counts_total[k] = outcome_counts_total.get(k, 0) + v

    archetype_counts: dict[str, int] = {}
    for r in results:
        archetype_counts[r["opponent_archetype"]] = archetype_counts.get(r["opponent_archetype"], 0) + 1

    summary = {
        "ogerpon_q_critic_config": ACTIVE_Q_CONFIG,
        "games": n_games,
        "n_decisions_total": n_decisions_total,
        "n_errors": n_errors,
        "n_illegal_actions": n_illegal,
        "n_integrity_violations_total": n_integrity_violations_total,
        "n_exceptions_total": n_exceptions_total,
        "shadow_outcome_counts": outcome_counts_total,
        "n_active_overrides_total": n_active_overrides_total,
        "n_games_with_active_override": sum(1 for r in results if r["n_active_overrides"] > 0),
        "win_rate_directional_only": (sum(wins) / len(wins)) if wins else None,
        "n_games_with_result": len(wins),
        "opponent_archetype_counts": archetype_counts,
        "wall_seconds": time.time() - t0,
        "output": str(resolve_path(args.output)),
    }

    atomic_write_json(resolve_path(args.output), {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
