"""Stage7A: baseline / q_gate_strict の154試合ずつのpaired Safety A/B。

design.mdの当初計画は「効果検証のためのA/B」だったが、Stage6のsmoke結果
(40試合・435評価中override 1件)から、154試合/armで見込めるoverrideは
1/435*154*2(2arm合計)程度 ≈ 数件しかなく、勝率差をモデル効果の判定に使うことは
できないとユーザーが判断した。したがって本スクリプトの目的は
「strict activeの実戦安全性・非劣化確認」に限定する(勝率は参考記録として残すが、
結論には使わない)。

両armは同一task list(learner_index, seed, opponent_index)を使い、決定論的に
ペアリングする(相手構成・先後・seedを完全一致させる)。baseline armも
`OGERPON_Q_SHADOW_LOG` を有効にして診断ログを取る(ただし`shadow_only=True`なので
行動には一切影響しない、Stage5で検証済みの経路)。

安全性の合格条件(勝率の有意性は評価対象にしない):
- engine error: 0
- illegal action: 0
- integrity violation: 0
- unsupported triggerでのoverride: 0
- lethalを上書きした事例: 0
- model/contract error(weights_not_ready): 0
- shadow推論例外: 0
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

BASELINE_Q_CONFIG = {
    "shadow_only": True,
    "supported_triggers": ["main_attach"],
    "decision_thresholds": {
        "strict_override_threshold": 0.02, "near_tie_enabled": False,
        "near_tie_epsilon": 0.015, "loop_threshold": 0.45, "uncertainty_coef": 1.0,
    },
}
# !!! 実験専用設定。sample_submission/configs/ 配下の本番configには絶対にコピーしないこと !!!
# `shadow_only: False` はこのスクリプト(オフラインA/B)内でのみ使う。本番提出パス
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

    base_config = dict(load_config(ml_config_name))
    base_config["policy_weights_path"] = weights_path
    _W["base_config"] = base_config
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
        return {"error": f"start {sd.errorType}", "n_decisions": 0, "win": None,
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
                # lethalを上書きしていないことは、agent()内部でis_lethalガード(一次防御は
                # `_ogerpon_q_shadow_impl`、二次防御は`agent()`自体)により構造的に保証され、
                # ログの`is_lethal_baseline`/`active_override_applied`から直接確認できる
                # (`_try_lethal`を監査目的で再実行すると探索コストが実質2倍になるため、
                # ここでは再実行しない。集計時にログのis_lethal_baselineを直接検査する)。
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


def _process_task(packed):
    from matchup_common import read_deck

    task, arm = packed
    learner_index, seed, opp_idx = task
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    config = dict(_W["base_config"])
    config["ogerpon_q_critic"] = dict(ACTIVE_Q_CONFIG if arm == "q_gate_strict" else BASELINE_Q_CONFIG)

    result = _play_one_game(deck0, deck1, learner_index, opp_pm, config, seed)
    n_active_overrides = sum(1 for rec in result["shadow_log"] if rec.get("active_override_applied"))
    n_lethal_override_violations = sum(
        1 for rec in result["shadow_log"]
        if rec.get("active_override_applied") and rec.get("is_lethal_baseline"))
    outcome_counts: dict[str, int] = {}
    for rec in result["shadow_log"]:
        outcome_counts[rec["outcome"]] = outcome_counts.get(rec["outcome"], 0) + 1
    overrides = [rec for rec in result["shadow_log"] if rec.get("active_override_applied")]
    unsupported_active_triggers = [
        rec for rec in overrides if rec["trigger_kind"] not in ACTIVE_Q_CONFIG["active_triggers"]]

    return {
        "arm": arm, "seed": seed, "learner_index": learner_index, "opponent_archetype": arch,
        "error": result["error"], "win": result["win"], "n_decisions": result["n_decisions"],
        "n_shadow_log_entries": len(result["shadow_log"]),
        "outcome_counts": outcome_counts,
        "n_active_overrides": n_active_overrides,
        "n_lethal_override_violations": n_lethal_override_violations,
        "n_unsupported_trigger_overrides": len(unsupported_active_triggers),
        "overrides": overrides,
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
    ap.add_argument("--games", type=int, default=154)
    ap.add_argument("--seed", type=int, default=13571113)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from matchup_common import read_deck

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)

    workdir = tempfile.mkdtemp(prefix="ogerq_ab_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    packed_tasks = [(t, arm) for arm in ("baseline", "q_gate_strict") for t in tasks]

    t0 = time.time()
    results = []
    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        for res in pool.imap_unordered(_process_task, packed_tasks):
            results.append(res)

    def _arm_summary(arm):
        rs = [r for r in results if r["arm"] == arm]
        wins = [r["win"] for r in rs if r["win"] is not None]
        outcome_counts_total: dict[str, int] = {}
        for r in rs:
            for k, v in r["outcome_counts"].items():
                outcome_counts_total[k] = outcome_counts_total.get(k, 0) + v
        return {
            "games": len(rs),
            "n_decisions_total": sum(r["n_decisions"] for r in rs),
            "n_errors": sum(1 for r in rs if r["error"]),
            "n_illegal_actions": sum(1 for r in rs if r["error"] and "illegal_action" in str(r["error"])),
            "n_integrity_violations_total": sum(r["n_integrity_violations"] for r in rs),
            "n_exceptions_total": sum(r["n_exceptions"] for r in rs),
            "shadow_outcome_counts": outcome_counts_total,
            "n_active_overrides_total": sum(r["n_active_overrides"] for r in rs),
            "n_lethal_override_violations_total": sum(r["n_lethal_override_violations"] for r in rs),
            "n_unsupported_trigger_overrides_total": sum(r["n_unsupported_trigger_overrides"] for r in rs),
            "win_rate_reference_only": (sum(wins) / len(wins)) if wins else None,
            "n_games_with_result": len(wins),
        }

    all_overrides = [
        {**o, "arm": r["arm"], "seed": r["seed"], "learner_index": r["learner_index"],
         "opponent_archetype": r["opponent_archetype"], "game_win": r["win"]}
        for r in results if r["arm"] == "q_gate_strict"
        for o in r["overrides"]
    ]

    summary = {
        "baseline": _arm_summary("baseline"),
        "q_gate_strict": _arm_summary("q_gate_strict"),
        "baseline_q_config": BASELINE_Q_CONFIG,
        "active_q_config": ACTIVE_Q_CONFIG,
        "all_active_overrides": all_overrides,
        "wall_seconds": time.time() - t0,
        "output": str(resolve_path(args.output)),
        "note": ("勝率(win_rate_reference_only)は参考記録であり、override発生数が"
                "少ない場合はモデル効果の判定には使用しないこと(ユーザー指示)。"),
    }

    atomic_write_json(resolve_path(args.output), {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
