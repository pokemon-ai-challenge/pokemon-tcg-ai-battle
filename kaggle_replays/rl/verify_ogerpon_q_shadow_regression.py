"""design.md Phase3 item4完了条件: shadow-only Q-critic配線がbaseline行動を一切変えない
ことを、実self-playで検証する。

## 検証方式(ユーザー指摘による改訂版)

当初案は「同じ(deck, seed, 相手)で shadow off/on それぞれ独立に1試合ずつ通し、行動列の
ハッシュを比較する」設計だったが、これは誤りだった。実際に検証した結果、shadow off の
まま同じ試合を2回リプレイしただけでも行動列が食い違うことが分かった(原因:
`search.pipeline`/PIMCが壁時計ベースのdeadline(`time_limit_ms`・動的`time_budget`)を
使っており、2回目の実行は壁時計上の実時間差だけで探索が異なる深さで打ち切られうるため)。
これはこのshadow配線とは無関係な、既存の本番エージェントが元々持つ非決定性であり、
「試合まるごと2回リプレイしてハッシュ比較」という検証方法そのものが壁時計ジッタを拾って
しまい、shadow配線の影響を切り分けられない(calibration runで実測: RNGを揃えても
`_selects_seen`/`_match_start_perf`を揃えても、baseline同士の別実行間で15%程度の
意思決定が食い違った)。

正しい検証は「同一試合を1本だけ進め、各意思決定点で"確定済みのbaseline_action"と
"shadow評価を挟んだ後にagent()が実際に返したfinal_action"を直接比較する」方式
(decision-level・同一呼び出し内比較)。これは壁時計ジッタの影響を受けない
(`ml_policy_agent._ogerpon_q_shadow`が内部で ``final_action``/``random`` 状態の
スナップショット比較を行い、``OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS`` へ記録する仕組みを
使う。このスクリプトは試合ごとにその記録を読み出して集計するだけ)。

## 受入条件(改訂版、Phase3 item4完了条件)

- `shadow_decision_count > 0`(shadowが実際に評価まで到達した意思決定が1件以上ある)
- `final_action != captured_baseline_action` の件数: 0
- shadow起因のRNG状態変化: 0
- engine error: 0
- illegal action: 0
- (weights/contract不一致/推論例外時はbaselineへ自然にフォールバックすることをログで確認)
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
SLOW_SHADOW_MS_THRESHOLD = 50.0  # 1手あたりのshadow推論時間がこれを超えたら「時間制限超過」として計上
_W: dict = {}


def _init(weights_path: str, ml_config_name: str, workdir: str, opponents):
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
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
    """試合を1本だけ進める。shadowは常に有効(``OGERPON_Q_SHADOW_LOG = []``)。
    各意思決定の非干渉性は ``ml_policy_agent`` 内部の整合性安全網
    (``OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS``)が保証し、このゲームループはその
    記録を試合単位で読み出すだけ(壁時計ジッタの影響を受けない、decision-level比較)。
    """
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
        return {
            "error": f"start {sd.errorType}", "n_decisions": 0,
            "shadow_log": [], "integrity_violations": [], "exceptions": [],
        }

    n = 0
    err = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"
                break
            if cur.result != -1:
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

    return {
        "error": err, "n_decisions": n_decisions,
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
    outcome_counts: dict[str, int] = {}
    for rec in result["shadow_log"]:
        outcome_counts[rec["outcome"]] = outcome_counts.get(rec["outcome"], 0) + 1
    durations = [rec["duration_ms"] for rec in result["shadow_log"] if "duration_ms" in rec]

    return {
        "seed": seed, "learner_index": learner_index, "opponent_archetype": arch,
        "error": result["error"], "n_decisions": result["n_decisions"],
        "n_shadow_log_entries": len(result["shadow_log"]),
        "outcome_counts": outcome_counts,
        "n_integrity_violations": len(result["integrity_violations"]),
        "integrity_violations": result["integrity_violations"],
        "n_exceptions": len(result["exceptions"]),
        "exceptions": result["exceptions"][:5],
        "durations_ms": durations,
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

    workdir = tempfile.mkdtemp(prefix="ogerq_shadow_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    t0 = time.time()
    results = []
    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        for res in pool.imap_unordered(_process_task, tasks):
            results.append(res)

    n_games = len(results)
    n_decisions_total = sum(r["n_decisions"] for r in results)
    n_shadow_log_entries_total = sum(r["n_shadow_log_entries"] for r in results)
    n_errors = sum(1 for r in results if r["error"])
    n_illegal = sum(1 for r in results if r["error"] and "illegal_action" in str(r["error"]))
    n_integrity_violations_total = sum(r["n_integrity_violations"] for r in results)
    n_exceptions_total = sum(r["n_exceptions"] for r in results)

    outcome_counts_total: dict[str, int] = {}
    for r in results:
        for k, v in r["outcome_counts"].items():
            outcome_counts_total[k] = outcome_counts_total.get(k, 0) + v

    all_durations = [d for r in results for d in r["durations_ms"]]
    all_durations.sort()

    def _pct(p):
        if not all_durations:
            return None
        idx = min(len(all_durations) - 1, int(len(all_durations) * p))
        return all_durations[idx]

    n_slow = sum(1 for d in all_durations if d > SLOW_SHADOW_MS_THRESHOLD)
    archetype_counts: dict[str, int] = {}
    for r in results:
        archetype_counts[r["opponent_archetype"]] = archetype_counts.get(r["opponent_archetype"], 0) + 1

    violating_games = [r for r in results if r["n_integrity_violations"] > 0]
    exception_games = [r for r in results if r["n_exceptions"] > 0]

    summary = {
        "games": n_games,
        "n_decisions_total": n_decisions_total,
        "n_shadow_log_entries_total": n_shadow_log_entries_total,
        "shadow_outcome_counts": outcome_counts_total,
        "shadow_decision_count": outcome_counts_total.get("evaluated", 0),
        "n_errors": n_errors,
        "n_illegal_actions": n_illegal,
        "n_integrity_violations_total": n_integrity_violations_total,
        "n_final_action_mutated": sum(
            1 for r in results for v in r["integrity_violations"] if v["kind"] == "final_action_mutated"),
        "n_rng_state_changed": sum(
            1 for r in results for v in r["integrity_violations"] if v["kind"] == "rng_state_changed"),
        "n_exceptions_total": n_exceptions_total,
        "example_exceptions": [e for r in exception_games for e in r["exceptions"]][:20],
        "example_integrity_violations": [v for r in violating_games for v in r["integrity_violations"]][:20],
        "shadow_duration_ms": {
            "p50": _pct(0.50), "p95": _pct(0.95), "p99": _pct(0.99),
            "max": max(all_durations) if all_durations else None,
            "n_exceeding_threshold_ms": SLOW_SHADOW_MS_THRESHOLD, "n_slow": n_slow,
        },
        "opponent_archetype_counts": archetype_counts,
        "wall_seconds": time.time() - t0,
        "output": str(resolve_path(args.output)),
    }

    atomic_write_json(resolve_path(args.output), {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
