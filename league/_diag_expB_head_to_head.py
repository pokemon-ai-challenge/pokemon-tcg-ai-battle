"""Diagnostic (throwaway, not shipped): Tier3 Experiment B(グループA consequence特徴)
vs 統制群(control)のミラー head-to-head。

tier3-consequence-features-design-and-implementation-plan.md §7 Experiment B /
§8 評価プロトコル(5. ミラー対戦、採用の最終ゲート)に対応する。

両側とも同一データ・同一既定レシピ(既定weight scheme)で学習した重みを使い、
consequence特徴(opp_hp_loss/opp_energy_removed/opp_special_energy_removed)の
有無だけを変える(league/_diag_tier1abc_head_to_head.py と同じ設計方針: config を
直接dictで組み立てて2エージェントをプロセス内で混線なく比較)。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_expB_head_to_head.py \\
        --candidate-name expB --games 300 --seed-start 50000 \\
        --out league/results/_diag_expB.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

_LEAGUE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _LEAGUE_DIR.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"

for _candidate in (str(_LEAGUE_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from run_match import play_match  # noqa: E402


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="統制群(consequence特徴なし) vs Experiment B(グループA consequence特徴) のミラー head-to-head。"
    )
    parser.add_argument("--config-base", default="ml_lethal_attackplan_v0only")
    parser.add_argument(
        "--control-weights", default="ptcg_ai/learning/policy_weights_base_ctrl_v2.json",
    )
    parser.add_argument(
        "--candidate-weights", default="ptcg_ai/learning/policy_weights_expB.json",
    )
    parser.add_argument(
        "--candidate-fields", default="opp_hp_loss,opp_energy_removed,opp_special_energy_removed",
    )
    parser.add_argument("--candidate-name", default="expB")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    control_weights = (_SAMPLE_SUBMISSION_DIR / args.control_weights).resolve()
    candidate_weights = (_SAMPLE_SUBMISSION_DIR / args.candidate_weights).resolve()
    for p in (control_weights, candidate_weights):
        if not p.exists():
            print(f"エラー: 重みファイルが存在しない: {p}", file=sys.stderr)
            sys.exit(1)

    import os
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from ptcg_ai.core.config import load_config
    from ptcg_ai.hidden_information import match_context
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

    base_config = load_config(args.config_base)

    config_control = copy.deepcopy(base_config)
    config_control["policy_weights_path"] = str(control_weights)

    config_candidate = copy.deepcopy(base_config)
    config_candidate["policy_weights_path"] = str(candidate_weights)

    def agent_control(obs):
        return ml_policy_agent.agent(obs, config=config_control)

    def agent_candidate(obs):
        return ml_policy_agent.agent(obs, config=config_candidate)

    deck = read_deck_csv()

    games = args.games
    records: list[dict] = []
    control_wins = 0
    candidate_wins = 0
    errors = 0
    error_reasons: dict[str, int] = {}
    turns_sum = 0
    steps_sum = 0
    valid = 0

    t_start = time.time()
    for i in range(games):
        seed = args.seed_start + i
        control_is_player0 = (i % 2 == 0)
        match_context.reset()
        if control_is_player0:
            agent0, agent1 = agent_control, agent_candidate
        else:
            agent0, agent1 = agent_candidate, agent_control

        result = play_match(agent0, agent1, deck, deck, seed=seed)

        winner_name = None
        if result.error is not None:
            errors += 1
            error_reasons[result.error] = error_reasons.get(result.error, 0) + 1
        else:
            valid += 1
            control_player_index = 0 if control_is_player0 else 1
            winner_name = "control" if result.winner == control_player_index else "candidate"
            if winner_name == "control":
                control_wins += 1
            else:
                candidate_wins += 1
            if result.turns is not None:
                turns_sum += result.turns
            steps_sum += result.steps

        records.append({
            "index": i, "seed": seed, "control_player_index": 0 if control_is_player0 else 1,
            "winner": winner_name, "turns": result.turns, "steps": result.steps,
            "seconds": result.seconds, "error": result.error,
        })

        completed = i + 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == games):
            elapsed = time.time() - t_start
            print(
                f"[progress] {completed}/{games} games "
                f"(control={control_wins} {args.candidate_name}={candidate_wins} errors={errors}) "
                f"elapsed={elapsed:.1f}s avg={elapsed / completed:.2f}s/game",
                file=sys.stderr,
            )

    elapsed_total = time.time() - t_start
    lo, hi = wilson_interval(candidate_wins, valid)

    summary = {
        "config_base": args.config_base,
        "control_weights": str(control_weights),
        "candidate_weights": str(candidate_weights),
        "candidate_fields": args.candidate_fields.split(","),
        "candidate_name": args.candidate_name,
        "games_requested": games,
        "games_valid": valid,
        "errors": errors,
        "error_reasons": error_reasons,
        "control_wins": control_wins,
        "candidate_wins": candidate_wins,
        "candidate_win_rate": (candidate_wins / valid) if valid else None,
        "candidate_win_rate_wilson_95ci": [lo, hi],
        "avg_turns": (turns_sum / valid) if valid else None,
        "avg_steps": (steps_sum / valid) if valid else None,
        "elapsed_seconds_total": elapsed_total,
        "avg_seconds_per_game": (elapsed_total / games) if games else None,
        "games": records,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== control vs {args.candidate_name} ===")
    print(f"games: requested={games} valid={valid} errors={errors}")
    print(f"{args.candidate_name} win rate: {summary['candidate_win_rate']} 95% CI {[lo, hi]}")
    if summary["avg_seconds_per_game"] is not None:
        print(f"avg seconds/game: {summary['avg_seconds_per_game']:.2f}")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
