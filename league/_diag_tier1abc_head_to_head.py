"""Diagnostic (throwaway, not shipped): 特徴量拡張 Tier1a/1b/1c vs 統制群(tierなし)の
ミラー head-to-head。

policy-feature-expansion/stage1-tier1abc-implementation-plan.md Step6 の採用ゲート。
`_diag_skillconc_head_to_head.py` を踏襲するが、**両側とも `policy_weights_path` を明示指定できる**
ように拡張している(統制群比較のため。baseline=tierなし統制重み、candidate=tier重み。
探索設定は同一 base config を共有し、"重みの違いだけ"を比較する)。

統制群比較(clean ablation): baseline と candidate は同一レシピ・同一データで学習し、
tier 特徴の有無だけが異なる。これにより tier 特徴の純粋な効果を切り分ける。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_tier1abc_head_to_head.py \\
        --baseline-weights ptcg_ai/learning/policy_weights_base_ctrl.json \\
        --candidate-weights ptcg_ai/learning/policy_weights_tier1abc.json \\
        --candidate-name tier1abc --games 200 --seed-start 40000 \\
        --out league/results/_diag_tier1abc.json

``--baseline-weights`` を省略すると現行本番(既定 policy_weights.json)が baseline になる。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    """Wilson score interval (95% default)。skillconc/determinization_sweep と同じ複製実装。"""
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


def _resolve_weights(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    p = Path(path_arg)
    if not p.is_absolute():
        p = (_ROOT_DIR / p).resolve()
    if not p.exists():
        print(f"エラー: 重みファイルが存在しない: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier1a/1b/1c 候補 vs 統制群(tierなし)のミラー head-to-head。"
    )
    parser.add_argument(
        "--config-base", default="ml_lethal_attackplan_v0only",
        help="両側で共有する base config 名(既定: 本番と同じ探索設定)。",
    )
    parser.add_argument(
        "--baseline-weights", default=None,
        help="統制群(tierなし)重みへの相対/絶対パス。省略時は現行本番 policy_weights.json。",
    )
    parser.add_argument("--baseline-name", default="control", help="レポート用の baseline 名。")
    parser.add_argument(
        "--candidate-weights", required=True,
        help="tier 候補重みへの相対/絶対パス。",
    )
    parser.add_argument("--candidate-name", required=True, help="レポート用の候補名(例: tier1abc)。")
    parser.add_argument("--games", type=int, required=True, help="総試合数。")
    parser.add_argument("--seed-start", type=int, default=0, help="i試合目は seed_start + i を使う。")
    parser.add_argument("--out", required=True, help="出力JSONパス。")
    parser.add_argument("--progress-every", type=int, default=10, help="進捗出力間隔(既定: 10試合)。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    baseline_weights = _resolve_weights(args.baseline_weights)
    candidate_weights = _resolve_weights(args.candidate_weights)

    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from ptcg_ai.core.config import load_config
    from ptcg_ai.hidden_information import match_context
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

    base_config = load_config(args.config_base)
    config_baseline = copy.deepcopy(base_config)
    if baseline_weights is not None:
        config_baseline["policy_weights_path"] = str(baseline_weights)
    config_candidate = copy.deepcopy(base_config)
    config_candidate["policy_weights_path"] = str(candidate_weights)

    def agent_baseline(obs):
        return ml_policy_agent.agent(obs, config=config_baseline)

    def agent_candidate(obs):
        return ml_policy_agent.agent(obs, config=config_candidate)

    deck = read_deck_csv()

    games = args.games
    records: list[dict] = []
    baseline_wins = 0
    candidate_wins = 0
    errors = 0
    error_reasons: dict[str, int] = {}
    turns_sum = 0
    steps_sum = 0
    valid = 0

    t_start = time.time()
    for i in range(games):
        seed = args.seed_start + i
        baseline_is_player0 = (i % 2 == 0)
        match_context.reset()
        if baseline_is_player0:
            agent0, agent1 = agent_baseline, agent_candidate
        else:
            agent0, agent1 = agent_candidate, agent_baseline

        result = play_match(agent0, agent1, deck, deck, seed=seed)

        winner_name = None
        if result.error is not None:
            errors += 1
            error_reasons[result.error] = error_reasons.get(result.error, 0) + 1
        else:
            valid += 1
            baseline_player_index = 0 if baseline_is_player0 else 1
            winner_name = "baseline" if result.winner == baseline_player_index else "candidate"
            if winner_name == "baseline":
                baseline_wins += 1
            else:
                candidate_wins += 1
            if result.turns is not None:
                turns_sum += result.turns
            steps_sum += result.steps

        records.append({
            "index": i,
            "seed": seed,
            "baseline_player_index": 0 if baseline_is_player0 else 1,
            "winner": winner_name,
            "turns": result.turns,
            "steps": result.steps,
            "seconds": result.seconds,
            "error": result.error,
        })

        completed = i + 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == games):
            elapsed = time.time() - t_start
            print(
                f"[progress] {completed}/{games} games "
                f"({args.baseline_name}={baseline_wins} {args.candidate_name}={candidate_wins} "
                f"errors={errors}) elapsed={elapsed:.1f}s avg={elapsed / completed:.2f}s/game",
                file=sys.stderr,
            )

    elapsed_total = time.time() - t_start
    lo, hi = wilson_interval(candidate_wins, valid)

    summary = {
        "config_base": args.config_base,
        "baseline_name": args.baseline_name,
        "baseline_weights_path": str(baseline_weights) if baseline_weights else "(default policy_weights.json)",
        "candidate_name": args.candidate_name,
        "candidate_weights_path": str(candidate_weights),
        "games_requested": games,
        "games_valid": valid,
        "errors": errors,
        "error_reasons": error_reasons,
        "baseline_wins": baseline_wins,
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

    print(f"\n=== {args.baseline_name} vs {args.candidate_name} ===")
    print(f"games: requested={games} valid={valid} errors={errors}")
    print(
        f"{args.candidate_name} win rate: {summary['candidate_win_rate']} "
        f"95% CI {[round(lo, 4), round(hi, 4)]}"
    )
    if summary["avg_seconds_per_game"] is not None:
        print(f"avg seconds/game: {summary['avg_seconds_per_game']:.2f}")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
