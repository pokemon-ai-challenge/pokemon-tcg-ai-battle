"""Diagnostic (throwaway, not shipped): ATTACK専用ハイブリッド(案B) vs 構成Cのミラー head-to-head。

attack-rulebased-hybrid-implementation-plan.md Step3: 構成C(`policy_weights_configC.json`)を
両側で固定し、`config["attack_hybrid"]["enabled"]` の有無だけを変えてミラー対戦させる
(`league/_diag_skillconc_head_to_head.py` と同じ構成上の方針: config を直接dictで組み立てて
2エージェントをプロセス内で混線なく比較、match_context.reset() を毎試合、Wilson 95% CI、
先手後手半々)。

変更する軸は`attack_hybrid`のON/OFFのみ(attack-rulebased-hybrid-strategy.md §3.2 の一軸原則)。
両側とも `policy_weights_path` は同じ構成Cに固定するため、この実験の結論(ATTACK委譲が
構成C比でさらに勝ち越すか)は構成C自体の検証状況に依存しない。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_attack_hybrid_head_to_head.py \\
        --weights-path kaggle_replays/policy_net/policy_weights_configC.json \\
        --candidate-name attack_hybrid_on --games 200 --seed-start 40000 \\
        --out league/results/_diag_attack_hybrid_configC.json
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
    """Wilson score interval (95% default). league/_diag_skillconc_head_to_head.py と同じ複製実装。"""
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
        description="構成C(重み固定) vs 構成C+ATTACK委譲(attack_hybrid.enabled=True) のミラー head-to-head。"
    )
    parser.add_argument(
        "--config-base", default="ml_lethal",
        help="両側のベースとなる config 名(既定: ml_lethal、本番と同じ探索設定)。",
    )
    parser.add_argument(
        "--weights-path", default="kaggle_replays/policy_net/policy_weights_configC.json",
        help="両側で固定する重みJSONへの相対/絶対パス(既定: 構成C)。",
    )
    parser.add_argument(
        "--candidate-name", required=True,
        help="レポート用の候補名(例: attack_hybrid_on)。",
    )
    parser.add_argument("--games", type=int, required=True, help="総試合数。")
    parser.add_argument("--seed-start", type=int, default=0, help="i試合目は seed_start + i を使う。")
    parser.add_argument("--out", required=True, help="出力JSONパス。")
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="標準エラーに進捗を出す間隔(既定: 10試合ごと)。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    weights_path = Path(args.weights_path)
    if not weights_path.is_absolute():
        weights_path = (_ROOT_DIR / weights_path).resolve()
    if not weights_path.exists():
        print(f"エラー: 重みファイルが存在しない: {weights_path}", file=sys.stderr)
        sys.exit(1)

    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from ptcg_ai.core.config import load_config
    from ptcg_ai.hidden_information import match_context
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

    base_config = load_config(args.config_base)
    base_config["policy_weights_path"] = str(weights_path)

    config_baseline = copy.deepcopy(base_config)  # 構成C、attack_hybrid無し(既定False)
    config_candidate = copy.deepcopy(base_config)
    config_candidate["attack_hybrid"] = {"enabled": True}

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
                f"(baseline={baseline_wins} {args.candidate_name}={candidate_wins} errors={errors}) "
                f"elapsed={elapsed:.1f}s avg={elapsed / completed:.2f}s/game",
                file=sys.stderr,
            )

    elapsed_total = time.time() - t_start
    lo, hi = wilson_interval(candidate_wins, valid)

    summary = {
        "config_base": args.config_base,
        "weights_path": str(weights_path),
        "candidate_name": args.candidate_name,
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

    print(f"\n=== baseline(configC) vs {args.candidate_name}(configC+attack_hybrid) ===")
    print(f"games: requested={games} valid={valid} errors={errors}")
    print(f"{args.candidate_name} win rate: {summary['candidate_win_rate']} 95% CI {[lo, hi]}")
    if summary["avg_seconds_per_game"] is not None:
        print(f"avg seconds/game: {summary['avg_seconds_per_game']:.2f}")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
