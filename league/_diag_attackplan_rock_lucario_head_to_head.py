"""Diagnostic (throwaway, not shipped): attack_plan v0 の実戦検証。

design-and-implementation-plan.md の Step0/Step1 相当を、狭く対象を絞って先行実施する。
「ロック闘エネルギー付きメガルカリオex」デッキ(local_decks/demo_lucario.csv)を相手に固定し、
自分側(フーディン, deck.csv)の config だけを ml_lethal(baseline) / ml_lethal_attackplan_v0only
(candidate) で切り替えて2バッチ対戦させる。対戦相手は rule_based に固定する(attack_plan は
ml_policy 側にしか無いため、rule_based を対戦相手にすることで比較対象の変数を自分側の
config だけに絞れる)。

各バッチの前後で attack_plan.get_stats() を reset/取得し、実戦で
- ATTACK を選んだ回数のうちダメージ0だった回数
- そのうち有益な副作用も無い「真の無駄攻撃」だった回数
- v0 のマスク再選択で代替行動が見つかった回数(v0_alternatives)
- タイムアウト回数
を記録する(design-and-implementation-plan.md Step0 の計測項目のサブセット)。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_attackplan_rock_lucario_head_to_head.py \\
        --games 150 --seed-start 90000 \\
        --out league/results/_diag_attackplan_rock_lucario.json
"""

from __future__ import annotations

import argparse
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
        description="ロック闘エネルギー入りルカリオデッキ相手に、自分側 config を "
        "ml_lethal(baseline) / ml_lethal_attackplan_v0only(candidate) で切り替えて2バッチ対戦。"
    )
    parser.add_argument("--own-deck", default="sample_submission/deck.csv")
    parser.add_argument("--opponent-deck", default="sample_submission/local_decks/demo_lucario.csv")
    parser.add_argument("--baseline-config", default="ml_lethal")
    parser.add_argument("--candidate-config", default="ml_lethal_attackplan_v0only")
    parser.add_argument("--games", type=int, required=True, help="バッチあたりの試合数")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (_ROOT_DIR / path).resolve()
    return path


def _read_deck(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    deck: list[int] = []
    for raw in text.replace(",", "\n").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    if len(deck) != 60:
        raise ValueError(f"deck must have exactly 60 cards, got {len(deck)}: {path}")
    return deck


def run_batch(*, own_agent, own_deck, opp_agent, opp_deck, games, seed_start, progress_every, label):
    from ptcg_ai.hidden_information import match_context

    own_wins = 0
    valid = 0
    errors = 0
    error_reasons: dict[str, int] = {}
    turns_sum = 0
    records: list[dict] = []

    t_start = time.time()
    for i in range(games):
        seed = seed_start + i
        own_is_player0 = (i % 2 == 0)
        match_context.reset()
        if own_is_player0:
            agent0, agent1 = own_agent, opp_agent
            deck0, deck1 = own_deck, opp_deck
        else:
            agent0, agent1 = opp_agent, own_agent
            deck0, deck1 = opp_deck, own_deck

        result = play_match(agent0, agent1, deck0, deck1, seed=seed)

        winner = None
        if result.error is not None:
            errors += 1
            error_reasons[result.error] = error_reasons.get(result.error, 0) + 1
        else:
            valid += 1
            own_player_index = 0 if own_is_player0 else 1
            winner = "own" if result.winner == own_player_index else "opponent"
            if winner == "own":
                own_wins += 1
            if result.turns is not None:
                turns_sum += result.turns

        records.append({
            "index": i, "seed": seed, "own_player_index": 0 if own_is_player0 else 1,
            "winner": winner, "turns": result.turns, "error": result.error,
        })

        completed = i + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == games):
            elapsed = time.time() - t_start
            print(
                f"[{label}] {completed}/{games} games (own_wins={own_wins} errors={errors}) "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    lo, hi = wilson_interval(own_wins, valid)
    return {
        "label": label, "games_requested": games, "games_valid": valid, "errors": errors,
        "error_reasons": error_reasons, "own_wins": own_wins,
        "own_win_rate": (own_wins / valid) if valid else None,
        "own_win_rate_wilson_95ci": [lo, hi],
        "avg_turns": (turns_sum / valid) if valid else None,
        "elapsed_seconds": time.time() - t_start,
        "games": records,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from ptcg_ai.core.config import load_config
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent
    from ptcg_ai.search import attack_plan

    own_deck_path = _resolve(args.own_deck)
    opp_deck_path = _resolve(args.opponent_deck)
    own_deck = _read_deck(own_deck_path)
    opp_deck = _read_deck(opp_deck_path)

    baseline_config = load_config(args.baseline_config)
    candidate_config = load_config(args.candidate_config)

    def own_agent_baseline(obs):
        return ml_policy_agent.agent(obs, config=baseline_config)

    def own_agent_candidate(obs):
        return ml_policy_agent.agent(obs, config=candidate_config)

    def opponent_agent(obs):
        return rule_based_agent(obs)

    batches = {}
    for label, own_agent, cfg_name in (
        ("baseline", own_agent_baseline, args.baseline_config),
        ("candidate", own_agent_candidate, args.candidate_config),
    ):
        attack_plan.reset_stats()
        batch = run_batch(
            own_agent=own_agent, own_deck=own_deck,
            opp_agent=opponent_agent, opp_deck=opp_deck,
            games=args.games, seed_start=args.seed_start,
            progress_every=args.progress_every, label=label,
        )
        batch["config_name"] = cfg_name
        batch["attack_plan_stats"] = attack_plan.get_stats()
        batches[label] = batch
        print(
            f"\n=== {label} ({cfg_name}) vs rule_based(demo_lucario) ===\n"
            f"own win rate: {batch['own_win_rate']} 95% CI {batch['own_win_rate_wilson_95ci']} "
            f"(valid={batch['games_valid']} errors={batch['errors']})\n"
            f"attack_plan stats: {batch['attack_plan_stats']}"
        )

    summary = {
        "own_deck_path": str(own_deck_path), "opponent_deck_path": str(opp_deck_path),
        "opponent_agent": "rule_based",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": batches["baseline"], "candidate": batches["candidate"],
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to: {out_path}")


if __name__ == "__main__":
    main()
