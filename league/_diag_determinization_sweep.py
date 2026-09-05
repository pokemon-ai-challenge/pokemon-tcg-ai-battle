"""Diagnostic (throwaway, not shipped): PIMC ``num_determinizations`` sweep.

Part of Step2 of
``sample_submission/docs/plans/individual/shogo/pimc-null-diagnosis-implementation-plan.md``:
isolate whether *sampling noise* (too few determinizations) contributed to the null
result found in
``sample_submission/results/2026-07-21_pimc_prod_validation_main.md``
(``ml_lethal`` 52/100 vs ``ml_pimc`` num_determinizations=4 48/100, no significant
difference). This script reruns the same ``ml_lethal`` vs ``ml_pimc`` mirror
head-to-head at higher ``num_determinizations`` settings (16, 64), scaling
``time_limit_ms`` roughly proportionally so the higher settings actually get a
fair chance to run more determinizations rather than just hitting the same
wall-clock deadline as the baseline.

Not part of the shipped submission; not imported by ``ptcg_ai``/``main.py``.
Follows the same pattern as
``sample_submission/results/2026-07-21_pimc_prod_validation_main.md`` ("実験方法"):

- ``ptcg_ai.ml_policy.ml_policy_agent.agent(obs, config=...)`` accepts an explicit
  ``config`` dict that bypasses the module's internal cached-config singleton, so
  two independently-configured closures can run in the same process without
  clobbering each other's caches.
- ``ptcg_ai.hidden_information.match_context.reset()`` is called once per game
  (new match => discard all per-match hidden-info estimation state).
- ``league.run_match.play_match`` drives one full match via the native ``cg``
  engine (does not go through ``run_league.py``, which only supports picking
  agents from a fixed registry by name and cannot inject two different configs
  for the same underlying ``ml_policy`` module).
- Half the games have ``ml_lethal`` as player_index=0 (first player), half as
  player_index=1, to cancel first/second-player bias (same convention as
  ``run_league.py``'s A/B alternation).
- Wilson 95% CI, replicated (not imported) from ``league/run_league.py``'s
  ``wilson_interval`` per existing team convention of small-helper duplication
  across these scripts.
- ``ptcg_ai.search.pimc.reset_stats()`` / ``get_stats()`` bracket the whole run so
  the printed/saved summary includes ``determinizations_run``, ``found``, and
  timeout counts for the ``ml_pimc`` side only (the ``ml_lethal`` side uses the
  independent ``lethal_simple`` module and never touches ``pimc``'s counters).

Usage (run from repo root; the script chdirs to ``sample_submission`` itself,
matching ``run_league.py``'s convention, since ``deck.csv`` and the native
``cg.dll``/``libcg.so`` load resolve relative to that cwd)::

    python league/_diag_determinization_sweep.py \\
        --config-b ml_pimc_det16 --games 60 --seed-start 10000 \\
        --out league/results/_diag_det16.json

    python league/_diag_determinization_sweep.py \\
        --config-b ml_pimc_det64 --games 40 --seed-start 20000 \\
        --out league/results/_diag_det64.json

``--config-a`` defaults to ``ml_lethal`` (current production baseline). Both
configs are loaded via ``ptcg_ai.core.config.load_config`` from
``sample_submission/configs/<name>.json`` -- nothing here modifies those
committed config files' *existing* contents; ``ml_pimc_det16.json`` /
``ml_pimc_det64.json`` are new files added alongside this script.
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
    """Wilson score interval (95% default). Copied from ``league/run_league.py``
    (existing team convention: duplicate this small helper rather than import it).
    """
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
        description="ml_lethal vs ml_pimc (parameterized num_determinizations/time_limit_ms) "
        "mirror head-to-head, for the determinization-sweep noise diagnostic."
    )
    parser.add_argument(
        "--config-a", default="ml_lethal",
        help="Config name for side A (default: ml_lethal, current production baseline).",
    )
    parser.add_argument(
        "--config-b", required=True,
        help="Config name for side B, e.g. ml_pimc_det16 / ml_pimc_det64.",
    )
    parser.add_argument("--games", type=int, required=True, help="Total games to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Game i uses seed_start + i.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--progress-every", type=int, default=5,
        help="Print progress to stderr every N games (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # deck.csv / native cg.dll load resolve relative to sample_submission/, same
    # requirement as league/run_league.py's os.chdir.
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    # Imported after chdir (mirrors run_league.load_agent's lazy-import timing;
    # these modules don't actually require the chdir to *import*, but agent()
    # calls made afterward do require it for deck.csv/cg.dll resolution).
    from ptcg_ai.core.config import load_config
    from ptcg_ai.hidden_information import match_context
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import read_deck_csv
    from ptcg_ai.search import pimc

    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)

    def agent_a(obs):
        return ml_policy_agent.agent(obs, config=config_a)

    def agent_b(obs):
        return ml_policy_agent.agent(obs, config=config_b)

    deck = read_deck_csv()

    pimc.reset_stats()

    games = args.games
    records: list[dict] = []
    a_wins = 0
    b_wins = 0
    errors = 0
    error_reasons: dict[str, int] = {}
    turns_sum = 0
    steps_sum = 0
    valid = 0

    t_start = time.time()
    for i in range(games):
        seed = args.seed_start + i
        a_is_player0 = (i % 2 == 0)
        match_context.reset()  # new match => discard all per-match hidden-info state
        if a_is_player0:
            agent0, agent1 = agent_a, agent_b
        else:
            agent0, agent1 = agent_b, agent_a

        result = play_match(agent0, agent1, deck, deck, seed=seed)

        winner_name = None
        if result.error is not None:
            errors += 1
            error_reasons[result.error] = error_reasons.get(result.error, 0) + 1
        else:
            valid += 1
            a_player_index = 0 if a_is_player0 else 1
            winner_name = args.config_a if result.winner == a_player_index else args.config_b
            if winner_name == args.config_a:
                a_wins += 1
            else:
                b_wins += 1
            if result.turns is not None:
                turns_sum += result.turns
            steps_sum += result.steps

        records.append({
            "index": i,
            "seed": seed,
            "a_player_index": 0 if a_is_player0 else 1,
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
                f"({args.config_a}={a_wins} {args.config_b}={b_wins} errors={errors}) "
                f"elapsed={elapsed:.1f}s avg={elapsed / completed:.2f}s/game",
                file=sys.stderr,
            )

    elapsed_total = time.time() - t_start
    pimc_stats = pimc.get_stats()
    lo, hi = wilson_interval(b_wins, valid)

    summary = {
        "config_a": args.config_a,
        "config_b": args.config_b,
        "config_b_lethal_search": config_b.get("lethal_search"),
        "games_requested": games,
        "games_valid": valid,
        "errors": errors,
        "error_reasons": error_reasons,
        f"{args.config_a}_wins": a_wins,
        f"{args.config_b}_wins": b_wins,
        "b_win_rate": (b_wins / valid) if valid else None,
        "b_win_rate_wilson_95ci": [lo, hi],
        "avg_turns": (turns_sum / valid) if valid else None,
        "avg_steps": (steps_sum / valid) if valid else None,
        "elapsed_seconds_total": elapsed_total,
        "avg_seconds_per_game": (elapsed_total / games) if games else None,
        "pimc_stats": pimc_stats,
        "games": records,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== {args.config_a} vs {args.config_b} ===")
    print(f"games: requested={games} valid={valid} errors={errors}")
    print(f"{args.config_b} win rate: {summary['b_win_rate']} 95% CI {[lo, hi]}")
    print(f"avg seconds/game: {summary['avg_seconds_per_game']:.2f}")
    print(f"pimc stats: {json.dumps(pimc_stats, ensure_ascii=False)}")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
