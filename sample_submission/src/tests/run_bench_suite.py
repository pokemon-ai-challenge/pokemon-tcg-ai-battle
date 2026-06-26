"""マルチデッキ一括ベンチマーク。

複数のデッキを連続で計測し、比較テーブルを出力する。
デッキ切り替え（deck.csv 書き換え・get_deck_plan() 変更）は不要。

Usage:
    cd sample_submission

    # 両デッキ × 全Tier-Sアーキタイプ（フル計測）
    python src/tests/run_bench_suite.py --games 30

    # 特定アーキタイプのみ両デッキで計測
    python src/tests/run_bench_suite.py --games 30 --decks maries_obstagoon_ex dragapult_ex

    # Lucario のみ全アーキタイプ計測
    python src/tests/run_bench_suite.py --games 30 --my-decks lucario

    # 3デッキ同時計測
    python src/tests/run_bench_suite.py --games 30 --my-decks maries lucario hydrapple
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.knowledge.deck_plan as dp
from src.tests.tier_s_benchmark import (
    MY_DECK_CONFIGS,
    DECK_CATALOG,
    _load_deck_csv,
    run_benchmark,
    print_report,
    tier_s_decks,
)

SUITE_CONFIGS: dict[str, tuple[str, dp.DeckPlan]] = {
    "maries":    ("deck_maries_obstagoon.csv", dp.MARIES_OBSTAGOON_PLAN),
    "lucario":   ("deck_lucario.csv",           dp.LUCARIO_PLAN),
    "hydrapple": ("deck_hydrapple.csv",          dp.HYDRAPPLE_PLAN),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-deck benchmark suite")
    parser.add_argument("--games", type=int, default=30, help="Games per opponent deck")
    parser.add_argument(
        "--my-decks",
        nargs="+",
        default=["maries", "lucario"],
        choices=list(SUITE_CONFIGS.keys()),
        help="My decks to benchmark (default: maries lucario)",
    )
    parser.add_argument(
        "--decks",
        nargs="*",
        default=None,
        help="Opponent decks (default: all Tier-S)",
    )
    args = parser.parse_args()

    opp_deck_names = args.decks if args.decks else tier_s_decks()
    invalid = [d for d in opp_deck_names if d not in DECK_CATALOG]
    if invalid:
        print(f"Unknown opponent deck names: {invalid}")
        sys.exit(1)

    all_results: dict[str, dict[str, dict]] = {}

    for my_deck_name in args.my_decks:
        csv_file, plan = SUITE_CONFIGS[my_deck_name]
        dp.set_active_plan(plan)
        our_deck = _load_deck_csv(csv_file)

        print(f"\n{'='*60}")
        print(f"SUITE: my_deck={my_deck_name} (plan={plan.name})")
        print(f"{'='*60}")

        results = run_benchmark(opp_deck_names, args.games, our_deck)
        print_report(results, args.games)
        all_results[my_deck_name] = results

    dp.set_active_plan(None)

    if len(args.my_decks) < 2:
        return

    # 比較テーブル
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print(f"{'Opponent':<25}", end="")
    for name in args.my_decks:
        print(f"  {name:>10}", end="")
    if len(args.my_decks) == 2:
        diff_label = f"  {'diff':>6}"
        print(diff_label, end="")
    print()
    print("-" * 70)

    for opp in sorted(opp_deck_names):
        print(f"{opp:<25}", end="")
        rates: list[float] = []
        for name in args.my_decks:
            r = all_results[name].get(opp, {})
            wr = r.get("win_rate", 0.0)
            rates.append(wr)
            print(f"  {wr:>9.0%}", end="")
        if len(rates) == 2:
            diff = rates[1] - rates[0]
            sign = "+" if diff >= 0 else ""
            print(f"  {sign}{diff:.0%}", end="")
        print()

    print("=" * 70)

    # 全体平均行
    print(f"{'AVERAGE':<25}", end="")
    for name in args.my_decks:
        rs = all_results[name]
        total = sum(v["wins"] + v["losses"] for v in rs.values())
        wins = sum(v["wins"] for v in rs.values())
        wr = wins / total if total else 0.0
        print(f"  {wr:>9.0%}", end="")
    print()


if __name__ == "__main__":
    main()
