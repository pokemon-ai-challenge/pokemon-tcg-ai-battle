"""Diagnostic (throwaway, not shipped): Tier3のStage3c投資判断のための事前チェック。

tier3-consequence-features-design-and-implementation-plan.md §2.3で決めた
「学習時もdummy隠れ状態のみを使う(train/runtime parity優先)」という制約が、
実際の局面で意味のある信号(delta_best_effective_attack_damage等)を残すか、
それとも相手の非公開情報が無いことでほぼ退化(常に0)してしまうかを、
本格的なencoder/build_features配線・再学習の前に安く確認する。

実際の対戦(自分のデッキ vs ロック闘エネルギー入りルカリオ)を通常通り走らせつつ、
ATTACH/PLAY/EVOLVE型の選択肢が存在するMAIN局面で consequence.option_consequence を
計測目的で追加呼び出しし(実際の行動選択には使わない)、結果の分布を集計する。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_consequence_distribution.py --games 40 --seed-start 98000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_LEAGUE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _LEAGUE_DIR.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"

for _candidate in (str(_LEAGUE_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from run_match import play_match  # noqa: E402

_TARGET_OPTION_TYPE_NAMES = ("ATTACH", "PLAY", "EVOLVE")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-deck", default="sample_submission/deck.csv")
    parser.add_argument("--opponent-deck", default="sample_submission/local_decks/demo_lucario.csv")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=98000)
    parser.add_argument("--time-limit-ms", type=float, default=100.0)
    parser.add_argument("--max-options-per-decision", type=int, default=3,
                         help="1局面あたり試す対象選択肢の上限(全部試すとコストが増えるため)")
    parser.add_argument("--out", default="league/results/_diag_consequence_distribution.json")
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (_ROOT_DIR / path).resolve()
    return path


def _read_deck(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    deck = [int(v.strip()) for v in text.replace(",", "\n").splitlines() if v.strip() and not v.strip().startswith("#")]
    if len(deck) != 60:
        raise ValueError(f"deck must have exactly 60 cards, got {len(deck)}: {path}")
    return deck


def main(argv=None) -> None:
    args = parse_args(argv)
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from cg.api import OptionType
    from ptcg_ai.board_evaluation import consequence
    from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent

    target_types = {getattr(OptionType, name) for name in _TARGET_OPTION_TYPE_NAMES}

    own_deck = _read_deck(_resolve(args.own_deck))
    opp_deck = _read_deck(_resolve(args.opponent_deck))

    records: list[dict] = []
    unresolved = 0
    attempted = 0
    errors = 0
    full_deck_cache = own_deck

    def measuring_agent(obs):
        nonlocal unresolved, attempted, errors
        action = ml_policy_agent.agent(obs)
        if obs.select is not None and obs.current is not None:
            tried = 0
            for i, opt in enumerate(obs.select.option):
                if opt.type not in target_types:
                    continue
                if tried >= args.max_options_per_decision:
                    break
                tried += 1
                attempted += 1
                deadline = time.perf_counter() + args.time_limit_ms / 1000
                factory = lambda: build_dummy_search_state(obs, full_deck_cache)
                try:
                    result = consequence.option_consequence(obs, i, factory, deadline)
                except Exception:
                    errors += 1
                    continue
                if result is None:
                    unresolved += 1
                    continue
                records.append({
                    "option_type": OptionType(opt.type).name,
                    "delta_best_effective_attack_damage": result.delta_best_effective_attack_damage,
                    "delta_can_ko": result.delta_can_ko,
                    "opp_hp_loss": result.opp_hp_loss,
                    "self_hp_gain": result.self_hp_gain,
                    "opp_energy_removed": result.opp_energy_removed,
                    "opp_special_energy_removed": result.opp_special_energy_removed,
                    "self_energy_added": result.self_energy_added,
                    "cards_drawn": result.cards_drawn,
                    "pokemon_evolved": result.pokemon_evolved,
                    "stadium_changed": result.stadium_changed,
                })
        return action

    def opponent_agent(obs):
        return rule_based_agent(obs)

    t_start = time.time()
    for i in range(args.games):
        seed = args.seed_start + i
        own_is_player0 = (i % 2 == 0)
        if own_is_player0:
            agent0, agent1, deck0, deck1 = measuring_agent, opponent_agent, own_deck, opp_deck
        else:
            agent0, agent1, deck0, deck1 = opponent_agent, measuring_agent, opp_deck, own_deck
        result = play_match(agent0, agent1, deck0, deck1, seed=seed)
        print(f"[{i+1}/{args.games}] winner={result.winner} error={result.error} "
              f"records_so_far={len(records)} unresolved_so_far={unresolved}", file=sys.stderr)

    elapsed_total = time.time() - t_start

    # --- 集計 ---
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["option_type"], []).append(r)

    summary = {
        "games": args.games,
        "attempted": attempted,
        "resolved": len(records),
        "unresolved": unresolved,
        "errors": errors,
        "elapsed_seconds": elapsed_total,
        "by_option_type": {},
    }

    print(f"\n=== consequence distribution over {args.games} real games ===")
    print(f"attempted={attempted} resolved={len(records)} unresolved={unresolved} errors={errors}")

    for opt_type, rs in sorted(by_type.items()):
        deltas = [r["delta_best_effective_attack_damage"] for r in rs]
        nonzero = [d for d in deltas if d != 0]
        positive = [d for d in deltas if d > 0]
        counter_fields = {
            "opp_energy_removed": sum(1 for r in rs if r["opp_energy_removed"]),
            "opp_special_energy_removed": sum(1 for r in rs if r["opp_special_energy_removed"]),
            "self_energy_added": sum(1 for r in rs if r["self_energy_added"]),
            "delta_can_ko": sum(1 for r in rs if r["delta_can_ko"]),
            "cards_drawn>0": sum(1 for r in rs if r["cards_drawn"] > 0),
            "pokemon_evolved": sum(1 for r in rs if r["pokemon_evolved"]),
        }
        stats = {
            "n": len(rs),
            "delta_nonzero": len(nonzero),
            "delta_positive": len(positive),
            "delta_min": min(deltas) if deltas else None,
            "delta_max": max(deltas) if deltas else None,
            "delta_mean": (sum(deltas) / len(deltas)) if deltas else None,
            **counter_fields,
        }
        summary["by_option_type"][opt_type] = stats
        print(f"\n[{opt_type}] n={stats['n']}")
        print(f"  delta_best_effective_attack_damage: nonzero={stats['delta_nonzero']}/{stats['n']} "
              f"positive={stats['delta_positive']}/{stats['n']} "
              f"min={stats['delta_min']} max={stats['delta_max']} mean={stats['delta_mean']:.2f}"
              if deltas else "  (no records)")
        for k, v in counter_fields.items():
            print(f"  {k}: {v}/{stats['n']}")

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to: {out_path}")
    print(f"elapsed_total={elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
