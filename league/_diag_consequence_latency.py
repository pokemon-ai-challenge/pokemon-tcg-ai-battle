"""Diagnostic (throwaway, not shipped): Tier3 Stage3a のレイテンシ実測。

tier3-consequence-features-design-and-implementation-plan.md Stage3a DoD:
「レイテンシ実測(p50/p95/p99、attack_planの計測項目を踏襲)」に対応する。

実際のゲーム進行(ロック闘エネルギー入りルカリオ相手)を通常通り走らせつつ、ATTACK型の
選択肢が存在するMAIN局面ごとに consequence.best_effective_attack_damage を計測目的だけで
追加呼び出しする(結果は使わない・実際の行動選択には一切影響しない、純粋なピギーバック計測)。

使い方(repo rootから実行、内部で sample_submission へ chdir する):
    python league/_diag_consequence_latency.py --games 30 --seed-start 95000
"""

from __future__ import annotations

import argparse
import os
import statistics
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-deck", default="sample_submission/deck.csv")
    parser.add_argument("--opponent-deck", default="sample_submission/local_decks/demo_lucario.csv")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=95000)
    parser.add_argument("--time-limit-ms", type=float, default=100.0)
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
    from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent, read_deck_csv as _  # noqa: F401

    own_deck = _read_deck(_resolve(args.own_deck))
    opp_deck = _read_deck(_resolve(args.opponent_deck))

    durations_ms: list[float] = []
    timeouts = 0
    calls = 0
    oc_durations_ms: list[float] = []  # option_consequence(Stage3b, transaction解決込み)
    oc_calls = 0
    full_deck_cache = own_deck  # build_dummy_search_state はこちらのdeckで十分(自分視点の仮実行)

    def measuring_agent(obs):
        nonlocal calls, timeouts, oc_calls
        action = ml_policy_agent.agent(obs)  # 実際の意思決定(こちらの結果だけ使う)
        if obs.select is not None and obs.current is not None:
            option_types = [opt.type for opt in obs.select.option]
            has_attack = OptionType.ATTACK in option_types
            if has_attack:
                deadline = time.perf_counter() + args.time_limit_ms / 1000
                t0 = time.perf_counter()
                factory = lambda: build_dummy_search_state(obs, full_deck_cache)
                result = consequence.best_effective_attack_damage(obs, factory, deadline)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                durations_ms.append(elapsed_ms)
                calls += 1
                if elapsed_ms >= args.time_limit_ms:
                    timeouts += 1
            # Stage3b: ATTACH型の選択肢が1つでもあれば、その先頭を仮実行してみる
            # (transaction解決込みのコスト実測。実際の行動選択には使わない)。
            for i, opt_type in enumerate(option_types):
                if opt_type == OptionType.ATTACH:
                    deadline = time.perf_counter() + args.time_limit_ms / 1000
                    t0 = time.perf_counter()
                    factory = lambda: build_dummy_search_state(obs, full_deck_cache)
                    consequence.option_consequence(obs, i, factory, deadline)
                    oc_durations_ms.append((time.perf_counter() - t0) * 1000)
                    oc_calls += 1
                    break
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
              f"calls_so_far={calls}", file=sys.stderr)

    elapsed_total = time.time() - t_start
    durations_ms.sort()

    def pct(p):
        if not durations_ms:
            return None
        idx = min(len(durations_ms) - 1, int((len(durations_ms) - 1) * p / 100))
        return durations_ms[idx]

    print(f"\n=== consequence.best_effective_attack_damage latency ({args.games} games) ===")
    print(f"calls: {calls}, timeouts(>={args.time_limit_ms}ms): {timeouts}")
    if durations_ms:
        print(f"min={min(durations_ms):.3f}ms max={max(durations_ms):.3f}ms "
              f"mean={statistics.mean(durations_ms):.3f}ms")
        print(f"p50={pct(50):.3f}ms p95={pct(95):.3f}ms p99={pct(99):.3f}ms")

    oc_durations_ms.sort()

    def oc_pct(p):
        if not oc_durations_ms:
            return None
        idx = min(len(oc_durations_ms) - 1, int((len(oc_durations_ms) - 1) * p / 100))
        return oc_durations_ms[idx]

    print(f"\n=== consequence.option_consequence(ATTACH) latency ({args.games} games) ===")
    print(f"calls: {oc_calls}")
    if oc_durations_ms:
        print(f"min={min(oc_durations_ms):.3f}ms max={max(oc_durations_ms):.3f}ms "
              f"mean={statistics.mean(oc_durations_ms):.3f}ms")
        print(f"p50={oc_pct(50):.3f}ms p95={oc_pct(95):.3f}ms p99={oc_pct(99):.3f}ms")

    print(f"\nelapsed_total={elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
