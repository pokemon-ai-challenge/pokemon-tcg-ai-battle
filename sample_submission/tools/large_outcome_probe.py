"""Step 1-22: 大規模 outcome（65+）での Phase 2 の挙動を実測する。

コーパスの chance ノードが最大 53 だったのは、採取が**終盤に偏っていた**ため。
outcome 数は「山札の残り枚数 × 異なるカード種 × ドロー枚数」で決まり、
実測（`deck.csv`, 22 種）では

    残り 60 / 22 種: k=2 -> 249, k=3 -> 1932, k=4 -> 11548
    残り 40 / 12 種: k=2 ->  77, k=3 ->  352, k=4 ->  1282
    残り 30 /  8 種: k=3 ->  120

なので、**序盤の 2〜3 枚ドローが自然な大規模 outcome 源**である。
局面は捏造せず、自己対戦の序盤から採取する。
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import read_deck_csv
from ptcg_ai.action_selection import router
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal import phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession

OUT = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_large_outcome.jsonl"
BUCKETS = ((16, "1-16"), (32, "17-32"), (64, "33-64"), (128, "65-128"),
           (256, "129-256"), (512, "257-512"))


def bucket_of(count: int) -> str:
    for limit, name in BUCKETS:
        if count <= limit:
            return name
    return ">512"


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def probe_position(obs, obs_dict, deck, rows, seen):
    """root の chance ノードを調べ、physical outcome 数を記録する。"""
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    if stub is None:
        return
    hidden = HiddenState.from_stub(stub)
    try:
        with SearchSession(obs, hidden) as session:
            backend = CgBackend(
                session, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            backend.attach_budget(
                Budget(time_limit_ms=8000.0, max_nodes=10**9, max_depth=8,
                       max_chance_depth=1).start()
            )
            state = backend.root()
            best = None
            for action in backend.legal_actions(state):
                transition = backend.apply(state, tuple(action))
                if transition.state is None or not transition.revealed:
                    continue
                enumeration = backend.enumerate_outcomes(state, tuple(action))
                if enumeration.outcome_set is None:
                    continue
                count = len(enumeration.outcome_set.outcomes)
                mass = sum(o.mass for o in enumeration.outcome_set.outcomes)
                if best is None or count > best["physical_outcomes"]:
                    best = {
                        "action": list(action),
                        "physical_outcomes": count,
                        "complete_enumeration": bool(
                            enumeration.outcome_set.coverage_certified and mass == 1
                        ),
                        "processed_mass": str(mass),
                    }
    except Exception:  # noqa: BLE001
        return
    if best is None or best["physical_outcomes"] < 17:
        return
    key = (obs.current.turn, best["physical_outcomes"],
           obs.current.players[0].deckCount)
    if key in seen:
        return
    seen[key] = True
    rows.append({
        "id": f"big{len(rows):04d}",
        "turn": obs.current.turn,
        "deck_count": obs.current.players[0].deckCount,
        "prizes_left": len(obs.current.players[0].prize),
        "chance": best,
        "bucket": bucket_of(best["physical_outcomes"]),
        "hidden_stub": {k: list(v) if isinstance(v, (list, tuple)) else v
                        for k, v in stub.items()},
        "obs": obs_dict,
    })


def collect(games: int) -> list[dict]:
    deck = read_deck_csv()
    rows: list[dict] = []
    seen: dict = {}
    for game in range(games):
        rng = random.Random(80_000 + game)
        obs_dict, start = battle_start(list(deck), list(deck))
        if start.errorType != 0:
            continue
        try:
            for _ in range(400):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    break
                if obs.select is None:
                    obs_dict = battle_select(list(deck))
                    continue
                if obs.current.yourIndex == 0:
                    # 山札が大きいうちだけ調べる（大規模 outcome の源）
                    if obs.current.turn >= 1 and obs.current.players[0].deckCount >= 20:
                        probe_position(obs, obs_dict, deck, rows, seen)
                    action = router.route(obs) or [0]
                else:
                    count = rng.randint(obs.select.minCount,
                                        min(obs.select.maxCount, len(obs.select.option)))
                    action = (rng.sample(range(len(obs.select.option)), count)
                              if count else [])
                obs_dict = battle_select(action)
        finally:
            battle_finish()
        if game % 5 == 4:
            print(f"  game {game + 1}/{games} collected={len(rows)} "
                  f"{dict(Counter(r['bucket'] for r in rows))}", flush=True)
    return rows


def measure(rows, deck) -> None:
    print("\n## Phase 2 の outcome 数別 scaling")
    for budget_ms in (100.0, 200.0, 500.0):
        buckets = defaultdict(list)
        for row in rows:
            observation = to_observation_class(row["obs"])
            hidden = HiddenState.from_stub(row["hidden_stub"])
            started = time.perf_counter()
            try:
                with SearchSession(observation, hidden) as session:
                    backend = CgBackend(
                        session, observation, hidden,
                        deck_composition=KnownDeckComposition.from_card_ids(deck),
                    )
                    if backend.is_win(backend.root()):
                        continue
                    result = phase2.search(
                        backend,
                        Budget(time_limit_ms=budget_ms, max_nodes=20_000,
                               max_depth=8, max_chance_depth=1),
                    )
                    elapsed = (time.perf_counter() - started) * 1000.0
                    buckets[row["bucket"]].append({
                        "ms": elapsed,
                        "proof": result.proof.name,
                        "stop": {s.name for s in result.stop_reasons},
                        "nodes": result.nodes,
                        "search_begin": getattr(session, "acquired", 0),
                        "complete": row["chance"]["complete_enumeration"],
                    })
            except Exception:  # noqa: BLE001
                continue
        print(f"\n### budget {budget_ms:.0f}ms")
        print(f"  {'outcomes':>10}{'fixture':>8}{'complete':>10}{'WIN':>5}"
              f"{'NOWIN':>7}{'UNK':>5}{'TIME':>6}{'NOTENUM':>9}"
              f"{'sbegin':>8}{'p50':>8}{'p95':>8}{'max':>8}")
        for _limit, name in list(BUCKETS) + [(0, ">512")]:
            group = buckets.get(name)
            if not group:
                continue
            times = [g["ms"] for g in group]
            proofs = Counter(g["proof"] for g in group)
            print(f"  {name:>10}{len(group):>8}"
                  f"{sum(1 for g in group if g['complete']):>10}"
                  f"{proofs['PROVEN_WIN']:>5}{proofs['PROVEN_NO_WIN']:>7}"
                  f"{proofs['UNKNOWN']:>5}"
                  f"{sum(1 for g in group if 'TIME_LIMIT' in g['stop']):>6}"
                  f"{sum(1 for g in group if 'OUTCOMES_NOT_ENUMERABLE' in g['stop']):>9}"
                  f"{sum(g['search_begin'] for g in group):>8}"
                  f"{percentile(times, 0.5):>8.1f}{percentile(times, 0.95):>8.1f}"
                  f"{max(times):>8.1f}")
        worst = max((g["ms"] for group in buckets.values() for g in group), default=0.0)
        print(f"  予算遵守: 実測最大 {worst:.1f}ms （予算 {budget_ms:.0f}ms）")


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    rows = collect(games)
    OUT.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\n収集: {len(rows)} 件 -> {OUT}")
    print("  bucket:", dict(Counter(r["bucket"] for r in rows)))
    print("  complete enumeration:",
          f"{sum(1 for r in rows if r['chance']['complete_enumeration'])}/{len(rows)}")
    if rows:
        measure(rows, read_deck_csv())


if __name__ == "__main__":
    main()
