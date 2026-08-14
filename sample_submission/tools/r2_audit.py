"""Step 1-25: R2（既知 listing を decision として扱う）の適用可能性と Before/After。

solver は変更しない。R2 の 4 条件を **監査側で独立に再計算**して内訳を出し、
そのうえで `known_listing_as_decision` を OFF / ON にして同一条件で比較する。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import OptionType, SelectContext, to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import _KNOWN_LISTING_CONTEXTS, CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"
BUDGET_MS = 500.0


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def make_backend(row, deck, *, r2: bool):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    session = SearchSession(observation, hidden)
    session.__enter__()
    backend = CgBackend(
        session, observation, hidden,
        deck_composition=KnownDeckComposition.from_card_ids(deck),
        known_listing_as_decision=r2,
    )
    return session, backend


def r2_reason(backend, child_observation, state, me: int) -> str:
    """R2 の 4 条件を**監査側で独立に**判定して、落ちた理由を返す。"""
    select = child_observation.select
    current = child_observation.current
    if select is None or select.deck is None or current is None:
        return "NOT_A_LISTING"
    if current.yourIndex != me:
        return "NOT_OUR_TURN"
    if int(select.context) not in _KNOWN_LISTING_CONTEXTS:
        return "WRONG_CONTEXT"
    if any(card is None for card in select.deck):
        return "HIDDEN_CARD_PRESENT"
    expected = backend._belief_from_supply(state)
    if expected is None:
        return "BELIEF_UNAVAILABLE"
    listing = Counter(card.id for card in select.deck)
    if listing - expected:
        return "LISTING_NOT_SUBSET"
    return "APPLICABLE"


def audit_applicability(rows, deck) -> None:
    print("\n## R2 applicability（115 件の 4 条件判定）")
    reasons = Counter()
    by_action = defaultdict(Counter)
    positive_reasons = Counter()
    for row in rows:
        session, backend = make_backend(row, deck, r2=False)
        try:
            backend.attach_budget(
                Budget(time_limit_ms=4000.0, max_nodes=10**9, max_depth=8,
                       max_chance_depth=1).start()
            )
            observation = to_observation_class(row["obs"])
            state = backend.root()
            node = session.root
            for action in backend.legal_actions(state):
                kinds = "+".join(
                    OptionType(int(observation.select.option[i].type)).name
                    for i in action if i < len(observation.select.option)
                ) or "(none)"
                if kinds not in ("PLAY", "ATTACH"):
                    continue
                transition = backend.apply(state, tuple(action))
                if transition.state is None or not transition.revealed:
                    continue
                try:
                    child, events = session.step(node, list(action))
                except Exception:  # noqa: BLE001
                    continue
                if events.drawn:
                    continue  # ドロー由来は R2 の対象ではない
                reason = r2_reason(backend, child.observation, state, 0)
                reasons[reason] += 1
                by_action[kinds][reason] += 1
                if row["oracle"]["oracle_win"]:
                    positive_reasons[reason] += 1
        except Exception:  # noqa: BLE001
            reasons["audit_error"] += 1
        finally:
            session.__exit__(None, None, None)
    total = sum(reasons.values())
    print(f"  {'reason':>22}{'count':>8}{'positive 由来':>14}")
    for name, count in reasons.most_common():
        print(f"  {name:>22}{count:>8}{positive_reasons[name]:>14}")
    print(f"  合計 {total}")
    for action, counter in by_action.items():
        print(f"    {action}: {dict(counter)}")


def measure(rows, deck, *, r2: bool) -> dict:
    stats = Counter()
    times = []
    nodes = []
    per_row = {}
    for row in rows:
        session, backend = make_backend(row, deck, r2=r2)
        started = time.perf_counter()
        try:
            if backend.is_win(backend.root()):
                continue
            result = phase2.search(
                backend,
                Budget(time_limit_ms=BUDGET_MS, max_nodes=20_000, max_depth=8,
                       max_chance_depth=1),
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            stats[result.proof.name] += 1
            times.append(elapsed)
            nodes.append(result.nodes)
            names = {s.name for s in result.stop_reasons}
            for name in ("UNSUPPORTED_EFFECT", "OUTCOMES_NOT_ENUMERABLE",
                         "DECK_REVEALED_AT_ROOT", "TIME_LIMIT", "DEPTH_LIMIT"):
                if name in names:
                    stats[f"stop_{name}"] += 1
            per_row[row["id"]] = (result.proof.name, sorted(names))
        except Exception:  # noqa: BLE001
            stats["ERROR"] += 1
        finally:
            session.__exit__(None, None, None)
    return {"stats": stats, "times": times, "nodes": nodes, "per_row": per_row}


def report_before_after(rows, deck) -> tuple[dict, dict]:
    off = measure(rows, deck, r2=False)
    on = measure(rows, deck, r2=True)
    print("\n## Before / After（Phase 2, budget 500ms, 同一 fixture）")
    keys = ("PROVEN_WIN", "PROVEN_NO_WIN", "UNKNOWN",
            "stop_UNSUPPORTED_EFFECT", "stop_OUTCOMES_NOT_ENUMERABLE",
            "stop_DECK_REVEALED_AT_ROOT", "stop_TIME_LIMIT", "stop_DEPTH_LIMIT",
            "ERROR")
    print(f"  {'metric':>32}{'R2 OFF':>10}{'R2 ON':>10}")
    for key in keys:
        print(f"  {key:>32}{off['stats'][key]:>10}{on['stats'][key]:>10}")
    for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        print(f"  {label:>32}{percentile(off['times'], q):>10.1f}"
              f"{percentile(on['times'], q):>10.1f}")
    print(f"  {'max':>32}{max(off['times'] or [0]):>10.1f}"
          f"{max(on['times'] or [0]):>10.1f}")
    print(f"  {'nodes 平均':>32}"
          f"{sum(off['nodes']) / max(len(off['nodes']), 1):>10.0f}"
          f"{sum(on['nodes']) / max(len(on['nodes']), 1):>10.0f}")
    return off, on


def report_transitions(rows, off, on) -> None:
    """指示 14: 最初に当たる stop reason がどう変わったかを追跡する。"""
    print("\n## R2 ON で結果が変わった fixture")
    changed = Counter()
    newly_won = []
    by_id = {row["id"]: row for row in rows}
    for fixture_id, (proof_off, stop_off) in off["per_row"].items():
        proof_on, stop_on = on["per_row"].get(fixture_id, (None, []))
        if proof_on is None or proof_on == proof_off:
            continue
        changed[f"{proof_off} -> {proof_on}"] += 1
        if proof_on == "PROVEN_WIN":
            newly_won.append((fixture_id, stop_off))
    for name, count in changed.most_common():
        print(f"  {name:>34}: {count}")
    print(f"  新たに PROVEN_WIN になった: {len(newly_won)} {[i for i, _ in newly_won[:8]]}")
    # positive candidate のうち何件が動いたか
    positive_moved = sum(
        1 for fixture_id, _ in newly_won if by_id[fixture_id]["oracle"]["oracle_win"]
    )
    print(f"  うち oracle が positive と判定している fixture: {positive_moved}")
    # UNKNOWN のまま残ったものの stop reason
    still = Counter()
    for fixture_id, (proof_on, stop_on) in on["per_row"].items():
        if proof_on == "UNKNOWN":
            for name in stop_on:
                still[name] += 1
    print(f"  R2 ON でも UNKNOWN のままの stop reason: {dict(still.most_common(6))}")


def main() -> None:
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row["turn"] >= 1]
    deck = read_deck_csv()
    print(f"valid fixture = {len(rows)}")
    audit_applicability(rows, deck)
    off, on = report_before_after(rows, deck)
    report_transitions(rows, off, on)


if __name__ == "__main__":
    main()
