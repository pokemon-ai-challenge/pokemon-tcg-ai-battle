"""Step 1-20: P0 修正後の Phase 1 / Phase 2 を**完全に同一条件**で比較する。

固定するもの（指示 2）: corpus / fixture state / deck belief / config /
node budget / wall-clock budget / max_depth / max_chance_depth / backend / process。
唯一変えるのは探索モジュールだけ。

出力:
  * exact-positive subset と all-valid subset の両方
  * Case A/B/C/D 分類（指示 4）と Case B の全件明細（指示 5/6）
  * outcome 数バケット別の runtime / materialization（指示 13）
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

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# 両 phase で完全に同じ値を使う。Phase 1 は chance depth を使わないが、
# 「条件が違うから差が出た」と言われないように明示的に揃える。
MAX_NODES = 20_000
MAX_DEPTH = 8
MAX_CHANCE_DEPTH = 1
BUDGETS = (100.0, 200.0, 500.0)


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


class _Counting:
    """backend を包んで outcome 数と具体化回数を数えるだけ（挙動は変えない）。"""

    def __init__(self, inner):
        self._inner = inner
        self.outcomes = 0
        self.materializations = 0
        self.chance_nodes = 0

    def enumerate_outcomes(self, state, action):
        enumeration = self._inner.enumerate_outcomes(state, action)
        self.chance_nodes += 1
        if enumeration.outcome_set is not None:
            self.outcomes += len(enumeration.outcome_set.outcomes)
        return enumeration

    def apply_outcome(self, state, action, outcome):
        self.materializations += 1
        return self._inner.apply_outcome(state, action, outcome)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run(row, deck, module, budget_ms):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    started = time.perf_counter()
    with SearchSession(observation, hidden) as session:
        inner = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        if inner.is_win(inner.root()):
            return None
        backend = _Counting(inner)
        result = module.search(
            backend,
            Budget(time_limit_ms=budget_ms, max_nodes=MAX_NODES,
                   max_depth=MAX_DEPTH, max_chance_depth=MAX_CHANCE_DEPTH),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "proof": result.proof.name,
            "stop": sorted(s.name for s in result.stop_reasons),
            "nodes": result.nodes,
            "ms": elapsed,
            "outcomes": backend.outcomes,
            "materializations": backend.materializations,
            "chance_nodes": backend.chance_nodes,
            "search_begin": getattr(session, "acquired", 0),
        }


def classify(row, p1, p2) -> str:
    win1 = p1["proof"] == "PROVEN_WIN"
    win2 = p2["proof"] == "PROVEN_WIN"
    if win1 and win2:
        return "A_both_win"
    if not win1 and win2:
        return "B_phase2_only"
    if win1 and not win2:
        return "C_phase1_only"
    return "D_neither"


def added_capability_type(row, p1, p2) -> str:
    """指示 6: Phase 2 の追加能力がどこから来たのかを分類する。"""
    oracle = row["oracle"]
    stop1 = set(p1["stop"])
    if p2["chance_nodes"] > 0 and p2["outcomes"] > 1:
        return "Type3_multi_outcome"
    if p2["chance_nodes"] > 0:
        return "Type2_random_outcome"
    if oracle.get("has_opponent_choice"):
        return "Type4_and_node"
    if oracle.get("has_root_deck_reveal") or oracle.get("has_midturn_deck_reveal"):
        return "Type5_deck_reveal"
    if "DEPTH_LIMIT" in stop1 and "OUTCOMES_NOT_ENUMERABLE" not in stop1:
        return "Type1_phase1_depth"
    return "Type6_other"


def report_subset(name, rows, deck, budget_ms):
    print(f"\n### {name}  (n={len(rows)}, budget={budget_ms:.0f}ms, "
          f"nodes={MAX_NODES}, depth={MAX_DEPTH}, chance={MAX_CHANCE_DEPTH})")
    cases = Counter()
    tally = {"Phase 1": Counter(), "Phase 2": Counter()}
    times = {"Phase 1": [], "Phase 2": []}
    case_b = []
    case_c = []
    for row in rows:
        p1 = run(row, deck, phase1, budget_ms)
        p2 = run(row, deck, phase2, budget_ms)
        if p1 is None or p2 is None:
            continue
        tally["Phase 1"][p1["proof"]] += 1
        tally["Phase 2"][p2["proof"]] += 1
        times["Phase 1"].append(p1["ms"])
        times["Phase 2"].append(p2["ms"])
        case = classify(row, p1, p2)
        cases[case] += 1
        if case == "B_phase2_only":
            case_b.append((row, p1, p2))
        elif case == "C_phase1_only":
            case_c.append((row, p1, p2))
    for label in ("Phase 1", "Phase 2"):
        counter = tally[label]
        values = times[label]
        print(f"  {label}: WIN={counter['PROVEN_WIN']:>3} NO_WIN={counter['PROVEN_NO_WIN']:>3} "
              f"UNKNOWN={counter['UNKNOWN']:>3}  "
              f"p50={percentile(values, 0.5):6.1f} p95={percentile(values, 0.95):7.1f} "
              f"p99={percentile(values, 0.99):7.1f} max={max(values or [0]):7.1f}")
    print(f"  Case: A(both WIN)={cases['A_both_win']} "
          f"**B(Phase2 only)={cases['B_phase2_only']}** "
          f"C(Phase1 only)={cases['C_phase1_only']} D(neither)={cases['D_neither']}")
    return case_b, case_c


def report_case_b(case_b):
    print(f"\n### Case B の明細（Phase 2 だけが確定できた {len(case_b)} 件）")
    types = Counter()
    print(f"  {'id':>9}{'md':>4}{'chance':>7}{'B4':>4}{'reveal':>7}{'outc':>6}"
          f"{'mat':>5}  {'P1 stop':<34} type")
    for row, p1, p2 in case_b:
        oracle = row["oracle"]
        kind = added_capability_type(row, p1, p2)
        types[kind] += 1
        reveal = oracle.get("has_root_deck_reveal") or oracle.get("has_midturn_deck_reveal")
        print(f"  {row['id']:>9}{str(oracle['minimum_depth']):>4}"
              f"{'yes' if not oracle['chance_free'] else '-':>7}"
              f"{'yes' if oracle.get('has_opponent_choice') else '-':>4}"
              f"{'yes' if reveal else '-':>7}{p2['outcomes']:>6}{p2['materializations']:>5}"
              f"  {','.join(p1['stop'])[:32]:<34} {kind}")
    print("  分類:", dict(types.most_common()))


def report_buckets(rows, deck, budget_ms):
    print(f"\n### Phase 2 の outcome 数別 runtime（budget={budget_ms:.0f}ms）")
    buckets = defaultdict(list)
    for row in rows:
        record = run(row, deck, phase2, budget_ms)
        if record is None:
            continue
        count = record["outcomes"]
        for limit, name in ((0, "0 (chance なし)"), (16, "1-16"), (32, "17-32"),
                            (64, "33-64"), (128, "65-128"), (256, "129-256"),
                            (512, "257-512")):
            if count <= limit:
                buckets[name].append(record)
                break
        else:
            buckets[">512"].append(record)
    print(f"  {'bucket':>16}{'n':>5}{'chance':>8}{'mat':>7}{'sbegin':>8}"
          f"{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  proof")
    order = ["0 (chance なし)", "1-16", "17-32", "33-64", "65-128", "129-256",
             "257-512", ">512"]
    for name in order:
        group = buckets.get(name)
        if not group:
            continue
        values = [g["ms"] for g in group]
        proofs = Counter(g["proof"] for g in group)
        print(f"  {name:>16}{len(group):>5}"
              f"{sum(g['chance_nodes'] for g in group):>8}"
              f"{sum(g['materializations'] for g in group):>7}"
              f"{sum(g['search_begin'] for g in group):>8}"
              f"{percentile(values, 0.5):>8.1f}{percentile(values, 0.95):>8.1f}"
              f"{percentile(values, 0.99):>8.1f}{max(values):>8.1f}  "
              f"W{proofs['PROVEN_WIN']}/N{proofs['PROVEN_NO_WIN']}/U{proofs['UNKNOWN']}")


def main() -> None:
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    deck = read_deck_csv()
    valid = [r for r in rows if r["turn"] >= 1]
    exact_positive = [r for r in valid
                      if r["oracle"]["ground_truth"] == "GROUND_TRUTH_EXACT"]
    chance_positive = [r for r in valid
                       if r["oracle"]["ground_truth"] == "DETERMINIZATION_ONLY"]
    print(f"corpus: valid={len(valid)} exact_positive={len(exact_positive)} "
          f"chance_positive={len(chance_positive)}")

    for budget_ms in BUDGETS:
        print(f"\n{'=' * 70}\n## budget {budget_ms:.0f}ms")
        # 第一評価: exact positive（oracle replay 済み・chance-free）だけ（指示 10）
        report_subset("exact-positive subset（主評価）", exact_positive, deck, budget_ms)
        # 第二段階: chance を含む positive
        report_subset("chance-positive subset", chance_positive, deck, budget_ms)
        case_b, _case_c = report_subset("all-valid subset", valid, deck, budget_ms)
        if budget_ms == 500.0:
            report_case_b(case_b)
            report_buckets(valid, deck, budget_ms)


if __name__ == "__main__":
    main()
