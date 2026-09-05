"""`PROVEN_NO_WIN` の**共通契約**（Step 1-20 指示 7/8/12）。

Phase 1 と Phase 2 で「NO_WIN とは何か」が別々になってはいけない。共通の定義:

    PROVEN_NO_WIN
      = 必要な探索木を完全に展開し、勝利へ到達する枝が存在しないことを確認した

以下が 1 つでも混ざったら `UNKNOWN` であり、`PROVEN_NO_WIN` にしてはならない:

    time / depth / node / unsupported / incomplete /
    construction failure / unverified pruning / unexpanded reveal branch

Step 1-19 の P0 はこの契約が Phase 1 で破れていた事例（未展開の reveal 枝を
`PROVEN_NO_WIN` にしていた）。同じ誤りが再発しないよう、両 phase で固定する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.enumeration import (
    Outcome, OutcomeSet, certify_for_proof, draw_outcomes,
)
from ptcg_ai.search.lethal.types import ChanceClass, Proof

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# 「調べ尽くしていない」ことを表す停止理由。これらと PROVEN_NO_WIN は両立しない。
INCOMPLETE_REASONS = frozenset({
    "TIME_LIMIT",
    "NODE_LIMIT",
    "DEPTH_LIMIT",
    "CHANCE_DEPTH_LIMIT",
    "UNSUPPORTED_EFFECT",
    "INCOMPLETE_ACTION_SET",
    "OUTCOMES_NOT_ENUMERABLE",
    "SHUFFLE_ENCOUNTERED",
})


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip("能力評価コーパスが未生成")
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if row["turn"] >= 1]


@pytest.fixture(scope="module")
def deck():
    return read_deck_csv()


def _search(row, deck, module, *, ms=500.0, depth=8, chance=1, nodes=20_000):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        if backend.is_win(backend.root()):
            return None
        return module.search(
            backend,
            Budget(time_limit_ms=ms, max_nodes=nodes, max_depth=depth,
                   max_chance_depth=chance),
        )


@pytest.mark.parametrize("module", [phase1, phase2], ids=["phase1", "phase2"])
@pytest.mark.parametrize("budget_ms", [5.0, 50.0, 500.0])
def test_no_win_never_coexists_with_an_incomplete_reason(corpus, deck, module, budget_ms):
    """`PROVEN_NO_WIN` と「調べ尽くしていない」停止理由が同時に付かない。

    予算を変えて走らせることで、打ち切りが起きる状況を意図的に作る。
    """
    violations = []
    for row in corpus:
        result = _search(row, deck, module, ms=budget_ms)
        if result is None or result.proof is not Proof.PROVEN_NO_WIN:
            continue
        names = {s.name for s in result.stop_reasons}
        overlap = names & INCOMPLETE_REASONS
        if overlap:
            violations.append((row["id"], sorted(overlap)))
    assert not violations, (
        f"{module.__name__}: 未完了のまま PROVEN_NO_WIN を主張した: {violations[:5]}"
    )


@pytest.mark.parametrize("module", [phase1, phase2], ids=["phase1", "phase2"])
def test_tiny_budget_never_yields_no_win_by_timeout(corpus, deck, module):
    """予算をほぼゼロにしたとき、打ち切り由来の `PROVEN_NO_WIN` が出ない。"""
    for row in corpus[:60]:
        result = _search(row, deck, module, ms=0.5, nodes=1)
        if result is None:
            continue
        if result.proof is Proof.PROVEN_NO_WIN:
            names = {s.name for s in result.stop_reasons}
            assert not (names & INCOMPLETE_REASONS), row["id"]


def test_partial_outcome_collection_is_never_admissible():
    """未処理 outcome が残る集合・質量が 1 未満の集合を証明に使わない（指示 12）。"""
    full = draw_outcomes({1: 4, 2: 4, 3: 4}, 2)
    assert full is not None
    assert certify_for_proof(full, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER).admissible, "完全な集合が却下されている"

    # 一部だけ取り出した集合（= 未処理が残っている）は証明に使えない
    partial = OutcomeSet(
        outcomes=full.outcomes[: max(1, len(full.outcomes) - 1)],
        source=full.source,
        coverage_certified=full.coverage_certified,
    )
    assert not certify_for_proof(partial, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER).admissible, (
        "質量が 1 に満たない outcome 集合が証明に使えると判定された"
    )

    # 空集合も当然使えない
    empty = OutcomeSet(outcomes=(), source=full.source, coverage_certified=False)
    assert not certify_for_proof(empty, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER).admissible


def test_enumeration_timeout_returns_none_not_a_partial_set():
    """列挙の打ち切りは `None`。部分集合を返してはならない（指示 12）。"""
    belief = {card_id: 4 for card_id in range(1, 16)}
    assert draw_outcomes(belief, 3, should_stop=lambda: True) is None
    assert draw_outcomes(belief, 3, should_stop=lambda: False) is not None


def test_certification_rejects_a_mass_that_does_not_sum_to_one():
    """質量の合計が 1 でない集合を却下する。"""
    from fractions import Fraction
    broken = OutcomeSet(
        outcomes=(Outcome(label=((1, 1),), mass=Fraction(1, 3)),),
        source="test",
        coverage_certified=True,
    )
    assert not certify_for_proof(broken, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER).admissible
