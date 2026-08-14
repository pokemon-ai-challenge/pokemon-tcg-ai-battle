"""Step 1-19 P0 の回帰テスト。

1. `PROVEN_NO_WIN` の健全性（cap0197 / cap0199）
2. Phase 2 の予算制御（列挙・具体化を含めた合計で予算を守る）

どちらも「今後の最適化で元へ戻らないこと」を守るためのテスト。
"""

from __future__ import annotations

import json
import sys
import time
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
from ptcg_ai.search.lethal.enumeration import draw_outcomes
from ptcg_ai.search.lethal.types import Proof
from tools import lethal_oracle

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# P0-1 の最小再現。どちらも 1 手目でデッキを公開し、その先に検証済みの
# 4 手リーサルがある。旧実装は公開枝を `PROVEN_NO_WIN` として捨てていた。
P0_FIXTURES = ("cap0197", "cap0199")


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip("能力評価コーパスが未生成")
    return {json.loads(line)["id"]: json.loads(line)
            for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}


@pytest.fixture(scope="module")
def deck():
    return read_deck_csv()


def _search(row, deck, module, *, ms=500.0, depth=8, chance=1):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    started = time.perf_counter()
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        if backend.is_win(backend.root()):
            return None, 0.0
        result = module.search(
            backend,
            Budget(time_limit_ms=ms, max_nodes=20_000, max_depth=depth,
                   max_chance_depth=chance),
        )
    return result, (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------- P0-1

@pytest.mark.parametrize("fixture_id", P0_FIXTURES)
def test_proven_no_win_is_not_claimed_when_a_win_exists(corpus, deck, fixture_id):
    """勝ち筋があるのに `PROVEN_NO_WIN` を返さない。

    Phase 1 は公開枝を展開しない設計なので `PROVEN_WIN` は要求しない。
    要求するのは **`PROVEN_NO_WIN` ではないこと**。
    """
    row = corpus[fixture_id]
    observation = to_observation_class(row["obs"])
    # 前提: オラクルの勝ち筋が今も実エンジンで勝ちへ到達する
    assert lethal_oracle.replay_verify(
        observation, HiddenState.from_stub(row["hidden_stub"]), 0,
        row["oracle"]["oracle_sequence"],
    ), f"{fixture_id}: 前提が崩れている"
    result, _ = _search(row, deck, phase1)
    assert result.proof is not Proof.PROVEN_NO_WIN, (
        f"{fixture_id}: 検証済みの勝ち筋があるのに PROVEN_NO_WIN "
        f"(stop={[s.name for s in result.stop_reasons]})"
    )


def test_no_false_proven_no_win_on_the_whole_exact_corpus(corpus, deck):
    """exact な positive すべてで `PROVEN_NO_WIN` を出さない。"""
    targets = [row for row in corpus.values()
               if row["oracle"]["ground_truth"] == "GROUND_TRUTH_EXACT"]
    assert targets, "exact positive が無い"
    violations = [row["id"] for row in targets
                  if _search(row, deck, phase1)[0] is not None
                  and _search(row, deck, phase1)[0].proof is Proof.PROVEN_NO_WIN]
    assert not violations, f"false PROVEN_NO_WIN: {violations}"


def test_revealing_branches_yield_unknown_not_no_win(corpus, deck):
    """公開を伴う枝しか無い局面で `PROVEN_NO_WIN` を返さない。

    P0-1 の原因はこの分岐だった。`OUTCOMES_NOT_ENUMERABLE` が付いた結果は
    「展開しなかった枝がある」ことを意味するので `PROVEN_NO_WIN` と両立しない。
    """
    checked = 0
    for row in corpus.values():
        result, _ = _search(row, deck, phase1)
        if result is None:
            continue
        names = {s.name for s in result.stop_reasons}
        if "OUTCOMES_NOT_ENUMERABLE" in names:
            checked += 1
            assert result.proof is not Proof.PROVEN_NO_WIN, row["id"]
    assert checked > 0, "公開枝を含む fixture が無く、検出力が無い"


# --------------------------------------------------------------- P0-2

@pytest.mark.parametrize("budget_ms", [50.0, 100.0, 200.0, 500.0])
def test_phase2_respects_the_total_budget(corpus, deck, budget_ms):
    """予算は「列挙 + 構築 + 探索 + 検証 + 後始末」の合計で守る。

    修正前は 500ms 指定で実測 9,186ms（18 倍）だった。
    列挙の内側に予算チェックが無く、`draw_outcomes` が走り切っていたため。
    """
    tolerance = budget_ms * 0.6 + 60.0
    worst = 0.0
    worst_id = None
    for row in corpus.values():
        result, elapsed = _search(row, deck, phase2, ms=budget_ms)
        if result is None:
            continue
        if elapsed > worst:
            worst, worst_id = elapsed, row["id"]
    assert worst <= budget_ms + tolerance, (
        f"予算 {budget_ms}ms に対し実測 {worst:.1f}ms ({worst_id})"
    )


def test_enumeration_aborts_and_returns_none_when_stopped():
    """列挙は打ち切り時に**部分集合を返さない**。

    中途半端な outcome 集合を返すと「全 outcome を覆った」という
    誤った証明の根拠になりうる。
    """
    belief = {card_id: 4 for card_id in range(1, 16)}
    assert draw_outcomes(belief, 3, should_stop=lambda: False) is not None
    assert draw_outcomes(belief, 3, should_stop=lambda: True) is None


def test_budget_exhaustion_yields_unknown_not_no_win(corpus, deck):
    """予算切れは `UNKNOWN` であって `PROVEN_NO_WIN` ではない。"""
    for row in corpus.values():
        result, _ = _search(row, deck, phase2, ms=1.0)
        if result is None:
            continue
        if result.proof is Proof.PROVEN_NO_WIN:
            # 1ms でも即座に詰み切れる終端局面はありうる。
            # その場合でも打ち切り理由が付いていてはいけない。
            names = {s.name for s in result.stop_reasons}
            assert "TIME_LIMIT" not in names and "NODE_LIMIT" not in names, (
                f"{row['id']}: 予算切れなのに PROVEN_NO_WIN"
            )
