"""Phase 2 の capability golden（Step 1-33 指示 6）。

Phase 1 では確定できず Phase 2 だけが確定できた 9 件。
**9/9 replay green を必須回帰**とする。今後の最適化でどれか 1 つでも
落ちたら、それは能力の退行である。

replay は本番と同じ規律（first action だけ実行 → 再探索）で行う。
`limit` はエンジンの**選択回数**であって手数ではない（Step 1-21）。
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
from ptcg_ai.search.lethal.types import Proof

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# fixture -> 追加能力の分類（Step 1-21 で確定）
PHASE2_ONLY = {
    "cap0056": "TypeC_multi_outcome",
    "cap0057": "TypeC_multi_outcome",
    "cap0136": "TypeC_multi_outcome",
    "cap0179": "TypeC_multi_outcome",
    "cap0058": "TypeB_random_outcome",
    "cap0162": "TypeB_random_outcome",
    "cap0176": "TypeD_deck_reveal",
    "cap0008": "TypeE_phase1_efficiency",
    "cap0139": "TypeE_phase1_efficiency",
}


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip("能力評価コーパスが未生成")
    return {json.loads(line)["id"]: json.loads(line)
            for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}


@pytest.fixture(scope="module")
def deck():
    return read_deck_csv()


def _budget():
    """回帰テストは**能力**を見るので、予算は余裕を持たせる。

    500ms 固定にすると、スイート全体を走らせたときの負荷で
    `TIME_LIMIT` に当たり、能力の退行と区別できない失敗が出る
    （実測: 単体では 19/19 green、全体実行では 6 件が揺れた）。
    latency は `test_p0_soundness_and_budget.py` の予算遵守テストで別途固定する。
    """
    return Budget(time_limit_ms=5_000.0, max_nodes=200_000, max_depth=8,
                  max_chance_depth=1)


class _Rooted:
    def __init__(self, inner, state):
        self._inner, self._state = inner, state

    def root(self):
        return self._state

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _classification_budget():
    """9 件を分類したときの条件（500ms）。Phase 1/2 の役割分担はこの予算で定義した。

    TypeE の cap0008 / cap0139 は「500ms 予算下で Phase 1 が落とす」ケースなので、
    予算を緩めると Phase 1 も解ける（実測: 5s なら解ける）。
    分類の再現には**分類時と同じ予算**を使う必要がある。
    """
    return Budget(time_limit_ms=500.0, max_nodes=20_000, max_depth=8,
                  max_chance_depth=1)


def _search(row, deck, module, budget=None):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        if backend.is_win(backend.root()):
            return None
        return module.search(backend, budget or _budget())


@pytest.mark.parametrize("fixture_id", sorted(PHASE2_ONLY))
def test_phase2_still_proves_the_win(corpus, deck, fixture_id):
    result = _search(corpus[fixture_id], deck, phase2)
    assert result is not None and result.proof is Proof.PROVEN_WIN, (
        f"{fixture_id} ({PHASE2_ONLY[fixture_id]}): Phase 2 の確定能力が退行した"
    )


@pytest.mark.parametrize("fixture_id", sorted(PHASE2_ONLY))
def test_phase2_proof_replays_to_an_actual_win(corpus, deck, fixture_id):
    """first action だけ実行 → 再探索、を繰り返して実際に勝ち切れる。"""
    row = corpus[fixture_id]
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        state = backend.root()
        for _ in range(60):
            if backend.is_win(state):
                return
            assert not backend.is_terminal(state), fixture_id
            result = phase2.search(_Rooted(backend, state), _budget())
            assert result.proof is Proof.PROVEN_WIN, (
                f"{fixture_id}: 再探索で証明を失った ({result.proof.name})"
            )
            action = tuple(result.first_action)
            assert action in {tuple(a) for a in backend.legal_actions(state)}, fixture_id
            transition = backend.apply(state, action)
            assert transition.state is not None, fixture_id
            state = transition.state
    pytest.fail(f"{fixture_id}: 60 選択以内に勝ちへ到達しなかった")


def test_phase1_still_cannot_prove_these(corpus, deck):
    """この 9 件が Phase 2 固有の能力であることを保つ。

    Phase 1 が解けるようになったら、それは改善なので**このテストを更新**する
    （落ちたら退行ではなく、golden の見直しが必要というシグナル）。
    """
    solved_by_phase1 = []
    for fixture_id in PHASE2_ONLY:
        result = _search(corpus[fixture_id], deck, phase1, _classification_budget())
        if result is not None and result.proof is Proof.PROVEN_WIN:
            solved_by_phase1.append(fixture_id)
    assert not solved_by_phase1, (
        f"Phase 1 が解けるようになった: {solved_by_phase1}。"
        "退行ではなく改善なので、この golden を見直すこと"
    )
