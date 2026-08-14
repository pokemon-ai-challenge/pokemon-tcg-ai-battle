"""能力評価コーパスの回帰テスト（Step 1-18 指示 12）。

コーパスは「作ったときに正しかった」だけでは意味が無い。
毎回の回帰テストとして

    fixture -> oracle action sequence -> engine apply -> final result

を実行し、保存された ground-truth が今も成立することを確認する。

さらに **false `PROVEN_WIN` = 0** を negative / near miss で直接確認する。
near miss は「あと一歩でリーサル」なので、solver が最も誤りやすい。
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
from ptcg_ai.search.lethal import phase1
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.types import Proof
from tools import lethal_oracle

from main import read_deck_csv

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip("能力評価コーパスが未生成（tools/build_capability_corpus.py）")
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        pytest.skip("コーパスが空")
    return rows


@pytest.fixture(scope="module")
def deck():
    return read_deck_csv()


def _hidden(row):
    return HiddenState.from_stub(row["hidden_stub"])


def _run_phase1(row, deck, *, depth=8, ms=500.0, nodes=20_000):
    observation = to_observation_class(row["obs"])
    hidden = _hidden(row)
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        return phase1.search(
            backend, Budget(time_limit_ms=ms, max_nodes=nodes, max_depth=depth)
        )


def test_every_positive_fixture_still_replays_to_a_win(corpus):
    """保存した oracle 手順を実エンジンで再生し、今も勝ちへ到達することを確認する。"""
    positives = [row for row in corpus if row["oracle"]["expected_lethal"]]
    if not positives:
        pytest.skip("positive fixture が無い")
    failures = []
    for row in positives:
        observation = to_observation_class(row["obs"])
        ok = lethal_oracle.replay_verify(
            observation, _hidden(row), 0, row["oracle"]["oracle_sequence"]
        )
        if not ok:
            failures.append(row["id"])
    assert not failures, f"oracle 手順が勝ちへ到達しなくなった: {failures}"


def test_positive_fixtures_have_consistent_labels(corpus):
    """ラベルの内部整合。手順長 == minimum_depth、root action が手順の先頭。"""
    for row in corpus:
        oracle = row["oracle"]
        if not oracle["expected_lethal"]:
            assert oracle["minimum_depth"] is None
            continue
        sequence = oracle["oracle_sequence"]
        assert sequence, row["id"]
        assert len(sequence) == oracle["minimum_depth"], row["id"]
        assert list(sequence[0]) == list(oracle["expected_root_action"]), row["id"]


def test_no_false_proven_win_on_negative_and_near_miss(corpus, deck):
    """negative / near miss で `PROVEN_WIN` を返さない。

    これが破れたら **false PROVEN_WIN** であり、最優先の安全条件違反。
    """
    # オラクルは**ターン中の相手選択を経由する線を探索しない**ので、
    # `has_opponent_choice` の局面では `oracle_win=False` を
    # 「リーサルが存在しない」根拠に使えない。
    # 実測（Step 1-18）: cap0064 はこの理由で false PROVEN_WIN に見えていたが、
    # 実際には 1 手目でサイドを取って相手のバトル場を空にし、
    # ベンチ 5 体すべてを AND で詰める正しい PROVEN_WIN だった。
    targets = [row for row in corpus if row["category"] in ("negative", "near_miss")
               and not row["oracle"]["truncated"]
               and not row["oracle"].get("has_opponent_choice")]
    if not targets:
        pytest.skip("negative / near miss fixture が無い")
    violations = []
    for row in targets:
        result = _run_phase1(row, deck)
        if result.proof is Proof.PROVEN_WIN:
            violations.append((row["id"], row["category"]))
    assert not violations, f"リーサルが無い局面で PROVEN_WIN を返した: {violations}"


def test_phase1_never_contradicts_the_oracle_on_chance_free_positives(corpus, deck):
    """chance を含まない positive で `PROVEN_NO_WIN` を返さない。

    オラクルが再生検証済みの勝ち筋を持っているので、`PROVEN_NO_WIN` は誤り。
    `UNKNOWN` は正しい（予算内で証明しきれなかっただけ）。
    """
    targets = [row for row in corpus
               if row["oracle"]["expected_lethal"] and row["oracle"]["chance_free"]]
    if not targets:
        pytest.skip("chance-free positive が無い")
    violations = []
    for row in targets:
        result = _run_phase1(row, deck)
        if result.proof is Proof.PROVEN_NO_WIN:
            violations.append((row["id"], row["oracle"]["minimum_depth"]))
    assert not violations, f"勝ち筋があるのに PROVEN_NO_WIN を返した: {violations}"


def test_oracle_negative_scope_is_recorded(corpus):
    """負のラベルの**射程**が行ごとに判定できることを固定する。

    `has_opponent_choice=True` の行では、オラクルの `oracle_win=False` は
    「相手選択を経由しない勝ち筋が無い」だけを意味する。
    このフラグが失われると negative ラベルを誤用してしまうので、必ず存在させる。
    """
    for row in corpus:
        assert "has_opponent_choice" in row["oracle"], row["id"]
        assert "ground_truth" in row["oracle"], row["id"]
        assert row["oracle"]["ground_truth"] in (
            "GROUND_TRUTH_EXACT", "GROUND_TRUTH_CHANCE_FREE",
            "DETERMINIZATION_ONLY", "DEPTH_LIMITED",
        ), row["id"]


def test_chance_positives_are_not_used_as_exact_ground_truth(corpus):
    """指示 7: `chance_free=False` の positive を確定 ground truth 扱いしない。"""
    for row in corpus:
        oracle = row["oracle"]
        if oracle["oracle_win"] and not oracle["chance_free"]:
            assert oracle["outcome_complete"] is False, row["id"]
            assert oracle["ground_truth"] == "DETERMINIZATION_ONLY", row["id"]


def test_corpus_covers_more_than_one_category(corpus):
    """コーパスが 1 カテゴリへ偏っていないこと（評価基盤としての最低条件）。"""
    categories = {row["category"] for row in corpus}
    assert len(categories) >= 3, f"カテゴリが偏っている: {categories}"
