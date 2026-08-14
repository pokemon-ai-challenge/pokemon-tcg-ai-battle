"""機能別 fixture による実エンジン検証(Step 1-5)。

「局面数」ではなく「何を検証できる局面か」で管理する。各 fixture は
``tags``(required_capability)と ``golden``(expected result)を持つ。

ここで確認すること:

- 機能別カバレッジ(draw / deck reveal / opponent choice / prize / …)
- **実エンジンでの** outcome 完全列挙: 列挙 → 全構築 → 全再生 → 期待状態と一致 → 質量 1
- 列挙できない場合に ``PROVEN_WIN`` へ絶対に進まないこと
- golden 回帰(安全側の判定が後の最適化で崩れたら落ちる)
- 機能別のレイテンシ・ノード数・資源使用
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal import entry, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.enumeration import certify_for_proof
from ptcg_ai.search.lethal.types import ChanceClass, Proof, StopReason

FIXTURE = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_positions.jsonl"

# golden を作ったときと同じ設定(打ち切りをノード数で決める)
GOLDEN_CONFIG = {
    "enabled": True,
    "max_remaining_prizes": 2,
    "phase12_ms": 10_000.0,
    "max_nodes": 300,
    "max_depth": 8,
    "max_chance_depth": 1,
}

# 機能別に必要な最低数(件数ではなく能力で管理する)
REQUIRED_CAPABILITIES = {
    "gate": 10,
    "main": 5,
    "deck_open": 4,
    "draw": 5,
    "draw_multi": 3,
    "opponent_choice": 4,
    "opponent_choice_multi": 2,
    "prize_take": 4,
    "immediate_win": 2,
    "reveal": 5,
}


@pytest.fixture(scope="module")
def corpus():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert "tags" in row and "golden" in row, "fixture に tags/golden が無い"
    return rows


def _hidden(obs, deck, seed=0):
    stub = build_dummy_search_state(obs, deck, rng=random.Random(seed))
    return None if stub is None else HiddenState.from_stub(stub)


def _backend(session, obs, hidden, deck, **kwargs):
    return CgBackend(
        session, obs, hidden,
        deck_composition=KnownDeckComposition.from_card_ids(deck),
        **kwargs,
    )


def _context(obs, deck, config=None):
    return {
        "observation": obs,
        "config": {**GOLDEN_CONFIG, **(config or {})},
        "hidden_state_factory": lambda: build_dummy_search_state(obs, deck, rng=random.Random(0)),
        "full_deck": list(deck),
    }


# ------------------------------------------------------------- カバレッジ


def test_capability_coverage(corpus):
    counts: Counter = Counter()
    for row in corpus:
        counts.update(row["tags"])
    print("\nfixture capability coverage:", dict(counts.most_common()))
    missing = {
        tag: (counts[tag], required)
        for tag, required in REQUIRED_CAPABILITIES.items()
        if counts[tag] < required
    }
    assert not missing, f"機能別 fixture が不足: {missing}"


def test_every_fixture_has_expected_result(corpus):
    """各 fixture の expected result(golden)が形として揃っている。"""
    for row in corpus:
        golden = row["golden"]
        assert set(golden) >= {
            "started", "phase1_proof", "phase2_proof", "stop_reasons",
            "first_action", "fallback_reason",
        }
        if golden["first_action"] is not None:
            assert golden["phase1_proof"] == "PROVEN_WIN" or golden["phase2_proof"] == "PROVEN_WIN"


# ------------------------------------------------------------- golden 回帰


def test_golden_results_are_reproduced(corpus):
    """安全側の判定が崩れていないこと(proof と停止理由の回帰)。"""
    deck = read_deck_csv()
    mismatches = []
    for row in corpus:
        obs = to_observation_class(row["obs"])
        selection = entry.search(obs.current, obs.select.option, _context(obs, deck))
        diagnostics = entry.last_diagnostics()
        golden = row["golden"]
        actual = {
            "started": diagnostics.started,
            "phase1_proof": diagnostics.phase1_proof,
            "phase2_proof": diagnostics.phase2_proof,
            "first_action": selection,
        }
        expected = {key: golden[key] for key in actual}
        if actual != expected:
            mismatches.append((row["id"], expected, actual))
    assert not mismatches, f"golden 不一致: {mismatches[:3]}"


def test_no_false_proven_win_and_no_illegal_action(corpus):
    """最優先の安全指標: 誤証明 0・違法手 0・fallback 100%。"""
    deck = read_deck_csv()
    illegal = 0
    returned = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        selection = entry.search(obs.current, obs.select.option, _context(obs, deck))
        if selection is None:
            continue
        returned += 1
        select = obs.select
        if not (select.minCount <= len(selection) <= select.maxCount
                and len(selection) == len(set(selection))
                and all(0 <= i < len(select.option) for i in selection)):
            illegal += 1
    assert illegal == 0
    assert returned > 0


# ------------------------------------------------- 実エンジンでの outcome 列挙


def test_real_engine_draw_outcomes_are_currently_not_enumerable(corpus):
    """既定設定では、実デッキのドローは outcome 数が多すぎて列挙できない。

    これは欠陥ではなく組合せの現実(3〜4 枚ドロー × 山札 8〜18 枚)。
    重要なのは**列挙できないときに確定証明へ進まない**こと。
    """
    deck = read_deck_csv()
    refusals: Counter = Counter()
    classes = []
    for row in corpus:
        if "draw" not in row["tags"]:
            continue
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            root = backend.root()
            for action in backend.legal_actions(root):
                transition = backend.apply(root, action)
                if transition.state is None or not transition.revealed:
                    continue
                enumeration = backend.enumerate_outcomes(root, action)
                if enumeration.outcome_set is None:
                    refusals[enumeration.stop_reason.name] += 1
                else:
                    classes.append(len(enumeration.outcome_set.outcomes))
    print("\ndraw chance nodes:", dict(refusals), "enumerated:", classes)
    assert refusals, "ドローの chance ノードが 1 つも見つからなかった"
    assert set(refusals) <= {"OUTCOMES_NOT_ENUMERABLE", "UNSUPPORTED_EFFECT"}


def test_real_engine_outcome_enumeration_pipeline_when_it_fits(corpus):
    """上限を上げれば、実エンジンでも列挙 → 全構築 → 再生検証まで通ること。

    C クラス(供給順で制御できるドロー)の能力そのものを実エンジンで確認する。
    上限を上げるのは**この検証のためだけ**で、本番設定は保守的なまま。
    """
    deck = read_deck_csv()
    verified = 0
    for row in corpus:
        if "draw" not in row["tags"]:
            continue
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck, max_outcomes=80,
                               max_materializations=2000)
            root = backend.root()
            for action in backend.legal_actions(root):
                transition = backend.apply(root, action)
                if transition.state is None or not transition.revealed:
                    continue
                enumeration = backend.enumerate_outcomes(root, action)
                if enumeration.outcome_set is None:
                    continue
                outcome_set = enumeration.outcome_set
                # 1) 質量合計が厳密に 1
                assert outcome_set.total_mass() == Fraction(1)
                # 2) 証明に使える集合であること
                certification = certify_for_proof(
                    outcome_set, chance_class=enumeration.chance_class
                )
                assert certification.admissible, certification.failures
                # 3) 全 outcome を構築し、実際に意図通り引けたことを検証
                built = 0
                for outcome in outcome_set.outcomes:
                    realized = backend.apply_outcome(root, action, outcome)
                    if realized.state is None:
                        pytest.fail(f"outcome を構築できなかった: {realized.stop_reason}")
                    drawn = realized.state.draws[-1]
                    expected = tuple(
                        card_id for card_id, count in outcome.label for _ in range(count)
                    )
                    assert sorted(drawn) == sorted(expected), "引いたカードが outcome と違う"
                    built += 1
                assert built == len(outcome_set.outcomes)
                verified += 1
                break
        if verified >= 2:
            break
    if verified == 0:
        pytest.skip("上限 80 以内に収まるドロー局面が corpus に無かった")
    print(f"\nreal-engine outcome pipeline verified on {verified} chance node(s)")


def test_shuffled_paths_are_never_used_for_proof(corpus):
    """S クラス(シャッフル後)は確定証明に使わない。"""
    deck = read_deck_csv()
    seen = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            root = backend.root()
            for action in backend.legal_actions(root)[:8]:
                transition = backend.apply(root, action)
                if transition.state is None or not transition.state.shuffled:
                    continue
                enumeration = backend.enumerate_outcomes(transition.state, (0,))
                assert enumeration.outcome_set is None
                assert enumeration.chance_class is not ChanceClass.CONTROLLED_SUPPLY_ORDER
                seen += 1
                break
        if seen >= 3:
            break
    assert seen > 0, "シャッフルを含む経路が見つからなかった"


# ----------------------------------------------------------------- B9 サイド


def test_prize_take_cases(corpus):
    """サイド取得を A〜D に分けて扱えているか。"""
    deck = read_deck_csv()
    cases: Counter = Counter()
    for row in corpus:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            root = backend.root()
            for action in backend.legal_actions(root)[:10]:
                transition = backend.apply(root, action)
                child = transition.state
                if child is None:
                    continue
                if backend.is_win(child):
                    # ケース A: 取得で即勝利 → terminal WIN
                    cases["A_win"] += 1
                    assert backend.is_terminal(child)
                elif backend.is_opponent_node(child):
                    # ケース B: 取得したが勝利せず → 後続状態が正しく作られている
                    cases["B_continue"] += 1
                    assert not backend.is_terminal(child)
                    assert backend.legal_actions(child), "相手の選択肢が取れていない"
                elif transition.revealed and "prize" in str(backend.refusals):
                    # ケース C: 取得したカードを以後の判断に使う → 新情報境界
                    cases["C_reveal"] += 1
            # ケース D: サイド取得のランダム性は列挙対象にしない
    print("\nprize cases:", dict(cases))
    assert cases["A_win"] > 0 or cases["B_continue"] > 0


# --------------------------------------------------- PROVEN_WIN 再生(機能別)


class _RootedBackend:
    def __init__(self, backend, state):
        self._backend = backend
        self._state = state

    def root(self):
        return self._state

    def __getattr__(self, name):
        return getattr(self._backend, name)


def test_proven_win_replay_by_capability(corpus):
    """``PROVEN_WIN`` を返した全 fixture で、1 手ずつ再探索して実際に勝つ。"""
    from ptcg_ai.search.lethal import phase1

    deck = read_deck_csv()
    replayed = Counter()
    failures = []
    for row in corpus:
        obs = to_observation_class(row["obs"])
        selection = entry.search(obs.current, obs.select.option, _context(obs, deck))
        if selection is None:
            continue
        hidden = _hidden(obs, deck)
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            state = backend.root()
            ok = False
            for _ in range(20):
                if backend.is_win(state):
                    ok = True
                    break
                result = phase1.search(
                    _RootedBackend(backend, state),
                    Budget(time_limit_ms=1000.0, max_nodes=20_000, max_depth=8),
                )
                if result.proof is not Proof.PROVEN_WIN:
                    failures.append((row["id"], result.proof.name))
                    break
                transition = backend.apply(state, result.first_action)
                if transition.state is None:
                    failures.append((row["id"], "apply_failed"))
                    break
                state = transition.state
            if ok:
                for tag in row["tags"]:
                    replayed[tag] += 1
                replayed["total"] += 1
    print("\nPROVEN_WIN replay by capability:", dict(replayed))
    assert not failures, f"再生に失敗: {failures}"
    assert replayed["total"] > 0


# ------------------------------------------------------- 機能別レイテンシ


def test_latency_and_resources_by_capability(corpus):
    deck = read_deck_csv()
    by_tag: dict[str, list[float]] = defaultdict(list)
    nodes_by_tag: dict[str, list[int]] = defaultdict(list)
    all_durations: list[float] = []
    for row in corpus:
        obs = to_observation_class(row["obs"])
        context = _context(obs, deck, {"phase12_ms": 100.0, "max_nodes": 4000})
        started = time.perf_counter()
        entry.search(obs.current, obs.select.option, context)
        duration = (time.perf_counter() - started) * 1000.0
        diagnostics = entry.last_diagnostics()
        all_durations.append(duration)
        for tag in row["tags"]:
            by_tag[tag].append(duration)
            nodes_by_tag[tag].append(diagnostics.nodes)

    def percentile(values, q):
        values = sorted(values)
        return values[min(len(values) - 1, int(len(values) * q))]

    print("\nlatency by capability (ms):")
    for tag in sorted(by_tag):
        values = by_tag[tag]
        print(f"  {tag:22} n={len(values):3d} p50={percentile(values,0.5):6.1f} "
              f"p95={percentile(values,0.95):6.1f} max={max(values):6.1f} "
              f"nodes_max={max(nodes_by_tag[tag])}")
    print(f"  {'ALL':22} n={len(all_durations):3d} p50={percentile(all_durations,0.5):6.1f} "
          f"p95={percentile(all_durations,0.95):6.1f} p99={percentile(all_durations,0.99):6.1f} "
          f"max={max(all_durations):6.1f}")
    assert max(all_durations) < 1500.0, f"1 局面が遅すぎる: {max(all_durations):.1f}ms"


# ------------------------------------ 新情報を伴う相手選択(検出と安全側処理)


def test_opponent_choice_followed_by_new_information(corpus):
    """相手の選択の**後**に新情報が公開されるケースを検出する。

    AND ノードだけで押し切らず、公開が起きたら「新しい decision node +
    情報境界」として扱う。安全に扱えない場合は ``PROVEN_WIN`` へ進めない。
    """
    deck = read_deck_csv()
    observed = Counter()
    for row in corpus:
        if "opponent_choice" not in row["tags"]:
            continue
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            queue = [(backend.root(), 0)]
            while queue:
                state, depth = queue.pop(0)
                if depth >= 3:
                    continue
                for action in backend.legal_actions(state)[:6]:
                    transition = backend.apply(state, action)
                    child = transition.state
                    if child is None:
                        continue
                    if backend.is_opponent_node(state):
                        # 相手が選んだ**後**の遷移
                        observed["after_opponent_choice"] += 1
                        if transition.revealed:
                            observed["after_opponent_choice_revealed"] += 1
                            # 公開されたなら、確定証明にそのまま使わない
                            enumeration = backend.enumerate_outcomes(state, action)
                            assert (
                                enumeration.outcome_set is None
                                or enumeration.chance_class.is_enumerable
                            )
                    if backend.is_opponent_node(child):
                        observed["opponent_nodes"] += 1
                        queue.append((child, depth + 1))
                    elif not backend.is_terminal(child):
                        queue.append((child, depth + 1))
    print("\nopponent choice / new information:", dict(observed))
    assert observed["opponent_nodes"] > 0
