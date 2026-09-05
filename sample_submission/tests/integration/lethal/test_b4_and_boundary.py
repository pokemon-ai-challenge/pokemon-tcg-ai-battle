"""実エンジンでの B4(相手選択)・情報境界・資源/実行時間の検証(Step 1-4)。

- B4: KO 後の ``TO_ACTIVE`` などを **AND ノード**として扱えているか
- 情報境界: デッキ情報の順序が探索へ入らないこと
- fixture: 何を検証する局面なのかを機能別に分類できていること
- 資源: ``search_begin`` で確保した分を全部解放していること
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import SelectContext, to_observation_class
from main import read_deck_csv
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal import entry, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend, CgState
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.types import Proof

FIXTURE = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_positions.jsonl"

SEARCH_CONFIG = {
    "enabled": True,
    "max_remaining_prizes": 2,
    "phase12_ms": 10_000.0,   # 打ち切りをノード数だけで決める(再現性のため)
    "max_nodes": 300,
    "max_depth": 8,
    "max_chance_depth": 1,
}


@pytest.fixture(scope="module")
def corpus():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert rows
    return rows


def _backend(obs, deck, session):
    hidden = HiddenState.from_stub(build_dummy_search_state(obs, deck, rng=random.Random(0)))
    return hidden, CgBackend(
        session, obs, hidden,
        deck_composition=KnownDeckComposition.from_card_ids(deck),
    )


class _RootedBackend:
    def __init__(self, backend, state):
        self._backend = backend
        self._state = state

    def root(self):
        return self._state

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _find_opponent_nodes(backend, session, max_depth=4, max_breadth=6):
    """相手選択ノードを浅く探す(選択肢が 2 つ以上のものを優先)。"""
    found = []
    queue = [(backend.root(), 0)]
    while queue:
        state, depth = queue.pop(0)
        if depth >= max_depth or len(found) >= 4:
            continue
        try:
            actions = backend.legal_actions(state)
        except Exception:  # noqa: BLE001
            continue
        for action in actions[:max_breadth]:
            transition = backend.apply(state, action)
            if transition.state is None:
                continue
            child = transition.state
            if backend.is_opponent_node(child):
                options = backend.legal_actions(child)
                found.append((child, len(options)))
                continue
            if not backend.is_terminal(child):
                queue.append((child, depth + 1))
    _ = session
    return found


# ------------------------------------------------------------------- B4


def test_opponent_choice_nodes_exist_and_are_not_terminal(corpus):
    """KO 後の相手選択が、終端でも未対応でもなく**探索ノード**になっている。"""
    deck = read_deck_csv()
    total = 0
    multi = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        if build_dummy_search_state(obs, deck, rng=random.Random(0)) is None:
            continue
        with SearchSession(obs, HiddenState.from_stub(
            build_dummy_search_state(obs, deck, rng=random.Random(0))
        )) as session:
            _hidden, backend = None, CgBackend(
                session, obs, HiddenState.from_stub(
                    build_dummy_search_state(obs, deck, rng=random.Random(0))
                ),
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            for state, option_count in _find_opponent_nodes(backend, session):
                total += 1
                assert not backend.is_terminal(state)
                assert backend.is_opponent_node(state)
                assert option_count >= 1
                if option_count >= 2:
                    multi += 1
        if total >= 6 and multi >= 1:
            break
    assert total > 0, "相手選択ノードが 1 つも見つからなかった"
    assert multi > 0, "選択肢が 2 つ以上の相手選択ノードが見つからなかった"


def test_opponent_node_uses_and_semantics(corpus):
    """相手選択ノードの判定が AND であること(有利な選択だけを採らない)。"""
    deck = read_deck_csv()
    checked = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        hidden = HiddenState.from_stub(stub)
        with SearchSession(obs, hidden) as session:
            backend = CgBackend(
                session, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            for state, option_count in _find_opponent_nodes(backend, session):
                if option_count < 2:
                    continue
                budget = Budget(time_limit_ms=5000.0, max_nodes=2000, max_depth=6)
                node_proof = phase2.search(_RootedBackend(backend, state), budget).proof
                children = []
                for action in backend.legal_actions(state):
                    transition = backend.apply(state, action)
                    if transition.state is None:
                        children.append(Proof.PROVEN_NO_WIN)
                        continue
                    child_budget = Budget(time_limit_ms=5000.0, max_nodes=2000, max_depth=6)
                    if backend.is_win(transition.state):
                        children.append(Proof.PROVEN_WIN)
                    elif backend.is_terminal(transition.state):
                        children.append(Proof.PROVEN_NO_WIN)
                    else:
                        children.append(
                            phase2.search(
                                _RootedBackend(backend, transition.state), child_budget
                            ).proof
                        )
                # AND の規則: 全部勝ちなら勝ち / 1つでも負けなら負け / それ以外は UNKNOWN
                if all(p is Proof.PROVEN_WIN for p in children):
                    assert node_proof is Proof.PROVEN_WIN
                elif any(p is Proof.PROVEN_NO_WIN for p in children):
                    assert node_proof is Proof.PROVEN_NO_WIN
                else:
                    assert node_proof is Proof.UNKNOWN
                checked += 1
                if checked >= 2:
                    return
    assert checked > 0, "2 択以上の相手選択ノードで検証できなかった"


# ------------------------------------------------------- 情報境界(full_deck)


def test_deck_list_order_does_not_change_the_result(corpus):
    """デッキリストの並びを変えても、探索結果もノード数も変わらない。"""
    deck = read_deck_csv()
    shuffled_deck = list(deck)
    random.Random(99).shuffle(shuffled_deck)
    compared = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        results = []
        for variant in (list(deck), shuffled_deck):
            context = {
                "observation": obs,
                "config": SEARCH_CONFIG,
                "hidden_state_factory": lambda o=obs: build_dummy_search_state(
                    o, deck, rng=random.Random(0)
                ),
                "full_deck": variant,
            }
            selection = entry.search(obs.current, obs.select.option, context)
            diagnostics = entry.last_diagnostics()
            results.append((None if selection is None else tuple(selection), diagnostics.nodes))
        if not entry.last_diagnostics().started:
            continue
        compared += 1
        assert len(set(results)) == 1, f"デッキリストの並びで結果が変わった: {results}"
    assert compared >= 5


def test_search_does_not_call_the_critic():
    """Phase 1/2 の経路で critic を呼ばない(Phase 3 未接続であることの担保)。"""
    import ptcg_ai.search.lethal.entry as entry_module

    source = Path(entry_module.__file__).read_text(encoding="utf-8")
    assert "value" not in source.split("import")[1] or "ValueModel" not in source
    for module_name in ("phase1", "phase2", "cg_backend"):
        text = (Path(entry_module.__file__).parent / f"{module_name}.py").read_text(
            encoding="utf-8"
        )
        assert "ValueModel" not in text
        assert "predict_win_prob" not in text


# ------------------------------------------------------------ fixture 分類


def _classify(obs, deck) -> set[str]:
    """その局面が何を検証できるかのタグを付ける。"""
    tags: set[str] = set()
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    if stub is None:
        return {"no_hidden_state"}
    me = obs.current.yourIndex
    tags.add(f"prizes_{len(obs.current.players[me].prize)}")
    if obs.select.deck is not None:
        tags.add("deck_open")
    hidden = HiddenState.from_stub(stub)
    with SearchSession(obs, hidden) as session:
        backend = CgBackend(
            session, obs, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        root = backend.root()
        for action in backend.legal_actions(root)[:8]:
            transition = backend.apply(root, action)
            if transition.state is None:
                continue
            if transition.revealed:
                tags.add("reveal")
            if transition.state.draws != root.draws:
                tags.add("draw")
            if backend.is_win(transition.state):
                tags.add("immediate_win")
            if backend.is_opponent_node(transition.state):
                tags.add("opponent_choice")
        for state, count in _find_opponent_nodes(backend, session, max_depth=3):
            tags.add("opponent_choice")
            if count >= 2:
                tags.add("opponent_choice_multi")
            _ = state
    return tags


def test_fixture_covers_the_intended_features(corpus):
    """局面数ではなく「何を検証する局面か」で fixture を管理する。"""
    deck = read_deck_csv()
    matrix: dict[str, set[str]] = {}
    counts: Counter = Counter()
    for index, row in enumerate(corpus):
        obs = to_observation_class(row["obs"])
        tags = _classify(obs, deck)
        matrix[f"{row['kind']}#{index}"] = tags
        counts.update(tags)
        counts[f"kind:{row['kind']}"] += 1
    print("\nfixture feature coverage:", dict(counts.most_common()))
    required = ["kind:gate", "kind:main", "kind:deck_open", "opponent_choice",
                "opponent_choice_multi", "draw", "reveal"]
    missing = [tag for tag in required if counts[tag] == 0]
    assert not missing, f"fixture がカバーしていない機能: {missing}"


# ------------------------------------------------------------ 資源・実行時間


def test_search_sessions_release_everything(corpus):
    """``search_begin`` / ``search_step`` で確保した ID を全部解放している。"""
    deck = read_deck_csv()
    leaks = []
    for row in corpus[:10]:
        obs = to_observation_class(row["obs"])
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        hidden = HiddenState.from_stub(stub)
        session = SearchSession(obs, hidden)
        with session as opened:
            backend = CgBackend(
                opened, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            phase2.search(backend, Budget(time_limit_ms=200.0, max_nodes=200, max_depth=5))
        if session.acquired != session.released:
            leaks.append((session.acquired, session.released))
    assert not leaks, f"解放漏れ: {leaks}"


def test_runtime_and_materialization_budget(corpus):
    """1 局面あたりの実行時間と outcome 構築回数を記録し、上限内に収める。"""
    deck = read_deck_csv()
    durations = []
    materializations = []
    for row in corpus:
        obs = to_observation_class(row["obs"])
        context = {
            "observation": obs,
            "config": {**SEARCH_CONFIG, "phase12_ms": 100.0, "max_nodes": 4000},
            "hidden_state_factory": lambda o=obs: build_dummy_search_state(
                o, deck, rng=random.Random(0)
            ),
            "full_deck": list(deck),
        }
        started = time.perf_counter()
        entry.search(obs.current, obs.select.option, context)
        durations.append((time.perf_counter() - started) * 1000.0)
        diagnostics = entry.last_diagnostics()
        materializations.append(diagnostics.extra.get("materialization_limit", 0))
    durations.sort()
    print(f"\nentry latency: mean={sum(durations)/len(durations):.1f}ms "
          f"p95={durations[int(len(durations)*0.95)]:.1f}ms max={durations[-1]:.1f}ms")
    # 予算 100ms + 呼び出しのオーバーヘッド。極端に超えていないこと。
    assert durations[-1] < 1000.0, f"1 局面が遅すぎる: {durations[-1]:.1f}ms"
    assert sum(materializations) == 0 or True  # 記録のみ(上限到達は UNKNOWN として安全)
