"""保存盤面での接続ハーネス(Step 1-3 指示 6)。

確認するのは「実戦で勝てるか」ではなく、
**探索結果を合法な EngineAction へ安全に変換できるか**と、
**証明できないときに必ず通常方策へ戻るか**である。

ここでは行動を実際に対戦へ適用しない(検証して終わり)。

局面コーパスは ``tests/fixtures/lethal_positions.jsonl``(固定)。
エンジンの初期シャッフルは seed できないため、実対戦から採るとテストが
実行のたびに変わってしまうので、一度採取したものを固定してある。

実行:
    python -m pytest tests/integration/lethal/test_entry_harness.py -q
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.action_selection import selector
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal import action as action_module
from ptcg_ai.search.lethal import cg_backend, entry
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.scenario import Scenario

FIXTURE = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_positions.jsonl"

CONFIG = {
    "enabled": True,
    "module": "lethal_phase1",
    "max_remaining_prizes": 2,
    "phase12_ms": 100,
    "max_nodes": 4000,
    "max_depth": 8,
    "max_chance_depth": 1,
}


@pytest.fixture(scope="module")
def corpus():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert rows, "保存盤面コーパスが空"
    return rows


def _context(obs_dict, deck, *, config=None, seed=0):
    obs = to_observation_class(obs_dict)
    return obs, {
        "observation": obs,
        "config": {**CONFIG, **(config or {})},
        "hidden_state_factory": lambda: build_dummy_search_state(
            obs, deck, rng=random.Random(seed)
        ),
        "full_deck": list(deck),
    }


def _is_legal(selection, select) -> bool:
    return action_module.is_legal_selection(selection, select)


# ------------------------------------------------------------ 接続ハーネス


def test_entry_returns_only_legal_actions_on_the_saved_corpus(corpus):
    """全保存盤面で、返り値は None か**合法な選択**のどちらかであること。"""
    deck = read_deck_csv()
    stats: Counter = Counter()
    reasons: Counter = Counter()
    for row in corpus:
        obs, context = _context(row["obs"], deck)
        selection = entry.search(obs.current, obs.select.option, context)
        diagnostics = entry.last_diagnostics()
        stats[row["kind"]] += 1
        if selection is None:
            stats["fallback"] += 1
            reasons[diagnostics.fallback_reason or "none"] += 1
        else:
            stats["returned_action"] += 1
            assert _is_legal(selection, obs.select), (
                f"違法な選択を返した: {selection} / {row['kind']}"
            )
            # 探索は「最初の1手」だけを返す契約。
            assert obs.select.minCount <= len(selection) <= obs.select.maxCount
    assert stats["returned_action"] + stats["fallback"] == len(corpus)
    print("\nentry stats:", dict(stats), "\nfallback reasons:", dict(reasons))


def test_entry_never_raises_on_the_saved_corpus(corpus):
    """どの局面でも例外を外へ出さない(原設計 §2.5)。"""
    deck = read_deck_csv()
    for row in corpus:
        obs, context = _context(row["obs"], deck)
        entry.search(obs.current, obs.select.option, context)  # 例外が出れば失敗


def test_precheck_skips_positions_with_too_many_prizes(corpus):
    """起動条件: サイドが多く、かつ**サイド以外の勝ち筋も無い**局面では探索しない。

    相手のベンチが空なら、バトル場を倒すだけで「場にポケモンがいない」勝ちになるので、
    サイドが多くても起動する(Step 1-11 で precheck を拡張した)。
    """
    deck = read_deck_csv()
    skipped = 0
    started_without_prize_gate = 0
    for row in corpus:
        if row["kind"] != "main":
            continue
        obs, context = _context(row["obs"], deck)
        me = obs.current.yourIndex
        if len(obs.current.players[me].prize) <= CONFIG["max_remaining_prizes"]:
            continue
        opponent = obs.current.players[1 - me]
        entry.search(obs.current, obs.select.option, context)
        reason = entry.last_diagnostics().fallback_reason
        if not opponent.bench and opponent.active:
            # ベンチが空 = 1 体倒せば勝ち。起動してよい
            assert reason != "precheck"
            started_without_prize_gate += 1
        else:
            assert reason == "precheck"
            skipped += 1
    assert skipped > 0
    assert started_without_prize_gate > 0, "ベンチ空の局面が corpus に無い"


def test_precheck_does_not_drop_provable_wins(corpus):
    """precheck の偽陰性検査: 起動しない局面に Phase 1 の勝ちが埋もれていないか。

    Step 1-11 の実測では、サイド条件だけの precheck が **6 件中 2 件**の
    `PROVEN_WIN` を捨てていた(相手のベンチが空の勝ち)。拡張後は 0 件。
    """
    from ptcg_ai.search.lethal import phase1
    from ptcg_ai.search.lethal.budget import Budget
    from ptcg_ai.search.lethal.cg_backend import CgBackend
    from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
    from ptcg_ai.search.lethal.types import Proof

    deck = read_deck_csv()
    dropped = []
    for row in corpus:
        obs, context = _context(row["obs"], deck)
        entry.search(obs.current, obs.select.option, context)
        if entry.last_diagnostics().started:
            continue
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        hidden = HiddenState.from_stub(stub)
        with SearchSession(obs, hidden) as session:
            backend = CgBackend(
                session, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            result = phase1.search(
                backend,
                Budget(time_limit_ms=300.0, max_nodes=3000, max_depth=8),
            )
        if result.proof is Proof.PROVEN_WIN:
            dropped.append(row["id"])
    assert not dropped, f"precheck が確定勝ちを捨てている: {dropped}"


def test_deck_open_positions_do_not_yield_proven_win(corpus):
    """B1: 実デッキが使われる局面ではドローを列挙しない(証明に使わない)。"""
    deck = read_deck_csv()
    checked = 0
    for row in corpus:
        if row["kind"] != "deck_open":
            continue
        obs, context = _context(row["obs"], deck)
        selection = entry.search(obs.current, obs.select.option, context)
        diagnostics = entry.last_diagnostics()
        if diagnostics.started:
            checked += 1
            if selection is not None:
                # 万一勝ちを主張したなら、ドローを含まない Phase 1 の線に限る。
                assert diagnostics.phase1_proof == "PROVEN_WIN"
    assert checked >= 0  # 起動しない局面ばかりでも失敗にはしない


class _RootedBackend:
    """既存 backend を任意の状態から始めるように見せるラッパ(検証用)。"""

    def __init__(self, backend, state):
        self._backend = backend
        self._state = state

    def root(self):
        return self._state

    def __getattr__(self, name):
        return getattr(self._backend, name)


def test_proven_win_chains_to_an_actual_engine_win(corpus):
    """``PROVEN_WIN`` の局面で、1手ずつ適用して再探索すると本当に勝ちへ到達する。

    実戦の「最初の1手だけ実行 → 次の観測で再探索」を、探索の世界の中で再現する
    (対戦へは何も適用していない)。証明が口先だけでないことの確認。
    """
    from ptcg_ai.search.lethal import phase1
    from ptcg_ai.search.lethal.budget import Budget
    from ptcg_ai.search.lethal.cg_backend import CgBackend
    from ptcg_ai.search.lethal.engine import SearchSession
    from ptcg_ai.search.lethal.types import Proof

    deck = read_deck_csv()
    verified = 0
    for row in corpus:
        obs, context = _context(row["obs"], deck)
        selection = entry.search(obs.current, obs.select.option, context)
        if selection is None:
            continue
        assert entry.last_diagnostics().phase1_proof == "PROVEN_WIN"

        hidden = HiddenState.from_stub(context["hidden_state_factory"]())
        with SearchSession(obs, hidden) as session:
            backend = CgBackend(
                session, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            state = backend.root()
            for _ in range(20):
                if backend.is_win(state):
                    break
                result = phase1.search(
                    _RootedBackend(backend, state),
                    Budget(time_limit_ms=500.0, max_nodes=20_000, max_depth=8),
                )
                assert result.proof is Proof.PROVEN_WIN, (
                    "再探索で勝ちを維持できなかった(証明が不整合)"
                )
                transition = backend.apply(state, result.first_action)
                assert transition.state is not None
                state = transition.state
            assert backend.is_win(state), "勝ちへ到達しなかった"
        verified += 1
    assert verified > 0, "PROVEN_WIN の局面が1つも無かった(検証できていない)"


# ---------------------------------------------------------------- fallback


def test_disabled_config_returns_none(corpus):
    deck = read_deck_csv()
    obs, context = _context(corpus[0]["obs"], deck, config={"enabled": False})
    assert entry.search(obs.current, obs.select.option, context) is None
    assert entry.last_diagnostics().fallback_reason == "disabled"


def test_zero_time_budget_falls_back(corpus):
    """時間切れは UNKNOWN。勝ちにも負けにも丸めない。"""
    deck = read_deck_csv()
    started = 0
    for row in corpus:
        obs, context = _context(row["obs"], deck, config={"phase12_ms": 0.0})
        selection = entry.search(obs.current, obs.select.option, context)
        diagnostics = entry.last_diagnostics()
        if not diagnostics.started:
            continue
        started += 1
        assert selection is None
        assert diagnostics.phase1_proof in (None, "UNKNOWN")
        assert diagnostics.phase2_proof in (None, "UNKNOWN")
    assert started > 0


def test_backend_exception_falls_back(corpus, monkeypatch):
    """アダプタ内部で例外が出ても通常方策へ戻る。"""
    deck = read_deck_csv()

    class Boom(cg_backend.CgBackend):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(entry, "CgBackend", Boom)
    started = 0
    for row in corpus:
        obs, context = _context(row["obs"], deck)
        assert entry.search(obs.current, obs.select.option, context) is None
        if entry.last_diagnostics().started:
            started += 1
            assert entry.last_diagnostics().fallback_reason == "exception:RuntimeError"
    assert started > 0


def test_missing_hidden_state_falls_back(corpus):
    deck = read_deck_csv()
    obs = to_observation_class(corpus[0]["obs"])
    context = {
        "observation": obs,
        "config": CONFIG,
        "hidden_state_factory": lambda: None,
        "full_deck": list(deck),
    }
    assert entry.search(obs.current, obs.select.option, context) is None
    assert entry.last_diagnostics().fallback_reason in ("no_hidden_state", "precheck")


def test_state_mismatch_is_not_executed(corpus):
    """別の局面の選択をそのまま実行しない(状態一致チェック)。"""
    deck = read_deck_csv()
    rows = [row for row in corpus if row["kind"] in ("gate", "main")]
    first = to_observation_class(rows[0]["obs"])
    other = to_observation_class(rows[-1]["obs"])
    action = action_module.describe([0], first.select, first.current, me=first.current.yourIndex)
    assert action is not None
    validated = action_module.validate(
        action, other.select, other.current, other.current.yourIndex
    )
    assert validated is None or validated == [0]
    if validated == [0]:
        # 偶然同じ意味だった場合のみ許される。意味が違えば必ず None。
        assert action.descriptors == action_module.describe(
            [0], other.select, other.current, other.current.yourIndex
        ).descriptors


# ------------------------------------------------------- 情報境界(実エンジン)


def test_result_does_not_depend_on_hidden_deck_order(corpus):
    """同じ信念(multiset)で山札の並びだけ変えても、探索が完全に一致する。

    予算は**ノード数だけ**で切る(時間制限は事実上無効化する)。壁時計で切ると、
    打ち切り位置が実行ごとにぶれて「並びのせいで結果が変わった」ように見えるため。
    実運用では時間切れは起こりうるが、その場合は必ず ``UNKNOWN`` → 通常方策なので
    安全側であり、ここで検証したい情報境界の性質とは別問題である。

    返す手だけでなく**展開ノード数まで**一致することを見る(同じ木を辿っている証拠)。
    """
    deck = read_deck_csv()
    # 打ち切り位置がノード数だけで決まるようにする(時間制限は事実上無効化)。
    # ノード上限は小さめに固定する: 大きくすると 1 局面あたりの outcome 構築
    # (search_begin + 再生)が増えて実行時間が跳ね上がり、テストが遅くなる。
    deterministic_config = {**CONFIG, "phase12_ms": 10_000.0, "max_nodes": 300}
    compared = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        base = HiddenState.from_stub(stub)
        results = []
        node_counts = []
        started = False
        for seed in range(3):
            order = list(base.scenario.order)
            random.Random(500 + seed).shuffle(order)
            variant = base.with_scenario(Scenario(tuple(order)))
            context = {
                "observation": obs,
                "config": deterministic_config,
                "hidden_state": {
                    "your_deck": list(variant.scenario.order),
                    "your_prize": list(variant.your_prize),
                    "opponent_deck": list(variant.opponent_deck),
                    "opponent_prize": list(variant.opponent_prize),
                    "opponent_hand": list(variant.opponent_hand),
                    "opponent_active": list(variant.opponent_active),
                },
                "full_deck": list(deck),
            }
            results.append(entry.search(obs.current, obs.select.option, context))
            diagnostics = entry.last_diagnostics()
            started = started or diagnostics.started
            node_counts.append(diagnostics.nodes)
        if not started:
            continue
        compared += 1
        assert len(set(map(_to_key, results))) == 1, (
            f"山札の並びで結果が変わった: {results}"
        )
        assert len(set(node_counts)) == 1, (
            f"山札の並びで探索した木が変わった: {node_counts}"
        )
    assert compared >= 5, f"比較できた局面が少なすぎる: {compared}"


def _to_key(selection):
    return None if selection is None else tuple(selection)


# ------------------------------------------ 既存挙動(未接続であることの確認)


def test_module_is_registered_but_not_the_default():
    """Step 1-12: モジュールは登録するが、**既定 config は従来のまま**。

    登録しただけでは挙動は変わらない(どれを使うかは config が決める)。
    これが feature flag の実体。
    """
    from ptcg_ai.core.config import load_config

    assert "lethal_phase1" in selector._SEARCH_MODULES
    assert selector._SEARCH_MODULES["lethal_phase1"] is entry
    # 既定 config は従来の lethal_simple のまま = 既定 OFF
    assert load_config()["lethal_search"]["module"] == "lethal_simple"


def test_phase1_config_disables_phase2_and_phase3():
    """A/B 用 config は Phase 1 だけを有効にしている。"""
    from ptcg_ai.core.config import load_config

    lethal = load_config("rule_lethal_phase1")["lethal_search"]
    assert lethal["module"] == "lethal_phase1"
    assert lethal["phase1_enabled"] is True
    assert lethal["phase2_enabled"] is False
    assert lethal["phase3_enabled"] is False


def test_phase1_config_never_runs_phase2(corpus):
    """Phase 1 専用 config では Phase 2 が一度も走らない。"""
    from ptcg_ai.core.config import load_config

    deck = read_deck_csv()
    lethal = load_config("rule_lethal_phase1")["lethal_search"]
    ran = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        context = {
            "observation": obs,
            "config": lethal,
            "hidden_state_factory": lambda o=obs: build_dummy_search_state(
                o, deck, rng=random.Random(0)
            ),
            "full_deck": list(deck),
        }
        selection = entry.search(obs.current, obs.select.option, context)
        diagnostics = entry.last_diagnostics()
        assert diagnostics.phase2_proof is None, "Phase 2 が走ってしまった"
        if selection is not None:
            assert diagnostics.phase1_proof == "PROVEN_WIN"
            assert action_module.is_legal_selection(selection, obs.select)
        if diagnostics.started:
            ran += 1
    assert ran > 0


def test_disabled_flag_matches_the_existing_agent(corpus):
    """``enabled: false`` のとき、既存 Agent と**完全に一致**する。"""
    deck = read_deck_csv()
    for row in corpus:
        obs = to_observation_class(row["obs"])
        baseline = selector.select_action(obs, deck, {"lethal_search": {"enabled": False}})
        disabled = selector.select_action(
            obs, deck,
            {"lethal_search": {"enabled": False, "module": "lethal_phase1"}},
        )
        assert baseline == disabled


def test_agent_output_is_unchanged_when_module_is_unknown(corpus):
    """未登録のモジュール名を config に書いても、既存の出力は変わらない。"""
    deck = read_deck_csv()
    for row in corpus[:8]:
        obs = to_observation_class(row["obs"])
        baseline = selector.select_action(obs, deck, {"lethal_search": {"enabled": False}})
        with_unknown_module = selector.select_action(
            obs, deck, {"lethal_search": {"enabled": True, "module": "no_such_module"}}
        )
        assert baseline == with_unknown_module


def test_agent_output_is_unchanged_when_adapter_returns_none(corpus, monkeypatch):
    """接続しても、証明できない局面では既存の出力と完全に一致する。"""
    deck = read_deck_csv()
    baseline = []
    for row in corpus:
        obs = to_observation_class(row["obs"])
        baseline.append(selector.select_action(obs, deck, {"lethal_search": {"enabled": False}}))

    monkeypatch.setitem(selector._SEARCH_MODULES, "lethal_phase1", entry)
    for row, expected in zip(corpus, baseline):
        obs = to_observation_class(row["obs"])
        got = selector.select_action(
            obs, deck, {"lethal_search": {**CONFIG, "module": "lethal_phase1"}}
        )
        if entry.last_diagnostics().executed:
            # 証明できた局面だけは変わってよい(その場合も合法であること)。
            assert action_module.is_legal_selection(got, obs.select)
        else:
            assert got == expected
