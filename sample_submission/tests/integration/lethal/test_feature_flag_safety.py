"""feature flag の安全性(Step 1-13 指示 1・2)。

`_SEARCH_MODULES` へ登録済みでも、**config が選ばない限り Phase 1 は一度も動かない**。
壊れた config・欠落した config でも危険側へ倒れないことを固定する。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.action_selection import selector
from ptcg_ai.core.config import load_config
from ptcg_ai.search.lethal import action as action_module
from ptcg_ai.search.lethal import entry

FIXTURE = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_positions.jsonl"


@pytest.fixture(scope="module")
def corpus():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


class _Tripwire:
    """呼ばれたら記録する(呼ばれてはいけないことの検査用)。"""

    def __init__(self):
        self.calls = 0

    def search(self, state, legal_actions, context):
        self.calls += 1
        return None


def test_default_config_never_invokes_phase1(corpus, monkeypatch):
    """既定 config では Phase 1 が**一度も呼ばれない**。"""
    deck = read_deck_csv()
    tripwire = _Tripwire()
    monkeypatch.setitem(selector._SEARCH_MODULES, "lethal_phase1", tripwire)
    default = load_config()
    assert default["lethal_search"]["module"] == "lethal_simple"
    for row in corpus:
        obs = to_observation_class(row["obs"])
        action = selector.select_action(obs, deck, default)
        assert action_module.is_legal_selection(action, obs.select)
    assert tripwire.calls == 0, "既定 config で Phase 1 が呼ばれた"


def test_phase1_config_does_invoke_phase1(corpus, monkeypatch):
    """逆に、専用 config では確かに呼ばれる(テスト自体の検出力の確認)。"""
    deck = read_deck_csv()
    tripwire = _Tripwire()
    monkeypatch.setitem(selector._SEARCH_MODULES, "lethal_phase1", tripwire)
    config = load_config("rule_lethal_phase1")
    for row in corpus[:5]:
        obs = to_observation_class(row["obs"])
        selector.select_action(obs, deck, config)
    assert tripwire.calls == 5


@pytest.mark.parametrize(
    "broken",
    [
        {},                                              # lethal_search が無い
        {"lethal_search": {}},                           # 項目が空
        {"lethal_search": {"enabled": True}},            # module 未指定
        {"lethal_search": {"module": "lethal_phase1"}},  # enabled 未指定
        {"lethal_search": {"enabled": True, "module": "does_not_exist"}},
        {"lethal_search": {"enabled": "yes", "module": "lethal_phase1"}},  # 型違い
        {"lethal_search": None},
    ],
    ids=["no_section", "empty", "no_module", "no_enabled", "unknown_module",
         "wrong_type", "null_section"],
)
def test_malformed_config_falls_back_safely(corpus, broken, monkeypatch):
    """壊れた config でも、危険側(勝手に Phase 1 が動く)へ倒れない。

    ``enabled`` が未指定なら動かない。``module`` が未知なら動かない。
    いずれの場合も**合法な行動**は必ず返る。
    """
    deck = read_deck_csv()
    _ = monkeypatch
    lethal = (broken or {}).get("lethal_search") or {}
    should_run = lethal.get("enabled") is True and lethal.get("module") == "lethal_phase1"
    executed = 0
    for row in corpus[:6]:
        obs = to_observation_class(row["obs"])
        # 前回の診断が残っていると判定できないので、毎回リセットする
        entry._LAST_DIAGNOSTICS = entry.Diagnostics()
        action = selector.select_action(obs, deck, broken)
        assert action_module.is_legal_selection(action, obs.select)
        if entry.last_diagnostics().executed:
            executed += 1
    if not should_run:
        # 実際に行動を上書きしたかどうかで判定する(呼ばれること自体は無害)。
        assert executed == 0, f"危険側へ倒れた: {broken}"


def test_entry_itself_defaults_to_off_when_config_is_missing(corpus):
    """entry を直接呼んだ場合も、``enabled`` が無ければ動かない。"""
    deck = read_deck_csv()
    obs = to_observation_class(corpus[0]["obs"])
    context = {
        "observation": obs,
        "config": {"enabled": False},
        "hidden_state_factory": lambda: None,
        "full_deck": list(deck),
    }
    assert entry.search(obs.current, obs.select.option, context) is None
    assert entry.last_diagnostics().fallback_reason == "disabled"


def test_rollback_by_config_only(corpus):
    """config を戻すだけで baseline へ戻れる(コード変更不要)。"""
    deck = read_deck_csv()
    baseline = {"lethal_search": {"enabled": False}}
    for row in corpus[:10]:
        obs = to_observation_class(row["obs"])
        before = selector.select_action(obs, deck, baseline)
        selector.select_action(obs, deck, load_config("rule_lethal_phase1"))
        after = selector.select_action(obs, deck, baseline)
        assert before == after, "Phase 1 を挟むと baseline の出力が変わった"


def test_phase1_config_keeps_phase2_and_phase3_off():
    lethal = load_config("rule_lethal_phase1")["lethal_search"]
    assert lethal["phase2_enabled"] is False
    assert lethal["phase3_enabled"] is False


def test_selector_passes_the_deck_list_not_a_shuffled_order(corpus):
    """selector が渡す ``full_deck`` は deck.csv の並びで、実際の山札順ではない。

    注意: selector 経由の出力そのものを比較してはいけない。
    ``build_dummy_search_state`` は呼び出しごとに **グローバル random で**
    未確認カードをデッキ/サイドへ振り分けるため、同じ入力でも信念が変わる。
    デッキ並びへの非依存は、信念を固定した状態で
    ``test_b4_and_boundary.test_deck_list_order_does_not_change_the_result``
    が検証している。ここでは「渡している値が deck.csv の内容そのもの」だけを見る。
    """
    deck = read_deck_csv()
    captured = {}

    class _Capture:
        def search(self, state, legal_actions, context):
            captured["full_deck"] = context.get("full_deck")
            return None

    original = selector._SEARCH_MODULES.get("lethal_phase1")
    selector._SEARCH_MODULES["lethal_phase1"] = _Capture()
    try:
        obs = to_observation_class(corpus[0]["obs"])
        selector.select_action(obs, deck, load_config("rule_lethal_phase1"))
    finally:
        selector._SEARCH_MODULES["lethal_phase1"] = original
    assert captured["full_deck"] == deck
