"""Scenario の隔離(規則 R2)と診断ログの検閲(規則 R2b)。

観点:
1. ``Scenario`` / ``HiddenState`` の ``repr`` に順序が出ない
2. ``Scenario`` を import してよいモジュールが allow-list に限られている
3. 診断ログに順列・禁止キー・長い int 列を書けない
4. 正常な診断は普通に書ける
"""

from pathlib import Path

import pytest

from ptcg_ai.search.lethal.diagnostics import (
    DiagnosticsLeakError,
    DiagnosticsRecorder,
)
from ptcg_ai.search.lethal.engine import HiddenState
from ptcg_ai.search.lethal.scenario import (
    _ALLOWED_MODULES,
    _ORDER_READER_MODULES,
    Scenario,
)
from ptcg_ai.search.lethal.types import ChanceClass

LETHAL_DIR = Path(__file__).resolve().parents[3] / "ptcg_ai" / "search" / "lethal"

# 4桁のIDだけを使う。repr に出る size / digest の文字と偶然一致しないようにするため。
ORDER = (743, 1182, 1264, 1086, 1079)


def _without_digest(text: str, scenario: Scenario) -> str:
    """digest(16進)は順序を復元できないので、照合前に取り除く。"""
    return text.replace(scenario.digest(), "")


def hidden() -> HiddenState:
    return HiddenState(
        scenario=Scenario(ORDER),
        your_prize=(1, 2),
        opponent_deck=(3,),
        opponent_prize=(4,),
        opponent_hand=(5,),
        opponent_active=(),
    )


def test_scenario_repr_hides_the_order():
    # 観点1: 例外メッセージやログへ紛れ込んでも順序が漏れない。
    scenario = Scenario(ORDER)
    text = _without_digest(f"{scenario!r} {scenario!s}", scenario)
    for card_id in ORDER:
        assert str(card_id) not in text
    assert "digest" in repr(scenario)


def test_hidden_state_repr_hides_the_order():
    state = hidden()
    text = _without_digest(repr(state), state.scenario)
    for card_id in ORDER:
        assert str(card_id) not in text


def test_scenario_exposes_multiset_but_order_only_for_the_engine():
    scenario = Scenario(ORDER)
    assert scenario.multiset()[743] == 1
    assert scenario.size() == len(ORDER)
    # engine へ渡す形は list。これを使ってよいのは engine 層だけ(観点2 で担保)。
    assert scenario.to_engine_list() == list(ORDER)


def test_only_allowed_modules_import_scenario():
    # 観点2a: allow-list 外のモジュールが Scenario を import していないこと。
    allowed = {name.rsplit(".", 1)[-1] for name in _ALLOWED_MODULES}
    offenders = []
    for path in sorted(LETHAL_DIR.glob("*.py")):
        if path.stem in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "import Scenario" in text or "scenario import" in text:
            offenders.append(path.name)
    assert offenders == [], f"Scenario leaked into: {offenders}"


def test_only_order_readers_touch_the_permutation():
    # 観点2b: import できることと「順序を読めること」は別の権限。
    # 決定側はもちろん、診断モジュールも順序そのものには触れない。
    allowed = {name.rsplit(".", 1)[-1] for name in _ORDER_READER_MODULES}
    offenders = []
    for path in sorted(LETHAL_DIR.glob("*.py")):
        if path.stem in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in (".order", "to_engine_list"):
            if marker in text:
                offenders.append(f"{path.name}:{marker}")
    assert offenders == [], f"deck order read outside the allow-list: {offenders}"


def test_diagnostics_rejects_scenario():
    recorder = DiagnosticsRecorder()
    with pytest.raises(DiagnosticsLeakError):
        recorder.record("node", value=Scenario(ORDER))
    assert recorder.events == []


def test_diagnostics_rejects_forbidden_keys():
    recorder = DiagnosticsRecorder()
    for key in ("your_deck", "scenario", "deck_order", "opponent_hand"):
        with pytest.raises(DiagnosticsLeakError):
            recorder.record("node", **{key: "anything"})


def test_diagnostics_rejects_long_int_sequences():
    # 山札順の実体になりうる長さの int 列は書かせない。
    recorder = DiagnosticsRecorder()
    with pytest.raises(DiagnosticsLeakError):
        recorder.record("node", ids=list(range(20)))
    # 短いものは許す(選択インデックス列など)。
    recorder.record("node", selection=[0, 2])
    assert recorder.events[-1][1]["selection"] == [0, 2]


def test_diagnostics_rejects_nested_forbidden_keys():
    recorder = DiagnosticsRecorder()
    with pytest.raises(DiagnosticsLeakError):
        recorder.record("node", detail={"your_deck": [1, 2, 3]})


def test_diagnostics_accepts_normal_fields():
    # 観点4: 設計 §8 で記録すると決めた項目は普通に書ける。
    recorder = DiagnosticsRecorder()
    recorder.record(
        "phase2",
        info_key="ab12cd34",
        chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER.name,
        p_lethal_lower=0.0,
        p_lethal_upper=1.0,
        outcomes_expanded=12,
        elapsed_ms=41.2,
    )
    assert recorder.as_dicts()[0]["event"] == "phase2"
    assert recorder.as_dicts()[0]["outcomes_expanded"] == 12
