"""ptcg_ai.search.boss_target_eval の単体テスト(r9 Fix-E)。

対象別KO判定の本体はエンジン(`search_begin`/`search_step`)に委ねるので、ここでは
**判定不能を確実に判定不能として返す**防御パス(呼び出し側が「介入しない」を選べる契約)と、
対象選択スキーマの識別だけを検証する。実局面での KO 判定は
`kaggle_replays/_snapshot_gate_r8.py` の G7(93503044 T13)で固定している。

sample_submission/ から:
    python -m pytest tests/unit/test_boss_target_eval.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))


@pytest.fixture(scope="module")
def bte():
    try:
        from ptcg_ai.search import boss_target_eval as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"boss_target_eval / cg engine unavailable: {exc}")
    return mod


class _Select:
    def __init__(self, type_, context, option):
        self.type = type_
        self.context = context
        self.option = option


class _State:
    def __init__(self, result=-1):
        self.result = result
        self.yourIndex = 0
        self.players = []


class _Obs:
    def __init__(self, current, select):
        self.current = current
        self.select = select


def _main_obs():
    from cg.api import SelectContext, SelectType

    select = _Select(SelectType.MAIN, SelectContext.MAIN, [object(), object()])
    state = _State()
    state.players = [type("P", (), {"prize": [None, None]})(), type("P", (), {"prize": []})()]
    return _Obs(state, select)


def test_evaluate_targets_returns_none_without_observation(bte):
    """obs/current/select が無ければ判定不能(None)。"""
    assert bte.evaluate_targets(None, lambda: {}, 0) is None
    assert bte.evaluate_targets(_Obs(None, None), lambda: {}, 0) is None


def test_evaluate_targets_returns_none_without_hidden_state(bte):
    """隠れ状態が組めない(factory が None を返す/None)なら判定不能。"""
    obs = _main_obs()
    assert bte.evaluate_targets(obs, None, 0) is None
    assert bte.evaluate_targets(obs, lambda: None, 0) is None


def test_evaluate_targets_returns_none_for_out_of_range_play_index(bte):
    """play_index が選択肢の範囲外なら探索を始めない。"""
    obs = _main_obs()
    called = {"n": 0}

    def _factory():
        called["n"] += 1
        return {}

    assert bte.evaluate_targets(obs, _factory, 99) is None
    assert called["n"] == 0  # 範囲チェックが先(隠れ状態の構築コストを払わない)


def test_evaluate_targets_returns_none_when_game_is_over(bte):
    """すでに決着している局面では評価しない。"""
    obs = _main_obs()
    obs.current.result = 0
    assert bte.evaluate_targets(obs, lambda: {}, 0) is None


def test_is_target_select_matches_boss_schema(bte):
    """実測スキーマ: SelectType.CARD かつ SelectContext.SWITCH かつ選択肢あり。"""
    from cg.api import SelectContext, SelectType

    assert bte._is_target_select(_Select(SelectType.CARD, SelectContext.SWITCH, [object()]))
    assert not bte._is_target_select(_Select(SelectType.CARD, SelectContext.SWITCH, []))
    assert not bte._is_target_select(_Select(SelectType.MAIN, SelectContext.SWITCH, [object()]))
    assert not bte._is_target_select(_Select(SelectType.CARD, SelectContext.TO_HAND, [object()]))
    assert not bte._is_target_select(None)


def test_can_ko_from_node_reports_abort_on_broken_node(bte):
    """壊れたノードを渡しても例外を投げず (False, True=判定不能) を返す。"""
    from ptcg_ai.search import ko_search

    can_ko, aborted = ko_search.can_ko_from_node(object(), 0, 2, {"time_limit_ms": 5})
    assert can_ko is False and aborted is True
