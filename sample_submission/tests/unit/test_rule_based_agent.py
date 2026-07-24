"""rule_based_agent.agent が新しい試合の開始を検知して opponent_modeling.tracker を
リセットすることを検証する。

`obs.select is None` は初回のデッキ選択ターンを表す（`match_context` が新しい試合の
開始検知に使っているのと同じシグナル、`hidden_information/match_context.py` docstring参照）。
同一プロセスで複数試合を連続実行した場合に前の試合の相手デッキ予測が次の試合へ
持ち越されないことを保証する。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.rule_based import rule_based_agent


def _deck_select_obs() -> Observation:
    return Observation(select=None, logs=[], current=None)


@pytest.fixture(autouse=True)
def reset_tracker():
    opponent_tracker.reset()
    yield
    opponent_tracker.reset()


def test_new_deck_selection_resets_opponent_tracker(monkeypatch):
    # 前の試合の残り(観測済み相手情報・予測結果)を模擬する。
    monkeypatch.setattr(opponent_tracker, "_knowledge", object())
    monkeypatch.setattr(
        opponent_tracker,
        "_last_prediction",
        {"status": "confident", "deck_type": "alakazam"},
    )

    # match_context.update(obs) が例外を出しても本テストの対象(tracker)には影響しない。
    monkeypatch.setattr(rule_based_agent.match_context, "update", lambda obs: None)

    rule_based_agent.agent(_deck_select_obs())

    assert opponent_tracker.current_prediction() is None
    assert opponent_tracker.current_matchup_plan() is None
