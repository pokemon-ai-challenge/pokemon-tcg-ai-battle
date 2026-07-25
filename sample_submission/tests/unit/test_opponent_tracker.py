"""クラスタ③/⑤ 検証（相手デッキ予測の配線）／担当B

opponent_modeling.tracker が:
    - 毎ターン OpponentKnowledge を logs→state の順で更新してから rough_predictor.predict を呼ぶこと
    - 予測が確信を持てない（status != "confident"）場合は current_matchup_plan が None を返すこと
    - 確信を持てる場合、DeckPlan.matchup_plans から対応する MatchupPlan を引けること
を検証する。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import PlayerState, State
from ptcg_ai.opponent_modeling import tracker
from ptcg_ai.shared.profile_types import DeckPlan, MatchupPlan


def make_player_state(**kwargs) -> PlayerState:
    defaults = dict(
        active=[],
        bench=[],
        benchMax=5,
        deckCount=50,
        discard=[],
        prize=[None] * 6,
        handCount=5,
        hand=None,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )
    defaults.update(kwargs)
    return PlayerState(**defaults)


def make_state(your_index=0, turn=1) -> State:
    players = [None, None]
    players[your_index] = make_player_state()
    players[1 - your_index] = make_player_state()
    return State(
        turn=turn,
        turnActionCount=0,
        yourIndex=your_index,
        firstPlayer=your_index,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=-1,
        stadium=[],
        looking=None,
        players=players,
    )


class _FakeKnowledge:
    """OpponentKnowledge の代わりに呼び出し順序を記録するだけのテストダブル。"""

    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []

    def update_from_logs(self, logs):
        self.calls.append("logs")

    def update_from_state(self, state):
        self.calls.append("state")


@pytest.fixture(autouse=True)
def reset_tracker():
    tracker.reset()
    yield
    tracker.reset()


def test_update_calls_logs_then_state_and_returns_prediction(monkeypatch):
    fake_knowledge = _FakeKnowledge()
    monkeypatch.setattr(tracker, "OpponentKnowledge", lambda *a, **k: fake_knowledge)

    captured = {}

    def fake_predict(state, knowledge):
        captured["state"] = state
        captured["knowledge"] = knowledge
        return {"status": "confident", "deck_type": "alakazam"}

    monkeypatch.setattr(tracker.rough_predictor, "predict", fake_predict)

    state = make_state()

    class _FakeObs:
        logs = ["log1", "log2"]
        current = state

    result = tracker.update(_FakeObs())

    assert fake_knowledge.calls == ["logs", "state"]
    assert captured["knowledge"] is fake_knowledge
    assert captured["state"] is state
    assert result == {"status": "confident", "deck_type": "alakazam"}
    assert tracker.current_prediction() == result


@pytest.mark.parametrize("status", ["ambiguous", "insufficient_evidence", "no_candidate"])
def test_current_matchup_plan_is_none_when_not_confident(monkeypatch, status):
    monkeypatch.setattr(tracker, "OpponentKnowledge", lambda *a, **k: _FakeKnowledge())
    monkeypatch.setattr(
        tracker.rough_predictor, "predict", lambda state, knowledge: {"status": status, "deck_type": "alakazam"}
    )

    class _FakeObs:
        logs: list = []
        current = make_state()

    tracker.update(_FakeObs())

    assert tracker.current_matchup_plan() is None


def test_current_matchup_plan_returns_matching_entry_when_confident(monkeypatch):
    monkeypatch.setattr(tracker, "OpponentKnowledge", lambda *a, **k: _FakeKnowledge())
    monkeypatch.setattr(
        tracker.rough_predictor, "predict", lambda state, knowledge: {"status": "confident", "deck_type": "alakazam"}
    )

    expected_plan = MatchupPlan(attack_priority_boost={743: 5.0}, note="test entry")
    fake_deck_plan = DeckPlan(matchup_plans={"alakazam": expected_plan})
    monkeypatch.setattr(tracker.profile_registry, "get_deck_plan", lambda: fake_deck_plan)

    class _FakeObs:
        logs: list = []
        current = make_state()

    tracker.update(_FakeObs())

    assert tracker.current_matchup_plan() is expected_plan


def test_current_matchup_plan_is_none_when_no_entry_for_predicted_archetype(monkeypatch):
    monkeypatch.setattr(tracker, "OpponentKnowledge", lambda *a, **k: _FakeKnowledge())
    monkeypatch.setattr(
        tracker.rough_predictor, "predict", lambda state, knowledge: {"status": "confident", "deck_type": "dragapult_ex"}
    )
    monkeypatch.setattr(tracker.profile_registry, "get_deck_plan", lambda: DeckPlan(matchup_plans={}))

    class _FakeObs:
        logs: list = []
        current = make_state()

    tracker.update(_FakeObs())

    assert tracker.current_matchup_plan() is None


def test_current_prediction_and_matchup_plan_are_none_before_first_update():
    assert tracker.current_prediction() is None
    assert tracker.current_matchup_plan() is None
