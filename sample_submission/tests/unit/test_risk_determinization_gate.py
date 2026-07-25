"""``risk_determinization.search()`` のゲート条件（要件書 §5.3）の回帰テスト。

各ゲート条件を1項目ずつ崩し、期待どおり ``None`` を返し、``get_stats()`` の
``reject_reasons`` に対応する理由が記録されることを確認する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import Observation, Option, OptionType, PlayerState, SelectContext, SelectData, SelectType, State
from ptcg_ai.search import risk_determinization


# ---------------------------------------------------------------------------
# Builders(test_lethal_simple.pyと同じ最小構成)
# ---------------------------------------------------------------------------

def make_player(prizes: int = 6) -> PlayerState:
    return PlayerState(
        active=[], bench=[], benchMax=5, deckCount=40, discard=[], prize=[None] * prizes,
        handCount=3, hand=[], poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def make_state(*, turn: int = 5, your_index: int = 0, my_prizes: int = 6, opp_prizes: int = 6) -> State:
    mine = make_player(prizes=my_prizes)
    theirs = make_player(prizes=opp_prizes)
    players = [mine, theirs] if your_index == 0 else [theirs, mine]
    return State(
        turn=turn, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False, retreated=False,
        result=-1, stadium=[], looking=None, players=players,
    )


def make_select(select_type: SelectType = SelectType.MAIN, option_count: int = 2) -> SelectData:
    return SelectData(
        type=select_type, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=OptionType.ATTACK) for _ in range(option_count)],
        deck=None, contextCard=None, effect=None,
    )


def make_obs(state: State | None, select: SelectData | None) -> Observation:
    return Observation(select=select, logs=[], current=state, search_begin_input="{}")


class FakeOpponentState:
    def __init__(self, is_ready: bool):
        self.is_ready = is_ready


class FakeValueModel:
    def __init__(self, is_ready: bool = True, value: float = 0.6):
        self.is_ready = is_ready
        self._value = value

    def predict_win_prob_from_state(self, state):
        return self._value


class FakeSearchState:
    def __init__(self, current):
        self.searchId = 1
        self.observation = type("Obs", (), {"current": current})()


class FakeCgApi:
    """search_begin/step/end/release の最小フェイク(risk_determinization用)。"""

    def search_begin(self, *args, **kwargs):
        return FakeSearchState(make_state())

    def search_step(self, search_id, select):
        return FakeSearchState(make_state())

    def search_end(self):
        pass

    def search_release(self, search_id):
        pass


@pytest.fixture(autouse=True)
def _isolated_module_state(monkeypatch):
    """各テストで risk_determinization のモジュール状態を初期化する。"""
    risk_determinization.reset_stats()
    monkeypatch.setattr(risk_determinization, "_match_start_time", None)
    monkeypatch.setattr(risk_determinization, "_last_seen_turn", None)
    monkeypatch.setattr(risk_determinization, "cg_api", FakeCgApi())
    monkeypatch.setattr(
        risk_determinization.match_context, "get_opponent_state", lambda: FakeOpponentState(is_ready=True)
    )
    monkeypatch.setattr(risk_determinization, "_get_value_model", lambda: FakeValueModel(is_ready=True))
    monkeypatch.delenv(risk_determinization._ENV_DISABLE, raising=False)
    yield


def good_context(obs: Observation, **config_overrides) -> dict:
    config = {
        "enabled": True,
        "mode": "mean",
        "determinizations": 2,
        "top_k": 2,
        "min_turn": 3,
        "time_limit_ms": 1000,
        "min_remaining_game_ms": 0,  # デフォルトでは時間ゲートに引っかからないようにする
        "max_evaluations": 32,
    }
    config.update(config_overrides)
    hidden_state = {
        "your_deck": [], "your_prize": [], "opponent_deck": [],
        "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
    }
    return {
        "observation": obs,
        "config": config,
        "hidden_state_factory": lambda: hidden_state,
        # tie_ratio/tie_abs 既定(0.05/1.0)で T に入るよう、スコア差を margin 以内にする
        # (top=10.0 → margin=max(1.0, 0.5)=1.0)。
        "candidate_provider": lambda: [([0], 10.0), ([1], 9.5)],
    }


def test_happy_path_fires_and_returns_action_or_none():
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    result = risk_determinization.search(state, obs.select.option, good_context(obs))
    # 結果は None(ルール第1候補と同じ)か、合法な候補のどちらか。いずれにせよ例外は出ない。
    assert result is None or result in ([0], [1])
    stats = risk_determinization.get_stats()
    assert stats["invocations"] == 1
    assert stats["fired"] == 1


def test_not_enabled_rejects():
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs, enabled=False)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"not_enabled": 1}


def test_env_var_disable_overrides_enabled(monkeypatch):
    monkeypatch.setenv(risk_determinization._ENV_DISABLE, "1")
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"env_disabled": 1}


def test_non_main_select_type_rejects():
    state = make_state(turn=5)
    obs = make_obs(state, make_select(select_type=SelectType.YES_NO))
    ctx = good_context(obs)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"not_main": 1}


def test_no_current_state_rejects():
    obs = make_obs(None, make_select())
    ctx = good_context(obs)
    assert risk_determinization.search(None, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"no_observation": 1}


def test_insufficient_candidates_rejects():
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs)
    ctx["candidate_provider"] = lambda: [([0], 10.0)]
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"insufficient_candidates": 1}


def test_turn_too_early_rejects():
    state = make_state(turn=1)
    obs = make_obs(state, make_select())
    ctx = good_context(obs, min_turn=3)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"turn_too_early": 1}


def test_opponent_not_ready_rejects(monkeypatch):
    monkeypatch.setattr(
        risk_determinization.match_context, "get_opponent_state", lambda: FakeOpponentState(is_ready=False)
    )
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"opponent_not_ready": 1}


def test_value_model_not_ready_rejects(monkeypatch):
    monkeypatch.setattr(risk_determinization, "_get_value_model", lambda: FakeValueModel(is_ready=False))
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"value_not_ready": 1}


def test_time_budget_rejects():
    state = make_state(turn=5)
    obs = make_obs(state, make_select())
    ctx = good_context(obs, min_remaining_game_ms=10 ** 12)  # 絶対に満たせない残り時間要求
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"time_budget": 1}
