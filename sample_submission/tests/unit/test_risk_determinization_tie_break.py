"""``risk_determinization.search()`` のタイブレーク方式（v2, 要件書 §5.2改訂）の回帰テスト。

初版の「スコア混合」方式を棄却し、``tie_ratio``/``tie_abs`` で先頭候補と「互角」な候補集合 T
を先に確定し、T の中だけをリスク集約スコアで並べ替える方式に作り直した。ここでは:

- |T| < 2 のとき None を返し、評価（search_begin 等）が一切行われないこと
- |T| >= 2 のとき、T の外の候補には一切触れず（評価もしない）、T の中だけで並べ替えること
- ``get_stats()`` に T のサイズ分布が記録されること
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import Observation, Option, OptionType, PlayerState, SelectContext, SelectData, SelectType, State
from ptcg_ai.search import risk_determinization


def make_player(prizes: int = 6) -> PlayerState:
    return PlayerState(
        active=[], bench=[], benchMax=5, deckCount=40, discard=[], prize=[None] * prizes,
        handCount=3, hand=[], poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def make_state(*, turn: int = 5) -> State:
    players = [make_player(), make_player()]
    return State(
        turn=turn, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False, retreated=False,
        result=-1, stadium=[], looking=None, players=players,
    )


def make_select(option_count: int = 4) -> SelectData:
    return SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=OptionType.ATTACK) for _ in range(option_count)],
        deck=None, contextCard=None, effect=None,
    )


def make_obs(state: State, select: SelectData) -> Observation:
    return Observation(select=select, logs=[], current=state, search_begin_input="{}")


class FakeOpponentState:
    is_ready = True


class FakeValueModel:
    is_ready = True

    def __init__(self, value_by_action=None, default: float = 0.5):
        self._value_by_action = value_by_action or {}
        self._default = default

    def predict_win_prob_from_state(self, state):
        return getattr(state, "_risk_value", self._default)


class FakeSearchState:
    def __init__(self, current):
        self.searchId = 1
        self.observation = type("Obs", (), {"current": current})()


class SpyingCgApi:
    """search_begin/step の呼び出しに使われた action を記録する(T外を評価しないことの検証用)。"""

    def __init__(self, value_by_action_key):
        self.begin_calls = 0
        self.evaluated_actions: list[tuple[int, ...]] = []
        self._value_by_action_key = value_by_action_key

    def search_begin(self, *args, **kwargs):
        self.begin_calls += 1
        return FakeSearchState(make_state())

    def search_step(self, search_id, select):
        key = tuple(select)
        self.evaluated_actions.append(key)
        state = make_state()
        state._risk_value = self._value_by_action_key.get(key, 0.5)
        return FakeSearchState(state)

    def search_end(self):
        pass

    def search_release(self, search_id):
        pass


@pytest.fixture(autouse=True)
def _isolated_module_state(monkeypatch):
    risk_determinization.reset_stats()
    monkeypatch.setattr(risk_determinization, "_match_start_time", None)
    monkeypatch.setattr(risk_determinization, "_last_seen_turn", None)
    monkeypatch.setattr(
        risk_determinization.match_context, "get_opponent_state", lambda: FakeOpponentState()
    )
    monkeypatch.setattr(risk_determinization, "_get_value_model", lambda: FakeValueModel())
    monkeypatch.delenv(risk_determinization._ENV_DISABLE, raising=False)
    yield


def base_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "mode": "mean",
        "determinizations": 2,
        "top_k": 4,
        "tie_ratio": 0.05,
        "tie_abs": 1.0,
        "min_turn": 3,
        "time_limit_ms": 1000,
        "min_remaining_game_ms": 0,
        "max_evaluations": 64,
    }
    config.update(overrides)
    return config


def test_tie_margin_uses_max_of_abs_and_ratio():
    # top=10 → ratio項=0.5, abs項=1.0 → margin=1.0
    assert risk_determinization._tie_margin(10.0, tie_ratio=0.05, tie_abs=1.0) == 1.0
    # top=100 → ratio項=5.0, abs項=1.0 → margin=5.0
    assert risk_determinization._tie_margin(100.0, tie_ratio=0.05, tie_abs=1.0) == 5.0


def test_tie_set_size_less_than_two_returns_none_without_any_evaluation(monkeypatch):
    """1位が2位以下を大きく引き離す（KO_BONUSのような構造）と T は先頭候補のみになり、
    評価(search_begin)は一切呼ばれず None を返す。"""
    cg_api = SpyingCgApi({})
    monkeypatch.setattr(risk_determinization, "cg_api", cg_api)

    state = make_state(turn=5)
    obs = make_obs(state, make_select(option_count=2))
    ctx = {
        "observation": obs,
        "config": base_config(),
        "hidden_state_factory": lambda: {
            "your_deck": [], "your_prize": [], "opponent_deck": [],
            "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
        },
        # 1000 対 70 のような、有界でないルールスコアの典型例(要件書§5.2の教訓)。
        "candidate_provider": lambda: [([0], 1070.0), ([1], 70.0)],
    }

    result = risk_determinization.search(state, obs.select.option, ctx)
    assert result is None
    assert cg_api.begin_calls == 0
    stats = risk_determinization.get_stats()
    assert stats["reject_reasons"] == {"tie_set_too_small": 1}
    assert stats["fired"] == 0
    assert stats["tie_set_sizes"] == [1]


def test_tie_set_excludes_out_of_margin_candidates_from_evaluation(monkeypatch):
    """T の外（margin超過）の候補は search_step にすら渡されない。"""
    value_by_action = {(0,): 0.9, (1,): 0.1}  # [1]の方がValue的には高評価だがTの外
    cg_api = SpyingCgApi(value_by_action)
    monkeypatch.setattr(risk_determinization, "cg_api", cg_api)

    state = make_state(turn=5)
    obs = make_obs(state, make_select(option_count=3))
    ctx = {
        "observation": obs,
        "config": base_config(),
        "hidden_state_factory": lambda: {
            "your_deck": [], "your_prize": [], "opponent_deck": [],
            "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
        },
        # top=10.0, margin=max(1.0, 0.5)=1.0 → [1](diff=0.5)はT内、[2](diff=5.0)はT外。
        "candidate_provider": lambda: [([0], 10.0), ([1], 9.5), ([2], 5.0)],
    }

    result = risk_determinization.search(state, obs.select.option, ctx)
    # [2]はT外なので評価されず、結果として選ばれることも無い。
    assert result != [2]
    assert (2,) not in cg_api.evaluated_actions
    stats = risk_determinization.get_stats()
    assert stats["tie_set_sizes"] == [2]
    assert stats["fired"] == 1


def test_tie_set_reorders_within_t_only(monkeypatch):
    """T内の候補どうしはリスク集約スコアで並べ替わり得る（ルール2位が採用されることがある）。"""
    value_by_action = {(0,): 0.2, (1,): 0.9}  # ルール1位[0]よりValueでは[1]が高評価
    cg_api = SpyingCgApi(value_by_action)
    monkeypatch.setattr(risk_determinization, "cg_api", cg_api)

    state = make_state(turn=5)
    obs = make_obs(state, make_select(option_count=2))
    ctx = {
        "observation": obs,
        "config": base_config(),
        "hidden_state_factory": lambda: {
            "your_deck": [], "your_prize": [], "opponent_deck": [],
            "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
        },
        "candidate_provider": lambda: [([0], 10.0), ([1], 9.5)],
    }

    result = risk_determinization.search(state, obs.select.option, ctx)
    assert result == [1]
    stats = risk_determinization.get_stats()
    assert stats["overrides"] == 1
    assert stats["fired"] == 1
