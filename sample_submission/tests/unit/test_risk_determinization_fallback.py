"""``risk_determinization.search()`` のフォールバック（例外安全性）回帰テスト。

要件書 §11: ``hidden_state_factory`` が例外/None、Value未ロード、予算0 のとき ``None`` を返す。
どのケースでも ``search()`` 自体は例外を外へ伝播させない（呼び出し側 ``selector.select_action``
の「例外は握り潰してNoneに落ちる」契約と同じ）。
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


def make_select() -> SelectData:
    return SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=OptionType.ATTACK), Option(type=OptionType.ATTACK)],
        deck=None, contextCard=None, effect=None,
    )


def make_obs(state: State, select: SelectData) -> Observation:
    return Observation(select=select, logs=[], current=state, search_begin_input="{}")


class FakeOpponentState:
    is_ready = True


class FakeValueModel:
    is_ready = True

    def predict_win_prob_from_state(self, state):
        return 0.5


class FakeSearchState:
    def __init__(self):
        self.searchId = 1
        self.observation = type("Obs", (), {"current": make_state()})()


class FakeCgApiOk:
    def search_begin(self, *args, **kwargs):
        return FakeSearchState()

    def search_step(self, search_id, select):
        return FakeSearchState()

    def search_end(self):
        pass

    def search_release(self, search_id):
        pass


@pytest.fixture(autouse=True)
def _isolated_module_state(monkeypatch):
    risk_determinization.reset_stats()
    monkeypatch.setattr(risk_determinization, "_match_start_time", None)
    monkeypatch.setattr(risk_determinization, "_last_seen_turn", None)
    monkeypatch.setattr(risk_determinization, "cg_api", FakeCgApiOk())
    monkeypatch.setattr(
        risk_determinization.match_context, "get_opponent_state", lambda: FakeOpponentState()
    )
    monkeypatch.setattr(risk_determinization, "_get_value_model", lambda: FakeValueModel())
    monkeypatch.delenv(risk_determinization._ENV_DISABLE, raising=False)
    yield


def base_context(obs: Observation, **overrides) -> dict:
    config = {
        "enabled": True,
        "mode": "mean",
        "determinizations": 3,
        "top_k": 2,
        "min_turn": 1,
        "time_limit_ms": 1000,
        "min_remaining_game_ms": 0,
        "max_evaluations": 32,
    }
    config.update(overrides.pop("config_overrides", {}))
    ctx = {
        "observation": obs,
        "config": config,
        "candidate_provider": lambda: [([0], 10.0), ([1], 5.0)],
    }
    ctx.update(overrides)
    return ctx


def test_hidden_state_factory_raises_every_call_returns_none():
    state = make_state()
    obs = make_obs(state, make_select())

    def factory():
        raise RuntimeError("boom")

    ctx = base_context(obs, hidden_state_factory=factory)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    stats = risk_determinization.get_stats()
    assert stats["reject_reasons"].get("no_samples") == 1


def test_hidden_state_factory_returns_none_every_call_returns_none():
    state = make_state()
    obs = make_obs(state, make_select())
    ctx = base_context(obs, hidden_state_factory=lambda: None)
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"].get("no_samples") == 1


def test_missing_hidden_state_factory_key_returns_none():
    state = make_state()
    obs = make_obs(state, make_select())
    ctx = base_context(obs)  # hidden_state_factory を渡さない
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"no_hidden_state_factory": 1}


def test_max_evaluations_zero_returns_none_without_exception():
    state = make_state()
    obs = make_obs(state, make_select())
    hidden_state = {
        "your_deck": [], "your_prize": [], "opponent_deck": [],
        "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
    }
    ctx = base_context(
        obs,
        hidden_state_factory=lambda: hidden_state,
        config_overrides={"max_evaluations": 0},
    )
    # max_evaluations=0 は max(1, 0)=1 に丸められるため最低1回は評価される仕様だが、
    # time_limit_ms=0 にすれば即座に予算切れとなり0サンプルで安全にNoneへ落ちることを確認する。
    ctx["config"]["time_limit_ms"] = 0
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    stats = risk_determinization.get_stats()
    assert stats["reject_reasons"].get("no_samples") == 1


def test_value_model_predict_raises_is_swallowed():
    """個々の候補評価で例外が起きても search() 全体は落ちず None にフォールバックする。"""
    state = make_state()
    obs = make_obs(state, make_select())
    hidden_state = {
        "your_deck": [], "your_prize": [], "opponent_deck": [],
        "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
    }

    class RaisingValueModel:
        is_ready = True

        def predict_win_prob_from_state(self, state):
            raise RuntimeError("boom")

    import ptcg_ai.search.risk_determinization as rd
    # このテストだけ value_model を差し替える(autouse fixtureのFakeValueModelを上書き)。
    orig = rd._get_value_model
    rd._get_value_model = lambda: RaisingValueModel()
    try:
        ctx = base_context(obs, hidden_state_factory=lambda: hidden_state)
        result = rd.search(state, obs.select.option, ctx)
        assert result is None
        assert rd.get_stats()["reject_reasons"].get("no_samples") == 1
    finally:
        rd._get_value_model = orig


def test_candidate_provider_exception_returns_none():
    state = make_state()
    obs = make_obs(state, make_select())

    def raising_provider():
        raise RuntimeError("boom")

    ctx = base_context(obs, candidate_provider=raising_provider)
    ctx["hidden_state_factory"] = lambda: {
        "your_deck": [], "your_prize": [], "opponent_deck": [],
        "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
    }
    assert risk_determinization.search(state, obs.select.option, ctx) is None
    assert risk_determinization.get_stats()["reject_reasons"] == {"candidate_provider_error": 1}
