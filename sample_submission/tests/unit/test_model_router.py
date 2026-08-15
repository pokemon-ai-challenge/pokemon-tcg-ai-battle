"""opponent_modeling.model_router のユニットテスト。

rough_predictor.predict() 自体の正しさは test_rough_predictor.py で別途カバー済みなので、
ここでは monkeypatch で結果を直接差し替え、model_router 側のロジック(config gate・
レジストリ引き・一度確定したら試合中は戻さない・reset())だけを見る
(test_match_context.py と同じ「実装を差し替えて単体を切り出す」スタイル)。
"""

from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation, PlayerState, State
from ptcg_ai.opponent_modeling import model_router

ENABLED_CONFIG = {"opponent_model_routing": {"enabled": True}}
FAKE_REGISTRY = {"crustle": "policy_weights_kamitsuorochi_ex_vs_crustle_rl.json", "ogerpon_teal_ex": None}


def _player_state():
    return PlayerState(
        active=[], bench=[], benchMax=5, deckCount=50, discard=[], prize=[None] * 6,
        handCount=5, hand=None, poisoned=False, burned=False, asleep=False,
        paralyzed=False, confused=False,
    )


def make_obs(your_index=0) -> Observation:
    players = [None, None]
    players[your_index] = _player_state()
    players[1 - your_index] = _player_state()
    state = State(
        turn=1, turnActionCount=0, yourIndex=your_index, firstPlayer=your_index,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False, retreated=False,
        result=-1, stadium=[], looking=None, players=players,
    )
    return Observation(select=None, logs=[], current=state)


def _patch(monkeypatch, *, status: str, deck_type: str, registry=FAKE_REGISTRY):
    monkeypatch.setattr(model_router, "_load_registry", lambda: registry)
    monkeypatch.setattr(
        model_router.rough_predictor, "predict",
        lambda state, knowledge: {"status": status, "deck_type": deck_type},
    )


def test_disabled_by_default_returns_none(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="crustle")
    assert model_router.route(make_obs(), None) is None
    assert model_router.route(make_obs(), {}) is None


def test_not_confident_returns_none(monkeypatch):
    model_router.reset()
    for status in ("insufficient_evidence", "ambiguous", "no_candidate"):
        _patch(monkeypatch, status=status, deck_type="crustle")
        assert model_router.route(make_obs(), ENABLED_CONFIG) is None


def test_confident_and_registered_returns_weights_path(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="crustle")
    result = model_router.route(make_obs(), ENABLED_CONFIG)
    assert result is not None
    assert result.endswith("policy_weights_kamitsuorochi_ex_vs_crustle_rl.json")


def test_confident_but_unregistered_returns_none(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="ogerpon_teal_ex")
    assert model_router.route(make_obs(), ENABLED_CONFIG) is None


def test_confident_but_unknown_archetype_returns_none(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="some_未登録_archetype")
    assert model_router.route(make_obs(), ENABLED_CONFIG) is None


def test_route_locks_in_for_rest_of_match(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="crustle")
    first = model_router.route(make_obs(), ENABLED_CONFIG)
    assert first is not None

    # rough_predictor がもう confident を返さなくなっても(=盤面が変わっても)、
    # 一度確定した対面のままであり続ける(プランの不連続を避けるため)。
    monkeypatch.setattr(
        model_router.rough_predictor, "predict",
        lambda state, knowledge: {"status": "ambiguous", "deck_type": "unknown"},
    )
    second = model_router.route(make_obs(), ENABLED_CONFIG)
    assert second == first


def test_reset_clears_lock_in(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="crustle")
    assert model_router.route(make_obs(), ENABLED_CONFIG) is not None

    model_router.reset()
    _patch(monkeypatch, status="ambiguous", deck_type="unknown")
    assert model_router.route(make_obs(), ENABLED_CONFIG) is None


def test_none_current_obs_returns_none(monkeypatch):
    model_router.reset()
    _patch(monkeypatch, status="confident", deck_type="crustle")
    obs = Observation(select=None, logs=[], current=None)
    assert model_router.route(obs, ENABLED_CONFIG) is None
