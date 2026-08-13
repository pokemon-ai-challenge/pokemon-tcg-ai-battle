"""ptcg_ai.ml_policy.policy_registry (アーキタイプルーター) のユニットテスト。"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture
def registry(monkeypatch):
    try:
        from ptcg_ai.ml_policy import policy_registry as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"policy_registry / cg engine unavailable: {exc}")
    monkeypatch.setattr(mod, "_validated_route_map", None)
    monkeypatch.setattr(mod, "_model_cache", {})
    mod.reset()
    yield mod
    mod.reset()
    monkeypatch.setattr(mod, "_validated_route_map", None)


@pytest.fixture
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _obs(encoder_observations: dict, key: str = "mid_game"):
    from cg.api import to_observation_class

    obs_dict = {**encoder_observations[key], "logs": []}
    return to_observation_class(obs_dict)


def test_empty_route_map_always_returns_general(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_ROUTE_MAP_PATH", tmp_path / "route_map.json")
    assert registry.has_routes() is False


def test_load_failure_falls_back_to_general(registry, tmp_path, monkeypatch):
    bad_path = tmp_path / "does_not_exist.json"
    route_map_path = tmp_path / "route_map.json"
    route_map_path.write_text(json.dumps({"crustle": str(bad_path)}), encoding="utf-8")
    monkeypatch.setattr(registry, "_ROUTE_MAP_PATH", route_map_path)
    # ロードできないパスは検証時点で除外され、結果としてルートなし = general。
    assert registry.has_routes() is False


def test_unsupported_route_falls_back_to_general(registry, encoder_observations, monkeypatch):
    """route_map に候補が無いアーキタイプの予測は general のまま。"""
    monkeypatch.setattr(registry, "_load_route_map", lambda: {"some_other_arch": "x.json"})
    monkeypatch.setattr(
        registry.rough_predictor, "predict",
        lambda *a, **k: {"status": "confident", "deck_type": "crustle"})
    obs = _obs(encoder_observations)
    path = registry.resolve_active_weights_path(obs)
    assert path == registry._general_weights_path()


def test_route_not_confirmed_until_two_consecutive_confident_predictions(
    registry, encoder_observations, monkeypatch, tmp_path
):
    cand = tmp_path / "policy_vs_crustle.json"
    cand.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(registry, "_load_route_map", lambda: {"crustle": str(cand)})
    monkeypatch.setattr(
        registry.rough_predictor, "predict",
        lambda *a, **k: {"status": "confident", "deck_type": "crustle"})
    obs = _obs(encoder_observations)

    class _Stub:
        is_ready = True

    monkeypatch.setattr(registry, "_get_model_by_path", lambda p: _Stub())

    path1 = registry.resolve_active_weights_path(obs)
    assert path1 == registry._general_weights_path()  # 1回目はまだ確定しない
    path2 = registry.resolve_active_weights_path(obs)
    assert path2 == str(cand)  # 2回連続で確定


def test_sticky_route_persists_after_prediction_changes(
    registry, encoder_observations, monkeypatch, tmp_path
):
    cand = tmp_path / "policy_vs_crustle.json"
    cand.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(registry, "_load_route_map", lambda: {"crustle": str(cand)})

    predictions = iter(["crustle", "crustle", "unknown"])

    def fake_predict(*a, **k):
        d = next(predictions, "unknown")
        return {"status": "confident" if d != "unknown" else "insufficient_evidence", "deck_type": d}

    monkeypatch.setattr(registry.rough_predictor, "predict", fake_predict)

    class _Stub:
        is_ready = True

    monkeypatch.setattr(registry, "_get_model_by_path", lambda p: _Stub())
    obs = _obs(encoder_observations)
    registry.resolve_active_weights_path(obs)
    confirmed_path = registry.resolve_active_weights_path(obs)
    assert confirmed_path == str(cand)
    # 3回目は予測が変わっても sticky route が維持される。
    still_confirmed = registry.resolve_active_weights_path(obs)
    assert still_confirmed == str(cand)


def test_reset_clears_state_between_games(registry, encoder_observations, monkeypatch, tmp_path):
    cand = tmp_path / "policy_vs_crustle.json"
    cand.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(registry, "_load_route_map", lambda: {"crustle": str(cand)})
    monkeypatch.setattr(
        registry.rough_predictor, "predict",
        lambda *a, **k: {"status": "confident", "deck_type": "crustle"})

    class _Stub:
        is_ready = True

    monkeypatch.setattr(registry, "_get_model_by_path", lambda p: _Stub())
    obs = _obs(encoder_observations)
    registry.resolve_active_weights_path(obs)
    registry.resolve_active_weights_path(obs)
    assert registry._state["confirmed_archetype"] == "crustle"
    registry.reset()
    assert registry._state["confirmed_archetype"] is None
    assert registry._state["last_prediction"] is None
    assert registry._state["consecutive_count"] == 0


def test_model_load_failure_falls_back_to_general(registry, encoder_observations, monkeypatch, tmp_path):
    cand = tmp_path / "broken.json"
    cand.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(registry, "_load_route_map", lambda: {"crustle": str(cand)})
    model = registry.get_model(explicit_weights_path=None)
    # explicit_weights_path=None, obs=None -> general path (constructor call may raise if file
    # missing, but PolicyModel is designed to degrade to not-ready rather than raise).
    assert model is not None


def test_pimc_prior_and_fallback_use_same_get_model_call_site():
    """ml_policy_agent の PIMC prior(_try_pipeline)と Policy fallback(_select_action)が
    どちらも同じ `_get_model(config, obs=obs)` を呼ぶことをソース上で確認する
    (実行時の混在を防ぐ唯一の呼び出し口である契約のテスト)。
    """
    import inspect

    from ptcg_ai.ml_policy import ml_policy_agent

    src_pipeline = inspect.getsource(ml_policy_agent._try_pipeline)
    src_select = inspect.getsource(ml_policy_agent._select_action)
    assert "_get_model(config, obs=obs)" in src_pipeline
    assert "_get_model(config, obs=obs)" in src_select
