"""クラスタ⑨ 検証（学習基盤）／状態エンコーダ

ptcg_ai.learning.encoder の単体テスト。実リプレイから取り出した observation dict を
フィクスチャ(tests/fixtures/encoder_observations.json)として同梱し、以下を確認する:

- ベクトル長が FEATURE_NAMES と一致し、複数局面で不変であること
- NaN / inf を含まないこと
- logs / select を削除した dict と元の dict で出力が一致すること(学習データは logs 削除済み)
- active が None(伏せ)・ベンチ空のケースで例外にならないこと

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import copy
import json
import math
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture(scope="module")
def observations() -> dict:
    """実リプレイ由来の observation dict 群(mid_game / early_active_none)。"""
    with _FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def encoder():
    """encoder モジュール。cg エンジンがロードできなければスキップ。"""
    try:
        from ptcg_ai.learning import encoder as enc
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder / cg engine unavailable: {exc}")
    return enc


def _obs_keys(observations: dict) -> list[str]:
    return [k for k in ("mid_game", "early_active_none") if k in observations]


def test_feature_names_length_matches_base_count(encoder):
    """FEATURE_NAMES の長さと BASE_FEATURE_COUNT が一致する。"""
    assert len(encoder.FEATURE_NAMES) == encoder.BASE_FEATURE_COUNT
    # 名前に重複が無い(学習側で係数解釈に使うため)。
    assert len(set(encoder.FEATURE_NAMES)) == len(encoder.FEATURE_NAMES)


def test_vector_length_is_fixed_across_positions(encoder, observations):
    """複数局面でベクトル長が FEATURE_NAMES と一致し不変であること。"""
    for key in _obs_keys(observations):
        vec = encoder.encode_obs_dict(observations[key])
        assert len(vec) == len(encoder.FEATURE_NAMES), key


def test_no_nan_or_inf(encoder, observations):
    """出力に NaN / inf が含まれないこと。"""
    for key in _obs_keys(observations):
        vec = encoder.encode_obs_dict(observations[key])
        assert all(isinstance(x, float) for x in vec), key
        assert all(math.isfinite(x) for x in vec), key


def test_logs_and_select_do_not_affect_output(encoder, observations):
    """logs / select を削除しても出力が一致すること(学習データは logs 削除済み)。"""
    for key in _obs_keys(observations):
        original = observations[key]
        stripped = copy.deepcopy(original)
        stripped.pop("logs", None)
        stripped.pop("select", None)

        vec_full = encoder.encode_obs_dict(original)
        vec_stripped = encoder.encode_obs_dict(stripped)
        assert vec_full == vec_stripped, key


def test_deterministic_same_input_same_output(encoder, observations):
    """同じ入力に常に同じ出力を返すこと。"""
    for key in _obs_keys(observations):
        v1 = encoder.encode_obs_dict(observations[key])
        v2 = encoder.encode_obs_dict(observations[key])
        assert v1 == v2, key


def test_active_none_and_empty_bench_do_not_raise(encoder, observations):
    """active が None(伏せ)・ベンチ空のケースで例外にならず、当該スロットがゼロ埋めされること。"""
    obs = observations["early_active_none"]
    vec = encoder.encode_obs_dict(obs)
    names = encoder.FEATURE_NAMES

    # 伏せ active は present=0 でゼロ埋め。
    assert vec[names.index("self_active_present")] == 0.0
    assert vec[names.index("self_active_hp_ratio")] == 0.0
    # 空ベンチの各スロットも present=0。
    assert vec[names.index("self_bench0_present")] == 0.0
    assert vec[names.index("opp_bench0_present")] == 0.0


def test_encode_state_matches_encode_obs_dict(encoder, observations):
    """encode_state(Observation) と encode_obs_dict(dict) が同一の出力になること(同一コードパス)。"""
    from cg.api import to_observation_class

    for key in _obs_keys(observations):
        obs_dict = observations[key]
        vec_dict = encoder.encode_obs_dict(obs_dict)
        vec_state = encoder.encode_state(to_observation_class(obs_dict))
        assert vec_dict == vec_state, key


def test_extra_features_appended_at_tail(encoder, observations):
    """extra_features は基本ベクトルの末尾へそのまま連結される(FEATURE_NAMES には含まれない)。"""
    obs = observations["mid_game"]
    base = encoder.encode_obs_dict(obs)
    extra = [0.5, -1.25, 3.0]
    with_extra = encoder.encode_obs_dict(obs, extra_features=extra)
    assert len(with_extra) == len(base) + len(extra)
    assert with_extra[: len(base)] == base
    assert with_extra[len(base):] == extra


def test_current_none_returns_zero_vector(encoder):
    """current が None(初回デッキ選択など)ではゼロベクトルを返し、壊れないこと。"""
    vec = encoder.encode_obs_dict({"current": None, "logs": [], "select": None})
    assert len(vec) == encoder.BASE_FEATURE_COUNT
    assert all(x == 0.0 for x in vec)


def test_missing_logs_key_is_tolerated(encoder, observations):
    """logs キー自体が欠損した dict でも例外にならず変換できること(学習データ想定)。"""
    obs = copy.deepcopy(observations["mid_game"])
    obs.pop("logs", None)
    vec = encoder.encode_obs_dict(obs)
    assert len(vec) == encoder.BASE_FEATURE_COUNT


def test_encode_options_is_stub(encoder, observations):
    """encode_options は Step2 スコープで未実装(NotImplementedError)であること。"""
    from cg.api import to_observation_class

    obs = to_observation_class(
        {**observations["mid_game"], "logs": [], "select": observations["mid_game"].get("select")}
    )
    with pytest.raises(NotImplementedError):
        encoder.encode_options(obs)
