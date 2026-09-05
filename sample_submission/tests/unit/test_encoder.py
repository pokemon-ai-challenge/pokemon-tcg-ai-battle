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


def _obs_with_select(observations: dict, key: str):
    """logs を空にし select を保持した Observation を作る(学習データと同じ形)。"""
    from cg.api import to_observation_class

    obs_dict = {**observations[key], "logs": []}
    return to_observation_class(obs_dict)


def test_option_feature_names_length_matches_count(encoder):
    """OPTION_FEATURE_NAMES の長さと OPTION_FEATURE_COUNT が一致し、重複が無いこと。"""
    assert len(encoder.OPTION_FEATURE_NAMES) == encoder.OPTION_FEATURE_COUNT
    assert len(set(encoder.OPTION_FEATURE_NAMES)) == len(encoder.OPTION_FEATURE_NAMES)


def test_encode_options_returns_one_row_per_option(encoder, observations):
    """select.option と同じ件数・順序で、各行が OPTION_FEATURE_COUNT 長のベクトルになること。"""
    for key in _obs_keys(observations):
        obs = _obs_with_select(observations, key)
        rows = encoder.encode_options(obs)
        assert len(rows) == len(obs.select.option), key
        for row in rows:
            assert len(row) == encoder.OPTION_FEATURE_COUNT, key


def test_encode_options_no_nan_or_inf(encoder, observations):
    for key in _obs_keys(observations):
        obs = _obs_with_select(observations, key)
        for row in encoder.encode_options(obs):
            assert all(isinstance(x, float) for x in row), key
            assert all(math.isfinite(x) for x in row), key


def test_encode_options_deterministic(encoder, observations):
    for key in _obs_keys(observations):
        obs = _obs_with_select(observations, key)
        rows1 = encoder.encode_options(obs)
        rows2 = encoder.encode_options(obs)
        assert rows1 == rows2, key


def test_encode_options_empty_when_select_missing(encoder, observations):
    """select が None なら空リストを返し、例外にならないこと。"""
    from cg.api import to_observation_class

    obs_dict = {**observations["mid_game"], "logs": [], "select": None}
    obs = to_observation_class(obs_dict)
    assert encoder.encode_options(obs) == []


def test_encode_options_empty_when_current_missing(encoder):
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    assert encoder.encode_options(obs) == []


def test_encode_options_mid_game_option_types(encoder, observations):
    """mid_game フィクスチャ(PLAY/PLAY/RETREAT/END の4択)で type one-hot が選択肢ごとに
    正しく立つこと(実データでの回帰確認)。
    """
    obs = _obs_with_select(observations, "mid_game")
    rows = encoder.encode_options(obs)
    names = encoder.OPTION_FEATURE_NAMES

    expected_types = ["opttype_play", "opttype_play", "opttype_retreat", "opttype_end"]
    for row, expected in zip(rows, expected_types):
        assert row[names.index(expected)] == 1.0
        # 他の opttype_* フラグは立っていないこと。
        other_flags = [
            row[names.index(n)] for n in names if n.startswith("opttype_") and n != expected
        ]
        assert all(flag == 0.0 for flag in other_flags)


def test_encode_option_card_ids_mid_game(encoder, observations):
    """mid_game(PLAY/PLAY/RETREAT/END)で PLAY は手札カードの card_id を、
    RETREAT/END は対象が無いため 0 を返すこと。
    """
    obs = _obs_with_select(observations, "mid_game")
    ids = encoder.encode_option_card_ids(obs.current, obs.select)
    assert len(ids) == len(obs.select.option) == 4
    assert ids[0] > 0
    assert ids[1] > 0
    assert ids[0] != ids[1]
    assert ids[2] == 0  # RETREAT
    assert ids[3] == 0  # END


def test_encode_option_card_ids_empty_when_no_target(encoder, observations):
    """COUNT選択(NUMBER選択肢、対象カード/ポケモン無し)は全て0になること。"""
    obs = _obs_with_select(observations, "early_active_none")
    ids = encoder.encode_option_card_ids(obs.current, obs.select)
    assert ids == [0, 0]


def test_encode_option_card_ids_empty_when_select_missing(encoder, observations):
    from cg.api import to_observation_class

    obs_dict = {**observations["mid_game"], "logs": [], "select": None}
    obs = to_observation_class(obs_dict)
    assert encoder.encode_option_card_ids(obs.current, obs.select) == []
    assert encoder.encode_option_card_ids(None, None) == []


def test_encode_options_count_select_number_values(encoder, observations):
    """early_active_none フィクスチャ(COUNT選択、number=0/1)で number_norm が反映されること。"""
    obs = _obs_with_select(observations, "early_active_none")
    rows = encoder.encode_options(obs)
    names = encoder.OPTION_FEATURE_NAMES

    assert len(rows) == 2
    assert rows[0][names.index("number_norm")] == 0.0
    assert rows[1][names.index("number_norm")] == pytest.approx(0.1)
    for row in rows:
        assert row[names.index("opttype_number")] == 1.0
        assert row[names.index("seltype_count")] == 1.0
