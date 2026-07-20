"""ptcg_ai.learning.policy_model.PolicyModel のユニットテスト。

ゴールデンテスト: 独立実装(numpy)で計算した選択肢スコアを収めた
``tests/fixtures/policy_model_predictions.json``(生成手順は同ファイル生成スクリプト参照)を
読み込み、``PolicyModel.score_options`` が同じ入力に対して 1e-6 以内で一致することを
確認する。これがフォワードパス(標準化・重み行列の向き・状態/選択肢特徴の連結順)の実装が
学習側の重みJSONスキーマと一致していることの担保になる(value_model.py の golden test と
同じ狙い)。fixture が未生成の場合はスキップする。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "policy_model_predictions.json"
_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"
_WEIGHTS = SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "policy_weights.json"

_TOL = 1e-6


@pytest.fixture(scope="module")
def policy_model():
    """デフォルト重み(policy_weights.json)をロードした PolicyModel。cg が無ければスキップ。"""
    try:
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"policy_model / cg engine unavailable: {exc}")
    model = PolicyModel()
    if not model.is_ready:
        pytest.skip(f"policy_weights.json not found at {_WEIGHTS}")
    return model


@pytest.fixture(scope="module")
def golden_predictions() -> list[dict]:
    if not _FIXTURE.exists():
        pytest.skip(f"policy_model_predictions.json not found at {_FIXTURE}")
    with _FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_scores_match(policy_model, golden_predictions):
    """独立実装(numpy)のスコアと 1e-6 以内で一致する(全選択肢)。"""
    from cg.api import to_observation_class

    assert len(golden_predictions) > 0
    mismatches = []
    for row in golden_predictions:
        obs = to_observation_class({**row["observation"], "logs": []})
        got = policy_model.score_options(obs)
        expected = row["expected_scores"]
        if len(got) != len(expected) or any(abs(g - e) > _TOL for g, e in zip(got, expected)):
            mismatches.append({"episode_id": row.get("episode_id"), "got": got, "expected": expected})
    assert not mismatches, (
        f"{len(mismatches)}/{len(golden_predictions)} 件が許容誤差 {_TOL} を超過: "
        + json.dumps(mismatches[:3], ensure_ascii=False)
    )


def test_missing_weights_is_safe():
    """重みファイルが存在しないパスを渡すと is_ready=False になり、例外を出さない。"""
    from cg.api import to_observation_class
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel(
        weights_path=SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "__no_such_weights__.json"
    )
    assert model.is_ready is False

    obs = to_observation_class({"logs": [], "select": None, "current": None})
    assert model.score_options(obs) == []
    assert model.select_option(obs) is None


def test_missing_weights_select_option_falls_back_to_zero(encoder_observations):
    """未ロードでも select.option があれば index 0 を返す(常に有効な選択を返す安全側)。"""
    from cg.api import to_observation_class
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel(
        weights_path=SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "__no_such_weights__.json"
    )
    obs = to_observation_class({**encoder_observations["mid_game"], "logs": []})
    assert model.select_option(obs) == 0


def test_select_option_returns_valid_index(policy_model, encoder_observations):
    """実データ(mid_game / early_active_none)で選択肢の範囲内のインデックスを返すこと。"""
    from cg.api import to_observation_class

    for key in ("mid_game", "early_active_none"):
        obs = to_observation_class({**encoder_observations[key], "logs": []})
        idx = policy_model.select_option(obs)
        assert idx is not None
        assert 0 <= idx < len(obs.select.option)


def test_select_option_none_when_no_select(policy_model):
    """select が None の場合は None を返す(例外にならない)。"""
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    assert policy_model.select_option(obs) is None
    assert policy_model.score_options(obs) == []
