"""ptcg_ai.learning.value_model.ValueModel のユニットテスト。

ゴールデンテスト: 学習パイプラインが出力した ``kaggle_replays/value_net/sample_predictions.json``
(val split から30件、各行に元 observation dict と較正後の最終予測確率 expected_win_prob)を
読み込み、``ValueModel.predict_win_prob_from_dict(row["observation"])`` が
``row["expected_win_prob"]`` と 1e-4 以内で一致することを確認する。これがフォワードパス
(標準化・W の向き・活性化)+ ターン帯較正の実装が学習側と一致していることの担保になる。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_REPO_ROOT = SAMPLE_SUBMISSION_ROOT.parent
_SAMPLE_PREDICTIONS = _REPO_ROOT / "kaggle_replays" / "value_net" / "sample_predictions.json"
_WEIGHTS = SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "value_weights.json"

_TOL = 1e-4


@pytest.fixture(scope="module")
def value_model():
    """デフォルト重み(value_weights.json)をロードした ValueModel。cg が無ければスキップ。"""
    try:
        from ptcg_ai.learning.value_model import ValueModel
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"value_model / cg engine unavailable: {exc}")
    model = ValueModel()
    if not model.is_ready:
        pytest.skip(f"value_weights.json not found at {_WEIGHTS}")
    return model


@pytest.fixture(scope="module")
def sample_predictions() -> list[dict]:
    if not _SAMPLE_PREDICTIONS.exists():
        pytest.skip(f"sample_predictions.json not found at {_SAMPLE_PREDICTIONS}")
    with _SAMPLE_PREDICTIONS.open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_predictions_match(value_model, sample_predictions):
    """30件すべてで較正後予測が expected_win_prob と 1e-4 以内で一致する。"""
    assert len(sample_predictions) > 0
    mismatches = []
    for row in sample_predictions:
        got = value_model.predict_win_prob_from_dict(row["observation"])
        expected = row["expected_win_prob"]
        if abs(got - expected) > _TOL:
            mismatches.append(
                {
                    "episode_id": row.get("episode_id"),
                    "turn": row.get("turn"),
                    "got": got,
                    "expected": expected,
                    "diff": abs(got - expected),
                }
            )
    assert not mismatches, (
        f"{len(mismatches)}/{len(sample_predictions)} 件が許容誤差 {_TOL} を超過: "
        + json.dumps(mismatches[:5], ensure_ascii=False)
    )


def test_prediction_in_unit_interval(value_model, sample_predictions):
    """全予測が [0, 1] に収まる(sigmoid 出力 + 較正の健全性)。"""
    for row in sample_predictions:
        got = value_model.predict_win_prob_from_dict(row["observation"])
        assert 0.0 <= got <= 1.0


def test_missing_weights_is_safe():
    """重みファイルが存在しないパスを渡すと is_ready=False で 0.5 を返し、例外を出さない。"""
    from ptcg_ai.learning.value_model import ValueModel

    model = ValueModel(weights_path=SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "__no_such_weights__.json")
    assert model.is_ready is False
    # obs_dict / Observation どちらの経路でも例外を出さず 0.5。
    assert model.predict_win_prob_from_dict({"current": {"turn": 5}}) == 0.5

    from cg.api import to_observation_class

    obs = to_observation_class({"logs": [], "select": None, "current": None})
    assert model.predict_win_prob(obs) == 0.5


def test_current_none_returns_half(value_model):
    """盤面未確定(current=None)の obs_dict は 0.5 を返す。"""
    assert value_model.predict_win_prob_from_dict({"current": None}) == 0.5
    assert value_model.predict_win_prob_from_dict({}) == 0.5
