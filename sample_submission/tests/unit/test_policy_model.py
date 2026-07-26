"""``ptcg_ai.learning.policy_model.PolicyModel`` の推論を検証する。

設計は ``ml_predictor.MLDeckPredictor`` / ``value_model.ValueModel`` と同じ:
重みファイルが無ければ例外を出さず「未ロード」のまま無害に動作する。
"""

import json
from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.learning.policy_model import PolicyModel


def _actions(n: int) -> list[dict]:
    return [
        {"option_type": 7, "from_area": None, "from_player": None, "card_id": None,
         "target_card_id": None, "count": None, "number": None, "attack_id": None}
        for _ in range(n)
    ]


# --- 未ロード時の安全なフォールバック ---------------------------------------


def test_is_ready_false_when_weights_file_missing(tmp_path):
    model = PolicyModel(weights_path=tmp_path / "does_not_exist.json")
    assert model.is_ready is False


def test_score_options_returns_neutral_scores_when_not_ready(tmp_path):
    model = PolicyModel(weights_path=tmp_path / "missing.json")
    scores = model.score_options({}, _actions(3))
    assert scores == [0.0, 0.0, 0.0]


def test_select_returns_none_when_not_ready(tmp_path):
    model = PolicyModel(weights_path=tmp_path / "missing.json")
    assert model.select({}, _actions(3)) is None


def test_select_returns_none_on_empty_actions(tmp_path):
    weights_path = tmp_path / "policy_weights.json"
    weights_path.write_text(
        json.dumps({"feature_names": [], "weights": [], "intercept": 0.0}),
        encoding="utf-8",
    )
    model = PolicyModel(weights_path=weights_path)
    assert model.is_ready is True
    assert model.select({}, []) is None


def test_malformed_weights_file_is_treated_as_not_ready(tmp_path):
    """壊れた重みJSON(例: feature_names と weights の長さ不一致)でも例外にしない。"""
    weights_path = tmp_path / "policy_weights.json"
    weights_path.write_text(
        json.dumps({"feature_names": ["a", "b"], "weights": [1.0], "intercept": 0.0}),
        encoding="utf-8",
    )
    model = PolicyModel(weights_path=weights_path)
    assert model.is_ready is False


# --- ロード成功時のスコアリング ---------------------------------------------


def _write_weights(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "model": "linear_pointwise",
        "feature_names": ["turn", "option_type_7", "card_attr_hp"],
        "weights": [0.1, 1.0, 0.01],
        "intercept": 0.0,
        "frequent_card_ids": [7],
        "card_attributes": {"7": {"hp": 60.0}},
        "meta": {"train_rows": 100, "test_accuracy": 0.5},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_score_options_matches_number_of_actions(tmp_path):
    weights_path = tmp_path / "policy_weights.json"
    _write_weights(weights_path)
    model = PolicyModel(weights_path=weights_path)
    assert model.is_ready is True

    state = {"turn": 5, "own": {}, "opponent": {}}
    actions = _actions(4)
    scores = model.score_options(state, actions)
    assert len(scores) == len(actions)
    assert all(isinstance(s, float) for s in scores)


def test_select_picks_the_highest_scoring_option(tmp_path):
    weights_path = tmp_path / "policy_weights.json"
    _write_weights(weights_path)
    model = PolicyModel(weights_path=weights_path)

    state = {"turn": 1, "own": {}, "opponent": {}}
    low = {"option_type": 14, "from_area": None, "from_player": None, "card_id": None,
           "target_card_id": None, "count": None, "number": None, "attack_id": None}
    high = {"option_type": 7, "from_area": None, "from_player": None, "card_id": 7,
            "target_card_id": None, "count": None, "number": None, "attack_id": None}
    assert model.select(state, [low, high]) == 1


def test_unrecognised_feature_names_are_ignored(tmp_path):
    """学習データに現れなかった特徴名は無視され、例外にならないこと。"""
    weights_path = tmp_path / "policy_weights.json"
    _write_weights(weights_path)
    model = PolicyModel(weights_path=weights_path)
    score = model.score({"totally_unknown_feature": 999.0})
    assert score == 0.0  # intercept のみ
