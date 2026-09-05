"""opponent_belief_features (belief-conditioned RL, docs/plans/belief-conditioned-rl-plan.md) の
parity/決定性テスト。

parity の肝は「学習側(collect_field)と推論側(PolicyModel)が同一 predictor・同一関数を呼ぶ」ことと、
その関数が現在 State のみの純関数で**決定的**であること。ここでは決定性(同一 State→同一ベクトル)・
形状(len==classes数)・確率制約を確認する。実 predictor 重みが無い環境では skip。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture(scope="module")
def deps():
    try:
        from cg.api import to_observation_class
        from ptcg_ai.hidden_information import match_context
        from ptcg_ai.learning.extra_features import opponent_belief_features
    except Exception as exc:  # noqa: BLE001 - cg/依存が無い環境ではスキップ
        pytest.skip(f"deps unavailable: {exc}")
    return opponent_belief_features, match_context, to_observation_class


def _state(to_obs):
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        d = json.load(f)
    obs = to_obs({**d["mid_game"], "logs": []})
    return obs.current


def test_belief_features_deterministic_and_shaped(deps):
    opponent_belief_features, match_context, to_obs = deps
    predictor = match_context._get_predictor()
    if predictor is None or not getattr(predictor, "is_ready", False):
        pytest.skip("deck predictor not available/ready in this env")
    state = _state(to_obs)
    v1 = opponent_belief_features(state, predictor)
    v2 = opponent_belief_features(state, predictor)
    n = len(predictor._classes)
    assert n > 0
    assert len(v1) == n
    # parity の核心: 同一 State -> 同一ベクトル(学習側/推論側で一致する保証)。
    assert v1 == v2
    assert all(0.0 <= x <= 1.0 for x in v1)
    assert sum(v1) <= 1.0 + 1e-6


def test_belief_features_safe_on_missing_predictor():
    from ptcg_ai.learning.extra_features import opponent_belief_features
    # predictor None(classes 取得不可) -> 空。State None も例外なく安全側。
    assert opponent_belief_features(None, None) == []
