"""collect_ogerpon_counterfactuals.py の純粋関数のテスト。design.md Phase2 item2。

search_begin/search_step を使う本体のrollout処理は自己対戦を伴うため、ここでは
self-playを回さずに検証できる部分(state_idのハッシュ、config合成、相手行動選択の
フォールバック)だけを対象にする。パイプライン全体の疎通は
runs/以下へのsmokeスクリプト実行で別途確認する。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _RL_DIR.parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _p in (str(_RL_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENCODER_FIXTURE = _SAMPLE_SUBMISSION_DIR / "tests" / "fixtures" / "encoder_observations.json"


@pytest.fixture
def mod():
    try:
        import collect_ogerpon_counterfactuals as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"collect_ogerpon_counterfactuals / cg engine unavailable: {exc}")
    return m


@pytest.fixture
def real_states():
    if not _ENCODER_FIXTURE.exists():
        pytest.skip("encoder_observations.json fixture not available")
    from cg.api import to_observation_class

    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue  # 例: "rewards" は観測ペイロードではない補助データ
        obs = to_observation_class({**payload, "logs": []})
        if obs.current is not None:
            out[key] = obs.current
    if not out:
        pytest.skip("no usable state in encoder_observations.json fixture")
    return out


# --- state_id_of ------------------------------------------------------------------

def test_state_id_is_deterministic(mod, real_states):
    state = next(iter(real_states.values()))
    assert mod.state_id_of(state) == mod.state_id_of(state)


def test_state_id_differs_for_different_states(mod, real_states):
    if len(real_states) < 2:
        pytest.skip("need at least 2 distinct fixture states")
    ids = {mod.state_id_of(s) for s in real_states.values()}
    assert len(ids) == len(real_states), "異なる盤面は異なるstate_idを持つべき"


def test_state_id_is_a_sha256_hex_digest(mod, real_states):
    state = next(iter(real_states.values()))
    sid = mod.state_id_of(state)
    assert isinstance(sid, str) and len(sid) == 64
    int(sid, 16)  # ValueErrorにならないこと(16進文字列であること)


# --- _forced_planner_config ---------------------------------------------------------

def test_forced_planner_config_enables_regardless_of_input(mod):
    cfg = mod._forced_planner_config({"ogerpon_planner": {"enabled": False, "attach_enabled": False}})
    assert cfg["ogerpon_planner"]["enabled"] is True
    assert cfg["ogerpon_planner"]["attach_enabled"] is True


def test_forced_planner_config_handles_none_and_missing_key(mod):
    assert mod._forced_planner_config(None)["ogerpon_planner"]["enabled"] is True
    assert mod._forced_planner_config({})["ogerpon_planner"]["attach_enabled"] is True


def test_forced_planner_config_preserves_other_keys(mod):
    cfg = mod._forced_planner_config({"pipeline": {"enabled": True}, "ogerpon_planner": {"enabled": False}})
    assert cfg["pipeline"] == {"enabled": True}


# --- _opponent_rollout_action --------------------------------------------------------

class _StubPolicy:
    def __init__(self, scores):
        self._scores = scores

    def select_option(self, obs):
        return max(range(len(self._scores)), key=lambda i: self._scores[i])

    def score_options(self, obs):
        return list(self._scores)


def test_opponent_action_empty_select_returns_empty(mod):
    obs = SimpleNamespace(select=None)
    assert mod._opponent_rollout_action(obs, _StubPolicy([])) == []
    obs2 = SimpleNamespace(select=SimpleNamespace(option=[]))
    assert mod._opponent_rollout_action(obs2, _StubPolicy([])) == []


def test_opponent_action_single_select_uses_select_option(mod):
    select = SimpleNamespace(option=[0, 1, 2], maxCount=1, minCount=1)
    obs = SimpleNamespace(select=select)
    assert mod._opponent_rollout_action(obs, _StubPolicy([0.1, 0.9, 0.2])) == [1]


def test_opponent_action_multi_select_uses_score_ranking(mod):
    select = SimpleNamespace(option=[0, 1, 2, 3], maxCount=2, minCount=1)
    obs = SimpleNamespace(select=select)
    action = mod._opponent_rollout_action(obs, _StubPolicy([0.1, 0.9, 0.5, 0.2]))
    assert action == [1, 2]
