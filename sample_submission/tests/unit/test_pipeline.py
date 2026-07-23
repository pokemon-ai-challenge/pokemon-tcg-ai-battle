"""ptcg_ai.search.pipeline のユニットテスト。

実 cg エンジンで完全な先読みを回すと重く非決定的なので、Step3〜5(決定化ループ・
末端評価・平均集約)は cg_api を軽量なフェイクに差し替えて決定的に検証する。Step2
(top-k 絞り込み・top1 集中の即返し・適用範囲ゲート)は実 obs フィクスチャで検証する。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture(scope="module")
def pipeline():
    try:
        from ptcg_ai.search import pipeline as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pipeline / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def fixtures() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _obs(fixtures: dict, key: str):
    from cg.api import to_observation_class

    return to_observation_class({**fixtures[key], "logs": []})


class _StubPolicy:
    """score_options / score_options_from_state を固定スコアで返すスタブ。"""

    def __init__(self, scores):
        self._scores = scores

    def score_options(self, obs, factory=None, deadline=None):
        return list(self._scores)

    def score_options_from_state(self, state, select):
        return list(self._scores)


def _base_config(**overrides):
    cfg = {"enabled": True, "num_determinizations": 2, "top_k": 4, "top1_shortcut_prob": 0.9}
    cfg.update(overrides)
    return cfg


def test_disabled_returns_none(pipeline, fixtures):
    obs = _obs(fixtures, "mid_game")
    ctx = {"observation": obs, "config": {"enabled": False}, "policy_model": _StubPolicy([1, 0, 0, 0])}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None


def test_out_of_scope_select_returns_none(pipeline, fixtures):
    """MAIN 以外の選択(early_active_none: type=8)は enabled でも None を返す。"""
    obs = _obs(fixtures, "early_active_none")
    ctx = {"observation": obs, "config": _base_config(), "policy_model": _StubPolicy([1, 0])}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None


def test_top1_shortcut_skips_search(pipeline, fixtures):
    """top1 が top1_shortcut_prob 以上に集中していれば、決定化(factory)を呼ばずに top1 を返す。"""
    obs = _obs(fixtures, "mid_game")

    def _factory_must_not_be_called():
        raise AssertionError("hidden_state_factory must not be called on the top1 shortcut path")

    ctx = {
        "observation": obs,
        "config": _base_config(),
        "policy_model": _StubPolicy([10.0, 0.0, 0.0, 0.0]),  # softmax top1 ~ 1.0
        "hidden_state_factory": _factory_must_not_be_called,
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [0]


def _fake_cg(result_by_first_move, me, your_index_leaf=None):
    """search_begin/step/release/end のフェイク。search_step の最初の呼び出し(root からの
    first move)で、その選択インデックスに応じた result を持つ末端ノードを返す。
    """
    def search_begin(obs, *args, **kwargs):
        return SimpleNamespace(searchId="root", observation=obs)

    def search_step(search_id, selection):
        idx = selection[0]
        result = result_by_first_move.get(idx, -1)
        yi = your_index_leaf if your_index_leaf is not None else me
        leaf_state = SimpleNamespace(
            result=result, yourIndex=yi, players=[None, None],
        )
        leaf_obs = SimpleNamespace(current=leaf_state, select=None)
        return SimpleNamespace(searchId=f"n{idx}", observation=leaf_obs)

    def search_release(search_id):
        pass

    def search_end():
        pass

    return SimpleNamespace(
        search_begin=search_begin, search_step=search_step,
        search_release=search_release, search_end=search_end,
    )


def test_averaging_picks_highest_eval_candidate(pipeline, fixtures, monkeypatch):
    """候補1が全決定化で自分の勝ち(1.0)、他が相手勝ち(0.0)なら候補1を選ぶ。

    末端評価は handcrafted(result==me→1.0, result==other→0.0)。search_step が即決着
    ノードを返すので rollout は1手で終わる。
    """
    obs = _obs(fixtures, "mid_game")
    me = obs.current.yourIndex
    opp = 1 - me
    # first-move index 1 -> 自分の勝ち、他 -> 相手の勝ち。
    fake = _fake_cg({0: opp, 1: me, 2: opp, 3: opp}, me=me)
    monkeypatch.setattr(pipeline, "cg_api", fake)

    ctx = {
        "observation": obs,
        "config": _base_config(num_determinizations=3),
        "policy_model": _StubPolicy([1.0, 0.9, 0.8, 0.1]),  # 集中していない(top1<0.9)
        "hidden_state_factory": lambda: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                          "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [1]


def test_tie_break_prefers_higher_policy(pipeline, fixtures, monkeypatch):
    """全候補が同じ末端評価(引き分け近傍)なら Policy スコア上位を採用する。"""
    obs = _obs(fixtures, "mid_game")
    me = obs.current.yourIndex
    # どの first move も result 未決(-1)→ leaf は select=None の中立盤面 → 全候補同スコア。
    fake = _fake_cg({}, me=me)
    monkeypatch.setattr(pipeline, "cg_api", fake)

    ctx = {
        "observation": obs,
        "config": _base_config(num_determinizations=2, tie_eps=1.0),
        "policy_model": _StubPolicy([0.1, 5.0, 0.2, 0.3]),  # index1 が最高
        "hidden_state_factory": lambda: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                          "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [1]


def test_missing_policy_model_returns_none(pipeline, fixtures):
    obs = _obs(fixtures, "mid_game")
    ctx = {"observation": obs, "config": _base_config()}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None
