"""ptcg_ai.ml_policy.ml_policy_agent のユニットテスト。

`rule_based/action_selection` を一切呼ばずに、選択肢インデックスを返せることを確認する
(step2-design.md §1 の接続点方針: rule_based とは独立に完結する)。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
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
def ml_policy_agent():
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"ml_policy_agent / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _obs(encoder_observations: dict, key: str, select_override=None):
    from cg.api import to_observation_class

    obs_dict = {**encoder_observations[key], "logs": []}
    if select_override is not None:
        obs_dict["select"] = select_override
    return to_observation_class(obs_dict)


def test_deck_selection_returns_60_card_ids(ml_policy_agent):
    """select が None(初回)なら deck.csv 由来の60枚のカードIDリストを返す。"""
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    deck = ml_policy_agent.agent(obs)
    assert len(deck) == 60
    assert all(isinstance(card_id, int) for card_id in deck)


def test_single_choice_returns_one_valid_index(ml_policy_agent, encoder_observations):
    """maxCount==1 の局面(mid_game / early_active_none)で範囲内の1要素リストを返す。"""
    for key in ("mid_game", "early_active_none"):
        obs = _obs(encoder_observations, key)
        result = ml_policy_agent.agent(obs)
        assert len(result) == 1
        assert 0 <= result[0] < len(obs.select.option)


def test_multi_select_greedy_fallback_respects_min_max_count(ml_policy_agent, encoder_observations):
    """maxCount>1(Step2学習スコープ外)は minCount〜maxCount 件・重複無しで返す。"""
    base = encoder_observations["mid_game"]
    select = dict(base["select"])
    # mid_game は4択(PLAY,PLAY,RETREAT,END)。minCount=2,maxCount=3 の複数選択に仕立てる。
    select["minCount"] = 2
    select["maxCount"] = 3

    obs = _obs(encoder_observations, "mid_game", select_override=select)
    result = ml_policy_agent.agent(obs)

    assert 2 <= len(result) <= 3
    assert len(set(result)) == len(result)
    assert all(0 <= i < len(obs.select.option) for i in result)
