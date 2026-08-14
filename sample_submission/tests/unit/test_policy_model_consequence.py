"""Unit tests for PolicyModel の consequence 特徴対応(Tier3 Stage3c)。

meta.consequence_fields を持たない旧重みでの完全後方互換と、持つ重みでの新しい
score_options(obs, hidden_state_factory, deadline) 経路を検証する。

Run from sample_submission/:
    python -m pytest tests/unit/test_policy_model_consequence.py -q
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import Observation, Option, OptionType, PlayerState, SelectContext, SelectData, SelectType, State
from ptcg_ai.board_evaluation import consequence
from ptcg_ai.learning import encoder
from ptcg_ai.learning.policy_model import PolicyModel

_STATE_N = encoder.BASE_FEATURE_COUNT
_OPTION_N = encoder.OPTION_FEATURE_COUNT
_EMBED_DIM = 8


def _write_weights(path: Path, *, consequence_fields: list[str] | None, extra_dim: int) -> None:
    in_dim = _STATE_N + _OPTION_N + extra_dim + _EMBED_DIM
    weights = 1.0 if extra_dim else 0.0
    payload = {
        "meta": {
            "state_feature_count": _STATE_N,
            "option_feature_count": _OPTION_N + extra_dim,
            **({"consequence_fields": consequence_fields} if consequence_fields else {}),
        },
        "standardization": {
            "state_mean": [0.0] * _STATE_N,
            "state_std": [1.0] * _STATE_N,
            "option_mean": [0.0] * (_OPTION_N + extra_dim),
            "option_std": [1.0] * (_OPTION_N + extra_dim),
        },
        "card_embedding": {
            "dim": _EMBED_DIM,
            "card_id_max": 2000,
            "table": [[0.0] * _EMBED_DIM for _ in range(2001)],
        },
        # 隠れ層無しの単純な線形和(最終層のみ)。consequence特徴部分の重みだけ1.0にして、
        # スコアがconsequence特徴の合計値そのものになるようにする(検証しやすくするため)。
        "layers": [
            {
                "weight": [[0.0] * (_STATE_N) + [0.0] * _OPTION_N + [weights] * extra_dim + [0.0] * _EMBED_DIM],
                "bias": [0.0],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_player() -> PlayerState:
    return PlayerState(
        active=[None], bench=[], benchMax=5, deckCount=10, discard=[],
        prize=[None], handCount=0, hand=[], poisoned=False, burned=False,
        asleep=False, paralyzed=False, confused=False,
    )


def make_obs(option_types: list[OptionType]) -> Observation:
    state = State(
        turn=3, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None,
        players=[make_player(), make_player()],
    )
    select = SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=t) for t in option_types],
        deck=None, contextCard=None, effect=None,
    )
    return Observation(select=select, logs=[], current=state, search_begin_input="{}")


def test_backward_compat_no_consequence_fields(tmp_path):
    """meta.consequence_fields が無い旧重みは _consequence_fields が空で、
    score_options_from_state がそのまま使える(例外にならない)。"""
    weights_path = tmp_path / "old.json"
    _write_weights(weights_path, consequence_fields=None, extra_dim=0)

    model = PolicyModel(weights_path)
    assert model.is_ready
    assert model._consequence_fields == []

    obs = make_obs([OptionType.END])
    scores = model.score_options_from_state(obs.current, obs.select)
    assert scores == [0.0]  # 全重み0なので0


def test_score_options_from_state_rejects_consequence_weights(tmp_path):
    """meta.consequence_fields を持つ重みで score_options_from_state を呼ぶと例外
    (search_begin_input を持つ完全な Observation が必要なため)。"""
    weights_path = tmp_path / "consequence.json"
    _write_weights(weights_path, consequence_fields=["opp_hp_loss"], extra_dim=1)

    model = PolicyModel(weights_path)
    obs = make_obs([OptionType.PLAY])
    with pytest.raises(ValueError):
        model.score_options_from_state(obs.current, obs.select)


def test_score_options_without_factory_zero_fills(tmp_path, monkeypatch):
    """hidden_state_factory を渡さない場合、consequence特徴は全て0で埋まる(fail-soft)。"""
    weights_path = tmp_path / "consequence.json"
    _write_weights(weights_path, consequence_fields=["opp_hp_loss"], extra_dim=1)
    model = PolicyModel(weights_path)

    def fail_if_called(*a, **k):
        raise AssertionError("option_consequence should not be called without a factory")

    monkeypatch.setattr(consequence, "option_consequence", fail_if_called)

    obs = make_obs([OptionType.PLAY])
    scores = model.score_options(obs)  # hidden_state_factory=None
    assert scores == [0.0]  # 重み1.0 * consequence特徴0 = 0


def test_score_options_with_factory_uses_consequence_value(tmp_path, monkeypatch):
    """hidden_state_factory を渡すと consequence 特徴が計算され、スコアに反映される。"""
    weights_path = tmp_path / "consequence.json"
    _write_weights(weights_path, consequence_fields=["opp_hp_loss"], extra_dim=1)
    model = PolicyModel(weights_path)

    class FakeResult:
        opp_hp_loss = 42
        self_hp_gain = 0
        opp_energy_removed = False
        opp_special_energy_removed = False
        self_energy_added = False
        cards_drawn = 0
        pokemon_evolved = False
        stadium_changed = False
        delta_best_effective_attack_damage = 0
        delta_can_ko = False
        delta_attack_ready = False
        delta_energy_shortfall = 0.0

    monkeypatch.setattr(consequence, "option_consequence", lambda obs, i, f, d: FakeResult())

    obs = make_obs([OptionType.PLAY])
    scores = model.score_options(obs, hidden_state_factory=lambda: {}, deadline=time.perf_counter() + 1)
    # 重み1.0 * opp_hp_loss(42) = 42(他の特徴は全て重み0)。
    assert scores == [42.0]


def test_select_option_threads_factory_through(tmp_path, monkeypatch):
    weights_path = tmp_path / "consequence.json"
    _write_weights(weights_path, consequence_fields=["opp_hp_loss"], extra_dim=1)
    model = PolicyModel(weights_path)

    class FakeResult:
        opp_hp_loss = 10
        self_hp_gain = 0
        opp_energy_removed = False
        opp_special_energy_removed = False
        self_energy_added = False
        cards_drawn = 0
        pokemon_evolved = False
        stadium_changed = False
        delta_best_effective_attack_damage = 0
        delta_can_ko = False
        delta_attack_ready = False
        delta_energy_shortfall = 0.0

    monkeypatch.setattr(consequence, "option_consequence", lambda obs, i, f, d: FakeResult())

    obs = make_obs([OptionType.END, OptionType.PLAY])  # index1がPLAY(consequence対象)
    idx = model.select_option(obs, hidden_state_factory=lambda: {}, deadline=time.perf_counter() + 1)
    assert idx == 1  # ENDは0点、PLAYは10点なのでPLAYが選ばれる
