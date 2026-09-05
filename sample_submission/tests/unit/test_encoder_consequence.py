"""Unit tests for ptcg_ai.learning.encoder.encode_option_consequence_features (Tier3 Stage3c).

consequence.option_consequence 自体のロジックは tests/unit/test_consequence.py で
検証済みのため、ここでは encoder 側の「選択肢の型で対象を絞る」「解決不能/対象外は
全特徴0で埋める」というディスパッチだけを、consequence.option_consequence を
monkeypatch した軽量フェイクで検証する。

Run from sample_submission/:
    python -m pytest tests/unit/test_encoder_consequence.py -q
"""

from __future__ import annotations

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


def make_player() -> PlayerState:
    return PlayerState(
        active=[None], bench=[], benchMax=5, deckCount=10, discard=[],
        prize=[None], handCount=0, hand=[], poisoned=False, burned=False,
        asleep=False, paralyzed=False, confused=False,
    )


def make_state() -> State:
    return State(
        turn=3, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None,
        players=[make_player(), make_player()],
    )


def make_select(option_types: list[OptionType]) -> SelectData:
    return SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=t) for t in option_types],
        deck=None, contextCard=None, effect=None,
    )


def make_obs(select: SelectData | None) -> Observation:
    return Observation(select=select, logs=[], current=make_state(), search_begin_input="{}")


def hidden_factory():
    return {}


def test_non_target_option_types_are_zero_filled(monkeypatch):
    """ATTACK/ENDのようなconsequence対象外の型は、option_consequenceを呼ばず全特徴0。"""
    calls = []

    def fake_option_consequence(obs, i, factory, deadline):
        calls.append(i)
        raise AssertionError("should not be called for non-target option types")

    monkeypatch.setattr(consequence, "option_consequence", fake_option_consequence)

    obs = make_obs(make_select([OptionType.ATTACK, OptionType.END]))
    result = encoder.encode_option_consequence_features(obs, hidden_factory, time.perf_counter() + 1)

    assert len(result) == 2
    for vec in result:
        assert len(vec) == encoder.CONSEQUENCE_FEATURE_COUNT
        assert vec == [0.0] * encoder.CONSEQUENCE_FEATURE_COUNT
    assert calls == []


def test_target_option_type_uses_option_consequence(monkeypatch):
    """ATTACH/EVOLVE/PLAYはoption_consequenceの結果がベクトルに反映される。"""
    class FakeResult:
        opp_hp_loss = 0
        self_hp_gain = 0
        opp_energy_removed = True
        opp_special_energy_removed = True
        self_energy_added = False
        cards_drawn = 0
        pokemon_evolved = False
        stadium_changed = False
        delta_best_effective_attack_damage = 120
        delta_can_ko = False
        delta_attack_ready = False
        delta_energy_shortfall = 0.0

    def fake_option_consequence(obs, i, factory, deadline):
        assert i == 0
        return FakeResult()

    monkeypatch.setattr(consequence, "option_consequence", fake_option_consequence)

    obs = make_obs(make_select([OptionType.PLAY]))
    result = encoder.encode_option_consequence_features(obs, hidden_factory, time.perf_counter() + 1)

    assert len(result) == 1
    vec = result[0]
    names = encoder.CONSEQUENCE_FEATURE_NAMES
    as_dict = dict(zip(names, vec))
    assert as_dict["opp_special_energy_removed"] == 1.0
    assert as_dict["opp_energy_removed"] == 1.0
    assert as_dict["delta_best_effective_attack_damage"] == 120.0
    assert as_dict["delta_can_ko"] == 0.0


def test_unresolved_is_zero_filled(monkeypatch):
    """option_consequenceがNone(解決不能)を返したら全特徴0(fail-soft)。"""
    monkeypatch.setattr(consequence, "option_consequence", lambda obs, i, factory, deadline: None)

    obs = make_obs(make_select([OptionType.ATTACH]))
    result = encoder.encode_option_consequence_features(obs, hidden_factory, time.perf_counter() + 1)

    assert result == [[0.0] * encoder.CONSEQUENCE_FEATURE_COUNT]


def test_exception_is_zero_filled(monkeypatch):
    """option_consequenceが例外を投げても呼び出し側は止めない(fail-soft)。"""
    def boom(obs, i, factory, deadline):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(consequence, "option_consequence", boom)

    obs = make_obs(make_select([OptionType.EVOLVE]))
    result = encoder.encode_option_consequence_features(obs, hidden_factory, time.perf_counter() + 1)

    assert result == [[0.0] * encoder.CONSEQUENCE_FEATURE_COUNT]


def test_empty_when_no_select():
    obs = make_obs(None)
    assert encoder.encode_option_consequence_features(obs, hidden_factory, time.perf_counter() + 1) == []


def test_vector_length_matches_names():
    assert encoder.CONSEQUENCE_FEATURE_COUNT == len(encoder.CONSEQUENCE_FEATURE_NAMES)
