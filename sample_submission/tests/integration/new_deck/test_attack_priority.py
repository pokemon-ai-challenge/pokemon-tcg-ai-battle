"""クラスタ⑨ 検証（デッキ固有シナリオ）／担当A下書き→担当Bとすり合わせ

このデッキの攻撃選択（リーサル優先、複数ワザがある場合の選び方など）を固定シナリオで検証する。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Option, OptionType, PlayerState, Pokemon, State
from ptcg_ai.rule_based.main_turn_parts.priorities import attack as attack_priority
from ptcg_ai.shared.profile_types import MatchupPlan


def test_lethal_attack_is_preferred():
    """相手をきぜつさせられるワザがある場合、それが最優先で選ばれることを確認する。"""
    pytest.skip("TODO: 新デッキ確定・③実装後にシナリオを書く")


def _make_pokemon(**kwargs) -> Pokemon:
    defaults = dict(
        id=1, serial=1, hp=200, maxHp=200, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]
    )
    defaults.update(kwargs)
    return Pokemon(**defaults)


def _make_player_state(**kwargs) -> PlayerState:
    defaults = dict(
        active=[], bench=[], benchMax=5, deckCount=50, discard=[], prize=[None] * 6,
        handCount=5, hand=None, poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )
    defaults.update(kwargs)
    return PlayerState(**defaults)


class _FakeCardData:
    """attack.py が defender_card.weakness/resistance を読むためだけのスタブ。"""

    weakness = None
    resistance = None


def _make_state(attacker: Pokemon, defender: Pokemon) -> State:
    players = [_make_player_state(active=[attacker]), _make_player_state(active=[defender])]
    return State(
        turn=1, turnActionCount=0, yourIndex=0, firstPlayer=0, supporterPlayed=False, stadiumPlayed=False,
        energyAttached=False, retreated=False, result=-1, stadium=[], looking=None, players=players,
    )


def test_matchup_plan_boost_breaks_a_damage_tie_between_attacks(monkeypatch):
    """opponent_tracker.current_matchup_plan() に加点があるワザは、ダメージが同点でも優先される。"""
    attacker = _make_pokemon(id=743, serial=1)
    defender = _make_pokemon(id=999, serial=2, hp=200)
    state = _make_state(attacker, defender)

    no_boost_attack_id, boosted_attack_id = 100, 200
    options = [
        Option(type=OptionType.ATTACK, attackId=no_boost_attack_id),
        Option(type=OptionType.ATTACK, attackId=boosted_attack_id),
    ]

    class _FakeObs:
        current = state

        class select:
            option = options

    # 両ワザともダメージ・追加効果は同点（=50）にし、matchup_plan の加点だけが差になるようにする。
    monkeypatch.setattr(attack_priority.card_cache, "get_card", lambda card_id: _FakeCardData())
    monkeypatch.setattr(attack_priority.card_cache, "get_attack", lambda attack_id: attack_id)
    monkeypatch.setattr(attack_priority.energy_requirements, "is_energy_sufficient", lambda attack, energies: True)
    monkeypatch.setattr(
        attack_priority.attack_features,
        "resolve_damage",
        lambda attack, attacker, weakness, resistance, hand_size: 50,
    )
    monkeypatch.setattr(attack_priority.profile_registry, "get_attack_profile", lambda attack_id: None)
    monkeypatch.setattr(
        attack_priority.opponent_tracker,
        "current_matchup_plan",
        lambda: MatchupPlan(attack_priority_boost={boosted_attack_id: 20.0}),
    )

    proposal = attack_priority.propose(_FakeObs())

    assert proposal is not None
    assert proposal.select == [1]  # boosted_attack_id (index 1) が同ダメージのもう一方より優先される


def test_matchup_plan_none_falls_back_to_damage_only_scoring(monkeypatch):
    """current_matchup_plan() が None（未確信/対策データ無し）なら加点は発生しない。"""
    attacker = _make_pokemon(id=743, serial=1)
    defender = _make_pokemon(id=999, serial=2, hp=200)
    state = _make_state(attacker, defender)

    options = [Option(type=OptionType.ATTACK, attackId=100)]

    class _FakeObs:
        current = state

        class select:
            option = options

    monkeypatch.setattr(attack_priority.card_cache, "get_card", lambda card_id: _FakeCardData())
    monkeypatch.setattr(attack_priority.card_cache, "get_attack", lambda attack_id: attack_id)
    monkeypatch.setattr(attack_priority.energy_requirements, "is_energy_sufficient", lambda attack, energies: True)
    monkeypatch.setattr(
        attack_priority.attack_features,
        "resolve_damage",
        lambda attack, attacker, weakness, resistance, hand_size: 50,
    )
    monkeypatch.setattr(attack_priority.profile_registry, "get_attack_profile", lambda attack_id: None)
    monkeypatch.setattr(attack_priority.opponent_tracker, "current_matchup_plan", lambda: None)

    proposal = attack_priority.propose(_FakeObs())

    assert proposal is not None
    assert proposal.score == 50.0
