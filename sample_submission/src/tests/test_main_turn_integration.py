from cg.api import AreaType, EnergyType, SelectContext, SelectType

from src.decision.main_turn import choose_main_action
from src.decision.main_turn_parts import buckets as main_buckets
from src.decision.main_turn_parts import energy_eval
from src.decision.main_turn_parts.priorities import attack as attack_priority
from src.decision import switch_eval
from src.tests.helpers_decision import (
    ability_option,
    attack,
    attack_option,
    attach_option,
    build_observation,
    end_option,
    energy_card,
    hand_card,
    harmless_opponent_card,
    pokemon,
    pokemon_card,
    retreat_option,
)


def test_choose_main_action_prefers_ability_over_energy_and_attack(monkeypatch):
    patch_main_card_lookups(
        monkeypatch,
        [
            pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
            energy_card(900, EnergyType.FIGHTING),
            harmless_opponent_card(300),
        ],
        [
            attack(979, name="Power Gem", damage=50, energies=[EnergyType.FIGHTING]),
            attack(3, name="Soft Tap", damage=10),
        ],
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        select_type=SelectType.MAIN,
        options=[
            ability_option(area=AreaType.ACTIVE, index=0),
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attack_option(979),
            end_option(),
        ],
        active=pokemon(675, 110),
        bench=[],
        hand=[hand_card(900)],
        opponent_active=pokemon(300, 120),
    )

    assert choose_main_action(obs) == [0]


def test_choose_main_action_prefers_energy_over_attack_when_no_ability(monkeypatch):
    patch_main_card_lookups(
        monkeypatch,
        [
            pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
            energy_card(900, EnergyType.FIGHTING),
            harmless_opponent_card(300),
        ],
        [
            attack(982, name="Aura Jab", damage=130, energies=[EnergyType.FIGHTING]),
            attack(3, name="Soft Tap", damage=10),
        ],
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        select_type=SelectType.MAIN,
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attack_option(982),
            end_option(),
        ],
        active=pokemon(678, 340),
        bench=[],
        hand=[hand_card(900)],
        opponent_active=pokemon(300, 200),
    )

    assert choose_main_action(obs) == [0]


def test_choose_main_action_prefers_attack_when_no_setup_action_is_valid(monkeypatch):
    patch_main_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1]),
            energy_card(900, EnergyType.FIRE),
            harmless_opponent_card(300),
        ],
        [
            attack(1, name="Ready Swing", damage=80, energies=[EnergyType.FIRE]),
            attack(3, name="Soft Tap", damage=10),
        ],
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        select_type=SelectType.MAIN,
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.BENCH, in_play_index=0),
            attack_option(1),
            end_option(),
        ],
        active=pokemon(100, 120, energies=[EnergyType.FIRE]),
        bench=[pokemon(200, 120, serial=2)],
        hand=[hand_card(900)],
        opponent_active=pokemon(300, 150),
    )

    assert choose_main_action(obs) == [1]


def test_choose_main_action_returns_end_when_no_other_meaningful_action_exists(monkeypatch):
    patch_main_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1]),
            harmless_opponent_card(300),
        ],
        [
            attack(1, name="Needs Fire", damage=80, energies=[EnergyType.FIRE]),
            attack(3, name="Soft Tap", damage=10),
        ],
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        select_type=SelectType.MAIN,
        options=[end_option()],
        active=pokemon(100, 120),
        bench=[],
        opponent_active=pokemon(300, 150),
    )

    assert choose_main_action(obs) == [0]


def test_choose_main_action_prefers_energy_over_retreat_with_current_weights(monkeypatch):
    patch_main_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=150, attacks=[2]),
            energy_card(900, EnergyType.FIRE),
            pokemon_card(300, hp=120, attacks=[3]),
        ],
        [
            attack(1, name="Needs Fire", damage=60, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            attack(2, name="Ready Swing", damage=90, energies=[EnergyType.FIRE]),
            attack(3, name="Threat", damage=80, energies=[EnergyType.FIRE]),
        ],
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        select_type=SelectType.MAIN,
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            retreat_option(),
            end_option(),
        ],
        active=pokemon(100, 70),
        bench=[pokemon(200, 150, serial=2, energies=[EnergyType.FIRE])],
        hand=[hand_card(900)],
        opponent_active=pokemon(300, 120, energies=[EnergyType.FIRE]),
    )

    assert choose_main_action(obs) == [0]


def patch_main_card_lookups(monkeypatch, cards, attacks) -> None:
    attack_lookup = {candidate.attackId: candidate for candidate in attacks}
    monkeypatch.setattr(main_buckets, "load_card_data", lambda: cards)
    monkeypatch.setattr(energy_eval, "load_card_data", lambda: cards)
    monkeypatch.setattr(energy_eval, "build_attack_lookup", lambda: attack_lookup)
    monkeypatch.setattr(attack_priority, "build_attack_lookup", lambda: attack_lookup)
    monkeypatch.setattr(switch_eval, "load_card_data", lambda: cards)
    monkeypatch.setattr(switch_eval, "build_attack_lookup", lambda: attack_lookup)
