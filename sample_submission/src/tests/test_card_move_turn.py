from cg.api import AreaType, Card, EnergyType, Option, OptionType, SelectContext, SelectType

from src.decision.card_move import bench_field as card_move_bench_field
from src.decision.card_move import common as card_move_common
from src.decision.card_move import discard as card_move_discard
from src.decision.card_move import hand_like as card_move_hand_like
from src.decision.card_move import not_move_or_look as card_move_not_move_or_look
from src.decision.card_move import to_hand_eval as card_move_to_hand_eval
from src.decision.card_move import hidden_zone as card_move_hidden_zone
from src.decision.handlers import card_move_turn
from src.tests.helpers_decision import (
    attack,
    build_observation,
    energy_card,
    hand_card,
    item_card,
    pokemon,
    pokemon_card,
    supporter_card,
)


def test_prefers_lunatone_when_solrock_and_fighting_energy_are_ready(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        676: pokemon_card(676, hp=110, attacks=[980], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        999: pokemon_card(999, hp=100, attacks=[999], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_bench_field, "_deck_evolution_targets", lambda: frozenset({"Riolu"}))

    obs = build_observation(
        context=SelectContext.TO_BENCH,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        bench=[pokemon(676, hp=110)],
        hand=[hand_card(675), hand_card(999, serial=11), hand_card(6, serial=12)],
    )

    assert card_move_turn.choose_to_bench_or_field_action(obs) == [0]


def test_prefers_riolu_over_second_solrock_when_one_slot_remains(monkeypatch):
    cards = {
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        676: pokemon_card(676, hp=110, attacks=[980], energy_type=EnergyType.FIGHTING),
        677: pokemon_card(677, hp=80, attacks=[981], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(
            678,
            hp=340,
            attacks=[982],
            energy_type=EnergyType.FIGHTING,
            ex=True,
        ),
        500: pokemon_card(500, hp=110, attacks=[500], energy_type=EnergyType.FIGHTING),
        501: pokemon_card(501, hp=110, attacks=[501], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_bench_field, "_deck_evolution_targets", lambda: frozenset({"Riolu"}))

    obs = build_observation(
        context=SelectContext.TO_BENCH,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        bench=[
            pokemon(676, hp=110),
            pokemon(675, hp=110),
            pokemon(500, hp=110),
            pokemon(501, hp=110),
        ],
        hand=[hand_card(676), hand_card(677, serial=11)],
    )

    assert card_move_turn.choose_to_bench_or_field_action(obs) == [1]


def test_selects_lunatone_and_solrock_together_when_both_are_offered(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        676: pokemon_card(676, hp=110, attacks=[980], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        999: pokemon_card(999, hp=100, attacks=[999], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_bench_field, "_deck_evolution_targets", lambda: frozenset())

    obs = build_observation(
        context=SelectContext.TO_BENCH,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(675), hand_card(676, serial=11), hand_card(999, serial=12), hand_card(6, serial=13)],
    )
    obs.select.minCount = 0
    obs.select.maxCount = 2

    result = card_move_turn.choose_to_bench_or_field_action(obs)

    assert len(result) == 2
    assert set(result) == {0, 1}


def test_multi_pick_consumes_remaining_bench_slots(monkeypatch):
    cards = {
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        676: pokemon_card(676, hp=110, attacks=[980], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        800: pokemon_card(800, hp=100, attacks=[800], energy_type=EnergyType.FIGHTING),
        801: pokemon_card(801, hp=100, attacks=[801], energy_type=EnergyType.FIGHTING),
        802: pokemon_card(802, hp=100, attacks=[802], energy_type=EnergyType.FIGHTING),
        803: pokemon_card(803, hp=100, attacks=[803], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_bench_field, "_deck_evolution_targets", lambda: frozenset())

    obs = build_observation(
        context=SelectContext.TO_BENCH,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        bench=[
            pokemon(800, hp=100),
            pokemon(801, hp=100),
            pokemon(802, hp=100),
            pokemon(803, hp=100),
        ],
        hand=[hand_card(675), hand_card(676, serial=11)],
    )
    obs.select.minCount = 0
    obs.select.maxCount = 2

    result = card_move_turn.choose_to_bench_or_field_action(obs)

    assert len(result) == 1


def test_bench_action_falls_back_when_min_count_cannot_be_met(monkeypatch):
    cards = {
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        900: energy_card(900, EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_bench_field, "_deck_evolution_targets", lambda: frozenset())

    obs = build_observation(
        context=SelectContext.TO_BENCH,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=1, cardId=900),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(675)],
    )
    obs.select.minCount = 2
    obs.select.maxCount = 2

    result = card_move_turn.choose_to_bench_or_field_action(obs)

    assert sorted(result) == [0, 1]


def test_returns_all_cards_when_hand_is_forced_back_to_bottom(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_DECK_BOTTOM,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(675), hand_card(6, serial=11)],
    )
    obs.select.minCount = 2
    obs.select.maxCount = 2

    assert card_move_turn.choose_card_move_action(obs) == [0, 1]


def test_prefers_extra_energy_when_returning_own_hand_to_bottom(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        675: pokemon_card(675, hp=110, attacks=[979], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_DECK_BOTTOM,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(678), hand_card(6, serial=11), hand_card(6, serial=12)],
    )

    assert card_move_turn.choose_card_move_action(obs) in ([1], [2])


def test_prefers_opponent_energy_for_hand_disruption(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        1192: supporter_card(1192),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        700: pokemon_card(700, hp=140, attacks=[700], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_DECK_BOTTOM,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=6),
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=1192),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(700, hp=140, energies=[EnergyType.FIGHTING]),
    )

    assert card_move_turn.choose_card_move_action(obs) == [0]


def test_discard_prefers_extra_fighting_energy_when_bench_acceleration_is_online(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        677: pokemon_card(677, hp=80, attacks=[981], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_discard, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_discard,
        "build_attack_lookup",
        lambda: {
            981: attack(981, name="Kick", damage=20, energies=[EnergyType.FIGHTING]),
            982: attack(982, name="Aura Jab", damage=130, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.DISCARD,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(678, hp=340, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
        opponent_active=pokemon(300, hp=120),
        bench=[pokemon(677, hp=80)],
        hand=[hand_card(6), hand_card(6, serial=11), hand_card(677, serial=12)],
    )

    assert card_move_turn.choose_card_move_action(obs) in ([0], [1])


def test_discard_keeps_last_energy_when_active_needs_manual_attachment(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        700: pokemon_card(700, hp=120, attacks=[13], energy_type=EnergyType.FIGHTING),
        1123: item_card(1123),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_discard, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_discard,
        "build_attack_lookup",
        lambda: {
            13: attack(13, name="Threat", damage=90, energies=[EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.DISCARD,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(700, hp=120),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(6), hand_card(1123, serial=11)],
    )

    assert card_move_turn.choose_card_move_action(obs) == [1]


def test_discard_protects_riolu_line_when_lucario_is_waiting_in_hand(monkeypatch):
    cards = {
        677: pokemon_card(677, hp=80, attacks=[981], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(
            678,
            hp=340,
            attacks=[982],
            energy_type=EnergyType.FIGHTING,
            ex=True,
            basic=False,
            stage1=True,
            evolves_from="Pokemon 677",
        ),
        1123: item_card(1123),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_discard, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_discard,
        "build_attack_lookup",
        lambda: {
            981: attack(981, name="Kick", damage=20, energies=[EnergyType.FIGHTING]),
            982: attack(982, name="Aura Jab", damage=130, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.DISCARD,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(300, hp=120),
        opponent_active=pokemon(400, hp=120),
        hand=[hand_card(677), hand_card(1123, serial=11), hand_card(678, serial=12)],
    )

    assert card_move_turn.choose_card_move_action(obs) == [1]


def test_discard_re_evaluates_energy_one_by_one_for_optional_multi_pick(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        677: pokemon_card(677, hp=80, attacks=[981], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_discard, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_discard,
        "build_attack_lookup",
        lambda: {
            981: attack(981, name="Kick", damage=20, energies=[EnergyType.FIGHTING]),
            982: attack(982, name="Aura Jab", damage=130, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.DISCARD,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=3, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=4, playerIndex=0),
        ],
        active=pokemon(678, hp=340, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
        opponent_active=pokemon(300, hp=120),
        bench=[pokemon(677, hp=80)],
        hand=[
            hand_card(6),
            hand_card(6, serial=11),
            hand_card(6, serial=12),
            hand_card(6, serial=13),
            hand_card(677, serial=14),
        ],
    )
    obs.select.minCount = 0
    obs.select.maxCount = 3

    result = card_move_turn.choose_card_move_action(obs)

    assert len(result) == 2
    assert set(result).issubset({0, 1, 2, 3})


def test_discard_prefers_opponent_energy_over_supporter_for_hand_disruption(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        700: pokemon_card(700, hp=120, attacks=[13], energy_type=EnergyType.FIGHTING),
        1192: supporter_card(1192),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_discard, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_discard,
        "build_attack_lookup",
        lambda: {
            13: attack(13, name="Threat", damage=90, energies=[EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.DISCARD,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=6),
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=1192),
        ],
        active=pokemon(300, hp=120),
        opponent_active=pokemon(700, hp=120),
    )

    assert card_move_turn.choose_card_move_action(obs) == [0]


def test_prefers_strong_self_discard_recovery_when_returning_to_deck(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        1123: item_card(1123),
        677: pokemon_card(677, hp=80, attacks=[10], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(
            678,
            hp=340,
            attacks=[11],
            energy_type=EnergyType.FIGHTING,
            ex=True,
            basic=False,
            stage1=True,
            evolves_from="Pokemon 677",
        ),
        700: pokemon_card(700, hp=120, attacks=[12], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_hidden_zone, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_DECK,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=1, playerIndex=0),
        ],
        active=pokemon(700, hp=120),
        opponent_active=pokemon(300, hp=120),
        bench=[pokemon(677, hp=80)],
    )
    obs.current.players[0].discard = [
        Card(id=678, serial=20, playerIndex=0),
        Card(id=1123, serial=21, playerIndex=0),
    ]

    assert card_move_turn.choose_card_move_action(obs) == [0]


def test_avoids_helping_opponent_when_returning_their_discard_to_deck(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        1123: item_card(1123),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        700: pokemon_card(700, hp=140, attacks=[700], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_hidden_zone, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_DECK,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=0, playerIndex=1),
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=1, playerIndex=1),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(700, hp=140, energies=[EnergyType.FIGHTING]),
    )
    obs.current.players[1].discard = [
        Card(id=6, serial=30, playerIndex=1),
        Card(id=1123, serial=31, playerIndex=1),
    ]

    assert card_move_turn.choose_card_move_action(obs) == [1]


def test_forced_full_hand_return_does_not_apply_to_to_prize():
    obs = build_observation(
        context=SelectContext.TO_PRIZE,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(678), hand_card(6, serial=11)],
    )
    obs.select.minCount = 2
    obs.select.maxCount = 2
    views = [
        card_move_common.analyze_card_move_option(obs, 0),
        card_move_common.analyze_card_move_option(obs, 1),
    ]

    assert card_move_hidden_zone._forced_full_hand_return(obs, [view for view in views if view is not None]) is None


def test_to_hand_prefers_energy_from_deck_that_unlocks_active_attack(monkeypatch):
    cards = {
        100: pokemon_card(100, hp=120, attacks=[1], energy_type=EnergyType.FIRE),
        200: pokemon_card(200, hp=100, attacks=[2], energy_type=EnergyType.FIRE),
        900: energy_card(900, EnergyType.FIRE),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_to_hand_eval,
        "build_attack_lookup",
        lambda: {
            1: attack(1, name="Flare", damage=80, energies=[EnergyType.FIRE]),
            2: attack(2, name="Small Hit", damage=10, energies=[]),
        },
    )

    obs = build_observation(
        context=SelectContext.TO_HAND,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DECK, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.DECK, index=1, playerIndex=0),
        ],
        active=pokemon(100, hp=120),
        opponent_active=pokemon(300, hp=120),
    )
    obs.select.deck = [
        Card(id=900, serial=1, playerIndex=0),
        Card(id=200, serial=2, playerIndex=0),
    ]

    assert card_move_turn.choose_to_hand_like_action(obs) == [0]


def test_to_hand_prefers_lucario_recovery_when_riolu_is_in_play(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        677: pokemon_card(677, hp=80, attacks=[10], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(
            678,
            hp=340,
            attacks=[11],
            energy_type=EnergyType.FIGHTING,
            ex=True,
            basic=False,
            stage1=True,
            evolves_from="Pokemon 677",
        ),
        700: pokemon_card(700, hp=120, attacks=[12], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_to_hand_eval,
        "build_attack_lookup",
        lambda: {
            10: attack(10, name="Kick", damage=20, energies=[EnergyType.FIGHTING]),
            11: attack(11, name="Aura", damage=130, energies=[EnergyType.FIGHTING, EnergyType.FIGHTING]),
            12: attack(12, name="Threat", damage=40, energies=[EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.TO_HAND,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=1, playerIndex=0),
        ],
        active=pokemon(700, hp=120),
        opponent_active=pokemon(300, hp=120),
        bench=[pokemon(677, hp=80)],
    )
    obs.current.players[0].discard = [
        Card(id=678, serial=20, playerIndex=0),
        Card(id=6, serial=21, playerIndex=0),
    ]

    assert card_move_turn.choose_to_hand_like_action(obs) == [0]


def test_to_hand_avoids_discard_energy_recovery_when_bench_acceleration_is_online(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        1123: item_card(1123),
        700: pokemon_card(700, hp=120, attacks=[13], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_to_hand_eval,
        "build_attack_lookup",
        lambda: {
            982: attack(982, name="Aura Jab", damage=130, energies=[EnergyType.FIGHTING]),
            13: attack(13, name="Threat", damage=90, energies=[EnergyType.FIGHTING]),
        },
    )

    obs = build_observation(
        context=SelectContext.TO_HAND,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.DISCARD, index=1, playerIndex=0),
        ],
        active=pokemon(700, hp=70, energies=[EnergyType.FIGHTING]),
        opponent_active=pokemon(300, hp=120, energies=[EnergyType.FIGHTING]),
        bench=[pokemon(678, hp=340, energies=[EnergyType.FIGHTING])],
    )
    obs.current.players[0].discard = [
        Card(id=6, serial=30, playerIndex=0),
        Card(id=1123, serial=31, playerIndex=0),
    ]

    assert card_move_turn.choose_to_hand_like_action(obs) == [1]


def test_to_hand_bounces_opponent_active_that_is_ready_to_attack(monkeypatch):
    cards = {
        100: pokemon_card(100, hp=120, attacks=[21], energy_type=EnergyType.FIGHTING),
        300: pokemon_card(300, hp=140, attacks=[23], energy_type=EnergyType.FIGHTING),
        301: pokemon_card(301, hp=90, attacks=[24], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(
        card_move_to_hand_eval,
        "build_attack_lookup",
        lambda: {
            21: attack(21, name="Punch", damage=30, energies=[EnergyType.FIGHTING]),
            23: attack(23, name="Big Swing", damage=120, energies=[EnergyType.FIGHTING]),
            24: attack(24, name="Bench Nudge", damage=20, energies=[]),
        },
    )

    obs = build_observation(
        context=SelectContext.TO_HAND,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.ACTIVE, index=0, playerIndex=1),
            Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=1),
        ],
        active=pokemon(100, hp=120),
        opponent_active=pokemon(300, hp=140, energies=[EnergyType.FIGHTING]),
        bench=[],
    )
    obs.current.players[1].bench = [pokemon(301, hp=90)]

    assert card_move_turn.choose_to_hand_like_action(obs) == [0]


def test_prefers_extra_energy_when_sending_card_to_prize(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_hidden_zone, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.TO_PRIZE,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(678), hand_card(6, serial=11), hand_card(6, serial=12)],
    )

    assert card_move_turn.choose_card_move_action(obs) in ([1], [2])


def test_to_hand_falls_back_to_random_legal_action_when_specialized_eval_is_unavailable(monkeypatch):
    monkeypatch.setattr(card_move_hand_like, "choose_to_hand_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(card_move_hand_like, "choose_random_legal_action", lambda obs: [1])

    obs = build_observation(
        context=SelectContext.TO_HAND,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.DECK, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.DECK, index=1, playerIndex=0),
        ],
        active=pokemon(100, hp=120),
        opponent_active=pokemon(300, hp=120),
    )
    obs.select.deck = [
        Card(id=900, serial=1, playerIndex=0),
        Card(id=901, serial=2, playerIndex=0),
    ]

    assert card_move_turn.choose_to_hand_like_action(obs) == [1]


def test_not_move_keeps_own_lucario_over_extra_energy(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        677: pokemon_card(677, hp=80, attacks=[10], energy_type=EnergyType.FIGHTING),
        678: pokemon_card(
            678,
            hp=340,
            attacks=[11],
            energy_type=EnergyType.FIGHTING,
            ex=True,
            basic=False,
            stage1=True,
            evolves_from="Pokemon 677",
        ),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_hidden_zone, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_not_move_or_look, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.NOT_MOVE,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
        ],
        active=pokemon(677, hp=80),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(678), hand_card(6, serial=11)],
    )

    assert card_move_turn.choose_card_move_action(obs) == [0]


def test_not_move_leaves_opponent_energy_instead_of_supporter(monkeypatch):
    cards = {
        6: energy_card(6, EnergyType.FIGHTING),
        1192: supporter_card(1192),
        678: pokemon_card(678, hp=340, attacks=[982], energy_type=EnergyType.FIGHTING, ex=True),
        700: pokemon_card(700, hp=140, attacks=[700], energy_type=EnergyType.FIGHTING),
    }
    monkeypatch.setattr(card_move_common, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_hidden_zone, "_card_data_lookup", lambda: cards)
    monkeypatch.setattr(card_move_not_move_or_look, "_card_data_lookup", lambda: cards)

    obs = build_observation(
        context=SelectContext.NOT_MOVE,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=6),
            Option(type=OptionType.CARD, area=AreaType.HAND, playerIndex=1, cardId=1192),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(700, hp=140, energies=[EnergyType.FIGHTING]),
    )

    assert card_move_turn.choose_card_move_action(obs) == [0]


def test_look_uses_deterministic_prefix_order():
    obs = build_observation(
        context=SelectContext.LOOK,
        select_type=SelectType.CARD,
        options=[
            Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.HAND, index=2, playerIndex=0),
        ],
        active=pokemon(678, hp=340),
        opponent_active=pokemon(300, hp=120),
        hand=[hand_card(678), hand_card(6, serial=11), hand_card(6, serial=12)],
    )
    obs.select.minCount = 1
    obs.select.maxCount = 2

    assert card_move_turn.choose_card_move_action(obs) == [0, 1]
