from cg.api import Attack, CardData, CardType, EnergyType, Pokemon

from src.decision.evaluation.board_features import attacker_priority_score
from src.knowledge.deck_profiles import (
    get_attack_effect_profile,
    get_energy_profile,
    get_item_profile,
    get_pokemon_profile,
    pokemon_active_role_bonus,
    get_stadium_profile,
    get_supporter_profile,
    get_tool_profile,
)


def test_attack_profile_requirements_reference_shared_deck_knowledge():
    profile = get_attack_effect_profile(980)

    assert profile is not None
    assert profile.fails_without_requirement is True


def test_pokemon_role_bonus_is_available_to_non_attack_evaluators():
    attack_by_id = {
        982: Attack(attackId=982, name="Aura Jab", text="", damage=130, energies=[]),
        999: Attack(attackId=999, name="Heavy Swing", text="", damage=170, energies=[]),
    }
    card_data_by_id = {
        678: pokemon_card(678, hp=340, attacks=[982], ex=True),
        9999: pokemon_card(9999, hp=150, attacks=[999]),
    }

    mega_score = attacker_priority_score(
        pokemon(678, hp=340),
        card_data_by_id,
        attack_by_id,
    )
    generic_score = attacker_priority_score(
        pokemon(9999, hp=150),
        card_data_by_id,
        attack_by_id,
    )

    assert pokemon_active_role_bonus(678) > 0
    assert mega_score > generic_score


def test_pair_profiles_capture_partner_and_draw_engine_rules():
    lunatone = get_pokemon_profile(675)
    solrock = get_pokemon_profile(676)

    assert lunatone is not None
    assert lunatone.partner_card_ids == frozenset({676})
    assert lunatone.draw_support_bonus > 0
    assert lunatone.hand_discard_energy_type_for_draw == EnergyType.FIGHTING

    assert solrock is not None
    assert solrock.partner_card_ids == frozenset({675})
    assert solrock.partner_required is True


def test_non_pokemon_profiles_cover_current_deck_cards():
    ultra_ball = get_item_profile(1121)
    premium_power_pro = get_item_profile(1141)
    boss = get_supporter_profile(1182)
    judge = get_supporter_profile(1213)
    air_balloon = get_tool_profile(1174)
    maximum_belt = get_tool_profile(1158)
    gravity_mountain = get_stadium_profile(1252)
    fighting_energy = get_energy_profile(6)

    assert ultra_ball is not None
    assert ultra_ball.searches_pokemon == 1
    assert ultra_ball.discard_cost == 2

    assert premium_power_pro is not None
    assert premium_power_pro.active_damage_bonus == 30
    assert premium_power_pro.only_for_fighting_pokemon is True

    assert boss is not None
    assert boss.gust_effect is True

    assert judge is not None
    assert judge.both_players_redraw is True
    assert judge.draw_cards == 4

    assert air_balloon is not None
    assert air_balloon.retreat_cost_reduction == 2

    assert maximum_belt is not None
    assert maximum_belt.active_damage_bonus_vs_ex == 50

    assert gravity_mountain is not None
    assert gravity_mountain.stage2_hp_modifier == -30

    assert fighting_energy is not None
    assert fighting_energy.is_basic_energy is True
    assert fighting_energy.provided_energy_type == EnergyType.FIGHTING


def pokemon(card_id: int, hp: int) -> Pokemon:
    return Pokemon(
        id=card_id,
        serial=1,
        hp=hp,
        maxHp=hp,
        appearThisTurn=False,
        energies=[],
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def pokemon_card(
    card_id: int,
    hp: int,
    attacks: list[int],
    ex: bool = False,
) -> CardData:
    return CardData(
        cardId=card_id,
        name=f"Pokemon {card_id}",
        cardType=CardType.POKEMON,
        retreatCost=1,
        hp=hp,
        weakness=None,
        resistance=None,
        energyType=EnergyType.FIGHTING,
        basic=True,
        stage1=False,
        stage2=False,
        ex=ex,
        megaEx=False,
        tera=False,
        aceSpec=False,
        evolvesFrom=None,
        skills=[],
        attacks=attacks,
    )
