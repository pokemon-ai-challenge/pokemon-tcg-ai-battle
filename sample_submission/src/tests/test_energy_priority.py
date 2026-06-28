from cg.api import (
    AreaType,
    Attack,
    Card,
    CardData,
    CardType,
    EnergyType,
    Observation,
    Option,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    SelectData,
    SelectType,
    State,
)

from src.decision.main_turn_parts import energy_eval
from src.decision.main_turn_parts.energy_eval import AttachEvaluation
from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.priorities import energy as energy_priority
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def test_skips_energy_when_turn_already_used_attachment():
    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[],
        hand=[Card(id=900, serial=10, playerIndex=0)],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        energy_attached=True,
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0]))

    assert proposal is None


def test_prefers_attach_that_unlocks_attack_now(monkeypatch):
    monkeypatch.setattr(energy_eval, "load_card_data", lambda: [
        pokemon_card(100, hp=120, attacks=[1]),
        pokemon_card(200, hp=120, attacks=[2]),
        energy_card(900, EnergyType.FIRE),
        harmless_opponent_card(300),
    ])
    monkeypatch.setattr(
        energy_eval,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Quick Hit", text="", damage=50, energies=[EnergyType.FIRE]),
            2: Attack(attackId=2, name="Big Swing", text="", damage=50, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            3: Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        },
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        ],
        hand=[Card(id=900, serial=10, playerIndex=0)],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attach_option(hand_index=0, in_play_area=AreaType.BENCH, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0, 1]))

    assert proposal is not None
    assert proposal.action == [0]


def test_prefers_energy_type_that_matches_attack_cost(monkeypatch):
    monkeypatch.setattr(energy_eval, "load_card_data", lambda: [
        pokemon_card(100, hp=100, attacks=[1]),
        energy_card(900, EnergyType.FIRE),
        energy_card(901, EnergyType.WATER),
        harmless_opponent_card(300),
    ])
    monkeypatch.setattr(
        energy_eval,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Flare", text="", damage=60, energies=[EnergyType.FIRE]),
            3: Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        },
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[],
        hand=[
            Card(id=900, serial=10, playerIndex=0),
            Card(id=901, serial=11, playerIndex=0),
        ],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attach_option(hand_index=1, in_play_area=AreaType.ACTIVE, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0, 1]))

    assert proposal is not None
    assert proposal.action == [0]


def test_avoids_attaching_to_pokemon_that_is_likely_to_be_knocked_out(monkeypatch):
    monkeypatch.setattr(energy_eval, "load_card_data", lambda: [
        pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
        pokemon_card(200, hp=150, attacks=[2]),
        energy_card(900, EnergyType.FIRE),
        pokemon_card(300, hp=120, attacks=[3]),
    ])
    monkeypatch.setattr(
        energy_eval,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Rush", text="", damage=50, energies=[EnergyType.FIRE]),
            2: Attack(attackId=2, name="Cannon", text="", damage=120, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            3: Attack(attackId=3, name="Threat", text="", damage=80, energies=[EnergyType.FIRE]),
        },
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=70, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=150, maxHp=150, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        ],
        hand=[Card(id=900, serial=10, playerIndex=0)],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attach_option(hand_index=0, in_play_area=AreaType.BENCH, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0, 1]))

    assert proposal is not None
    assert proposal.action == [1]


def test_does_not_offer_energy_when_best_attach_is_not_meaningful(monkeypatch):
    monkeypatch.setattr(
        energy_priority,
        "choose_best_attach_option",
        lambda obs, attach_options: AttachEvaluation(
            option_index=attach_options[0],
            score=-3,
            should_offer=False,
            enables_attack_now=False,
            reaches_main_attack_next_turn=False,
        ),
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[],
        hand=[Card(id=900, serial=10, playerIndex=0)],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0]))

    assert proposal is None


def test_energy_proposal_uses_category_weight_instead_of_internal_score(monkeypatch):
    monkeypatch.setattr(
        energy_priority,
        "choose_best_attach_option",
        lambda obs, attach_options: AttachEvaluation(
            option_index=attach_options[1],
            score=999,
            should_offer=True,
            enables_attack_now=True,
            reaches_main_attack_next_turn=False,
        ),
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[],
        hand=[
            Card(id=900, serial=10, playerIndex=0),
            Card(id=901, serial=11, playerIndex=0),
        ],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
            attach_option(hand_index=1, in_play_area=AreaType.ACTIVE, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = energy_priority.propose_energy_action(obs, MainOptionBuckets(attach=[0, 1]))

    assert proposal is not None
    assert proposal.action == [1]
    assert proposal.score == MAIN_ACTION_BASE_WEIGHTS["energy"]


def test_zero_damage_payable_attack_still_counts_as_attack_enabled(monkeypatch):
    monkeypatch.setattr(energy_eval, "load_card_data", lambda: [
        pokemon_card(100, hp=100, attacks=[1]),
        energy_card(900, EnergyType.FIRE),
        harmless_opponent_card(300),
    ])
    monkeypatch.setattr(
        energy_eval,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Setup Beam", text="", damage=0, energies=[EnergyType.FIRE]),
            3: Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        },
    )

    obs = build_observation(
        active=Pokemon(id=100, serial=1, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[],
        hand=[Card(id=900, serial=10, playerIndex=0)],
        options=[
            attach_option(hand_index=0, in_play_area=AreaType.ACTIVE, in_play_index=0),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    evaluation = energy_eval.choose_best_attach_option(obs, [0])

    assert evaluation.enables_attack_now is True
    assert evaluation.should_offer is True


def build_observation(
    active: Pokemon,
    bench: list[Pokemon],
    hand: list[Card],
    options: list[Option],
    opponent_active: Pokemon,
    energy_attached: bool = False,
) -> Observation:
    select = SelectData(
        type=SelectType.MAIN,
        context=SelectContext.MAIN,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )
    current = State(
        turn=3,
        turnActionCount=0,
        yourIndex=0,
        firstPlayer=0,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=energy_attached,
        retreated=False,
        result=-1,
        stadium=[],
        looking=None,
        players=[
            PlayerState(
                active=[active],
                bench=bench,
                benchMax=5,
                deckCount=40,
                discard=[],
                prize=[None] * 6,
                handCount=len(hand),
                hand=hand,
                poisoned=False,
                burned=False,
                asleep=False,
                paralyzed=False,
                confused=False,
            ),
            PlayerState(
                active=[opponent_active],
                bench=[],
                benchMax=5,
                deckCount=40,
                discard=[],
                prize=[None] * 6,
                handCount=0,
                hand=None,
                poisoned=False,
                burned=False,
                asleep=False,
                paralyzed=False,
                confused=False,
            ),
        ],
    )
    return Observation(select=select, logs=[], current=current)


def attach_option(
    hand_index: int,
    in_play_area: AreaType,
    in_play_index: int,
) -> Option:
    return Option(
        type=OptionType.ATTACH,
        area=AreaType.HAND,
        index=hand_index,
        inPlayArea=in_play_area,
        inPlayIndex=in_play_index,
    )


def pokemon_card(
    card_id: int,
    hp: int,
    attacks: list[int],
    retreat_cost: int = 1,
) -> CardData:
    return CardData(
        cardId=card_id,
        name=f"Pokemon {card_id}",
        cardType=CardType.POKEMON,
        retreatCost=retreat_cost,
        hp=hp,
        weakness=None,
        resistance=None,
        energyType=EnergyType.FIRE,
        basic=True,
        stage1=False,
        stage2=False,
        ex=False,
        megaEx=False,
        tera=False,
        aceSpec=False,
        evolvesFrom=None,
        skills=[],
        attacks=attacks,
    )


def energy_card(card_id: int, energy_type: EnergyType) -> CardData:
    return CardData(
        cardId=card_id,
        name=f"Energy {card_id}",
        cardType=CardType.BASIC_ENERGY,
        retreatCost=0,
        hp=0,
        weakness=None,
        resistance=None,
        energyType=energy_type,
        basic=False,
        stage1=False,
        stage2=False,
        ex=False,
        megaEx=False,
        tera=False,
        aceSpec=False,
        evolvesFrom=None,
        skills=[],
        attacks=[],
    )


def harmless_opponent_card(card_id: int) -> CardData:
    return pokemon_card(card_id, hp=100, attacks=[3], retreat_cost=1)
