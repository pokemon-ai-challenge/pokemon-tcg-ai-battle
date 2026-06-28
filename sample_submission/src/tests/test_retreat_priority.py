from cg.api import (
    AreaType,
    Attack,
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

from src.decision import switch_eval
from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.priorities.retreat import propose_retreat_action
from src.decision.router import choose_action


def test_skips_retreat_when_bench_does_not_improve_board(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=120, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Almost Ready", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=2, name="Also Slow", text="", damage=70, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0]))

    assert proposal is None


def test_skips_retreat_when_attack_option_already_exists(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=140, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Ready Swing", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=2, name="Bench Swing", text="", damage=100, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=140, maxHp=140, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0], attack=[1]))

    assert proposal is None


def test_offers_retreat_when_bench_can_attack_immediately(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=140, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Needs Fire", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Swing", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=140, maxHp=140, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0]))

    assert proposal is not None
    assert proposal.action == [0]


def test_skips_retreat_when_active_reaches_main_attack_next_turn(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(678, hp=340, attacks=[982], retreat_cost=1),
            pokemon_card(200, hp=120, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=982, name="Aura Jab", text="", damage=130, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=2, name="Small Bench Hit", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=678, serial=1, hp=340, maxHp=340, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0]))

    assert proposal is None


def test_offers_retreat_when_active_is_risky_but_bench_is_safe(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=150, attacks=[2]),
            pokemon_card(300, hp=120, attacks=[3]),
        ],
        [
            Attack(attackId=1, name="Slow Hit", text="", damage=40, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Swing", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Threat", text="", damage=80, energies=[EnergyType.FIRE]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=70, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=150, maxHp=150, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=3, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0]))

    assert proposal is not None
    assert proposal.action == [0]


def test_retreat_prediction_ignores_option_target_fields(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=140, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Needs Fire", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Swing", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=140, maxHp=140, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[Option(type=OptionType.RETREAT, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0]))

    assert proposal is not None
    assert proposal.action == [0]


def test_multiple_retreat_options_compare_same_predicted_target(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=140, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Needs Fire", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Cannon", text="", damage=100, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=140, maxHp=140, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[
            retreat_option(),
            retreat_option(),
        ],
        opponent_active=Pokemon(id=300, serial=3, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(obs, MainOptionBuckets(retreat=[0, 1]))

    assert proposal is not None
    assert proposal.action == [0]


def test_router_reuses_switch_evaluation_for_switch_context(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(200, hp=120, attacks=[1]),
            pokemon_card(201, hp=150, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Slow Bench", text="", damage=60, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Bench", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_switch_observation(
        active=Pokemon(id=200, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
            Pokemon(id=201, serial=3, hp=150, maxHp=150, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[
            switch_option(0),
            switch_option(1),
        ],
        opponent_active=Pokemon(id=300, serial=4, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
    )

    action = choose_action(obs)

    assert action == [1]


def test_router_reuses_switch_evaluation_for_to_active_context(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(200, hp=120, attacks=[1]),
            pokemon_card(201, hp=150, attacks=[2]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Slow Bench", text="", damage=60, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=2, name="Ready Bench", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    obs = build_switch_observation(
        active=Pokemon(id=200, serial=1, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
            Pokemon(id=201, serial=3, hp=150, maxHp=150, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[
            switch_option(0),
            switch_option(1),
        ],
        opponent_active=Pokemon(id=300, serial=4, hp=100, maxHp=100, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        context=SelectContext.TO_ACTIVE,
    )

    action = choose_action(obs)

    assert action == [1]


def test_retreat_prediction_matches_followup_switch_choice(monkeypatch):
    patch_card_lookups(
        monkeypatch,
        [
            pokemon_card(100, hp=120, attacks=[1], retreat_cost=1),
            pokemon_card(200, hp=120, attacks=[2]),
            pokemon_card(201, hp=150, attacks=[4]),
            harmless_opponent_card(300),
        ],
        [
            Attack(attackId=1, name="Slow Bench", text="", damage=40, energies=[EnergyType.FIRE, EnergyType.FIRE]),
            Attack(attackId=2, name="Medium Bench", text="", damage=60, energies=[EnergyType.FIRE]),
            Attack(attackId=4, name="Ready Bench", text="", damage=90, energies=[EnergyType.FIRE]),
            Attack(attackId=3, name="Soft Tap", text="", damage=10, energies=[]),
        ],
    )

    main_obs = build_main_observation(
        active=Pokemon(id=100, serial=1, hp=70, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
            Pokemon(id=201, serial=3, hp=150, maxHp=150, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[retreat_option()],
        opponent_active=Pokemon(id=300, serial=4, hp=100, maxHp=100, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
    )
    switch_obs = build_switch_observation(
        active=Pokemon(id=100, serial=1, hp=70, maxHp=120, appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[]),
        bench=[
            Pokemon(id=200, serial=2, hp=120, maxHp=120, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
            Pokemon(id=201, serial=3, hp=150, maxHp=150, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
        ],
        options=[switch_option(0), switch_option(1)],
        opponent_active=Pokemon(id=300, serial=4, hp=100, maxHp=100, appearThisTurn=False, energies=[EnergyType.FIRE], energyCards=[], tools=[], preEvolution=[]),
    )

    proposal = propose_retreat_action(main_obs, MainOptionBuckets(retreat=[0]))
    action = choose_action(switch_obs)

    assert proposal is not None
    assert proposal.action == [0]
    assert action == [1]


def patch_card_lookups(monkeypatch, cards: list[CardData], attacks: list[Attack]) -> None:
    monkeypatch.setattr(switch_eval, "load_card_data", lambda: cards)
    monkeypatch.setattr(
        switch_eval,
        "build_attack_lookup",
        lambda: {attack.attackId: attack for attack in attacks},
    )


def build_main_observation(
    active: Pokemon,
    bench: list[Pokemon],
    options: list[Option],
    opponent_active: Pokemon,
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
    return Observation(select=select, logs=[], current=build_state(active, bench, opponent_active))


def build_switch_observation(
    active: Pokemon,
    bench: list[Pokemon],
    options: list[Option],
    opponent_active: Pokemon,
    context: SelectContext = SelectContext.SWITCH,
) -> Observation:
    select = SelectData(
        type=SelectType.CARD,
        context=context,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )
    return Observation(select=select, logs=[], current=build_state(active, bench, opponent_active))


def build_state(
    active: Pokemon,
    bench: list[Pokemon],
    opponent_active: Pokemon,
) -> State:
    return State(
        turn=3,
        turnActionCount=0,
        yourIndex=0,
        firstPlayer=0,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
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
                handCount=0,
                hand=[],
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


def retreat_option(bench_index: int | None = None) -> Option:
    return Option(
        type=OptionType.RETREAT,
        inPlayArea=AreaType.BENCH if bench_index is not None else None,
        inPlayIndex=bench_index,
    )


def switch_option(bench_index: int) -> Option:
    return Option(
        type=OptionType.CARD,
        area=AreaType.BENCH,
        index=bench_index,
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


def harmless_opponent_card(card_id: int) -> CardData:
    return pokemon_card(card_id, hp=100, attacks=[3], retreat_cost=1)
