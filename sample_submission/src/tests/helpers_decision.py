from pathlib import Path

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
    Skill,
    State,
)


def pokemon(
    card_id: int,
    hp: int,
    *,
    serial: int = 1,
    max_hp: int | None = None,
    energies: list[EnergyType] | None = None,
) -> Pokemon:
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=hp,
        maxHp=hp if max_hp is None else max_hp,
        appearThisTurn=False,
        energies=[] if energies is None else energies,
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def player_state(
    *,
    active: Pokemon,
    bench: list[Pokemon] | None = None,
    hand: list[Card] | None = None,
) -> PlayerState:
    return PlayerState(
        active=[active],
        bench=[] if bench is None else bench,
        benchMax=5,
        deckCount=40,
        discard=[],
        prize=[None] * 6,
        handCount=0 if hand is None else len(hand),
        hand=[] if hand is None else hand,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def opponent_state(active: Pokemon) -> PlayerState:
    return PlayerState(
        active=[active],
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
    )


def build_observation(
    *,
    context: SelectContext,
    select_type: SelectType,
    options: list[Option],
    active: Pokemon,
    opponent_active: Pokemon,
    bench: list[Pokemon] | None = None,
    hand: list[Card] | None = None,
    energy_attached: bool = False,
) -> Observation:
    select = SelectData(
        type=select_type,
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
            player_state(active=active, bench=bench, hand=hand),
            opponent_state(opponent_active),
        ],
    )
    return Observation(select=select, logs=[], current=current)


def attack_option(attack_id: int) -> Option:
    return Option(type=OptionType.ATTACK, attackId=attack_id)


def attach_option(
    *,
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


def retreat_option(*, bench_index: int | None = None) -> Option:
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


def ability_option(*, area: AreaType, index: int) -> Option:
    return Option(type=OptionType.ABILITY, area=area, index=index)


def end_option() -> Option:
    return Option(type=OptionType.END)


def hand_card(card_id: int, *, serial: int = 10) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=0)


def pokemon_card(
    card_id: int,
    *,
    hp: int,
    attacks: list[int],
    energy_type: EnergyType = EnergyType.FIRE,
    retreat_cost: int = 1,
    ex: bool = False,
    skills: list[Skill] | None = None,
) -> CardData:
    return CardData(
        cardId=card_id,
        name=f"Pokemon {card_id}",
        cardType=CardType.POKEMON,
        retreatCost=retreat_cost,
        hp=hp,
        weakness=None,
        resistance=None,
        energyType=energy_type,
        basic=True,
        stage1=False,
        stage2=False,
        ex=ex,
        megaEx=False,
        tera=False,
        aceSpec=False,
        evolvesFrom=None,
        skills=[] if skills is None else skills,
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
    return pokemon_card(card_id, hp=100, attacks=[3])


def attack(
    attack_id: int,
    *,
    name: str,
    damage: int,
    energies: list[EnergyType] | None = None,
    text: str = "",
) -> Attack:
    return Attack(
        attackId=attack_id,
        name=name,
        text=text,
        damage=damage,
        energies=[] if energies is None else energies,
    )


def deck_card_ids() -> list[int]:
    deck_path = Path(__file__).resolve().parents[2] / "deck.csv"
    return [
        int(line.strip())
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
