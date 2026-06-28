from cg.api import (
    Attack,
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

from src.decision.attack_turn import choose_attack_action
from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.priorities import attack as attack_priority


def test_attack_phase_prefers_clean_knockout_over_heavier_recoil_attack(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Precise Finish", text="Draw a card.", damage=120, energies=[]),
            2: Attack(
                attackId=2,
                name="Wild Breaker",
                text="Discard all Energy from this Pokemon.",
                damage=180,
                energies=[],
            ),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=120,
    )

    assert choose_attack_action(obs) == [0]


def test_attack_phase_uses_secondary_effects_when_damage_is_close(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Heavy Swing", text="", damage=90, energies=[]),
            2: Attack(
                attackId=2,
                name="Spread Shot",
                text="This attack also does 10 damage to 1 of your opponent's Benched Pokemon.",
                damage=80,
                energies=[],
            ),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=200,
    )

    assert choose_attack_action(obs) == [1]


def test_attack_phase_penalizes_next_turn_lockout_when_not_knocking_out(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(
                attackId=1,
                name="Exhaust Cannon",
                text="During your next turn, this Pokemon can't use attacks.",
                damage=160,
                energies=[],
            ),
            2: Attack(attackId=2, name="Reliable Hit", text="", damage=150, energies=[]),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=220,
    )

    assert choose_attack_action(obs) == [1]


def test_main_phase_attack_proposal_reuses_attack_scoring(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Heavy Swing", text="", damage=90, energies=[]),
            2: Attack(
                attackId=2,
                name="Spread Shot",
                text="This attack also does 10 damage to 1 of your opponent's Benched Pokemon.",
                damage=80,
                energies=[],
            ),
        },
    )

    obs = build_observation(
        context=SelectContext.MAIN,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=200,
    )

    proposal = attack_priority.propose_attack_action(obs, MainOptionBuckets(attack=[0, 1]))

    assert proposal is not None
    assert proposal.action == [1]


def test_registered_profile_treats_solrock_as_whiff_without_lunatone(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            979: Attack(attackId=979, name="Power Gem", text="", damage=50, energies=[]),
            980: Attack(
                attackId=980,
                name="Cosmic Beam",
                text="If you don't have Lunatone on your Bench, this attack does nothing.",
                damage=70,
                energies=[],
            ),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(979),
            attack_option(980),
        ],
        opponent_hp=200,
    )

    assert choose_attack_action(obs) == [0]


def test_registered_profile_prefers_aura_jab_over_mega_brave_for_same_knockout(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            982: Attack(
                attackId=982,
                name="Aura Jab",
                text="Attach up to 3 Basic Energy cards from your discard pile to your Benched Pokemon in any way you like.",
                damage=130,
                energies=[],
            ),
            983: Attack(
                attackId=983,
                name="Mega Brave",
                text="During your next turn, this Pokemon can't use Mega Brave.",
                damage=270,
                energies=[],
            ),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(982),
            attack_option(983),
        ],
        opponent_hp=130,
    )

    assert choose_attack_action(obs) == [0]


def test_attack_phase_uses_remaining_hp_not_max_hp_for_knockout_check(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Exact Shot", text="", damage=50, energies=[]),
            2: Attack(attackId=2, name="Light Hit", text="", damage=30, energies=[]),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=40,
        opponent_max_hp=120,
    )

    assert choose_attack_action(obs) == [0]


def test_attack_phase_breaks_same_score_ties_by_lower_option_index(monkeypatch):
    monkeypatch.setattr(
        attack_priority,
        "build_attack_lookup",
        lambda: {
            1: Attack(attackId=1, name="Twin Strike A", text="", damage=90, energies=[]),
            2: Attack(attackId=2, name="Twin Strike B", text="", damage=90, energies=[]),
        },
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            attack_option(1),
            attack_option(2),
        ],
        opponent_hp=200,
    )

    assert choose_attack_action(obs) == [0]


def test_attack_phase_falls_back_when_no_attack_option_exists(monkeypatch):
    monkeypatch.setattr(
        "src.decision.attack_turn.choose_random_legal_action",
        lambda obs: [0],
    )

    obs = build_observation(
        context=SelectContext.ATTACK,
        options=[
            Option(type=OptionType.YES),
        ],
        opponent_hp=200,
    )

    assert choose_attack_action(obs) == [0]


def build_observation(
    context: SelectContext,
    options: list[Option],
    opponent_hp: int,
    opponent_max_hp: int | None = None,
    your_bench: list[Pokemon] | None = None,
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
    current = State(
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
                active=[
                    Pokemon(
                        id=100,
                        serial=1,
                        hp=150,
                        maxHp=150,
                        appearThisTurn=False,
                        energies=[],
                        energyCards=[],
                        tools=[],
                        preEvolution=[],
                    )
                ],
                bench=[] if your_bench is None else your_bench,
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
                active=[
                    Pokemon(
                        id=200,
                        serial=2,
                        hp=opponent_hp,
                        maxHp=opponent_hp if opponent_max_hp is None else opponent_max_hp,
                        appearThisTurn=False,
                        energies=[],
                        energyCards=[],
                        tools=[],
                        preEvolution=[],
                    )
                ],
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


def attack_option(attack_id: int) -> Option:
    return Option(type=OptionType.ATTACK, attackId=attack_id)
