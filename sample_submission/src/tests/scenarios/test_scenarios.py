"""固定シナリオテスト: 選択ロジックが意図したカードを選べるか確認する。

実行方法:
    cd sample_submission
    python -m pytest src/tests/scenarios/test_scenarios.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from unittest.mock import patch
from cg.api import (
    Observation, SelectData, SelectContext, SelectType,
    OptionType, Option, State, PlayerState, Card, Pokemon,
    AreaType, EnergyType,
)
from src.decision.setup_policy import choose_setup_active, choose_setup_bench
from src.decision.target_policy import choose_evolves_from, choose_attack
from src.knowledge.card_database import Attack


# --- ヘルパー ---

def _make_card(card_id: int, serial: int = 0, player: int = 0) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=player)


def _make_player(hand_ids: list[int], bench_ids: list[int] = None) -> PlayerState:
    hand = [_make_card(cid, i) for i, cid in enumerate(hand_ids)]
    bench = []
    if bench_ids:
        bench = [
            Pokemon(id=cid, serial=i, hp=100, maxHp=100, appearThisTurn=False,
                    energies=[], energyCards=[], tools=[], preEvolution=[])
            for i, cid in enumerate(bench_ids)
        ]
    return PlayerState(
        active=[], bench=bench, benchMax=5, deckCount=50,
        discard=[], prize=[], handCount=len(hand_ids), hand=hand,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def _make_state(my_hand: list[int], my_bench: list[int] = None,
                opp_active_hp: int = 100) -> State:
    opp_active_poke = Pokemon(
        id=999, serial=99, hp=opp_active_hp, maxHp=opp_active_hp,
        appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[],
    )
    opp_player = PlayerState(
        active=[opp_active_poke], bench=[], benchMax=5, deckCount=50,
        discard=[], prize=[], handCount=5, hand=None,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )
    return State(
        turn=1, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False,
        energyAttached=False, retreated=False,
        result=-1, stadium=[], looking=None,
        players=[_make_player(my_hand, my_bench), opp_player],
    )


def _make_card_options(card_ids: list[int]) -> list[Option]:
    return [
        Option(type=OptionType.CARD, area=AreaType.HAND, index=i,
               playerIndex=0, cardId=cid)
        for i, cid in enumerate(card_ids)
    ]


def _make_select(ctx: SelectContext, options: list[Option],
                 min_count: int = 1, max_count: int = 1) -> SelectData:
    return SelectData(
        type=SelectType.CARD, context=ctx,
        minCount=min_count, maxCount=max_count,
        remainDamageCounter=0, remainEnergyCost=0,
        option=options, deck=None, contextCard=None, effect=None,
    )


def _make_obs(ctx: SelectContext, options: list[Option],
              state: State, min_count: int = 1, max_count: int = 1) -> Observation:
    return Observation(
        select=_make_select(ctx, options, min_count, max_count),
        logs=[],
        current=state,
    )


# --- シナリオ 1: 初手バトルポケモン選択 ---

def test_setup_active_prefers_riolu_over_makuhita():
    """active_priority に従い、マクノシタよりリオルをバトル場に出す。"""
    RIOLU = 677
    MAKUHITA = 673

    # オプション: マクノシタ(index 0)、リオル(index 1) の順に並べる
    options = _make_card_options([MAKUHITA, RIOLU])
    state = _make_state(my_hand=[MAKUHITA, RIOLU])
    obs = _make_obs(SelectContext.SETUP_ACTIVE_POKEMON, options, state)

    result = choose_setup_active(obs)

    assert result == [1], f"リオル(index 1)を選ぶべきだが {result} が返された"


def test_setup_active_fallback_when_no_priority():
    """優先リストにないカードしかない場合は先頭を選ぶ。"""
    SOLROCK = 676

    options = _make_card_options([SOLROCK])
    state = _make_state(my_hand=[SOLROCK])
    obs = _make_obs(SelectContext.SETUP_ACTIVE_POKEMON, options, state)

    result = choose_setup_active(obs)

    assert result == [0], f"先頭(index 0)を選ぶべきだが {result} が返された"


def test_setup_bench_priority_order():
    """bench_priority に従い、リオル → マクノシタの順にベンチに出す。"""
    RIOLU = 677
    MAKUHITA = 673
    SOLROCK = 676

    # オプション: ソルロック(0), マクノシタ(1), リオル(2) の逆順
    options = _make_card_options([SOLROCK, MAKUHITA, RIOLU])
    state = _make_state(my_hand=[SOLROCK, MAKUHITA, RIOLU])
    obs = _make_obs(SelectContext.SETUP_BENCH_POKEMON, options, state, min_count=2, max_count=2)

    result = choose_setup_bench(obs)

    assert len(result) == 2, f"2枚選ぶべきだが {len(result)} 枚"
    assert 2 in result, "リオル(index 2)を選ぶべき"
    assert 1 in result, "マクノシタ(index 1)を選ぶべき"


# --- シナリオ 2: 進化選択 ---

def test_evolves_from_prefers_riolu_over_makuhita():
    """evolution_priority に従い、ハリテヤマ進化元よりメガルカリオex進化元を選ぶ。"""
    RIOLU = 677     # → メガルカリオex(678): 優先度 高
    MAKUHITA = 673  # → ハリテヤマ(674): 優先度 低

    # オプション: マクノシタ(index 0)、リオル(index 1) の順（逆順にして選択が実際に起きるか確認）
    options = [
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=0,
               playerIndex=0, cardId=MAKUHITA),
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=1,
               playerIndex=0, cardId=RIOLU),
    ]
    state = _make_state(my_hand=[], my_bench=[MAKUHITA, RIOLU])
    obs = _make_obs(SelectContext.EVOLVES_FROM, options, state)

    result = choose_evolves_from(obs)

    assert result == [1], f"リオル(index 1)を選ぶべきだが {result} が返された"


# --- シナリオ 3: 即倒しできる攻撃選択 ---

def test_attack_picks_lethal_over_low_damage():
    """相手HPに届く攻撃が存在する場合、ダメージが低い攻撃より一撃KO攻撃を選ぶ。"""
    ATTACK_ID_WEAK = 1001    # ダメージ 40
    ATTACK_ID_LETHAL = 1002  # ダメージ 70 = 相手HP と同等

    mock_attacks = {
        ATTACK_ID_WEAK: Attack(attackId=ATTACK_ID_WEAK, name="Weak", text="", damage=40, energies=[]),
        ATTACK_ID_LETHAL: Attack(attackId=ATTACK_ID_LETHAL, name="Strong", text="", damage=70, energies=[]),
    }

    options = [
        Option(type=OptionType.ATTACK, attackId=ATTACK_ID_WEAK),   # index 0
        Option(type=OptionType.ATTACK, attackId=ATTACK_ID_LETHAL),  # index 1
    ]
    select = SelectData(
        type=SelectType.ATTACK, context=SelectContext.ATTACK,
        minCount=1, maxCount=1, remainDamageCounter=0, remainEnergyCost=0,
        option=options, deck=None, contextCard=None, effect=None,
    )
    state = _make_state(my_hand=[], opp_active_hp=70)
    obs = Observation(select=select, logs=[], current=state)

    with patch("src.decision.target_policy.get_attack_db", return_value=mock_attacks):
        result = choose_attack(obs)

    assert result == [1], f"一撃KO攻撃(index 1)を選ぶべきだが {result} が返された"


def test_attack_picks_max_damage_when_no_lethal():
    """一撃KOできない場合は最大ダメージの攻撃を選ぶ。"""
    ATTACK_ID_LOW = 1001   # ダメージ 40
    ATTACK_ID_HIGH = 1002  # ダメージ 70

    mock_attacks = {
        ATTACK_ID_LOW: Attack(attackId=ATTACK_ID_LOW, name="Low", text="", damage=40, energies=[]),
        ATTACK_ID_HIGH: Attack(attackId=ATTACK_ID_HIGH, name="High", text="", damage=70, energies=[]),
    }

    options = [
        Option(type=OptionType.ATTACK, attackId=ATTACK_ID_LOW),   # index 0
        Option(type=OptionType.ATTACK, attackId=ATTACK_ID_HIGH),  # index 1
    ]
    select = SelectData(
        type=SelectType.ATTACK, context=SelectContext.ATTACK,
        minCount=1, maxCount=1, remainDamageCounter=0, remainEnergyCost=0,
        option=options, deck=None, contextCard=None, effect=None,
    )
    state = _make_state(my_hand=[], opp_active_hp=200)  # 一撃では倒せない
    obs = Observation(select=select, logs=[], current=state)

    with patch("src.decision.target_policy.get_attack_db", return_value=mock_attacks):
        result = choose_attack(obs)

    assert result == [1], f"高ダメージ攻撃(index 1)を選ぶべきだが {result} が返された"
