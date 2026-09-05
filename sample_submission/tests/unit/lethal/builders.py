"""合成 Observation ビルダ(単体テスト用)。

エンジンを起動せずに ``cg.api`` の dataclass を直接組み立てる。
``tests/conftest.py`` の方針(最小限の Observation を組んで渡す)に合わせている。
"""

from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import (
    AreaType,
    Card,
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


def card(card_id: int, serial: int, player_index: int = 0) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=player_index)


def pokemon(card_id: int, serial: int, hp: int = 100, max_hp: int = 100) -> Pokemon:
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=hp,
        maxHp=max_hp,
        appearThisTurn=False,
        energies=[],
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def player(
    *,
    hand: list[Card] | None = None,
    active: list[Pokemon | None] | None = None,
    bench: list[Pokemon] | None = None,
    deck_count: int = 40,
    discard: list[Card] | None = None,
    prize_facedown: int = 2,
) -> PlayerState:
    hand = [] if hand is None else hand
    return PlayerState(
        active=[pokemon(741, 900)] if active is None else active,
        bench=[] if bench is None else bench,
        benchMax=5,
        deckCount=deck_count,
        discard=[] if discard is None else discard,
        prize=[None] * prize_facedown,
        handCount=len(hand),
        hand=hand,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def opponent(**kwargs) -> PlayerState:
    """相手側。手札は常に見えない(``hand=None``)。"""
    state = player(**kwargs)
    state.hand = None
    state.handCount = kwargs.get("hand_count", 5)
    return state


def state(
    *,
    me: PlayerState,
    opp: PlayerState,
    your_index: int = 0,
    turn: int = 5,
    result: int = -1,
    looking: list[Card | None] | None = None,
) -> State:
    players = [me, opp] if your_index == 0 else [opp, me]
    return State(
        turn=turn,
        turnActionCount=1,
        yourIndex=your_index,
        firstPlayer=0,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=result,
        stadium=[],
        looking=looking,
        players=players,
    )


def play_options(hand: list[Card]) -> list[Option]:
    """手札の各カードを出す PLAY 選択肢(index は手札位置)。"""
    return [
        Option(type=OptionType.PLAY, area=AreaType.HAND, index=i, playerIndex=0)
        for i in range(len(hand))
    ]


def select(
    options: list[Option],
    *,
    select_type: SelectType = SelectType.MAIN,
    context: SelectContext = SelectContext.MAIN,
    min_count: int = 1,
    max_count: int = 1,
    deck: list[Card] | None = None,
    effect: Card | None = None,
) -> SelectData:
    return SelectData(
        type=select_type,
        context=context,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=deck,
        contextCard=None,
        effect=effect,
    )


def observation(select_data: SelectData, game_state: State) -> Observation:
    return Observation(select=select_data, logs=[], current=game_state)


def simple_observation(hand_ids: list[int], *, serial_base: int = 1) -> Observation:
    """手札 ``hand_ids`` を持ち、各手札を出す PLAY 選択肢を提示する観測。"""
    hand = [card(cid, serial_base + i) for i, cid in enumerate(hand_ids)]
    me = player(hand=hand)
    opp = opponent()
    return observation(select(play_options(hand)), state(me=me, opp=opp))
