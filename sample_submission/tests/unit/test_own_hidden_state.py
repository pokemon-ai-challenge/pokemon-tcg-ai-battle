from pathlib import Path
import sys
from collections import Counter

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import random

from cg.api import Card, PlayerState, Pokemon, SelectContext, SelectData, State
from ptcg_ai.hidden_information.own_hidden_state import OwnHiddenState

# 実カードIDを流用(test_opponent_knowledge.pyと同じ方針)。値そのものの意味は使わない。
SCRAFTY_A = 21   # "Scrafty" (reprint 1)
SCRAFTY_B = 611  # "Scrafty" (reprint 2, same name, different card_id) -- 同名別card_idケース
ENERGY_CARD = 1  # "Basic {G} Energy"
TOOL_CARD = 1154  # a Pokemon Tool card
ITEM_CARD = 2  # 任意の非ポケモンカード


def make_pokemon(card_id, serial, energy_cards=None, tools=None, pre_evolution=None):
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=60,
        maxHp=60,
        appearThisTurn=False,
        energies=[],
        energyCards=energy_cards or [],
        tools=tools or [],
        preEvolution=pre_evolution or [],
    )


def make_player_state(deck_count, prize, active=None, bench=None, discard=None, hand=None):
    return PlayerState(
        active=active or [],
        bench=bench or [],
        benchMax=5,
        deckCount=deck_count,
        discard=discard or [],
        prize=prize,
        handCount=len(hand) if hand else 0,
        hand=hand,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def make_state(your_player, opponent_player, your_index=0, turn=1, looking=None):
    players = [None, None]
    players[your_index] = your_player
    players[1 - your_index] = opponent_player
    return State(
        turn=turn,
        turnActionCount=0,
        yourIndex=your_index,
        firstPlayer=your_index,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=-1,
        stadium=[],
        looking=looking,
        players=players,
    )


def make_deck():
    """2xSCRAFTY_A, 2xSCRAFTY_B, 50xENERGY_CARD, 3xTOOL_CARD, 3xITEM_CARD = 60枚。"""
    return (
        [SCRAFTY_A] * 2
        + [SCRAFTY_B] * 2
        + [ENERGY_CARD] * 50
        + [TOOL_CARD] * 3
        + [ITEM_CARD] * 3
    )


def opponent_placeholder():
    return make_player_state(deck_count=50, prize=[None] * 6)


# ----------------------------------------------------------------------
# 観点2: update() が公開ゾーンを正しく引き算する(同名別card_idケース込み)


def test_update_subtracts_hand_board_and_discard():
    own = OwnHiddenState(make_deck())

    my_player = make_player_state(
        deck_count=48,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
        active=[make_pokemon(
            SCRAFTY_A, serial=1,
            energy_cards=[Card(id=ENERGY_CARD, serial=101, playerIndex=0)],
            tools=[Card(id=TOOL_CARD, serial=102, playerIndex=0)],
        )],
        bench=[make_pokemon(SCRAFTY_B, serial=2)],
        discard=[Card(id=ITEM_CARD, serial=103, playerIndex=0)],
    )
    state = make_state(my_player, opponent_placeholder())

    own.update(state)

    # プール = フル60枚 - 観測分。ENERGY_CARD: 50-2=48, SCRAFTY_A: 2-1=1, SCRAFTY_B: 2-1=1,
    # TOOL_CARD: 3-1=2, ITEM_CARD: 3-1=2。合計 = 48+1+1+2+2 = 54 = deckCount(48)+prize(6)。
    pool = own._pool
    assert pool[ENERGY_CARD] == 48
    assert pool[SCRAFTY_A] == 1
    assert pool[SCRAFTY_B] == 1
    assert pool[TOOL_CARD] == 2
    assert pool[ITEM_CARD] == 2
    assert sum(pool.values()) == 54


def test_update_subtracts_pre_evolution():
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(
        deck_count=52,  # 60 - (SCRAFTY_A本体1 + SCRAFTY_B進化元1として観測) - prize(6) = 52
        prize=[None] * 6,
        active=[make_pokemon(SCRAFTY_A, serial=1, pre_evolution=[Card(id=SCRAFTY_B, serial=2, playerIndex=0)])],
    )
    own.update(make_state(my_player, opponent_placeholder()))

    assert own._pool[SCRAFTY_A] == 1  # 2 - 1(active本体)
    assert own._pool[SCRAFTY_B] == 1  # 2 - 1(進化元として公開)


# ----------------------------------------------------------------------
# 観点3: 不変条件が毎回成り立つ


def test_invariant_holds_after_each_update_across_multiple_turns():
    own = OwnHiddenState(make_deck())

    turn1 = make_player_state(deck_count=54, prize=[None] * 6)
    own.update(make_state(turn1, opponent_placeholder(), turn=1))
    assert sum(own._pool.values()) == turn1.deckCount + len(turn1.prize)  # 54 + 6 = 60

    turn2 = make_player_state(
        # 観測3枚(ENERGY_CARD x1 hand, SCRAFTY_A x1 active, ITEM_CARD x1 discard) -> pool=60-3=57
        deck_count=51,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
        active=[make_pokemon(SCRAFTY_A, serial=1)],
        discard=[Card(id=ITEM_CARD, serial=101, playerIndex=0)],
    )
    own.update(make_state(turn2, opponent_placeholder(), turn=2))
    assert sum(own._pool.values()) == turn2.deckCount + len(turn2.prize)

    # サイドが1枚取られた(prizeが5枚に)ケースも成り立つ。
    turn3 = make_player_state(
        # 観測4枚(ENERGY_CARD x2, SCRAFTY_A x1, ITEM_CARD x1) -> pool=60-4=56
        deck_count=51,
        prize=[None] * 5,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
        active=[make_pokemon(SCRAFTY_A, serial=1)],
        discard=[Card(id=ITEM_CARD, serial=101, playerIndex=0), Card(id=ENERGY_CARD, serial=104, playerIndex=0)],
    )
    own.update(make_state(turn3, opponent_placeholder(), turn=3))
    assert sum(own._pool.values()) == turn3.deckCount + len(turn3.prize)


def test_update_assertion_fires_on_overcounted_zone():
    # 公開ゾーンの合計が元の60枚を超えるような壊れたStateは、不変条件アサートで検知される。
    own = OwnHiddenState([SCRAFTY_A])  # 1枚だけのデッキ
    broken_player = make_player_state(
        deck_count=0,
        prize=[],
        hand=[Card(id=SCRAFTY_A, serial=1, playerIndex=0)],
        active=[make_pokemon(SCRAFTY_A, serial=2)],  # 同じcard_idをもう1体、デッキに無い分まで観測
    )
    try:
        own.update(make_state(broken_player, opponent_placeholder()))
        assert False, "AssertionErrorが発生するはず"
    except AssertionError:
        pass


# ----------------------------------------------------------------------
# 観点4: resolve_deck_search() による決定的な消し込み


def _look_select(deck_cards, context=SelectContext.LOOK):
    return SelectData(
        type=1,
        context=context,
        minCount=0,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[],
        deck=deck_cards,
        contextCard=None,
        effect=None,
    )


def test_resolve_deck_search_confirms_cards_missing_from_deck_reveal():
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    own.update(make_state(my_player, opponent_placeholder()))

    # 未確認プール(合計60。updateでは観測ゼロなのでpoolはフル60のまま):
    #   SCRAFTY_A:2, SCRAFTY_B:2, ENERGY_CARD:50, TOOL_CARD:3, ITEM_CARD:3
    # 実データ検証どおり select.deck は「今の山札54枚全部」を写す(len(select.deck)==deckCount)。
    # SCRAFTY_A/SCRAFTY_Bは1枚も写っていない(4枚ともサイド確定)。ITEM_CARDは1枚だけ写っている
    # (2枚がサイド確定)。ENERGY_CARD/TOOL_CARDは全部写っている(サイドには無い)。
    # 確定枚数の合計 = 2+2+2 = 6 = prize枚数と一致(山札が丸ごと写る=その瞬間に全解決する)。
    deck_reveal = (
        [Card(id=ENERGY_CARD, serial=1000 + i, playerIndex=0) for i in range(50)]
        + [Card(id=TOOL_CARD, serial=2000 + i, playerIndex=0) for i in range(3)]
        + [Card(id=ITEM_CARD, serial=3000, playerIndex=0)]
    )
    assert len(deck_reveal) == 54 == my_player.deckCount

    select = _look_select(deck_reveal)
    own.resolve_deck_search(select)

    marginals = own.marginals()
    # SCRAFTY_A/SCRAFTY_B: 2枚とも写っていない -> 確定的にサイド落ち
    assert marginals[SCRAFTY_A] == {"deck": 0.0, "prize": 1.0}
    assert marginals[SCRAFTY_B] == {"deck": 0.0, "prize": 1.0}
    # ITEM_CARD: 3枚中1枚だけ写っている -> 少なくとも1枚はサイド確定(prize=1.0)。
    # 山札が丸ごと写った瞬間に全カード合わせて完全解決するため、残り1枚も消去法でdeck確定(deck=1.0)。
    assert marginals[ITEM_CARD] == {"deck": 1.0, "prize": 1.0}
    # ENERGY_CARD/TOOL_CARDは全部写っている -> サイド落ちなし、山札に確定
    assert marginals[ENERGY_CARD] == {"deck": 1.0, "prize": 0.0}
    assert marginals[TOOL_CARD] == {"deck": 1.0, "prize": 0.0}


def test_resolve_deck_search_ignores_when_deck_is_none():
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    own.update(make_state(my_player, opponent_placeholder()))

    before = own.marginals()
    select = _look_select(None)
    own.resolve_deck_search(select)
    after = own.marginals()

    assert before == after


def test_resolve_deck_search_fires_regardless_of_context_value():
    # 実データ検証結果: select.deck is not None であれば context がLOOK以外(TO_HAND/TO_BENCH等)でも
    # 発火させる必要がある(SelectContext.LOOKは検証中一度も発火しなかった)。
    own = OwnHiddenState([SCRAFTY_A, SCRAFTY_A])
    my_player = make_player_state(deck_count=2, prize=[None] * 0)
    own.update(make_state(my_player, opponent_placeholder()))
    assert sum(own._pool.values()) == 2

    my_player2 = make_player_state(deck_count=1, prize=[None])
    own.update(make_state(my_player2, opponent_placeholder()))

    deck_reveal = [Card(id=SCRAFTY_A, serial=1, playerIndex=0)]  # 山札に1枚だけ写っている
    select = _look_select(deck_reveal, context=SelectContext.TO_HAND)
    own.resolve_deck_search(select)

    assert own.marginals()[SCRAFTY_A] == {"deck": 1.0, "prize": 1.0}
    # (2枚中1枚は山札に見えている=deck確定寄りだが、もう1枚はサイド確定=prize確定寄り。
    #  同一card_idの一部だけ確定したケースなので、両方の"少なくとも1枚存在する"確率が1.0になる。)


# ----------------------------------------------------------------------
# 観点5: sample() が未確認プール全体と一致する


def test_sample_matches_pool_exactly_no_duplication_or_omission():
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(
        # 観測3枚(ENERGY_CARD x1 hand, SCRAFTY_A x1 active, ITEM_CARD x1 discard) -> pool=60-3=57
        deck_count=51,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
        active=[make_pokemon(SCRAFTY_A, serial=1)],
        discard=[Card(id=ITEM_CARD, serial=103, playerIndex=0)],
    )
    state = make_state(my_player, opponent_placeholder())
    own.update(state)

    rng = random.Random(42)
    for _ in range(20):
        deck_ids, prize_ids = own.sample(rng)

        assert len(prize_ids) == len(state.players[state.yourIndex].prize)
        assert len(deck_ids) == state.players[state.yourIndex].deckCount

        combined = Counter(deck_ids) + Counter(prize_ids)
        assert combined == own._pool  # 二重計上・欠落なし


def test_sample_prize_always_includes_confirmed_cards():
    own = OwnHiddenState(make_deck())
    # deckCount=56, prize=4枚: SCRAFTY_A/SCRAFTY_B(計4枚)だけがサイドで、残り56枚が山札という設定。
    my_player = make_player_state(deck_count=56, prize=[None] * 4)
    state = make_state(my_player, opponent_placeholder())
    own.update(state)

    deck_reveal = (
        [Card(id=ENERGY_CARD, serial=1000 + i, playerIndex=0) for i in range(50)]
        + [Card(id=TOOL_CARD, serial=2000 + i, playerIndex=0) for i in range(3)]
        + [Card(id=ITEM_CARD, serial=3000 + i, playerIndex=0) for i in range(3)]
    )  # SCRAFTY_A/SCRAFTY_Bは1枚も写っていない -> 4枚ともサイド確定
    assert len(deck_reveal) == 56 == my_player.deckCount
    own.resolve_deck_search(_look_select(deck_reveal))

    rng = random.Random(7)
    for _ in range(10):
        deck_ids, prize_ids = own.sample(rng)
        assert SCRAFTY_A not in deck_ids
        assert SCRAFTY_B not in deck_ids
        assert Counter(prize_ids)[SCRAFTY_A] == 2
        assert Counter(prize_ids)[SCRAFTY_B] == 2
        assert len(prize_ids) == len(state.players[state.yourIndex].prize)
        assert len(deck_ids) == state.players[state.yourIndex].deckCount


def test_sample_without_rng_uses_module_random():
    # rng省略時も例外なく動作し、正しいサイズになることを確認する。
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    state = make_state(my_player, opponent_placeholder())
    own.update(state)

    deck_ids, prize_ids = own.sample()
    assert len(prize_ids) == 6
    assert len(deck_ids) == 54
