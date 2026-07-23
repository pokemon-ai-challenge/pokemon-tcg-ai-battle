"""hidden_information.match_context のユニットテスト。

実装プラン Phase 4 のテスト方針: 手組みの ``Observation`` で「デッキ選択 → 数ターン進行 →
リセットして次の試合」を模擬する（cg エンジンの起動不要。``test_own_hidden_state.py`` /
``test_opponent_hidden_state.py`` と同じスタイル）。

``read_deck_csv()`` はカレントディレクトリの ``deck.csv`` に依存し、pytest の実行ディレクトリ
（リポジトリルート想定）によっては見つからない場合がある。本テストは deck.csv の中身に依存しない
ようにするため、``match_context._load_own_deck_ids`` を合成デッキへ差し替える
（``HybridDeckPredictor`` / ``OpponentHiddenState`` のデフォルト重み・プールJSONは
``Path(__file__).parent`` 基準の絶対パスなのでカレントディレクトリに依存せず、実物をそのまま使う）。
"""

from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Card, Observation, PlayerState, Pokemon, SelectContext, SelectData, State
from ptcg_ai.hidden_information import match_context

SCRAFTY_A = 21  # "Scrafty" (reprint 1) -- test_own_hidden_state.py と同じ実カードIDを流用
ENERGY_CARD = 1  # "Basic {G} Energy"
ITEM_CARD = 2


def make_deck() -> list[int]:
    """2xSCRAFTY_A, 50xENERGY_CARD, 8xITEM_CARD = 60枚の合成デッキ。"""
    return [SCRAFTY_A] * 2 + [ENERGY_CARD] * 50 + [ITEM_CARD] * 8


def make_pokemon(card_id, serial):
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=60,
        maxHp=60,
        appearThisTurn=False,
        energies=[],
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def make_player_state(deck_count, prize, active=None, bench=None, discard=None, hand=None, hand_count=None):
    return PlayerState(
        active=active or [],
        bench=bench or [],
        benchMax=5,
        deckCount=deck_count,
        discard=discard or [],
        prize=prize,
        handCount=hand_count if hand_count is not None else (len(hand) if hand else 0),
        hand=hand,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def make_state(your_player, opponent_player, your_index=0, turn=1, looking=None, stadium=None):
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
        stadium=stadium or [],
        looking=looking,
        players=players,
    )


def make_select(min_count=0, max_count=1, option=None, deck=None, effect=None, context=SelectContext.MAIN):
    return SelectData(
        type=1,
        context=context,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=option or [],
        deck=deck,
        contextCard=None,
        effect=effect,
    )


def _obs(select, current, logs=None) -> Observation:
    return Observation(select=select, logs=logs or [], current=current)


def _deck_select_obs() -> Observation:
    return _obs(select=None, current=None)


def _patch_deck_loader(monkeypatch) -> None:
    monkeypatch.setattr(match_context, "_load_own_deck_ids", make_deck)


# ----------------------------------------------------------------------
# デッキ選択ターン: reset() されて自分の60枚が登録される


def test_deck_selection_turn_registers_own_deck(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()

    match_context.update(_deck_select_obs())

    own = match_context.get_own_state()
    assert sum(own._pool.values()) == 60
    assert own._pool[SCRAFTY_A] == 2
    assert own._pool[ENERGY_CARD] == 50
    assert own._pool[ITEM_CARD] == 8


def test_update_is_noop_safe_before_any_call():
    # reset()すら呼ばれていない状態でも get_own_state()/get_opponent_state() は例外を出さない。
    match_context.reset()
    own = match_context.get_own_state()
    opponent = match_context.get_opponent_state()
    assert own is not None
    assert opponent is not None
    assert opponent.sample() == ([], [], [])


# ----------------------------------------------------------------------
# 通常ターンの進行: OwnHiddenState / OpponentHiddenState が蓄積される


def test_normal_turns_update_own_and_opponent_state(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()
    match_context.update(_deck_select_obs())

    # ターン1: 自分の手札にENERGY_CARDが1枚見えている。相手はまだ何も見えていない。
    my_player = make_player_state(
        deck_count=53,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
    )
    opp_player = make_player_state(deck_count=53, prize=[None] * 6, hand_count=7)
    state1 = make_state(my_player, opp_player, turn=1)
    select1 = make_select()
    match_context.update(_obs(select=select1, current=state1, logs=[]))

    own = match_context.get_own_state()
    assert sum(own._pool.values()) == 53 + 6  # deckCount + len(prize)
    assert own._pool[ENERGY_CARD] == 49  # 50 - 1(手札観測分)

    opponent = match_context.get_opponent_state()
    # update()済みなので、そのプレイヤーのゾーンサイズが反映されている(sample()の長さで確認)。
    deck, hand, prize = opponent.sample()
    assert len(deck) == 53
    assert len(hand) == 7
    assert len(prize) == 6

    # ターン2: 相手がSCRAFTY_A(基本ポケモン)をバトル場に出した(observed_card_idsに反映される)。
    opp_active = make_pokemon(SCRAFTY_A, serial=500)
    my_player2 = make_player_state(deck_count=52, prize=[None] * 6, hand=[])
    opp_player2 = make_player_state(deck_count=52, prize=[None] * 6, hand_count=6, active=[opp_active])
    state2 = make_state(my_player2, opp_player2, turn=2)
    select2 = make_select()
    match_context.update(_obs(select=select2, current=state2, logs=[]))

    opponent2 = match_context.get_opponent_state()
    deck2, hand2, prize2 = opponent2.sample()
    assert len(deck2) == 52
    assert len(hand2) == 6
    assert len(prize2) == 6


# ----------------------------------------------------------------------
# 山札サーチ検知: select.deck is not None で resolve_deck_search が呼ばれる


def test_deck_search_reveal_confirms_prize_and_deck(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()
    match_context.update(_deck_select_obs())

    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    opp_player = make_player_state(deck_count=53, prize=[None] * 6, hand_count=7)
    state = make_state(my_player, opp_player, turn=1)

    # 山札54枚のうちSCRAFTY_Aが1枚も写っていない -> 2枚ともサイド確定になる想定。
    deck_reveal = (
        [Card(id=ENERGY_CARD, serial=1000 + i, playerIndex=0) for i in range(50)]
        + [Card(id=ITEM_CARD, serial=2000 + i, playerIndex=0) for i in range(4)]
    )
    assert len(deck_reveal) == 54
    select = make_select(deck=deck_reveal, context=SelectContext.LOOK)
    match_context.update(_obs(select=select, current=state, logs=[]))

    own = match_context.get_own_state()
    marg = own.marginals()
    assert marg[SCRAFTY_A] == {"deck": 0.0, "prize": 1.0}


# ----------------------------------------------------------------------
# reset: 新しい試合の開始検知(obs.select is None)で状態が完全に作り直される


def test_new_deck_selection_resets_previous_match_state(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()
    match_context.update(_deck_select_obs())

    my_player = make_player_state(
        deck_count=53,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
    )
    opp_player = make_player_state(deck_count=53, prize=[None] * 6, hand_count=7)
    state = make_state(my_player, opp_player, turn=1)
    match_context.update(_obs(select=make_select(), current=state, logs=[]))

    own_before = match_context.get_own_state()
    assert own_before._pool[ENERGY_CARD] == 49  # 前の試合の観測が反映されている

    # 新しい試合が始まった(obs.select is None)。
    match_context.update(_deck_select_obs())

    own_after = match_context.get_own_state()
    assert own_after is not own_before  # 新しいインスタンスに作り直されている
    assert sum(own_after._pool.values()) == 60  # 観測前の状態(フル60枚)に戻っている
    assert own_after._pool[ENERGY_CARD] == 50

    opponent_after = match_context.get_opponent_state()
    assert opponent_after.sample() == ([], [], [])  # update()未呼び出し相当にリセットされている


# ----------------------------------------------------------------------
# 例外安全性: 推定レイヤー内部で例外が起きても update() は伝播させない


class _RaisingKnowledge:
    def update_from_logs(self, logs):
        raise RuntimeError("boom")

    def update_from_state(self, state):
        raise RuntimeError("boom")


def test_update_swallows_internal_exceptions(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()
    match_context.update(_deck_select_obs())

    my_player = make_player_state(deck_count=53, prize=[None] * 6)
    opp_player = make_player_state(deck_count=53, prize=[None] * 6, hand_count=7)
    state = make_state(my_player, opp_player, turn=1)

    # まず正常な1ターンを進めて_knowledgeを初期化させる。
    match_context.update(_obs(select=make_select(), current=state, logs=[]))
    own_before = match_context.get_own_state()

    # 内部の_knowledgeを壊れたものに差し替える。
    monkeypatch.setattr(match_context, "_knowledge", _RaisingKnowledge())

    # 例外が外に漏れないこと(戻り値はNoneのまま、呼び出し自体が失敗しない)。
    result = match_context.update(_obs(select=make_select(), current=state, logs=[]))
    assert result is None

    # 既存の推定状態はクラッシュしていない(引き続き参照できる)。
    own_after = match_context.get_own_state()
    assert own_after is own_before  # 例外発生時は更新されず、直前の状態のまま残る
