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

    own = match_context.get_own_state(0)
    assert sum(own._pool.values()) == 60
    assert own._pool[SCRAFTY_A] == 2
    assert own._pool[ENERGY_CARD] == 50
    assert own._pool[ITEM_CARD] == 8


def test_update_is_noop_safe_before_any_call():
    # reset()すら呼ばれていない状態でも get_own_state()/get_opponent_state() は例外を出さない。
    match_context.reset()
    own = match_context.get_own_state(0)
    opponent = match_context.get_opponent_state(0)
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

    own = match_context.get_own_state(0)
    assert sum(own._pool.values()) == 53 + 6  # deckCount + len(prize)
    assert own._pool[ENERGY_CARD] == 49  # 50 - 1(手札観測分)

    opponent = match_context.get_opponent_state(0)
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

    opponent2 = match_context.get_opponent_state(0)
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

    own = match_context.get_own_state(0)
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

    own_before = match_context.get_own_state(0)
    assert own_before._pool[ENERGY_CARD] == 49  # 前の試合の観測が反映されている

    # 新しい試合が始まった(obs.select is None)。
    match_context.update(_deck_select_obs())

    own_after = match_context.get_own_state(0)
    assert own_after is not own_before  # 新しいインスタンスに作り直されている
    assert sum(own_after._pool.values()) == 60  # 観測前の状態(フル60枚)に戻っている
    assert own_after._pool[ENERGY_CARD] == 50

    opponent_after = match_context.get_opponent_state(0)
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
    own_before = match_context.get_own_state(0)

    # 内部の_knowledge(player_index=0側)を壊れたものに差し替える。
    monkeypatch.setattr(match_context, "_knowledge", {0: _RaisingKnowledge()})

    # 例外が外に漏れないこと(戻り値はNoneのまま、呼び出し自体が失敗しない)。
    result = match_context.update(_obs(select=make_select(), current=state, logs=[]))
    assert result is None

    # 既存の推定状態はクラッシュしていない(引き続き参照できる)。
    own_after = match_context.get_own_state(0)
    assert own_after is own_before  # 例外発生時は更新されず、直前の状態のまま残る


# ----------------------------------------------------------------------
# プレイヤー分離: player0視点とplayer1視点を交互にupdate()しても互いに独立している
# (Stage2: match_context のプレイヤー分離、本Stageのバグ再発防止テスト)


def test_update_keeps_player_states_independent_when_interleaved(monkeypatch):
    _patch_deck_loader(monkeypatch)
    match_context.reset()
    match_context.update(_deck_select_obs())

    # player0視点: 手札にENERGY_CARDが1枚見えている。相手(player1)はdeck=53/hand=7。
    p0_own = make_player_state(
        deck_count=53,
        prize=[None] * 6,
        hand=[Card(id=ENERGY_CARD, serial=100, playerIndex=0)],
    )
    p0_opp = make_player_state(deck_count=53, prize=[None] * 6, hand_count=7)
    state_p0 = make_state(p0_own, p0_opp, your_index=0, turn=1)
    match_context.update(_obs(select=make_select(), current=state_p0, logs=[]))

    # player1視点: 同一プロセス内で別config(異なるエージェント)が交互に呼ばれる想定。
    # 手札にITEM_CARDが2枚見えている(deckCount+prize=58=60-観測2枚で不変条件と整合させる)。
    # 相手(player0視点から見たplayer0)はdeck=49/hand=5。
    p1_own = make_player_state(
        deck_count=52,
        prize=[None] * 6,
        hand=[
            Card(id=ITEM_CARD, serial=200, playerIndex=1),
            Card(id=ITEM_CARD, serial=201, playerIndex=1),
        ],
    )
    p1_opp = make_player_state(deck_count=49, prize=[None] * 6, hand_count=5)
    state_p1 = make_state(p1_own, p1_opp, your_index=1, turn=1)
    match_context.update(_obs(select=make_select(), current=state_p1, logs=[]))

    # player0視点をもう一度更新する(1枚ドローしてhandがENERGY+ITEMの2枚になった想定、
    # deckCountの53->52という変化がpool側の観測増加と整合するようにする)。
    # player1側の状態には影響しないはず。
    p0_own2 = make_player_state(
        deck_count=52,
        prize=[None] * 6,
        hand=[
            Card(id=ENERGY_CARD, serial=100, playerIndex=0),
            Card(id=ITEM_CARD, serial=300, playerIndex=0),
        ],
    )
    state_p0b = make_state(p0_own2, p0_opp, your_index=0, turn=2)
    match_context.update(_obs(select=make_select(), current=state_p0b, logs=[]))

    own0 = match_context.get_own_state(0)
    own1 = match_context.get_own_state(1)
    assert own0 is not own1
    assert sum(own0._pool.values()) == 52 + 6
    assert sum(own1._pool.values()) == 52 + 6
    assert own1._pool[ITEM_CARD] == 8 - 2  # player0側の2回目の更新の影響を受けていない

    opponent0 = match_context.get_opponent_state(0)
    deck0, hand0, prize0 = opponent0.sample()
    assert len(deck0) == 53
    assert len(hand0) == 7
    assert len(prize0) == 6

    opponent1 = match_context.get_opponent_state(1)
    deck1, hand1, prize1 = opponent1.sample()
    assert len(deck1) == 49
    assert len(hand1) == 5
    assert len(prize1) == 6


# --- 自山札の player_index 別上書き(ローカル評価harness用) ---
#
# 本番(1プロセス1エージェント)では set_own_deck_override を呼ばないので deck.csv のまま=不変。
# ローカルの field eval は1プロセスで両陣営を動かすため、両者が deck.csv を自分の山札だと
# 誤認して片側の search_begin が必ず失敗していた。その回帰防止。


def test_deck_override_is_per_player():
    match_context.reset()
    deck_a = [11] * 60
    deck_b = [22] * 60
    match_context.set_own_deck_override(0, deck_a)
    match_context.set_own_deck_override(1, deck_b)
    assert match_context._own_deck_ids_for(0) == deck_a
    assert match_context._own_deck_ids_for(1) == deck_b
    # 片側だけ解除すると、その player だけ従来経路(deck.csv)へ戻る。
    match_context.set_own_deck_override(0, None)
    assert match_context._own_deck_ids_for(1) == deck_b
    match_context.reset()


def test_deck_override_cleared_by_reset():
    """game 単位で宣言し直す契約。前の試合のデッキが次の試合へ漏れない。"""
    match_context.reset()
    match_context.set_own_deck_override(0, [33] * 60)
    assert match_context._own_deck_ids_for(0) == [33] * 60
    match_context.reset()
    assert 0 not in match_context._own_deck_override


def test_no_override_keeps_legacy_deck_csv_path(monkeypatch):
    """上書きが無ければ従来どおり read_deck_csv() を読む(=本番挙動が変わらない)。"""
    match_context.reset()
    sentinel = [99] * 60
    monkeypatch.setattr(match_context, "_load_own_deck_ids", lambda: sentinel)
    assert match_context._own_deck_ids_for(0) == sentinel
    assert match_context._own_deck_ids_for(None) == sentinel
