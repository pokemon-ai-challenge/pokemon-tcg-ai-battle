"""hidden_information.search_adapter.to_search_begin_kwargs() のユニットテスト。

実装プラン Phase 4 の記載どおり、``cg.api.search_begin()`` 自体は本フェーズでは呼ばない
（``ptcg_ai/search/`` = MCTS本体がまだ存在しないため）。本テストは、アダプタが返す各リストの
**長さ**が ``search_begin()`` の docstring に書かれた検証条件（``cg/api.py`` 参照）を満たすこと、
および ``None``（不明カードのプレースホルダ）が実カードIDへ変換され、``search_begin()`` に
そのまま ``list[int]`` として渡せる形になっていることを確認する。
"""

from pathlib import Path
import json
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Card, Observation, PlayerState, State
from ptcg_ai.hidden_information.opponent_hidden_state import OpponentHiddenState
from ptcg_ai.hidden_information.own_hidden_state import OwnHiddenState
from ptcg_ai.hidden_information.search_adapter import to_search_begin_kwargs

# 実カードID(cg.api.all_card_data()で確認済み): 22=Hippopotas(Basic), 21=Scrafty(非Basic,Stage1),
# 1=Basic {G} Energy。search_adapterのBasic判定・フォールバックロジックの検証に使う。
BASIC_POKEMON = 22
NON_BASIC_POKEMON = 21
ENERGY_CARD = 1
ITEM_CARD = 2


def make_deck() -> list[int]:
    return [ENERGY_CARD] * 50 + [ITEM_CARD] * 10


def make_player_state(deck_count, prize, hand_count=0, active=None):
    return PlayerState(
        active=active if active is not None else [],
        bench=[],
        benchMax=5,
        deckCount=deck_count,
        discard=[],
        prize=prize,
        handCount=hand_count,
        hand=None,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def make_state(your_player, opponent_player, your_index=0, turn=3):
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
        looking=None,
        players=players,
    )


def make_own_state(deck_count=54, prize_count=6) -> OwnHiddenState:
    own = OwnHiddenState(make_deck())
    my_player = make_player_state(deck_count=deck_count, prize=[None] * prize_count)
    opp_placeholder = make_player_state(deck_count=53, prize=[None] * 6)
    own.update(make_state(my_player, opp_placeholder))
    return own


def write_pool(tmp_path: Path, archetypes: dict[str, dict[int, int]]) -> Path:
    payload = {
        "meta": {"built_at": "test"},
        "archetypes": {
            name: {
                "card_counts": {
                    str(card_id): {"median": median, "inclusion_rate": 1.0}
                    for card_id, median in cards.items()
                }
            }
            for name, cards in archetypes.items()
        },
    }
    pool_path = tmp_path / "archetype_card_pool.json"
    with pool_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return pool_path


def make_opponent_state(
    tmp_path: Path,
    deck_count: int,
    hand_count: int,
    prize_count: int,
    archetype_cards: dict[int, int],
    archetype_name: str = "A",
) -> OpponentHiddenState:
    pool_path = write_pool(tmp_path, {archetype_name: archetype_cards})
    opponent = OpponentHiddenState(pool_path=pool_path)
    opponent.update(
        {archetype_name: 1.0},
        observed_card_ids={},
        player_state=make_player_state(deck_count=deck_count, prize=[None] * prize_count, hand_count=hand_count),
    )
    return opponent


# ----------------------------------------------------------------------
# 各リスト長がsearch_begin()の検証条件を満たす


def test_lengths_satisfy_search_begin_validation_when_active_visible(tmp_path):
    own = make_own_state(deck_count=54, prize_count=6)

    # 代表リスト: 埋め草多め(プール40) > ゾーン合計(deck10+hand5+prize4=19) -> 全部実カード。
    cards = {300 + i: 1 for i in range(40)}
    opponent = make_opponent_state(tmp_path, deck_count=10, hand_count=5, prize_count=4, archetype_cards=cards)

    opp_active_pokemon = Card(id=BASIC_POKEMON, serial=999, playerIndex=1)
    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    opp_player = make_player_state(deck_count=10, prize=[None] * 4, hand_count=5, active=[])
    state = make_state(my_player, opp_player)
    obs = Observation(select=None, logs=[], current=state)

    kwargs = to_search_begin_kwargs(own, opponent, obs)

    assert len(kwargs["your_deck"]) >= my_player.deckCount
    assert len(kwargs["your_prize"]) >= len(my_player.prize)
    assert len(kwargs["opponent_deck"]) >= opp_player.deckCount
    assert len(kwargs["opponent_prize"]) >= len(opp_player.prize)
    assert len(kwargs["opponent_hand"]) >= opp_player.handCount
    # activeが伏せでない(空リスト=盤面にまだ出ていない)ので推測不要 -> 空リスト。
    assert kwargs["opponent_active"] == []

    for key in ("your_deck", "your_prize", "opponent_deck", "opponent_prize", "opponent_hand"):
        assert all(isinstance(cid, int) for cid in kwargs[key]), f"{key} must be list[int] (no None)"


def test_opponent_active_guessed_from_most_likely_archetype_basic_pokemon(tmp_path):
    own = make_own_state()
    # 代表リストに非Basic(21)を多めに、Basic(22)を少なめに混ぜる -> Basic側だけが候補になる。
    cards = {NON_BASIC_POKEMON: 4, BASIC_POKEMON: 2}
    cards.update({400 + i: 1 for i in range(10)})
    opponent = make_opponent_state(tmp_path, deck_count=8, hand_count=4, prize_count=2, archetype_cards=cards)

    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    # activeが伏せ(None)のケース。
    opp_player = make_player_state(deck_count=8, prize=[None] * 2, hand_count=4, active=[None])
    state = make_state(my_player, opp_player)
    obs = Observation(select=None, logs=[], current=state)

    kwargs = to_search_begin_kwargs(own, opponent, obs)

    assert kwargs["opponent_active"] == [BASIC_POKEMON]


def test_none_placeholders_are_filled_with_real_card_ids(tmp_path):
    own = make_own_state()
    # 代表リスト4枚のみ。ゾーン合計(deck3+hand2+prize1=6) > プール4 -> sample()はNoneを含む。
    cards = {500: 1, 501: 1, 502: 1, 503: 1}
    opponent = make_opponent_state(tmp_path, deck_count=3, hand_count=2, prize_count=1, archetype_cards=cards)

    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    opp_player = make_player_state(deck_count=3, prize=[None] * 1, hand_count=2, active=[])
    state = make_state(my_player, opp_player)
    obs = Observation(select=None, logs=[], current=state)

    for _ in range(20):
        kwargs = to_search_begin_kwargs(own, opponent, obs)
        assert len(kwargs["opponent_deck"]) == 3
        assert len(kwargs["opponent_hand"]) == 2
        assert len(kwargs["opponent_prize"]) == 1
        combined = kwargs["opponent_deck"] + kwargs["opponent_hand"] + kwargs["opponent_prize"]
        assert all(cid is not None for cid in combined)
        assert all(isinstance(cid, int) for cid in combined)


def test_raises_on_deck_selection_turn():
    own = make_own_state()
    opponent = OpponentHiddenState(pool_path=Path("does_not_exist.json"))
    obs = Observation(select=None, logs=[], current=None)
    try:
        to_search_begin_kwargs(own, opponent, obs)
        assert False, "ValueErrorが発生するはず"
    except ValueError:
        pass


def test_opponent_active_empty_when_unready_state_has_no_pool(tmp_path):
    # プールJSON無し(is_ready=False)でも例外を出さず、伏せていてもフォールバックのBasicポケモンで埋まる。
    own = make_own_state()
    opponent = OpponentHiddenState(pool_path=tmp_path / "missing.json")
    opponent.update({}, observed_card_ids={}, player_state=make_player_state(deck_count=5, prize=[None] * 2, hand_count=3))

    my_player = make_player_state(deck_count=54, prize=[None] * 6)
    opp_player = make_player_state(deck_count=5, prize=[None] * 2, hand_count=3, active=[None])
    state = make_state(my_player, opp_player)
    obs = Observation(select=None, logs=[], current=state)

    kwargs = to_search_begin_kwargs(own, opponent, obs)
    assert len(kwargs["opponent_active"]) == 1
    assert isinstance(kwargs["opponent_active"][0], int)
