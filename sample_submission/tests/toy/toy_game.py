"""最小の人工ゲーム(Step 1-2a)。

実エンジンの構造のうち、リーサル探索の確定性に関係する部分だけを写した:

- 自分のターン中の**プリミティブ行動**しかない(相手の行動は無い)
- 行動回数に上限がある(= primitive depth)
- **ドローだけが新情報境界**であり、山札の順序が結果を決める
- 勝敗は公開情報(相手 HP)だけで決まる

カード:

| id | 名前   | 効果 |
|----|--------|------|
| 1  | STRIKE | 相手に (3 + ブースト) ダメージ |
| 2  | BOOST  | このターンの以後の STRIKE に +3 |
| 3  | DRAW2  | 山札の上から2枚引く(**新情報境界**) |
| 4  | DUD    | 何も起きない |

状態の可視部分は ``observable()``。``deck`` は隠れており、探索側は
**multiset しか見てはいけない**(順序を見たら determinization)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Hashable, Sequence

STRIKE = 1
BOOST = 2
DRAW2 = 3
DUD = 4

CARD_NAMES = {STRIKE: "STRIKE", BOOST: "BOOST", DRAW2: "DRAW2", DUD: "DUD"}

STRIKE_DAMAGE = 3
BOOST_AMOUNT = 3
DRAW_COUNT = 2

END = ("end",)


def play(card_id: int) -> tuple:
    return ("play", card_id)


@dataclass(frozen=True)
class ToyState:
    """toy の完全状態。``deck`` は隠れた順序を持つ(index 0 が山札の上)。

    B4(相手選択)のモデル化:
    バトルポケモンを倒すとサイドを 1 枚取る。サイドが 0 になれば勝ち。
    まだ残っていて相手にベンチがいる場合、**相手が次のバトルポケモンを選ぶ**
    (``pending_promotion``)。この選択は相手のものなので AND ノードになる。
    """

    hand: tuple[int, ...]
    deck: tuple[int, ...]
    opponent_hp: int
    bonus: int = 0
    actions_left: int = 3
    turn_over: bool = False
    prizes_left: int = 1
    opponent_bench: tuple[int, ...] = ()   # ベンチポケモンの HP
    pending_promotion: bool = False

    # -------------------------------------------------------------- 可視部分

    def observable(self) -> Hashable:
        """公開情報 + 自分の手札 multiset + 山札の枚数。**順序は含めない**。"""
        return (
            tuple(sorted(self.hand)),
            self.opponent_hp,
            self.bonus,
            self.actions_left,
            self.turn_over,
            len(self.deck),
            self.prizes_left,
            self.opponent_bench,
            self.pending_promotion,
        )

    def deck_multiset(self) -> Counter:
        """探索側が見てよい山札の情報。"""
        return Counter(self.deck)


class ToyBackend:
    """``ptcg_ai.search.lethal.backend.SearchBackend`` の toy 実装。"""

    def legal_actions(self, state: ToyState) -> Sequence[tuple]:
        if self.is_terminal(state):
            return ()
        if state.pending_promotion:
            # 相手の選択肢: どのベンチポケモンをバトル場に出すか。
            return tuple(("promote", i) for i in range(len(state.opponent_bench)))
        actions = [play(card_id) for card_id in sorted(set(state.hand))]
        actions.append(END)
        return tuple(actions)

    def is_opponent_node(self, state: ToyState) -> bool:
        return state.pending_promotion

    def apply(self, state: ToyState, action: tuple) -> ToyState:
        if action == END:
            return replace(state, turn_over=True, actions_left=0)
        kind, card_id = action
        if kind == "promote":
            bench = list(state.opponent_bench)
            promoted = bench.pop(card_id)
            return replace(
                state,
                opponent_hp=promoted,
                opponent_bench=tuple(bench),
                pending_promotion=False,
            )
        assert kind == "play"
        if card_id not in state.hand:
            raise ValueError(f"card {card_id} is not in hand")
        hand = list(state.hand)
        hand.remove(card_id)
        state = replace(state, hand=tuple(hand), actions_left=state.actions_left - 1)
        if card_id == STRIKE:
            state = replace(state, opponent_hp=state.opponent_hp - (STRIKE_DAMAGE + state.bonus))
            if state.opponent_hp <= 0:
                # きぜつ: サイドを 1 枚取る。まだ残っていてベンチがいるなら、
                # **相手が**次のバトルポケモンを選ぶ(B4)。
                state = replace(state, prizes_left=state.prizes_left - 1)
                if state.prizes_left > 0 and state.opponent_bench:
                    state = replace(state, pending_promotion=True)
        elif card_id == BOOST:
            state = replace(state, bonus=state.bonus + BOOST_AMOUNT)
        elif card_id == DRAW2:
            drawn = state.deck[:DRAW_COUNT]
            state = replace(
                state,
                hand=tuple(state.hand) + drawn,
                deck=state.deck[DRAW_COUNT:],
            )
        if state.actions_left <= 0 and not self.is_win(state) and not state.pending_promotion:
            state = replace(state, turn_over=True)
        return state

    def is_win(self, state: ToyState) -> bool:
        """サイドを取り切ったら勝ち(相手の場が空になる場合も同じ扱い)。"""
        return state.prizes_left <= 0

    def is_terminal(self, state: ToyState) -> bool:
        if self.is_win(state):
            return True
        if state.pending_promotion:
            return False  # 相手の選択ノードは終端ではない
        return state.turn_over or state.actions_left <= 0

    def reveals(self, state: ToyState, action: tuple) -> bool:
        """新情報境界か。山札が空なら引いても何も公開されない。"""
        return (
            action != END
            and action[0] == "play"
            and action[1] == DRAW2
            and len(state.deck) > 0
        )

    def observable_key(self, state: ToyState) -> Hashable:
        return state.observable()

    # ------------------------------------------------- multiset 上での適用

    def apply_without_reveal(self, state: ToyState, action: tuple) -> ToyState:
        """公開を伴わない行動を、山札の順序を使わずに適用する。

        探索側(候補実装)は順序を持たないので、``deck`` はダミーの並びで持つ。
        ``reveals()`` が False の行動しか渡してはいけない。
        """
        if self.reveals(state, action):
            raise ValueError("this action reveals information")
        return self.apply(state, action)


BACKEND = ToyBackend()


def make_state(
    hand: Sequence[int],
    deck: Sequence[int],
    opponent_hp: int,
    *,
    actions_left: int = 3,
    bonus: int = 0,
    prizes_left: int = 1,
    opponent_bench: Sequence[int] = (),
) -> ToyState:
    return ToyState(
        hand=tuple(hand),
        deck=tuple(deck),
        opponent_hp=opponent_hp,
        bonus=bonus,
        actions_left=actions_left,
        prizes_left=prizes_left,
        opponent_bench=tuple(opponent_bench),
    )


def describe(state: ToyState) -> str:
    hand = "+".join(CARD_NAMES[c] for c in sorted(state.hand)) or "-"
    return (
        f"hand={hand} hp={state.opponent_hp} bonus={state.bonus} "
        f"actions={state.actions_left} deck={len(state.deck)}"
    )
