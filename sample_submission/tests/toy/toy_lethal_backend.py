"""toy を ``LethalBackend`` へ適合させるアダプタ。

これにより、**製品の Phase 1/2 のコードをそのまま toy 上で動かして**
ground-truth oracle と突き合わせられる(``backend.py`` を置いた理由)。
toy 側の都合を製品コードへ持ち込まないよう、変換はここだけに閉じる。
"""

from __future__ import annotations

from typing import Hashable, Sequence

from ptcg_ai.search.lethal.backend import OutcomeEnumeration, Transition
from ptcg_ai.search.lethal.enumeration import Outcome, draw_outcomes
from ptcg_ai.search.lethal.types import ChanceClass, StopReason

from tests.toy.candidates import _label_to_cards, canonical, realize
from tests.toy.toy_game import BACKEND, DRAW_COUNT, ToyState


class ToyLethalBackend:
    """``LethalBackend`` の toy 実装。"""

    def __init__(
        self,
        state: ToyState,
        *,
        action_source=None,
        refuse_outcomes: StopReason | None = None,
        incomplete_actions: bool = False,
        key_wrapper=None,
    ) -> None:
        self._root = canonical(state)
        self._action_source = action_source or (lambda node: BACKEND.legal_actions(node))
        self._refuse_outcomes = refuse_outcomes
        self._incomplete_actions = incomplete_actions
        self._key_wrapper = key_wrapper

    def root(self) -> ToyState:
        return self._root

    def legal_actions(self, state: ToyState) -> Sequence[tuple]:
        if BACKEND.is_opponent_node(state):
            # 相手の選択肢を自分のマクロで並べ替えたり削ったりしない。
            return BACKEND.legal_actions(state)
        return self._action_source(state)

    def action_set_complete(self, state: ToyState) -> bool:
        return not self._incomplete_actions

    def apply(self, state: ToyState, action: tuple) -> Transition:
        revealed = BACKEND.reveals(state, action)
        try:
            child = BACKEND.apply(state, action)
        except ValueError:
            return Transition(None)
        return Transition(child, revealed=revealed)

    def is_win(self, state: ToyState) -> bool:
        return BACKEND.is_win(state)

    def is_terminal(self, state: ToyState) -> bool:
        return BACKEND.is_terminal(state)

    def is_opponent_node(self, state: ToyState) -> bool:
        return BACKEND.is_opponent_node(state)

    def observable_key(self, state: ToyState) -> Hashable:
        key = state.observable()
        return self._key_wrapper(key) if self._key_wrapper else key

    def enumerate_outcomes(self, state: ToyState, action: tuple) -> OutcomeEnumeration:
        if self._refuse_outcomes is not None:
            return OutcomeEnumeration(None, ChanceClass.ENGINE_RANDOM, self._refuse_outcomes)
        outcome_set = draw_outcomes(state.deck_multiset(), DRAW_COUNT)
        return OutcomeEnumeration(outcome_set, ChanceClass.CONTROLLED_SUPPLY_ORDER)

    def apply_outcome(self, state: ToyState, action: tuple, outcome: Outcome) -> Transition:
        drawn = _label_to_cards(outcome.label)
        try:
            realized = realize(state, drawn)
        except ValueError:
            return Transition(None, stop_reason=StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Transition(BACKEND.apply(realized, action), revealed=True)


class ConstantHashKey:
    """hash が必ず衝突するキー(transposition の健全性確認用)。"""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other) -> bool:
        return isinstance(other, ConstantHashKey) and self.value == other.value
