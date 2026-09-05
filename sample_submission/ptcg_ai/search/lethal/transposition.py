"""深さを考慮した transposition table。

キーは ``backend.observable_key(state)``(実エンジンでは ``InfoKey``)。
**digest ではなくキーオブジェクトそのもの**を dict のキーにするので、
digest の衝突が探索結果を壊すことはない(``InfoKey.__eq__`` は内容比較)。

再利用の規則(健全性):

- ``PROVEN_WIN`` は「より少ない残り深さ」で見つかった場合のみ再利用できる
- ``PROVEN_NO_WIN`` は「より多い残り深さ」で調べ尽くした場合のみ再利用できる
- ``UNKNOWN`` はキャッシュしない(打ち切り理由が状況依存のため)
"""

from __future__ import annotations

from typing import Hashable

from ptcg_ai.search.lethal.types import Proof


class Transposition:
    def __init__(self) -> None:
        self._win: dict[Hashable, int] = {}
        self._no_win: dict[Hashable, int] = {}

    def lookup(self, key: Hashable, depth_left: int) -> Proof | None:
        win_depth = self._win.get(key)
        if win_depth is not None and win_depth <= depth_left:
            return Proof.PROVEN_WIN
        no_win_depth = self._no_win.get(key)
        if no_win_depth is not None and no_win_depth >= depth_left:
            return Proof.PROVEN_NO_WIN
        return None

    def store_win(self, key: Hashable, depth_left: int) -> None:
        previous = self._win.get(key)
        if previous is None or depth_left < previous:
            self._win[key] = depth_left

    def store_no_win(self, key: Hashable, depth_left: int) -> None:
        previous = self._no_win.get(key)
        if previous is None or depth_left > previous:
            self._no_win[key] = depth_left

    def __len__(self) -> int:
        return len(self._win) + len(self._no_win)
