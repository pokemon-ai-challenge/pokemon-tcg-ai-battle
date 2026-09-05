"""山札順列(Scenario)を包む型。

Step 0 で確認した通り、``search_begin()`` に渡す ``your_deck`` の順序が
ドロー順を完全に決める。つまり順列は「隠れた山札順の代理」であり、
**決定ノードがこれを見た瞬間にカンニング(determinization)になる**。

そこで順列を裸の ``list[int]`` で持ち回らず、この型に閉じ込める:

- 順列を読めるのは ``_ALLOWED_MODULES`` のモジュールだけ(規則 R2)。
  違反はテスト ``test_scenario_confinement`` が検出する。
- ``repr()`` は順序を出さない。診断ログ・例外メッセージへ誤って
  流れ込んでも順序が漏れない(規則 R2b)。
- 安全に公開してよいのは **multiset** だけ。``multiset()`` を使うこと。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

# 型を import してよいモジュール。
#   scenario   : 定義そのもの
#   outcome    : outcome を実現する順列を作る側(Step 2 で追加予定)
#   engine     : ``search_begin()`` へ渡すためだけに順列を保持する層
#   diagnostics: 「ログへ書かせない」ための isinstance 判定にのみ使う
# 決定側(infokey / macro / phase1 / phase2 / phase3 / value / gate)は import 禁止。
#   cg_backend : outcome を実現する順列を**構築**する(信念 multiset から作るだけで、
#                隠れた順序を読むことはない。下の order-reader には入れない)
_ALLOWED_MODULES = (
    "ptcg_ai.search.lethal.scenario",
    "ptcg_ai.search.lethal.outcome",
    "ptcg_ai.search.lethal.engine",
    "ptcg_ai.search.lethal.diagnostics",
    "ptcg_ai.search.lethal.cg_backend",
)

# **順序そのもの**(``order`` / ``to_engine_list()``)を読んでよいモジュール。
# import できることと順序を読めることは別の権限として分ける。
_ORDER_READER_MODULES = (
    "ptcg_ai.search.lethal.scenario",
    "ptcg_ai.search.lethal.outcome",
    "ptcg_ai.search.lethal.engine",
)


@dataclass(frozen=True)
class Scenario:
    """自分の山札の1つの順列。末尾が山札の一番上(Step 0 §4.1)。"""

    order: tuple[int, ...]

    def __repr__(self) -> str:  # 順序を出さない
        return f"Scenario(size={len(self.order)}, digest={self.digest()})"

    __str__ = __repr__

    def digest(self) -> str:
        """順列の同一性比較用の短いハッシュ(順序そのものは復元できない)。"""
        raw = ",".join(str(card_id) for card_id in self.order).encode("ascii")
        return hashlib.sha1(raw).hexdigest()[:8]

    def multiset(self) -> Counter:
        """順序を落とした multiset。**これは公開してよい情報**。"""
        return Counter(self.order)

    def size(self) -> int:
        return len(self.order)

    def to_engine_list(self) -> list[int]:
        """``search_begin()`` へ渡す形。engine 層だけが呼ぶ。"""
        return list(self.order)
