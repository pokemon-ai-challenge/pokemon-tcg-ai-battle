"""自分のデッキ情報の**境界**(Step 1-4 指示 6)。

探索が必要とするのは次の3つだけで、**実際の山札順は一切必要ない**。

- 自分のデッキ構成(60 枚の multiset)
- 見えているカード(観測から分かる)
- 山札の残り枚数(観測から分かる)

そこで、探索へ渡す型を「順序を持てない型」に固定する。``list[int]`` を
そのまま持ち回ると、いつか誰かが ``deck[0]`` を読んでしまう。

境界の形:

```text
Agent / selector
   ├─ deck.csv の list[int]        ← 順序を持つ表現。ここで捨てる
   └─ KnownDeckComposition        ← 探索側へ渡すのはこれだけ(multiset)
```

``Scenario``(山札順列)との違い: ``Scenario`` は「我々が仮定した並び」で
outcome 構築のためだけに使う。``KnownDeckComposition`` は「プレイヤーが
正当に知っているデッキ構成」で、決定側が見てよい。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class KnownDeckComposition:
    """自分のデッキ構成(multiset)。**順序を保持しない**。

    ``counts`` は ``(card_id, 枚数)`` を card_id 昇順に並べたもの。
    元のリスト順は構築時に捨てられるので、ここから復元できない。
    """

    counts: tuple[tuple[int, int], ...]

    @classmethod
    def from_card_ids(cls, card_ids: Iterable[int]) -> "KnownDeckComposition":
        """デッキリスト(順序つき)から作る。**この時点で順序を捨てる**。"""
        counter: Counter = Counter(int(card_id) for card_id in card_ids)
        return cls(tuple(sorted(counter.items())))

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, int]) -> "KnownDeckComposition":
        return cls(tuple(sorted((int(k), int(v)) for k, v in mapping.items() if v > 0)))

    def as_counter(self) -> Counter:
        return Counter(dict(self.counts))

    def total(self) -> int:
        return sum(count for _card_id, count in self.counts)

    def __repr__(self) -> str:
        return f"KnownDeckComposition(cards={self.total()}, kinds={len(self.counts)})"


def coerce(source) -> KnownDeckComposition | None:
    """``list[int]`` / Mapping / ``KnownDeckComposition`` を境界で正規化する。

    ここが「順序を捨てる唯一の場所」。探索の内側はこの型しか受け取らない。
    """
    if source is None:
        return None
    if isinstance(source, KnownDeckComposition):
        return source
    if isinstance(source, Mapping):
        return KnownDeckComposition.from_mapping(source)
    return KnownDeckComposition.from_card_ids(source)
