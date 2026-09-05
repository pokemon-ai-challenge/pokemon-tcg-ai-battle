"""outcome 集合の構成と、**確定証明に使ってよいかの判定**。

Step 1-1 で入れた再生一致検査(規則 R3)は「同じ入力で 2 回再生して一致したら
供給順で制御できていそう」という検査だが、これは**反証にしか使えない**:
低確率で偶然一致すれば決定的だと誤判定しうるし、そもそも
「この事象の outcome がこれで全部か」は一切言っていない。

そこで Phase 2 の ``PROVEN_WIN`` に必要な条件を、分類とは別の軸として明示する。
原設計 §5.2/§8 とユーザ指示 (Step 1-2 #2) の 5 条件に対応:

1. 使用した outcome 集合が確定している        -> ``outcomes`` が有限・明示・ラベル一意
2. 各非0確率 outcome を漏れなく覆っている      -> ``coverage_certified`` かつ質量合計 == 1
3. 各 outcome を再現可能な状態として構築できる -> 各 ``Outcome.constructible``
4. 各 outcome の確率質量を正しく扱える         -> ``Fraction`` による厳密値
5. 未確認 outcome が存在する可能性を排除できる -> ``unprocessed_mass == 0``

1つでも満たさなければ ``StopReason.OUTCOMES_NOT_ENUMERABLE`` を返し、
Phase 2 は ``UNKNOWN`` にする。**「見つからなかった」を ``PROVEN_NO_WIN`` にしない**
のと同様に、「列挙しきれなかった」を ``PROVEN_WIN`` の根拠にしない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Hashable, Mapping

from ptcg_ai.search.lethal.types import ChanceClass, StopReason

# outcome 集合の導出方法。``coverage_certified`` を True にしてよいのは、
# 「モデル上、これ以外の outcome が存在しない」と言い切れる導出だけ。
SOURCE_MULTISET_HYPERGEOMETRIC = "multiset_hypergeometric"
SOURCE_EXPLICIT_CHOICE = "explicit_choice"      # コイン等、エンジンが選択肢で出すもの
SOURCE_SAMPLED = "sampled"                       # Phase 3 専用。証明には使えない
SOURCE_PARTIAL = "partial"                       # 打ち切り。証明には使えない


@dataclass(frozen=True)
class Outcome:
    """1つの outcome(等価類)。"""

    label: Hashable
    mass: Fraction
    constructible: bool = True
    payload: object = None  # 後続状態など。証明の判定には使わない

    def __post_init__(self) -> None:
        if not isinstance(self.mass, Fraction):
            raise TypeError("mass must be a Fraction (exact rational)")


@dataclass(frozen=True)
class OutcomeSet:
    """1つの chance 事象に対する outcome の集合。"""

    outcomes: tuple[Outcome, ...]
    source: str
    coverage_certified: bool
    unprocessed_mass: Fraction = field(default_factory=lambda: Fraction(0))

    def total_mass(self) -> Fraction:
        return sum((o.mass for o in self.outcomes), Fraction(0))

    def labels(self) -> tuple[Hashable, ...]:
        return tuple(o.label for o in self.outcomes)

    def labels_unique(self) -> bool:
        return len(set(self.labels())) == len(self.outcomes)


@dataclass(frozen=True)
class Certification:
    """``OutcomeSet`` が確定証明に使えるかどうか。"""

    admissible: bool
    stop_reason: StopReason | None = None
    failures: tuple[str, ...] = ()


def certify_for_proof(
    outcome_set: OutcomeSet, *, chance_class: ChanceClass
) -> Certification:
    """Phase 2 の ``PROVEN_WIN`` の根拠に使ってよいかを判定する。

    分類(C/M/S)が列挙可能クラスであることは**必要条件にすぎない**。
    outcome 集合そのものが上記 5 条件を満たすことを確認する。
    """
    failures: list[str] = []
    if not chance_class.is_enumerable:
        failures.append(f"chance_class_not_enumerable:{chance_class.name}")
    if outcome_set.source in (SOURCE_SAMPLED, SOURCE_PARTIAL):
        failures.append(f"source_not_provable:{outcome_set.source}")
    if not outcome_set.coverage_certified:
        failures.append("coverage_not_certified")
    if not outcome_set.outcomes:
        failures.append("empty_outcome_set")
    if not outcome_set.labels_unique():
        failures.append("duplicate_outcome_labels")
    if any(o.mass <= 0 for o in outcome_set.outcomes):
        failures.append("non_positive_mass")
    if any(not o.constructible for o in outcome_set.outcomes):
        failures.append("outcome_not_constructible")
    if outcome_set.unprocessed_mass != 0:
        failures.append("unprocessed_mass_present")
    if outcome_set.total_mass() != Fraction(1):
        failures.append(f"mass_sum_not_one:{outcome_set.total_mass()}")
    if failures:
        return Certification(False, StopReason.OUTCOMES_NOT_ENUMERABLE, tuple(failures))
    return Certification(True)


def draw_outcomes(
    deck_multiset: Mapping[int, int],
    count: int,
    *,
    should_stop=None,
) -> OutcomeSet | None:
    """既知の山札 multiset から ``count`` 枚引いたときの outcome を厳密列挙する。

    outcome の等価類は「引いたカードの multiset」。確率は多変量超幾何分布:

        P(c) = prod_i C(n_i, c_i) / C(N, k)

    山札が足りない場合は「引ける枚数だけ引く」1 通りになる(確率 1)。
    順序は使わない(情報境界: 我々が知ってよいのは multiset だけ)。

    ``should_stop`` は予算切れを問い合わせる述語。列挙は組合せ爆発しうるので、
    **ループの内側**で確認する(Step 1-19 指示 11)。打ち切った場合は
    部分集合を返さず ``None`` を返す。中途半端な outcome 集合を返すと
    「全 outcome を覆った」という誤った証明の根拠になりうるため。
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    items = tuple(sorted((int(k), int(v)) for k, v in deck_multiset.items() if v > 0))
    total = sum(v for _, v in items)
    draw = min(count, total)
    if draw == 0:
        return OutcomeSet(
            outcomes=(Outcome(label=(), mass=Fraction(1)),),
            source=SOURCE_MULTISET_HYPERGEOMETRIC,
            coverage_certified=True,
        )
    denominator = math.comb(total, draw)
    outcomes: list[Outcome] = []
    for index, combination in enumerate(_sub_multisets(items, draw)):
        if should_stop is not None and (index & 0x3F) == 0 and should_stop():
            return None
        numerator = 1
        for (card_id, available), taken in zip(items, combination):
            numerator *= math.comb(available, taken)
            _ = card_id
        if numerator == 0:
            continue
        label = tuple(
            (card_id, taken)
            for (card_id, _available), taken in zip(items, combination)
            if taken
        )
        outcomes.append(Outcome(label=label, mass=Fraction(numerator, denominator)))
    return OutcomeSet(
        outcomes=tuple(outcomes),
        source=SOURCE_MULTISET_HYPERGEOMETRIC,
        coverage_certified=True,
    )


def explicit_choice_outcomes(labels, mass_each: Fraction) -> OutcomeSet:
    """コインのように、エンジンが結果そのものを選択肢で出す事象の outcome 集合。"""
    outcomes = tuple(Outcome(label=label, mass=mass_each) for label in labels)
    return OutcomeSet(
        outcomes=outcomes,
        source=SOURCE_EXPLICIT_CHOICE,
        coverage_certified=True,
    )


def _sub_multisets(items: tuple[tuple[int, int], ...], draw: int):
    """``items``(card_id, 枚数) から合計 ``draw`` 枚取る取り方を列挙する。"""
    if not items:
        if draw == 0:
            yield ()
        return
    (_card_id, available), rest = items[0], items[1:]
    remaining_capacity = sum(v for _, v in rest)
    lower = max(0, draw - remaining_capacity)
    for taken in range(lower, min(available, draw) + 1):
        for tail in _sub_multisets(rest, draw - taken):
            yield (taken,) + tail
