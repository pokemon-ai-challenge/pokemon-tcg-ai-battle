"""リーサル探索の共通型(設計 §5)。

確定(Phase 1/2)と推定(Phase 3)を型の上で混同できないようにするのが目的。

- ``Proof``        : 確定探索の3値
- ``EstimateKind`` : 推定の由来(完全列挙 / 上下界 / サンプリング / 利用不可)
- ``ChanceClass``  : ランダム事象の**能力ベース**分類(§1.2)
- ``ValueKind``    : 価値の意味の分離。``V_policy`` / ``V_fail`` / ``Q_try`` / ``Q_base``
                     を同じ型で持ちつつ、取り違えをテストで検出できるようにする
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


class Proof(Enum):
    """Phase 1/2 の確定判定。推定値をここへ入れてはいけない。"""

    PROVEN_WIN = auto()
    PROVEN_NO_WIN = auto()
    UNKNOWN = auto()


class EstimateKind(Enum):
    """確率・価値の推定がどう作られたか。表示・ゲート判断で必ず参照する。"""

    EXACT = auto()       # 全 outcome を列挙し、確率が厳密に決まっている
    BOUNDED = auto()     # 未処理質量が残り、上下界のみ
    SAMPLED = auto()     # エンジンにサンプリングさせた推定(Phase 3 限定)
    UNAVAILABLE = auto()  # 値が得られなかった。ゲートに通してはいけない


class ChanceClass(Enum):
    """ランダム事象の分類。**ログの種類ではなく「制御できるか」で決める**(規則 R3)。

    - ``CONTROLLED_SUPPLY_ORDER``: 供給した ``your_deck`` の順序で outcome を
      指定できる(= 我々が任意 outcome を適用できる)。
    - ``CONTROLLED_EXPLICIT_CHOICE``: ``manual_coin`` のように、エンジンが
      結果を選択肢として提示してくる。
    - ``ENGINE_RANDOM``: エンジン内部 RNG。適用も再現もできない。
    - ``OPPONENT_CHOICE``: 相手が選ぶ。確率平均してはいけない(AND / min)。
    - ``UNCLASSIFIED``: 判定に必要な検証をまだ通していない。**確定探索では使えない**。
    """

    CONTROLLED_SUPPLY_ORDER = auto()
    CONTROLLED_EXPLICIT_CHOICE = auto()
    ENGINE_RANDOM = auto()
    OPPONENT_CHOICE = auto()
    UNCLASSIFIED = auto()

    @property
    def is_enumerable(self) -> bool:
        """全 outcome を列挙して確定判定に使えるクラスか。"""
        return self in (
            ChanceClass.CONTROLLED_SUPPLY_ORDER,
            ChanceClass.CONTROLLED_EXPLICIT_CHOICE,
        )


class StopReason(Enum):
    """探索を打ち切った/確定を主張しない理由。診断へそのまま出す。"""

    WIN = auto()
    TURN_ENDED = auto()
    TIME_LIMIT = auto()
    NODE_LIMIT = auto()
    DEPTH_LIMIT = auto()
    CHANCE_DEPTH_LIMIT = auto()
    INCOMPLETE_ACTION_SET = auto()
    OUTCOMES_NOT_ENUMERABLE = auto()
    UNSUPPORTED_EFFECT = auto()
    SHUFFLE_ENCOUNTERED = auto()
    REPLAY_NOT_VERIFIED = auto()   # 規則 R3 の再生一致検査を通していない
    REPLAY_MISMATCH = auto()       # 同一入力で結果が変わった = クラス S
    DECK_REVEALED_AT_ROOT = auto()  # B1: 探索開始時に実デッキが使われている
    DRAW_BEFORE_SHUFFLE_AFTER_REVEAL = auto()  # B1 ガード(規則 R4)違反
    CRITIC_UNAVAILABLE = auto()
    CRITIC_UNCALIBRATED = auto()
    STATE_MISMATCH = auto()
    CYCLE = auto()
    ENGINE_ERROR = auto()


class ValueKind(Enum):
    """価値の意味。混同を防ぐため、値そのものに種別を持たせる(設計 §7)。

    - ``V_POLICY``: ある局面から通常方策を続けたときの最終勝率(自分視点)
    - ``V_FAIL``  : リーサル失敗葉での ``V_POLICY``
    - ``Q_TRY``   : リーサルを試みたときの期待勝率(勝ち確率 + 失敗枝の価値)
    - ``Q_BASE``  : 通常方策をそのまま採ったときの期待勝率
    """

    V_POLICY = auto()
    V_FAIL = auto()
    Q_TRY = auto()
    Q_BASE = auto()


def _is_probability(x: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x) and 0.0 <= float(x) <= 1.0


@dataclass(frozen=True)
class ProbabilityInterval:
    """[lower, upper] の確率区間。NaN・範囲外・逆転は構築時に弾く。"""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not _is_probability(self.lower) or not _is_probability(self.upper):
            raise ValueError(f"probability out of range: {self.lower}, {self.upper}")
        if self.lower > self.upper:
            raise ValueError(f"lower > upper: {self.lower} > {self.upper}")

    @property
    def is_point(self) -> bool:
        return self.lower == self.upper

    @classmethod
    def unknown(cls) -> "ProbabilityInterval":
        return cls(0.0, 1.0)

    @classmethod
    def point(cls, value: float) -> "ProbabilityInterval":
        return cls(value, value)


@dataclass(frozen=True)
class ValueEstimate:
    """価値の推定値。``kind`` と ``estimate_kind`` を必ず持つ。

    ``estimate_kind == UNAVAILABLE`` のとき、``mean`` は意味を持たない
    (``None``)。採用ゲートは ``is_usable`` が False の値を通してはいけない。
    """

    kind: ValueKind
    estimate_kind: EstimateKind
    mean: float | None = None
    lower: float | None = None
    upper: float | None = None
    source: str = ""
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if self.estimate_kind is EstimateKind.UNAVAILABLE:
            return
        for name in ("mean", "lower", "upper"):
            value = getattr(self, name)
            if value is None:
                continue
            if not _is_probability(value):
                raise ValueError(f"{name} out of range: {value}")
        if self.mean is None:
            raise ValueError("mean is required unless estimate_kind is UNAVAILABLE")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower > upper: {self.lower} > {self.upper}")

    @property
    def is_usable(self) -> bool:
        """採用ゲートで比較してよい値か。"""
        return self.estimate_kind is not EstimateKind.UNAVAILABLE and self.mean is not None

    @classmethod
    def unavailable(
        cls, kind: ValueKind, *, source: str = "", stop_reason: StopReason | None = None
    ) -> "ValueEstimate":
        return cls(
            kind=kind,
            estimate_kind=EstimateKind.UNAVAILABLE,
            source=source,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True)
class LethalResult:
    """Phase 1/2 の結果。

    ``first_action`` は **root の最初の1手だけ**。手順全体を計画として持ち回らない
    (原設計 §2.4: 実行は最初の行動だけ)。
    """

    proof: Proof
    first_action: object | None = None
    depth: int | None = None
    stop_reasons: tuple[StopReason, ...] = ()
    nodes: int = 0
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.proof is Proof.PROVEN_WIN and self.first_action is None:
            raise ValueError("PROVEN_WIN requires a first action")

    @property
    def is_proven_win(self) -> bool:
        return self.proof is Proof.PROVEN_WIN
