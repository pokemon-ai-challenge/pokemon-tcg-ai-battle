"""critic の呼び出し契約と、価値の種別の分離(設計 §7 / 規則 R7)。

Step 1-1 で固定するのは**呼び方の安全性**だけである。校正(B3)は未解決なので、
ここで得た値を Phase 3 の採用判断に使ってはいけない。

固定する契約:

1. 葉の価値は常に **自分(``me``)視点**。観測が相手視点(``yourIndex != me``)なら
   ``1 - p`` へ変換する。``Q_try`` の失敗葉と ``Q_base`` のロールアウト葉で
   同じ規約を使う(混ぜると 0.1 規模のバイアスが入る。capability report §4.6)。
2. critic が例外・NaN・範囲外を返したら ``EstimateKind.UNAVAILABLE``。
   例外を対戦へ伝播させない(原設計 §2.5)。
3. 値には必ず ``ValueKind`` を付ける。``V_policy`` / ``V_fail`` / ``Q_try`` /
   ``Q_base`` を同じ float として持ち回らない。

まだ実装していないもの(意図的):
``Q_base`` のロールアウト計算、``Q_try`` の集約、採用ゲート。これらは
Phase 1/2 の本体と一緒に Step 1-3 以降で入れる。
"""

from __future__ import annotations

import math
from typing import Protocol

from cg.api import Observation

from ptcg_ai.search.lethal.types import (
    EstimateKind,
    StopReason,
    ValueEstimate,
    ValueKind,
)


class WinProbabilityModel(Protocol):
    """``ValueModel`` が満たす最小契約(テストで差し替えられるようにする)。"""

    def predict_win_prob(self, obs: Observation) -> float: ...


class LeafValueEvaluator:
    """葉の Observation を自分視点の勝率へ変換する、例外を出さない評価器。"""

    def __init__(self, model: WinProbabilityModel | None, *, source: str = "value_model"):
        self._model = model
        self._source = source

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def evaluate_leaf(
        self,
        obs: Observation,
        me: int,
        *,
        kind: ValueKind = ValueKind.V_POLICY,
    ) -> ValueEstimate:
        """葉の価値を返す。**どんな失敗でも例外を出さず UNAVAILABLE を返す**。"""
        if self._model is None:
            return ValueEstimate.unavailable(
                kind, source=self._source, stop_reason=StopReason.CRITIC_UNAVAILABLE
            )
        state = getattr(obs, "current", None)
        if state is None:
            return ValueEstimate.unavailable(
                kind, source=self._source, stop_reason=StopReason.CRITIC_UNAVAILABLE
            )
        try:
            raw = self._model.predict_win_prob(obs)
        except Exception:  # noqa: BLE001 - critic の失敗を対戦へ出さない
            return ValueEstimate.unavailable(
                kind, source=self._source, stop_reason=StopReason.CRITIC_UNAVAILABLE
            )
        value = _own_perspective(raw, state.yourIndex, me)
        if value is None:
            return ValueEstimate.unavailable(
                kind, source=self._source, stop_reason=StopReason.CRITIC_UNCALIBRATED
            )
        return ValueEstimate(
            kind=kind,
            estimate_kind=EstimateKind.EXACT,
            mean=value,
            lower=value,
            upper=value,
            source=self._source,
        )


def _own_perspective(raw: object, acting_index: int, me: int) -> float | None:
    """観測視点の勝率を自分視点へ直す。異常値は None。

    ``predict_win_prob`` は「その観測の持ち主」の勝率を返す
    (``encoder`` が ``state.yourIndex`` で視点正規化、学習ラベルも同じ定義)。
    したがって観測が相手のものなら ``1 - p``。
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        return None
    return value if acting_index == me else 1.0 - value
