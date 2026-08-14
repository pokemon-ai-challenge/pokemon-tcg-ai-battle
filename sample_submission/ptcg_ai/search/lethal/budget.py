"""探索予算(設計 §6)。

Step 1-3 では**現行と同じ 100 ms 相当**から始める。本番向けの引き上げ
(700 ms 等)は計測してからで、ここでは決めない(ブロッカー B2)。

予算切れは必ず ``StopReason`` を返し、Phase 側で ``UNKNOWN`` になる。
**予算切れを ``PROVEN_NO_WIN`` に丸めない**のがこのクラスの唯一の役目。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ptcg_ai.search.lethal.types import StopReason


@dataclass
class Budget:
    """時間・ノード数の上限。使い切ったら ``check()`` が理由を返す。"""

    time_limit_ms: float = 100.0
    max_nodes: int = 10_000
    max_depth: int = 20
    max_chance_depth: int = 1
    _deadline: float = field(default=0.0, init=False)
    nodes: int = field(default=0, init=False)
    exhausted_reason: StopReason | None = field(default=None, init=False)

    def start(self) -> "Budget":
        self._deadline = time.perf_counter() + self.time_limit_ms / 1000.0
        self.nodes = 0
        self.exhausted_reason = None
        return self

    def check(self) -> StopReason | None:
        if self.exhausted_reason is not None:
            return self.exhausted_reason
        if time.perf_counter() > self._deadline:
            self.exhausted_reason = StopReason.TIME_LIMIT
        elif self.nodes >= self.max_nodes:
            self.exhausted_reason = StopReason.NODE_LIMIT
        return self.exhausted_reason

    def spend_node(self) -> StopReason | None:
        self.nodes += 1
        return self.check()

    @property
    def elapsed_ms(self) -> float:
        return self.time_limit_ms - max(0.0, (self._deadline - time.perf_counter()) * 1000.0)
