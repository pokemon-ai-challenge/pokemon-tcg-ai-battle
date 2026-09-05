"""診断ログの記録器(設計 §8 / 規則 R2b)。

原設計 §14 は「非公開カード実体・実際の山札順・巨大な状態 dump を記録しない」と
定めている。運用でそれを守るのではなく、**書けないようにする**。

- 禁止キー名(``your_deck`` / ``scenario`` / ``deck_order`` など)を拒否
- ``Scenario`` / ``HiddenState`` インスタンスを拒否
- 長い int 列(山札順の実体になりうる)を拒否
- ``Card`` / ``Pokemon`` などの dataclass をそのまま渡すのも拒否
  (serial ごと落ちるため。必要なら呼び出し側が id へ落とす)

違反は ``DiagnosticsLeakError`` で**即座に失敗させる**。黙って握りつぶすと
「テストでは出ないが本番で漏れる」状態になるため。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ptcg_ai.search.lethal.scenario import Scenario

FORBIDDEN_KEYS = frozenset(
    {
        "your_deck",
        "deck",
        "deck_order",
        "deck_permutation",
        "scenario",
        "permutation",
        "hidden_deck",
        "hidden_state",
        "opponent_hand",
        "opponent_deck",
        "prize_order",
        "your_prize",
    }
)

# 山札順の実体になりうる長さ。これ以上の int 列は記録させない。
MAX_INT_SEQUENCE = 8


class DiagnosticsLeakError(ValueError):
    """診断ログへ書いてはいけない値が渡された。"""


@dataclass
class DiagnosticsRecorder:
    """検査付きのイベント記録器。テストからも本番からも同じものを使う。"""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def record(self, event: str, **fields: Any) -> None:
        for key, value in fields.items():
            _check_key(key)
            _check_value(key, value)
        self.events.append((event, dict(fields)))

    def clear(self) -> None:
        self.events.clear()

    def as_dicts(self) -> list[dict[str, Any]]:
        return [{"event": name, **fields} for name, fields in self.events]


def _check_key(key: str) -> None:
    if key.lower() in FORBIDDEN_KEYS:
        raise DiagnosticsLeakError(f"forbidden diagnostics key: {key}")


def _check_value(key: str, value: Any) -> None:
    if isinstance(value, Scenario):
        raise DiagnosticsLeakError(f"{key}: Scenario must never be logged")
    if type(value).__name__ in ("HiddenState", "Card", "Pokemon", "PlayerState", "State"):
        raise DiagnosticsLeakError(f"{key}: {type(value).__name__} must never be logged")
    if isinstance(value, (str, bytes)):
        return
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            _check_key(str(sub_key))
            _check_value(f"{key}.{sub_key}", sub_value)
        return
    if isinstance(value, Sequence) or isinstance(value, (set, frozenset)):
        items = list(value)
        if len(items) > MAX_INT_SEQUENCE and all(isinstance(i, int) for i in items):
            raise DiagnosticsLeakError(
                f"{key}: int sequence of length {len(items)} may encode a deck order"
            )
        for index, item in enumerate(items):
            _check_value(f"{key}[{index}]", item)
        return
    if isinstance(value, Iterable) and not isinstance(value, (int, float, bool)):
        raise DiagnosticsLeakError(f"{key}: unbounded iterable is not loggable")
