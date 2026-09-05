"""確定探索(Phase 1/2)が乗るゲーム側の最小インタフェース(Protocol のみ)。

toy(``tests/toy/``)と実エンジン(``cg_backend.py``)の両方をこの形にしてあるので、
**同じ Phase 1/2 のコードを toy oracle で検証できる**。

設計上の約束(実装側の責務):

- 情報境界: ``apply`` が返す ``Transition.revealed`` が True なら、そこは新情報境界。
  Phase 1 はそこで打ち切り、Phase 2 は ``enumerate_outcomes`` へ移る。
- **証明条件を backend 側で緩めない**: 列挙できない・扱えない事象は
  ``stop_reason`` を付けて返し、Phase 側が ``UNKNOWN`` にする。
  backend が「たぶん大丈夫」で成功を返してはいけない。
- 隠れ情報(実際の山札順・相手の非公開手札・エンジン内部 RNG)は
  ``observable_key`` に含めない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Protocol, Sequence, TypeVar

from ptcg_ai.search.lethal.enumeration import Outcome, OutcomeSet
from ptcg_ai.search.lethal.types import ChanceClass, StopReason

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Transition(Generic[StateT]):
    """1手適用の結果。``state is None`` は「その手は使えない」を意味する。"""

    state: StateT | None
    revealed: bool = False
    stop_reason: StopReason | None = None

    @property
    def ok(self) -> bool:
        return self.state is not None and self.stop_reason is None


@dataclass(frozen=True)
class OutcomeEnumeration:
    """chance 事象の列挙結果。列挙できないときは ``outcome_set is None``。"""

    outcome_set: OutcomeSet | None
    chance_class: ChanceClass
    stop_reason: StopReason | None = None


class LethalBackend(Protocol, Generic[StateT, ActionT]):
    """Phase 1/2 が必要とする操作の全部。ゲーム固有の仮定はここに入れない。"""

    def root(self) -> StateT:
        """探索の起点。"""

    def legal_actions(self, state: StateT) -> Sequence[ActionT]:
        """合法手。順序は探索順の既定値としてのみ使う(枝刈りではない)。"""

    def action_set_complete(self, state: StateT) -> bool:
        """``legal_actions`` がその状態の合法手を**漏れなく**返しているか。

        False の場合、``PROVEN_NO_WIN`` を主張してはいけない。
        """

    def apply(self, state: StateT, action: ActionT) -> Transition[StateT]:
        """1手適用。新情報が公開されたら ``revealed=True`` を返す。"""

    def is_win(self, state: StateT) -> bool: ...

    def is_terminal(self, state: StateT) -> bool: ...

    def is_opponent_node(self, state: StateT) -> bool:
        """自分のターン中に**相手が選ぶ**ノードか(原設計 §5.3)。

        True の場合、探索は OR ではなく **AND** で扱う:
        全ての合法な相手選択で勝てるときだけ ``PROVEN_WIN``。
        相手が自分に都合よく選ぶ前提を置いてはいけない。
        Phase 3 では同じノードを最悪値(min)で評価することになる。
        """

    def observable_key(self, state: StateT) -> Hashable:
        """状態同値判定用のキー。隠れ情報を含めてはいけない。"""

    def enumerate_outcomes(self, state: StateT, action: ActionT) -> OutcomeEnumeration:
        """``action`` が生む chance 事象の outcome を厳密列挙する。

        少しでも保証できない場合は ``outcome_set=None`` と ``stop_reason`` を返す。
        """

    def apply_outcome(
        self, state: StateT, action: ActionT, outcome: Outcome
    ) -> Transition[StateT]:
        """指定した outcome が実現した状態を**構築**する。

        構築・検証に失敗したら ``state=None`` と ``stop_reason`` を返すこと。
        """
