"""抽象アクションと ``EngineAction`` の対応(設計 §8 / Step 1-3 指示 7)。

このエンジンでは「行動 = ``obs.select.option`` のインデックス列」なので、
探索が扱う抽象アクションと ``EngineAction`` は**同じ選択インデックス列**である。
ただしインデックスは位置でしかないため、そのまま持ち回ると別の観測で
「同じ番号だが別のカード」になりうる。そこで実行時の照合用に、各インデックスの
**意味**(カードID・対象ポケモン・ワザID など)を一緒に保持する。

対応関係:

| 項目 | 表現 |
|---|---|
| action ID | ``Option.type``(PLAY / ATTACH / EVOLVE / ATTACK / ABILITY ...) |
| 対象カード | ``area`` + ``index`` を解決したカードID |
| 対象ポケモン | ``inPlayArea`` + ``inPlayIndex`` を解決したポケモンID |
| 選択インデックス | ``EngineAction.selection`` |
| 複数選択 | ``selection`` が複数要素(``minCount``〜``maxCount``) |
| 行動順序 | 保持しない(最初の1手だけ実行するため) |
| legality | ``validate()`` が実行直前に再検査する |

**変換が一意でない場合は解決しない**。``validate()`` は None を返し、
呼び出し側は通常方策へ戻す(``UNKNOWN`` 扱い)。
"""

from __future__ import annotations

from dataclasses import dataclass

from cg.api import SelectData, State

from ptcg_ai.search.lethal.infokey import option_descriptor


@dataclass(frozen=True)
class EngineAction:
    """エンジンへ渡す選択と、その意味。"""

    selection: tuple[int, ...]
    descriptors: tuple[tuple, ...]
    resolved: bool

    def as_list(self) -> list[int]:
        return list(self.selection)


def is_legal_selection(selection, select: SelectData) -> bool:
    """提出契約と同じ検査(``selector.is_valid_action`` と同値)。"""
    if not isinstance(selection, (list, tuple)):
        return False
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in selection):
        return False
    if not (select.minCount <= len(selection) <= select.maxCount):
        return False
    if len(selection) != len(set(selection)):
        return False
    return all(0 <= index < len(select.option) for index in selection)


def describe(selection, select: SelectData, state: State, me: int) -> EngineAction | None:
    """選択インデックス列から ``EngineAction`` を作る。違法なら None。"""
    if not is_legal_selection(selection, select):
        return None
    descriptors = []
    resolved = True
    for index in selection:
        parts, ok = option_descriptor(select.option[index], state, select, me)
        descriptors.append(parts)
        resolved = resolved and ok
    return EngineAction(tuple(selection), tuple(descriptors), resolved)


def validate(
    action: EngineAction,
    select: SelectData,
    state: State,
    me: int,
    *,
    same_observation: bool = False,
) -> list[int] | None:
    """実行直前の照合。安全に実行できる選択列を返す。できなければ None。

    検査するのは:

    - 選択列が現在の合法条件(個数・重複・範囲)を満たすこと
    - 各インデックスの**意味**が探索時と一致すること(状態不一致の検出)
    - 意味へ解決できない選択肢を含む場合は、探索時と同一観測だと
      呼び出し側が保証したときだけ実行を許す(``same_observation``)

    同じ意味の選択肢が複数ある場合(例: 同名カードが手札に2枚)は**曖昧ではない**。
    どちらを選んでも同じ意味なので、探索時のインデックスをそのまま使う。
    別の観測へ意味から引き直すのは :func:`reindex` の仕事で、そちらは
    一意に決まらなければ None を返す。
    """
    if not is_legal_selection(action.selection, select):
        return None
    current = describe(action.selection, select, state, me)
    if current is None:
        return None
    if current.descriptors != action.descriptors:
        return None
    if not action.resolved and not same_observation:
        return None
    return action.as_list()


def reindex(
    action: EngineAction, select: SelectData, state: State, me: int
) -> list[int] | None:
    """**別の観測**で、意味からインデックスを引き直す。

    1つでも「候補が0個 / 2個以上で区別できない」ものがあれば None を返す
    (勝手に解決しない)。呼び出し側はこれを ``UNKNOWN`` として扱うこと。
    """
    if not action.resolved:
        return None
    selection: list[int] = []
    used: set[int] = set()
    for descriptor in action.descriptors:
        matches = [
            index
            for index, option in enumerate(select.option)
            if index not in used
            and option_descriptor(option, state, select, me) == (descriptor, True)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            # 同じ意味が複数残っている場合、どれを選んでも等価だが、
            # 「等価であること」をここで証明できないので解決しない。
            return None
        selection.append(matches[0])
        used.add(matches[0])
    if not is_legal_selection(selection, select):
        return None
    return selection
