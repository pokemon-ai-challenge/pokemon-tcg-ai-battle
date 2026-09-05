"""B4（ターン中の相手選択）の**独立** ground-truth オラクル（Step 1-23）。

`phase1` / `phase2` / `InfoKey` / macro generator / candidate generator /
pruning / transposition を一切 import しない。使うのはエンジンの
状態取得・合法手列挙・適用・勝敗判定だけ。

狙いは「solver と同じ実装をもう一つ作ること」ではなく、
**solver が間違えたときに同じ間違いをしないこと**。そのため

  * 枝刈りをしない
  * 置換表を使わない
  * 相手選択を必ず**全列挙**して AND を取る
  * 相手が我々に有利な選択をする前提を置かない

AND 契約:

    全ての相手選択で勝てる            -> WIN
    1 つでも勝てない相手選択がある    -> NO_WIN
    勝てない選択は無いが不明がある    -> UNKNOWN
    相手選択を列挙し切れない          -> UNKNOWN
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import OptionType, SelectContext
from ptcg_ai.search.lethal.engine import EngineError, SearchSession

WIN = "WIN"
NO_WIN = "NO_WIN"
UNKNOWN = "UNKNOWN"

MAX_BRANCH = 20


@dataclass
class B4Case:
    """1 つの相手選択ノードについての独立判定。"""

    verdict: str = UNKNOWN
    choice_count: int = 0
    per_choice: list[str] = field(default_factory=list)
    context: str = ""
    option_types: tuple[str, ...] = ()
    our_action: list[int] | None = None
    losing_choice: int | None = None

    @property
    def case_label(self) -> str:
        """指示 2 の Case A / B / C に対応させる。"""
        if not self.per_choice:
            return "no_choice"
        if all(v == WIN for v in self.per_choice):
            return "CaseA_all_win"
        if any(v == NO_WIN for v in self.per_choice):
            return "CaseB_one_no_win"
        return "CaseC_one_unknown"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "case": self.case_label,
            "choice_count": self.choice_count,
            "per_choice": self.per_choice,
            "context": self.context,
            "option_types": list(self.option_types),
            "our_action": self.our_action,
            "losing_choice": self.losing_choice,
        }


def _actions(observation) -> list[list[int]]:
    select = observation.select
    if select is None:
        return []
    count = len(select.option)
    low = max(select.minCount, 0)
    if count == 0:
        return [[]] if low == 0 else []
    if low <= 1 and min(select.maxCount, count) >= 1:
        return [[i] for i in range(min(count, MAX_BRANCH))]
    return [list(range(min(low if low else 1, count)))]


def _is_opponent_turn_choice(observation, me: int) -> bool:
    state = observation.current
    return (state is not None and state.result == -1
            and state.yourIndex != me and observation.select is not None)


def _continuation(session, node, me: int, depth: int) -> str:
    """相手選択の**後**、我々が勝ち切れるか。枝刈りなしの素朴な探索。"""
    state = node.observation.current
    if state is None:
        return UNKNOWN
    if state.result == me:
        return WIN
    if state.result != -1:
        return NO_WIN
    if node.observation.select is None:
        return NO_WIN
    if state.yourIndex != me:
        # 相手の手番が続く。ここから我々が勝つ手段は今ターンには無い。
        return NO_WIN
    if depth <= 0:
        return UNKNOWN
    saw_unknown = False
    for action in _actions(node.observation):
        try:
            child, _events = session.step(node, action)
        except EngineError:
            continue
        verdict = _continuation(session, child, me, depth - 1)
        if verdict == WIN:
            return WIN
        if verdict == UNKNOWN:
            saw_unknown = True
    return UNKNOWN if saw_unknown else NO_WIN


def evaluate_opponent_node(session, node, me: int, depth: int = 4) -> B4Case:
    """相手選択ノードを AND で独立評価する。"""
    observation = node.observation
    select = observation.select
    case = B4Case(
        context=str(SelectContext(int(select.context))) if select is not None else "",
        option_types=tuple(
            OptionType(int(o.type)).name for o in (select.option if select else ())
        ),
    )
    choices = _actions(observation)
    case.choice_count = len(choices)
    if not choices:
        case.verdict = UNKNOWN
        return case
    for index, choice in enumerate(choices):
        try:
            child, _events = session.step(node, choice)
        except EngineError:
            # 列挙し切れない = AND を取れない
            case.verdict = UNKNOWN
            case.per_choice.append(UNKNOWN)
            continue
        verdict = _continuation(session, child, me, depth)
        case.per_choice.append(verdict)
        if verdict == NO_WIN and case.losing_choice is None:
            case.losing_choice = index
    if any(v == NO_WIN for v in case.per_choice):
        case.verdict = NO_WIN          # 相手が 1 つでも防げる
    elif all(v == WIN for v in case.per_choice):
        case.verdict = WIN             # 全選択で勝てる
    else:
        case.verdict = UNKNOWN
    return case


def find_opponent_nodes(observation, hidden, me: int, *, depth: int = 3,
                        max_cases: int = 6) -> list[B4Case]:
    """root から浅く進めて、ターン中の相手選択ノードを探す。"""
    found: list[B4Case] = []
    with SearchSession(observation, hidden) as session:

        def walk(node, left, prefix):
            if len(found) >= max_cases or left <= 0:
                return
            for action in _actions(node.observation):
                if len(found) >= max_cases:
                    return
                try:
                    child, events = session.step(node, action)
                except EngineError:
                    continue
                if events.turn_ended:
                    continue
                if _is_opponent_turn_choice(child.observation, me):
                    case = evaluate_opponent_node(session, child, me)
                    if case.choice_count >= 2:
                        case.our_action = list(prefix + [action])[-1]
                        found.append(case)
                    continue
                walk(child, left - 1, prefix + [action])

        walk(session.root, depth, [])
    return found
