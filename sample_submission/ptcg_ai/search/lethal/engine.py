"""``cg.api`` の探索 API の薄いラッパ(設計 §4 adapter)。

責務は4つだけで、探索ロジックは持たない。

1. 資源管理: ``search_release`` / ``search_end`` を必ず呼ぶ
2. 非公開情報の保持: 山札順列を ``HiddenState`` へ閉じ込める(規則 R2)
3. ログの正規化: 1ステップで起きた乱数事象を順序付きで取り出す
4. 再現性検査: 同一入力・同一手順の再生が一致するか(規則 R3)

エンジンは ``agent_ptr`` を共有するグローバル資源なので、セッションの入れ子は
禁止する(入れ子にすると ``search_end()`` が他方の状態も壊す)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterator, Mapping, Sequence

from cg import api as cg_api
from cg.api import LogType, Observation

from ptcg_ai.search.lethal.scenario import Scenario

DRAW = "draw"
SHUFFLE = "shuffle"
COIN = "coin"
PRIZE = "prize"


class EngineError(RuntimeError):
    """探索 API 呼び出しが失敗した。呼び出し側は通常方策へ戻すこと。"""


class SessionNestingError(RuntimeError):
    """探索セッションを入れ子にしようとした(共有資源が壊れる)。"""


@dataclass(frozen=True)
class HiddenState:
    """``search_begin()`` に渡す非公開情報一式。

    順序を持つのは ``scenario``(自分の山札)と ``your_prize`` で、どちらも
    「我々が仮定した並び」であって実物ではない。決定側へ漏らさないため、
    ``repr`` は中身を出さず、公開してよいのは multiset だけにしてある。
    """

    scenario: Scenario
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]

    def __repr__(self) -> str:  # 順序を出さない(規則 R2b)
        return (
            f"HiddenState(deck={self.scenario!r}, prize={len(self.your_prize)}, "
            f"opp_deck={len(self.opponent_deck)}, opp_hand={len(self.opponent_hand)})"
        )

    __str__ = __repr__

    @classmethod
    def from_stub(cls, stub: Mapping[str, Sequence[int]]) -> "HiddenState":
        """``search_state_stub.build_dummy_search_state()`` の dict から作る。"""
        return cls(
            scenario=Scenario(tuple(int(c) for c in stub["your_deck"])),
            your_prize=tuple(int(c) for c in stub["your_prize"]),
            opponent_deck=tuple(int(c) for c in stub["opponent_deck"]),
            opponent_prize=tuple(int(c) for c in stub["opponent_prize"]),
            opponent_hand=tuple(int(c) for c in stub["opponent_hand"]),
            opponent_active=tuple(int(c) for c in stub["opponent_active"]),
        )

    def deck_multiset(self) -> Counter:
        """自分の山札の multiset。**これは決定側へ渡してよい**。"""
        return self.scenario.multiset()

    def with_scenario(self, scenario: Scenario) -> "HiddenState":
        """順列だけ差し替える(outcome 適用用。multiset は呼び出し側が保つこと)。"""
        return replace(self, scenario=scenario)


@dataclass(frozen=True)
class StepEvents:
    """1ステップで観測された乱数関連の事象(ログ順)。

    ``sequence`` はログの並び順そのもの。B1 ガード(規則 R4)は
    「ドローがシャッフルより前か後か」を見るため、順序が必要になる。
    """

    sequence: tuple[str, ...] = ()
    drawn: tuple[int, ...] = ()
    coin_heads: tuple[bool, ...] = ()
    deck_listed_before: bool = False
    turn_ended: bool = False
    acting_is_me: bool = True

    @property
    def shuffled(self) -> bool:
        return SHUFFLE in self.sequence

    @property
    def drew(self) -> bool:
        return DRAW in self.sequence

    @property
    def took_prize(self) -> bool:
        return PRIZE in self.sequence


@dataclass
class _Node:
    """``SearchState`` のラッパ(search_id と observation)。"""

    search_id: int
    observation: Observation


class SearchSession:
    """1回の ``search_begin`` に対応するセッション。必ず ``with`` で使う。"""

    _active: "SearchSession | None" = None

    def __init__(
        self,
        obs: Observation,
        hidden: HiddenState,
        *,
        manual_coin: bool = False,
    ) -> None:
        self._obs = obs
        self._hidden = hidden
        self._manual_coin = manual_coin
        self._search_ids: list[int] = []
        self._root: _Node | None = None
        self._me = obs.current.yourIndex if obs.current is not None else 0
        # 資源リークの検出用カウンタ(begin/step で確保した数と解放した数)。
        self.acquired = 0
        self.released = 0

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "SearchSession":
        if SearchSession._active is not None:
            raise SessionNestingError(
                "search sessions must not be nested (agent_ptr is shared)"
            )
        SearchSession._active = self
        try:
            self._root = self._begin()
        except Exception:
            SearchSession._active = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for search_id in self._search_ids:
            try:
                cg_api.search_release(search_id)
                self.released += 1
            except Exception:  # noqa: BLE001 - 解放失敗は握りつぶす
                pass
        self._search_ids.clear()
        try:
            cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass
        SearchSession._active = None

    @property
    def me(self) -> int:
        return self._me

    @property
    def root(self) -> _Node:
        if self._root is None:
            raise EngineError("session is not open")
        return self._root

    # ------------------------------------------------------------- stepping

    def begin_with(self, hidden: HiddenState) -> _Node:
        """同じセッション内に、別の隠れ情報(別 scenario)の root を作る。

        outcome を適用するには「その outcome を実現する山札順で開き直して
        プレフィックスを再生する」しかない(状態複製 API が無いため)。
        同一セッション内で開くことで、``search_end()`` の呼び出しは 1 回で済む。
        """
        return self._begin(hidden)

    def _begin(self, hidden: HiddenState | None = None) -> _Node:
        hidden = hidden or self._hidden
        try:
            state = cg_api.search_begin(
                self._obs,
                hidden.scenario.to_engine_list(),
                list(hidden.your_prize),
                list(hidden.opponent_deck),
                list(hidden.opponent_prize),
                list(hidden.opponent_hand),
                list(hidden.opponent_active),
                self._manual_coin,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"search_begin failed: {type(exc).__name__}") from exc
        self._search_ids.append(state.searchId)
        self.acquired += 1
        return _Node(state.searchId, state.observation)

    def step(self, node: _Node, selection: Sequence[int]) -> tuple[_Node, StepEvents]:
        """1手進める。違法手は ``EngineError`` にせず例外を素通しさせない。"""
        deck_listed_before = (
            node.observation.select is not None and node.observation.select.deck is not None
        )
        try:
            state = cg_api.search_step(node.search_id, list(selection))
        except ValueError as exc:
            raise EngineError(f"illegal selection {list(selection)}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"search_step failed: {type(exc).__name__}") from exc
        self._search_ids.append(state.searchId)
        self.acquired += 1
        child = _Node(state.searchId, state.observation)
        return child, self._events(child, deck_listed_before)

    def walk(self, path: Sequence[Sequence[int]]) -> tuple[_Node, list[StepEvents]]:
        """root から ``path`` を再生する。"""
        node = self.root
        events: list[StepEvents] = []
        for selection in path:
            node, step_events = self.step(node, selection)
            events.append(step_events)
        return node, events

    # --------------------------------------------------------------- events

    def _events(self, node: _Node, deck_listed_before: bool) -> StepEvents:
        sequence: list[str] = []
        drawn: list[int] = []
        coins: list[bool] = []
        turn_ended = False
        for log in node.observation.logs:
            log_type = int(log.type)
            if log_type == int(LogType.DRAW) and log.playerIndex == self._me:
                sequence.append(DRAW)
                drawn.append(log.cardId)
            elif log_type == int(LogType.SHUFFLE):
                sequence.append(SHUFFLE)
            elif log_type == int(LogType.COIN):
                sequence.append(COIN)
                coins.append(bool(log.head))
            elif log_type == int(LogType.TURN_END) and log.playerIndex == self._me:
                turn_ended = True
            elif (
                log_type == int(LogType.MOVE_CARD)
                and log.playerIndex == self._me
                and log.fromArea is not None
                and int(log.fromArea) == 6  # AreaType.PRIZE
            ):
                sequence.append(PRIZE)
        state = node.observation.current
        acting_is_me = state is None or state.yourIndex == self._me
        return StepEvents(
            sequence=tuple(sequence),
            drawn=tuple(drawn),
            coin_heads=tuple(coins),
            deck_listed_before=deck_listed_before,
            turn_ended=turn_ended,
            acting_is_me=acting_is_me,
        )

    # ---------------------------------------------------------- fingerprint

    @staticmethod
    def fingerprint(node: _Node) -> str:
        """再現性比較用の指紋(``lethal_simple._state_key`` と同じ粒度)。"""
        obs = node.observation
        return f"{obs.current!r}|{obs.select!r}"


# ------------------------------------------------------------------ helpers


def deck_revealed_at_root(obs: Observation) -> bool:
    """探索開始時点でデッキが公開されているか(B1)。

    ``cg/api.py:553`` の通り、``obs.select.deck != None`` のとき
    ``search_begin()`` は我々の ``your_deck`` を捨てて**実物のデッキ**を使う。
    このとき供給順序による outcome 制御は効かない。
    """
    return obs.select is not None and obs.select.deck is not None


def replay_fingerprint(
    obs: Observation,
    hidden: HiddenState,
    path: Sequence[Sequence[int]],
    *,
    manual_coin: bool = False,
) -> tuple[str | None, list[StepEvents]]:
    """``path`` を1回再生し、終端の指紋と各ステップの事象を返す。

    途中で違法になった場合は ``(None, これまでの事象)`` を返す。
    """
    with SearchSession(obs, hidden, manual_coin=manual_coin) as session:
        node = session.root
        events: list[StepEvents] = []
        for selection in path:
            try:
                node, step_events = session.step(node, selection)
            except EngineError:
                return None, events
            events.append(step_events)
        return SearchSession.fingerprint(node), events


def is_replay_deterministic(
    obs: Observation,
    hidden: HiddenState,
    path: Sequence[Sequence[int]],
    *,
    repeats: int = 2,
    manual_coin: bool = False,
) -> bool:
    """規則 R3: 同一入力・同一手順の再生が一致するか。

    ``SHUFFLE`` ログの有無では判定しない。**実際に再生して比べる**。
    1度でも違法化・不一致があれば False。
    """
    if repeats < 2:
        raise ValueError("repeats must be >= 2 to compare anything")
    reference: str | None = None
    for _ in range(repeats):
        fingerprint, _ = replay_fingerprint(obs, hidden, path, manual_coin=manual_coin)
        if fingerprint is None:
            return False
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            return False
    return True


def iter_legal_single_selections(obs: Observation) -> Iterator[list[int]]:
    """``minCount <= 1 <= maxCount`` の選択肢を1つずつ返す(検査用の簡易列挙)。"""
    select = obs.select
    if select is None or not select.option:
        return
    if not (select.minCount <= 1 <= select.maxCount):
        return
    for index in range(len(select.option)):
        yield [index]
