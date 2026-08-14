"""実エンジン(``cg``)用の ``LethalBackend`` 実装。

状態は「root からの選択列」と「そこまでに実現したドロー列」で表す。
エンジンには状態複製 API が無いので、ある状態を作る唯一の方法は
``search_begin`` + プレフィックス再生である(Step 0 §1.1)。

安全側の設計(推測で通さない):

- コインが出たら ``UNSUPPORTED_EFFECT``(``manual_coin`` は Step 1-3 では使わない)
- サイド取得・自ターン中の相手選択も ``UNSUPPORTED_EFFECT``
- デッキ公開経路でシャッフルを挟まないドローは B1 ガードで ``UNSUPPORTED_EFFECT``
- 経路にシャッフルがあれば、その先のドローは ``SHUFFLE_ENCOUNTERED`` で列挙不可
- 探索開始時にデッキが公開されていたら(B1)、ドローは一切列挙不可
- outcome を構築したら**実際に引けたか検証**し、違えば列挙不可
- 複数選択の組合せを上限で打ち切ったら ``action_set_complete=False``
  (= ``PROVEN_NO_WIN`` を主張しない)

既知の非対応(Step 1-3 では扱わない): END を選んだ先(相手ターン開始時の
山札切れによる勝ちなど)は探索しない。既存 ``lethal_simple`` と同じ範囲。
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.search.lethal.backend import OutcomeEnumeration, Transition
from ptcg_ai.search.lethal.chance import RevealState, advance_reveal_state, detect_unsupported_events
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import (
    COIN,
    DRAW,
    SHUFFLE,
    EngineError,
    HiddenState,
    SearchSession,
    deck_revealed_at_root,
)
from ptcg_ai.search.lethal.enumeration import Outcome, OutcomeSet, draw_outcomes
from ptcg_ai.search.lethal.infokey import info_key, option_descriptor
from ptcg_ai.search.lethal.scenario import Scenario
from ptcg_ai.search.lethal.types import ChanceClass, StopReason

# R2: 既知 listing からの取得(取得先が確定している単純サーチ)。
# ここに含めるのは「公開されたカードから自分が選ぶだけ」のものに限る。
_KNOWN_LISTING_CONTEXTS = (int(SelectContext.TO_HAND), int(SelectContext.TO_BENCH))

MAX_COMBINATIONS_PER_SELECT = 128
MAX_OUTCOMES_PER_EVENT = 24
# outcome を適用するたびに ``search_begin`` + プレフィックス再生が走り、
# エンジン側の確保も増えていく。無制限にすると 1 回の探索が病的に遅くなるので、
# 回数で上限を切る(超えたら安全側に UNKNOWN)。
MAX_MATERIALIZATIONS = 200


@dataclass(frozen=True)
class CgState:
    """root からの選択列 + 実現したドロー列 + ガード状態。

    ``opponent_node`` は「自分のターン中に相手が選ぶ番」であることを示す
    (KO 後の ``TO_ACTIVE`` など)。探索側はここを AND ノードとして扱う。
    """

    path: tuple[tuple[int, ...], ...] = ()
    draws: tuple[tuple[int, ...], ...] = ()
    shuffled: bool = False
    reveal: RevealState = RevealState()
    opponent_node: bool = False


class _Refusal(Exception):
    def __init__(self, stop_reason: StopReason):
        super().__init__(stop_reason.name)
        self.stop_reason = stop_reason


class CgBackend:
    """``LethalBackend`` の実エンジン実装。"""

    def __init__(
        self,
        session: SearchSession,
        obs: Observation,
        hidden: HiddenState,
        *,
        deck_composition: KnownDeckComposition | None = None,
        max_combinations: int = MAX_COMBINATIONS_PER_SELECT,
        max_outcomes: int = MAX_OUTCOMES_PER_EVENT,
        max_materializations: int = MAX_MATERIALIZATIONS,
        known_listing_as_decision: bool = False,
    ) -> None:
        if deck_composition is not None and not isinstance(
            deck_composition, KnownDeckComposition
        ):
            # 順序を持つ表現(list[int] など)を探索の内側へ入れない(情報境界)。
            # 入力検査は他の何より先に行う。
            raise TypeError("deck_composition must be a KnownDeckComposition")
        self._session = session
        self._obs = obs
        self._hidden = hidden
        self._me = obs.current.yourIndex
        self._deck_composition = deck_composition
        self._full_deck = deck_composition.as_counter() if deck_composition else None
        self._max_combinations = max_combinations
        self._max_outcomes = max_outcomes
        self._max_materializations = max_materializations
        # R2(既知 listing からの取得)を decision node として扱うか。
        # **既定 False**: Step 1-9c の実測で、有効化しても新しい PROVEN_WIN は
        # 1 件も増えず(Phase 1 / Phase 2 とも)、Phase 1 の PROVEN_NO_WIN が
        # 6 件 UNKNOWN へ落ち、p50 が 42ms→122ms へ悪化しただけだった。
        # 機能とテストは残し、深さ・時間予算を増やせる段階で再評価する。
        self._known_listing_as_decision = known_listing_as_decision
        self._materializations = 0
        self._deck_revealed_at_root = deck_revealed_at_root(obs)
        self._nodes: dict[CgState, object] = {}
        self._complete: dict[CgState, bool] = {}
        # どの理由で枝を拒否したかの内訳(診断用。カード実体は入れない)。
        self.refusals: Counter = Counter()
        self._root = CgState(reveal=RevealState.at_root(self._deck_revealed_at_root))
        self._nodes[self._root] = session.root

    # ------------------------------------------------------------ 基本操作

    def root(self) -> CgState:
        return self._root

    def is_win(self, state: CgState) -> bool:
        observation = self._observation(state)
        current = observation.current
        return current is not None and current.result == self._me

    def is_terminal(self, state: CgState) -> bool:
        observation = self._observation(state)
        current = observation.current
        if current is None or current.result != -1:
            return True
        if current.yourIndex != self._me and not state.opponent_node:
            return True  # 自分のターンが終わった(相手の選択ノードは終端ではない)
        return observation.select is None or not observation.select.option

    def is_opponent_node(self, state: CgState) -> bool:
        return state.opponent_node

    def observable_key(self, state: CgState) -> Hashable:
        observation = self._observation(state)
        key = info_key(observation, self._me,
                       deck_multiset=self._deck_multiset(observation, state))
        current = observation.current
        if current is not None and current.players[self._me].hand is None:
            # 相手の選択ノードでは**自分の手札が観測に出ない**ので、キーだけでは
            # 別状態を区別できない。合流させないよう経路を混ぜる(健全側)。
            return (key, state.path, state.draws)
        return key

    def legal_actions(self, state: CgState) -> Sequence[tuple[int, ...]]:
        observation = self._observation(state)
        select = observation.select
        if select is None or not select.option:
            self._complete[state] = True
            return ()
        indices = [
            index
            for index, option in enumerate(select.option)
            if option.type != OptionType.END  # END はこのターン中の勝ちに繋がらない
        ]
        if self._is_known_listing_selection(observation, state):
            # デッキ listing には同名カードが何枚も並ぶ(45 枚中 15 種など)。
            # どのコピーを取っても、手札に入るカードIDも山札の残り multiset も同じなので、
            # **意味が同じ選択肢は 1 つに畳む**。
            # 健全性: 残る山札の「順序」だけが違うが、サーチ後のドローは
            # B1 ガード(R4)かシャッフル(クラス S)で必ず列挙対象外になるため、
            # 我々の情報モデルでは区別できず、区別する必要も無い。
            indices = self._dedupe_by_meaning(indices, observation)
        actions: list[tuple[int, ...]] = []
        complete = True
        low = max(select.minCount, 0)
        high = min(select.maxCount, len(indices))
        for count in range(low, high + 1):
            for combination in itertools.combinations(indices, count):
                if len(actions) >= self._max_combinations:
                    complete = False
                    break
                actions.append(tuple(combination))
            if not complete:
                break
        self._complete[state] = complete
        return tuple(actions)

    def action_set_complete(self, state: CgState) -> bool:
        if state not in self._complete:
            self.legal_actions(state)
        return self._complete.get(state, False)

    def apply(self, state: CgState, action: tuple[int, ...]) -> Transition:
        try:
            node = self._node(state)
        except _Refusal as refusal:
            return Transition(None, stop_reason=refusal.stop_reason)
        parent_turn = self._turn_of(node)
        try:
            child, events = self._session.step(node, list(action))
        except EngineError:
            return Transition(None)  # 違法手。単に使えないだけ

        reveal, violation = advance_reveal_state(state.reveal, events)
        opponent_node = self._opponent_chooses_mid_turn(child, events, parent_turn)
        child_state = CgState(
            path=state.path + (tuple(action),),
            draws=state.draws + ((tuple(events.drawn),) if events.drawn else ()),
            shuffled=state.shuffled or events.shuffled,
            reveal=reveal,
            opponent_node=opponent_node,
        )
        self._nodes[child_state] = child
        child_result = child.observation.current.result if child.observation.current else -1

        # 決着した局面は、その時点で全て確定している。サイド取得や手番移動を
        # 「未対応効果」として弾くと、**勝ち筋そのものを捨てることになる**。
        if child_result != -1:
            return Transition(child_state)

        if violation is not None:
            self.refusals[violation.name] += 1
            return Transition(None, stop_reason=violation)
        if COIN in events.sequence:
            # manual_coin を使っていないので、コイン結果は制御も列挙もできない。
            self.refusals["coin_without_manual_coin"] += 1
            return Transition(None, stop_reason=StopReason.UNSUPPORTED_EFFECT)
        if opponent_node:
            # B4: 自ターン中に相手が選ぶノード。**AND ノードとして探索側が扱う**。
            # 相手が我々に都合よく選ぶ前提は置かない(全選択で勝てるときだけ勝ち)。
            self.refusals["opponent_choice_node"] += 1
            return Transition(child_state, revealed=False)

        revealed = bool(events.drawn) or self._reveals_without_draw(child, child_state)
        if events.took_prize:
            # サイドを取ると、制御できない未知のカードが手札へ入る(B9)。
            # 拒否はしないが「新情報境界」として扱い、Phase 1 では展開せず、
            # Phase 2 では列挙できないので UNKNOWN になる。
            self.refusals["prize_taken_control_unverified"] += 1
            revealed = True
        return Transition(child_state, revealed=revealed)

    def _turn_of(self, node) -> int | None:
        current = node.observation.current
        return None if current is None else current.turn

    def _opponent_chooses_mid_turn(self, child, events, parent_turn) -> bool:
        """自分のターンが続いているのに相手へ手番が移ったか(B4)。

        攻撃などで**ターンが終わった**場合と区別する必要がある。ターン終了は
        ``TURN_END`` ログか、ターン番号の変化で判定できる。
        """
        current = child.observation.current
        if current is None or current.yourIndex == self._me:
            return False
        if events.turn_ended:
            return False
        return parent_turn is not None and current.turn == parent_turn

    # ------------------------------------------------------------ chance

    def attach_budget(self, budget) -> None:
        """探索の予算を列挙・具体化にも効かせる(Step 1-19 指示 10/11)。

        Phase 2 の予算は「列挙 + 構築 + 探索 + 検証 + 後始末の合計」で定義する。
        探索本体だけを測ると、列挙で予算の 18 倍(実測 9.0 秒 / 指定 500ms)を
        使い切ることがある。
        """
        self._budget = budget

    def _budget_exhausted(self) -> bool:
        budget = getattr(self, "_budget", None)
        return budget is not None and budget.check() is not None

    def enumerate_outcomes(self, state: CgState, action: tuple[int, ...]) -> OutcomeEnumeration:
        if self._deck_revealed_at_root:
            return OutcomeEnumeration(None, ChanceClass.ENGINE_RANDOM,
                                      StopReason.DECK_REVEALED_AT_ROOT)
        if state.shuffled:
            return OutcomeEnumeration(None, ChanceClass.ENGINE_RANDOM,
                                      StopReason.SHUFFLE_ENCOUNTERED)
        probe = self.apply(state, action)
        if probe.stop_reason is not None or probe.state is None:
            return OutcomeEnumeration(None, ChanceClass.UNCLASSIFIED,
                                      probe.stop_reason or StopReason.OUTCOMES_NOT_ENUMERABLE)
        if len(probe.state.draws) != len(state.draws) + 1:
            # ドロー以外の公開(デッキを見る等)は Step 1-3 では列挙しない。
            self.refusals["reveal_without_draw"] += 1
            return OutcomeEnumeration(None, ChanceClass.UNCLASSIFIED,
                                      StopReason.UNSUPPORTED_EFFECT)
        drawn_count = len(probe.state.draws[-1])
        try:
            belief = self._deck_multiset(self._observation(state), state)
        except _Refusal as refusal:
            return OutcomeEnumeration(None, ChanceClass.UNCLASSIFIED, refusal.stop_reason)
        if belief is None:
            return OutcomeEnumeration(None, ChanceClass.UNCLASSIFIED,
                                      StopReason.OUTCOMES_NOT_ENUMERABLE)
        if drawn_count <= 0 or drawn_count > sum(belief.values()):
            return OutcomeEnumeration(None, ChanceClass.UNCLASSIFIED,
                                      StopReason.OUTCOMES_NOT_ENUMERABLE)
        outcome_set = draw_outcomes(
            belief, drawn_count, should_stop=self._budget_exhausted
        )
        if outcome_set is None:
            # 列挙の途中で予算が尽きた。部分集合で証明してはならない。
            return OutcomeEnumeration(None, ChanceClass.CONTROLLED_SUPPLY_ORDER,
                                      StopReason.TIME_LIMIT)
        if len(outcome_set.outcomes) > self._max_outcomes:
            return OutcomeEnumeration(None, ChanceClass.CONTROLLED_SUPPLY_ORDER,
                                      StopReason.OUTCOMES_NOT_ENUMERABLE)
        return OutcomeEnumeration(outcome_set, ChanceClass.CONTROLLED_SUPPLY_ORDER)

    def apply_outcome(self, state: CgState, action: tuple[int, ...], outcome: Outcome) -> Transition:
        drawn = _label_to_cards(outcome.label)
        child_state = CgState(
            path=state.path + (tuple(action),),
            draws=state.draws + (drawn,),
            shuffled=state.shuffled,
            reveal=state.reveal,
        )
        try:
            self._node(child_state)  # ここで構築 + 検証が走る
        except _Refusal as refusal:
            return Transition(None, stop_reason=refusal.stop_reason)
        return Transition(child_state, revealed=True)

    # ------------------------------------------------------------ 内部

    def _observation(self, state: CgState) -> Observation:
        return self._node(state).observation

    def _node(self, state: CgState):
        cached = self._nodes.get(state)
        if cached is not None:
            return cached
        node = self._materialize(state)
        self._nodes[state] = node
        return node

    def _materialize(self, state: CgState):
        """``state`` を search_begin + プレフィックス再生で作り、ドローを検証する。"""
        if self._materializations >= self._max_materializations:
            self.refusals["materialization_limit"] += 1
            raise _Refusal(StopReason.NODE_LIMIT)
        self._materializations += 1
        scenario = self._scenario_for(state.draws)
        try:
            node = self._session.begin_with(self._hidden.with_scenario(scenario))
        except EngineError:
            raise _Refusal(StopReason.ENGINE_ERROR)
        observed: list[tuple[int, ...]] = []
        reveal = RevealState.at_root(self._deck_revealed_at_root)
        order_lost = False  # シャッフル後は供給順が効かない
        for selection in state.path:
            try:
                node, events = self._session.step(node, list(selection))
            except EngineError:
                raise _Refusal(StopReason.STATE_MISMATCH)
            reveal, violation = advance_reveal_state(reveal, events)
            if violation is not None:
                raise _Refusal(violation)
            if events.drawn:
                if not draw_is_supply_ordered(events.sequence, order_lost):
                    raise _Refusal(StopReason.SHUFFLE_ENCOUNTERED)
                observed.append(tuple(events.drawn))
            if events.shuffled:
                order_lost = True
        if tuple(observed) != tuple(state.draws):
            # 意図した outcome を再現できなかった = 完全列挙の前提が崩れている。
            raise _Refusal(StopReason.OUTCOMES_NOT_ENUMERABLE)
        return node

    def _scenario_for(self, draws: tuple[tuple[int, ...], ...]) -> Scenario:
        """``draws`` を順に引ける山札順を作る(末尾が山札の上、Step 0 §4.1)。"""
        remaining = Counter(self._hidden.scenario.multiset())
        for drawn in draws:
            for card_id in drawn:
                if remaining[card_id] <= 0:
                    raise _Refusal(StopReason.OUTCOMES_NOT_ENUMERABLE)
                remaining[card_id] -= 1
        order: list[int] = sorted(remaining.elements())
        for drawn in reversed(draws):
            order.extend(reversed(drawn))
        return Scenario(tuple(order))

    def _dedupe_by_meaning(self, indices, observation: Observation) -> list[int]:
        """意味(カードID・対象)が同じ選択肢を 1 つに畳む。

        解決できない選択肢が 1 つでもあれば**畳まない**(安全側)。
        """
        select = observation.select
        state = observation.current
        seen: dict = {}
        for index in indices:
            descriptor, resolved = option_descriptor(select.option[index], state, select, self._me)
            if not resolved:
                return list(indices)
            seen.setdefault(descriptor, index)
        return list(seen.values())

    def _deck_multiset(self, observation: Observation, state: CgState | None = None) -> Counter | None:
        """その局面での自分の山札 multiset(信念)。実際の山札順は使わない。

        Step 1-9b で導出方法を変更した。以前は
        「デッキリスト − 見えている自分のカード − 仮定した伏せサイド」で導いていたが、
        実測で**系統的に過少計上**していた(供給した山札 40 枚に対し belief 35 枚など)。
        列挙対象のプールが小さいと **outcome を取りこぼす = 偽証明の経路**になるため、
        導出を「我々が供給した山札から、引いた分を差し引く」方式へ変え、
        さらに ``deckCount`` と突き合わせて検証する。

        1. デッキが提示されている(``select.deck``)なら、それが確定情報
        2. そうでなければ「供給した multiset − ここまでに引いたカード」
        3. どちらも ``deckCount`` と枚数が一致しなければ **None**(安全側で諦める)
        """
        current = observation.current
        if current is None:
            return None
        deck_count = current.players[self._me].deckCount

        select = observation.select
        if select is not None and select.deck is not None:
            if any(card is None for card in select.deck):
                return None  # 伏せカードが混じる listing は使わない
            listing = Counter(card.id for card in select.deck)
            return listing if sum(listing.values()) == deck_count else None

        if state is None:
            return None
        remaining = self._belief_from_supply(state)
        if remaining is None:
            return None
        return remaining if sum(remaining.values()) == deck_count else None

    def _belief_from_supply(self, state: CgState) -> Counter | None:
        """供給した山札 multiset から、経路で引いたカードを差し引いたもの。"""
        remaining = Counter(self._hidden.scenario.multiset())
        for drawn in state.draws:
            for card_id in drawn:
                remaining[card_id] -= 1
        if any(count < 0 for count in remaining.values()):
            return None
        return +remaining

    def _reveals_without_draw(self, node, state: CgState | None = None) -> bool:
        """ドロー以外の情報公開(デッキを見る・looking)が起きたか。

        ただし **R2(既知 listing からの自分の選択)** は情報公開ではない(§Step 1-9a)。
        探索中に提示されるデッキ listing は「我々が供給した belief デッキ」なので、
        そこに我々の知らないカードは無い。それを**検証したうえで**通常の
        decision node として扱う。
        """
        observation = node.observation
        select = observation.select
        current = observation.current
        if current is not None and current.looking is not None:
            return True
        if select is None or select.deck is None:
            return False
        if self._is_known_listing_selection(observation, state):
            self.refusals["known_listing_selection"] += 1
            return False
        self.refusals["reveal_without_draw"] += 1
        return True

    def _is_known_listing_selection(
        self, observation: Observation, state: CgState | None
    ) -> bool:
        """R2 判定: 既知の belief デッキを見て**自分が**選ぶだけのノードか。

        以下をすべて**検証**する(仮定しない):

        1. 自分の手番である
        2. context が ``TO_HAND`` / ``TO_BENCH``(取得先が確定している単純サーチ)
        3. listing に伏せカードが無い
        4. listing の中身が、我々が供給した山札の残り(belief)の**部分集合**である
           = 我々がまだ知らなかったカードが 1 枚も現れていない

        4 を満たさない listing は「本当に新しい情報」なので、従来どおり公開扱いにする
        (将来そういう効果が出てきたら別 capability として分類する)。
        """
        if not self._known_listing_as_decision:
            return False
        select = observation.select
        current = observation.current
        if select is None or select.deck is None or current is None or state is None:
            return False
        if current.yourIndex != self._me:
            return False
        if int(select.context) not in _KNOWN_LISTING_CONTEXTS:
            return False
        if any(card is None for card in select.deck):
            return False
        expected = self._belief_from_supply(state)
        if expected is None:
            return False
        listing = Counter(card.id for card in select.deck)
        return not (listing - expected)


def draw_is_supply_ordered(sequence, order_lost: bool) -> bool:
    """このステップのドローが、供給した山札順どおりに起きたと言えるか。

    Step 1-7a で誤拒否の原因になった判定。**1 ステップ内でも順序が意味を持つ**:

    - ``DRAW, DRAW, DRAW, SHUFFLE`` → 引いてからシャッフル。ドローは供給順どおり(有効)
    - ``SHUFFLE, DRAW``             → シャッフル後の未知の順序から引いている(無効)
    - 経路上ですでにシャッフルが起きている(``order_lost``)→ 以降のドローは無効

    同一ステップに複数の draw / shuffle が混在する場合は、**最初のシャッフルより後に
    ドローがあるか**で判定する(1 つでもシャッフル後に引いていれば無効)。
    """
    if order_lost:
        return False
    shuffle_at = _first_index(sequence, SHUFFLE)
    if shuffle_at is None:
        return True
    last_draw = _last_index(sequence, DRAW)
    return last_draw is None or last_draw < shuffle_at


def _last_index(sequence, kind) -> int | None:
    found = None
    for index, item in enumerate(sequence):
        if item == kind:
            found = index
    return found


def _first_index(sequence, kind) -> int | None:
    for index, item in enumerate(sequence):
        if item == kind:
            return index
    return None


def _label_to_cards(label) -> tuple[int, ...]:
    cards: list[int] = []
    for card_id, count in label:
        cards.extend([card_id] * count)
    return tuple(cards)
