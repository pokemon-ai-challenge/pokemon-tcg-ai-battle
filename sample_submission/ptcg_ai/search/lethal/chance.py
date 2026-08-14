"""ランダム事象の分類(規則 R3)と B1 ガード(規則 R4)。

設計の要は「ログの種類から推論しない」こと。Step 0 で観測できたのは
「この 1 デッキ・限られたサンプルではこう見えた」であって、エンジンの仕様ではない。
そこで分類は**能力ベース**にする:

- ``CONTROLLED_SUPPLY_ORDER`` (C): 供給した山札順で outcome を指定できる
- ``CONTROLLED_EXPLICIT_CHOICE`` (M): エンジンが結果を選択肢として出す
- ``ENGINE_RANDOM`` (S): エンジン内部 RNG。適用も再現もできない

``SHUFFLE`` ログは**保守的な方向にだけ**使う。すなわち
「SHUFFLE があるから S にする」は許すが、「SHUFFLE が無いから C にする」は禁止。
C を主張するには ``is_replay_deterministic()``(規則 R3)の合格が必須である。

B1 ガード(規則 R4)は、デッキが公開された経路で SHUFFLE を挟まずにドローが
起きた場合に探索を止める。Step 0 ではそのような経路は観測されなかったが、
それは保証ではないので、実行時に必ず検査する。
"""

from __future__ import annotations

from dataclasses import dataclass

from cg.api import SelectContext, SelectData

from ptcg_ai.search.lethal.engine import DRAW, PRIZE, SHUFFLE, StepEvents
from ptcg_ai.search.lethal.types import ChanceClass, StopReason


# --------------------------------------------------------------- B1 ガード


@dataclass(frozen=True)
class RevealState:
    """「自分の山札が公開されたか」「その後シャッフルされたか」の追跡。"""

    deck_revealed: bool = False
    shuffled_since_reveal: bool = False

    @classmethod
    def at_root(cls, deck_revealed_at_root: bool) -> "RevealState":
        return cls(deck_revealed=deck_revealed_at_root, shuffled_since_reveal=False)


def advance_reveal_state(
    state: RevealState, events: StepEvents
) -> tuple[RevealState, StopReason | None]:
    """1ステップ分だけ B1 ガードを進める。

    Returns:
        (次の状態, 違反理由 or None)。違反が出たら呼び出し側はその枝を
        ``UNSUPPORTED_EFFECT`` として棄却し、確定探索を続けてはいけない。
    """
    deck_revealed = state.deck_revealed
    shuffled = state.shuffled_since_reveal
    if events.deck_listed_before:
        # このステップで答えた選択がデッキ公開選択だった = 実物のデッキを見ている。
        deck_revealed = True
        shuffled = False
    for kind in events.sequence:
        if kind == SHUFFLE:
            shuffled = True
        elif kind == DRAW:
            if deck_revealed and not shuffled:
                return (
                    RevealState(deck_revealed, shuffled),
                    StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL,
                )
    return RevealState(deck_revealed, shuffled), None


def check_reveal_guard(
    events_seq, *, deck_revealed_at_root: bool = False
) -> StopReason | None:
    """経路全体に B1 ガードを適用する。違反があればその理由を返す。"""
    state = RevealState.at_root(deck_revealed_at_root)
    for events in events_seq:
        state, violation = advance_reveal_state(state, events)
        if violation is not None:
            return violation
    return None


# ------------------------------------------------------------ 分類(能力ベース)


@dataclass(frozen=True)
class ChanceClassification:
    """分類結果。``evidence`` は診断用の根拠(順列などは入れない)。"""

    chance_class: ChanceClass
    stop_reason: StopReason | None = None
    evidence: tuple[str, ...] = ()

    @property
    def is_enumerable(self) -> bool:
        return self.chance_class.is_enumerable


def classify_pending_chance(
    *,
    select: SelectData | None,
    manual_coin: bool,
    deck_order_controlled: bool,
    shuffle_on_path: bool,
    replay_deterministic: bool | None,
    acting_is_me: bool = True,
    turn_ended: bool = False,
) -> ChanceClassification:
    """これから起きる乱数事象を分類する。

    Args:
        select: いま提示されている選択(``None`` なら選択待ちでない)。
        manual_coin: セッションが ``manual_coin=True`` で開かれているか。
        deck_order_controlled: 供給順序が効いているか
            (= 探索開始時にデッキが公開されていない)。
        shuffle_on_path: ここまでの経路で SHUFFLE が起きたか(**保守側の材料のみ**)。
        replay_deterministic: 規則 R3 の再生一致検査の結果。``None`` は未検査。
        acting_is_me: いま選ぶのが自分か。
        turn_ended: 自分のターンが終了済みか。

    Returns:
        ChanceClassification: ``is_enumerable`` が True のものだけ確定探索で使える。
    """
    evidence: list[str] = []
    if shuffle_on_path:
        evidence.append("shuffle_on_path")

    # 相手が選ぶ場面。確率平均してはいけない(原設計 §5.3)。初期版は未対応。
    if not acting_is_me and not turn_ended:
        return ChanceClassification(
            ChanceClass.OPPONENT_CHOICE,
            StopReason.UNSUPPORTED_EFFECT,
            tuple(evidence + ["opponent_to_act_during_our_turn"]),
        )

    # M: エンジンが結果そのものを選択肢として出している。
    if (
        manual_coin
        and select is not None
        and int(select.context) == int(SelectContext.COIN_HEAD)
    ):
        return ChanceClassification(
            ChanceClass.CONTROLLED_EXPLICIT_CHOICE,
            None,
            tuple(evidence + ["coin_head_select"]),
        )

    # ここから先は「供給順序で outcome を指定できるか」の判定。
    if shuffle_on_path:
        # シャッフル後は供給順序が効かない。SHUFFLE ログは S へ倒す方向にだけ使う。
        return ChanceClassification(
            ChanceClass.ENGINE_RANDOM,
            StopReason.SHUFFLE_ENCOUNTERED,
            tuple(evidence),
        )
    if not deck_order_controlled:
        return ChanceClassification(
            ChanceClass.ENGINE_RANDOM,
            StopReason.DECK_REVEALED_AT_ROOT,
            tuple(evidence + ["deck_revealed_at_root"]),
        )
    if replay_deterministic is None:
        # 未検査のものを C と呼ばない(規則 R3)。
        return ChanceClassification(
            ChanceClass.UNCLASSIFIED,
            StopReason.REPLAY_NOT_VERIFIED,
            tuple(evidence),
        )
    if not replay_deterministic:
        return ChanceClassification(
            ChanceClass.ENGINE_RANDOM,
            StopReason.REPLAY_MISMATCH,
            tuple(evidence + ["replay_mismatch"]),
        )
    return ChanceClassification(
        ChanceClass.CONTROLLED_SUPPLY_ORDER,
        None,
        tuple(evidence + ["replay_verified"]),
    )


# ----------------------------------------------- 未対応効果の検出(B4 / B8 / B9)


def detect_unsupported_events(
    events: StepEvents, *, answered_coin_select: bool = False
) -> tuple[StopReason | None, tuple[str, ...]]:
    """確定探索で扱えない事象を**検出**する(処理はしない)。

    Step 1 の方針(ユーザ指示 2)に従い、未解決ブロッカーは推測で処理せず、
    検出して ``UNSUPPORTED_EFFECT`` へ送る。

    - B8: 1つの ``COIN_HEAD`` 選択に対してコインが2枚以上出た
      → 混合 outcome へ到達できない可能性があるため列挙不能とする
    - B9: サイド取得が起きた → ``your_prize`` の供給順で制御できるか未検証
    - B4: 自ターン中に相手へ手番が移った → 相手選択ノード
    """
    reasons: list[str] = []
    if answered_coin_select and len(events.coin_heads) > 1:
        reasons.append("multi_coin_per_select")
    if PRIZE in events.sequence:
        reasons.append("prize_taken_control_unverified")
    if not events.acting_is_me and not events.turn_ended:
        reasons.append("opponent_to_act_during_our_turn")
    if reasons:
        return StopReason.UNSUPPORTED_EFFECT, tuple(reasons)
    return None, ()
