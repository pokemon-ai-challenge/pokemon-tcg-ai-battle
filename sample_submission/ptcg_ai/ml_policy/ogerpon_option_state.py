"""カプ・ブルル中継戦略(SINGLE_PRIZE_ROTATION)の固定Option Controller(Phase1)。

設計書(``ogerpon_single_prize_mlp_design.md``)の高位戦略 ``EX_TEMPO`` /
``SINGLE_PRIZE_ROTATION`` のうち、低位方策と終了条件をルールで固定した
「固定Option」部分だけを実装する(Q-critic本体はPhase3)。

## このモジュールの役割

``OgerponOptionState`` は「今どの段階か(BUILD/READY/ACTIVE/...)」だけを持つ状態機械。
盤面から高位戦略そのものを選ぶ判断(Q-critic)はまだ無いので、Phase1では
``force_start``(呼び出し側が明示的に渡すフラグ)または
``config["ogerpon_option_state"]["force_start_test_mode"]`` でのみ IDLE -> BUILD へ入る。
本番の意思決定へは配線しない(既定 OFF、``ml_policy_agent`` からは呼ばれない)。

## 状態遷移

    IDLE --(force_start)--> BUILD --(攻撃可能)--> READY --(昇格)--> ACTIVE --(気絶)--> COMPLETE
    BUILD/READY --(対象消失・置換/締切超過)--> ABORT
    ACTIVE --(対象がベンチ等へ想定外に移動)--> ABORT
    COMPLETE / ABORT --(呼び出し側が reset() するまで)--> そのまま

``COMPLETE``/``ABORT`` から ``IDLE`` への遷移は呼び出し側の責務(reset())。試合終了
(``obs.select is None``)では必ず ``reset()`` を呼び、試合間に状態を漏らさないこと
(``policy_registry`` と同じ規約)。

## 未実装(design.md §5.4 のうち、静的な盤面スナップショットだけでは判定できないもの)

以下は候補間の比較・仮実行(``search_step``)が要るため、Strategy Window Builder /
安全ゲート(Phase2以降、``ml_policy_agent`` 配線時)に持ち越す:

- Option継続により今ターンの攻撃が不可能になる。
- オーガポンへの手貼りでのみ確定KOへ到達できる。
- 次アタッカーが1体もおらず、オーガポンへの手貼りでのみ準備できる。

このモジュール単体では、盤面スナップショットと ``obs.logs`` だけから判定できる中断条件
(試合終了・対象消失/置換・締切超過)のみを扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

from cg.api import LogType, Observation

from ptcg_ai.ml_policy import ogerpon_planner

IDLE = "IDLE"
BUILD = "BUILD"
READY = "READY"
ACTIVE = "ACTIVE"
COMPLETE = "COMPLETE"
ABORT = "ABORT"

DEFAULTS: dict = {
    # BUILD開始から何回自分のターンを経ても4エネ完成しなければ ABORT するか。
    # State.turn は先攻/後攻ともに1手ごとに+1される共有カウンタなので、
    # 「自分のターン数」に換算するには x2 する。
    "max_build_turns": 4,
    # Phase1限定のテストフック。Q-criticが無いので、これが True の間は
    # IDLE の局面で対象になりうるカプ・ブルルが見つかり次第 BUILD へ入る。
    "force_start_test_mode": False,
}


def option_config(config: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    user = (config or {}).get("ogerpon_option_state") or {}
    if isinstance(user, dict):
        cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    return cfg


@dataclass
class OgerponOptionState:
    mode: str = IDLE
    phase: str = IDLE
    target_serial: int | None = None
    target_card_id: int | None = None
    started_turn: int | None = None
    last_decision_turn: int | None = None
    build_deadline_turn: int | None = None
    bulu_attack_count: int = 0
    abort_reason: str | None = None

    def reset_in_place(self) -> None:
        self.mode = IDLE
        self.phase = IDLE
        self.target_serial = None
        self.target_card_id = None
        self.started_turn = None
        self.last_decision_turn = None
        self.build_deadline_turn = None
        self.bulu_attack_count = 0
        self.abort_reason = None


# ---------------------------------------------------------------------------
# 純粋な遷移コア(prev を書き換えず、新しい OgerponOptionState を返す)。
# ユニットテストはこちらを直接使う(モジュールグローバルの reset() 管理が要らない)。
# ---------------------------------------------------------------------------

def _find_by_serial(mine, serial: int | None):
    if serial is None:
        return None
    for slot in list(mine.active or []) + list(mine.bench or []):
        if slot is not None and getattr(slot, "serial", None) == serial:
            return slot
    return None


def _find_new_bulu_target(mine):
    """BUILD開始時の対象候補を探す(ベンチのカプ・ブルルを優先し、無ければactiveを見る)。"""
    bench = [b for b in (mine.bench or []) if b is not None]
    for b in bench:
        if ogerpon_planner.is_bulu(b):
            return b
    active = next((s for s in (mine.active or []) if s is not None), None)
    if active is not None and ogerpon_planner.is_bulu(active):
        return active
    return None


def _count_bulu_attacks(obs: Observation, me: int, target_serial: int | None) -> int:
    """``obs.logs``(前回選択以降のイベント)から、対象がこの区間で攻撃した回数を数える。"""
    if target_serial is None:
        return 0
    count = 0
    for log in (getattr(obs, "logs", None) or []):
        try:
            if (int(getattr(log, "type", -1)) == int(LogType.ATTACK)
                    and getattr(log, "playerIndex", None) == me
                    and getattr(log, "serial", None) == target_serial):
                count += 1
        except Exception:  # noqa: BLE001
            continue
    return count


def _aborted(prev: OgerponOptionState, reason: str, turn: int | None) -> OgerponOptionState:
    nxt = OgerponOptionState(**vars(prev))
    nxt.mode = ABORT
    nxt.phase = ABORT
    nxt.abort_reason = reason
    nxt.last_decision_turn = turn
    return nxt


def advance_state(prev: OgerponOptionState, obs: Observation, me: int, config: dict | None,
                  *, force_start: bool = False) -> OgerponOptionState:
    """``prev`` を1手分進めた新しい ``OgerponOptionState`` を返す(``prev`` は変更しない)。

    例外時・obs不正時は ``prev`` をそのまま返す(安全側フォールバック。状態を壊さない)。
    """
    try:
        state = obs.current
        if state is None:
            return prev
        cfg = option_config(config)
        turn = getattr(state, "turn", None)
        mine = state.players[me]

        if getattr(state, "result", -1) != -1:
            # 試合終了。以降の判断は無意味なので現状維持で返す
            # (reset() は呼び出し側が新しい試合の開始で行う)。
            return prev

        if prev.mode == IDLE:
            if not (force_start or cfg["force_start_test_mode"]):
                nxt = OgerponOptionState(**vars(prev))
                nxt.last_decision_turn = turn
                return nxt
            target = _find_new_bulu_target(mine)
            if target is None:
                nxt = OgerponOptionState(**vars(prev))
                nxt.last_decision_turn = turn
                return nxt
            deadline = None
            if turn is not None:
                deadline = turn + int(cfg["max_build_turns"]) * 2
            return OgerponOptionState(
                mode=BUILD, phase=BUILD,
                target_serial=getattr(target, "serial", None),
                target_card_id=getattr(target, "id", None),
                started_turn=turn, last_decision_turn=turn,
                build_deadline_turn=deadline, bulu_attack_count=0, abort_reason=None,
            )

        if prev.mode in (COMPLETE, ABORT):
            # IDLE への復帰は呼び出し側の reset() 責務。ここでは終端状態を維持する。
            return prev

        target = _find_by_serial(mine, prev.target_serial)

        if prev.mode in (BUILD, READY):
            if target is None or getattr(target, "id", None) != prev.target_card_id:
                return _aborted(prev, "target_lost_or_replaced", turn)
            if prev.mode == BUILD and prev.build_deadline_turn is not None and turn is not None:
                if turn > prev.build_deadline_turn:
                    return _aborted(prev, "build_deadline_exceeded", turn)

        if prev.mode == BUILD:
            nxt = OgerponOptionState(**vars(prev))
            nxt.last_decision_turn = turn
            if ogerpon_planner.can_attack_now(target):
                nxt.mode = READY
                nxt.phase = READY
            return nxt

        if prev.mode == READY:
            nxt = OgerponOptionState(**vars(prev))
            nxt.last_decision_turn = turn
            active = next((s for s in (mine.active or []) if s is not None), None)
            if active is not None and getattr(active, "serial", None) == prev.target_serial:
                nxt.mode = ACTIVE
                nxt.phase = ACTIVE
            return nxt

        if prev.mode == ACTIVE:
            nxt = OgerponOptionState(**vars(prev))
            nxt.last_decision_turn = turn
            nxt.bulu_attack_count += _count_bulu_attacks(obs, me, prev.target_serial)
            if target is None:
                # 場のどこにも見つからない = 気絶して墓地へ(想定どおりの完遂)。
                nxt.mode = COMPLETE
                nxt.phase = COMPLETE
                return nxt
            still_active = any(getattr(s, "serial", None) == prev.target_serial
                               for s in (mine.active or []) if s is not None)
            if not still_active:
                # ACTIVE中の自発的な交代は原則拒否する設計(design.md §11.3)なので、
                # ベンチへ移動していたら想定外として安全側でABORTする。
                return _aborted(prev, "retreated_unexpectedly", turn)
            return nxt

        return prev
    except Exception:  # noqa: BLE001
        return prev


# ---------------------------------------------------------------------------
# モジュールグローバルの singleton(policy_registry と同じ規約)。
# 本番配線(Phase2以降)はこちらを使う想定。Phase1では ml_policy_agent からは呼ばれない。
# ---------------------------------------------------------------------------

_STATE = OgerponOptionState()


def reset() -> None:
    """対戦終了時に呼ぶ。試合間に状態を漏らさない。"""
    _STATE.reset_in_place()


def get_state() -> OgerponOptionState:
    return _STATE


def advance(obs: Observation, me: int, config: dict | None,
           *, force_start: bool = False) -> OgerponOptionState:
    """モジュールグローバルな状態を1手分進めて返す(``advance_state`` の副作用版)。"""
    global _STATE
    _STATE = advance_state(_STATE, obs, me, config, force_start=force_start)
    return _STATE
