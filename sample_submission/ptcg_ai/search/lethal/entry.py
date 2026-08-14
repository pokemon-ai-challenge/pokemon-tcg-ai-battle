"""チーム共通の探索モジュール契約に合わせた adapter(Step 1-3)。

``selector._SEARCH_MODULES`` に登録すれば使える形にしてあるが、
**Step 1-3 では登録していない**(既存経路は無改変)。

    def search(state, legal_actions, context) -> list[int] | None

``context`` のキー(``lethal_simple`` と互換 + 1つ追加):

- ``observation``          : 必須。エージェントが受け取った ``Observation``
- ``config``               : ``lethal_search`` セクション
- ``hidden_state`` / ``hidden_state_factory`` : どちらか必須
- ``full_deck``            : **追加**。自分の 60 枚のカードIDリスト。
  無い場合でも動くが、山札 multiset を作れないので Phase 2 の outcome 列挙は
  行わない(= ``UNKNOWN``)。

戻り値は「勝ちを**証明できた**ときの最初の1手」だけ。それ以外は必ず ``None`` を返し、
呼び出し側は通常方策へ進む。例外は外へ出さない(原設計 §2.5)。

既存 ``lethal_simple`` との関係:
別モジュール・別 config 名(``module: "lethal_phase1"``)で並存させる。
フラグや関数名を再利用して挙動を混ぜない(Step 1-3 指示 12)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from cg.api import Observation, State

from ptcg_ai.search.lethal import action as action_module
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import coerce as coerce_deck
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from ptcg_ai.search.lethal.types import Proof, StopReason

DEFAULTS: dict = {
    "enabled": True,
    "module": "lethal_phase1",
    "max_remaining_prizes": 2,
    # Step 1-3 は現行 lethal_simple と同じ予算から始める(引き上げは計測後)。
    "phase12_ms": 100,
    "max_nodes": 10_000,
    "max_depth": 20,
    "max_chance_depth": 1,
    "phase1_enabled": True,
    "phase2_enabled": True,
    # Phase 3 はこの adapter からは呼ばない(既定 OFF どころか未接続)。
    "phase3_enabled": False,
}


@dataclass
class Diagnostics:
    """1回の探索の結果概要(順列・非公開実体は入れない)。"""

    started: bool = False
    phase1_proof: str | None = None
    phase2_proof: str | None = None
    stop_reasons: tuple[str, ...] = ()
    nodes: int = 0
    elapsed_ms: float = 0.0
    executed: bool = False
    fallback_reason: str | None = None
    extra: dict = field(default_factory=dict)


_LAST_DIAGNOSTICS = Diagnostics()


def last_diagnostics() -> Diagnostics:
    """直近の呼び出しの診断(テスト・計測用)。"""
    return _LAST_DIAGNOSTICS


def search(state: State, legal_actions, context: dict) -> list[int] | None:
    """確定リーサルが**証明できた**ときだけ最初の1手を返す。"""
    global _LAST_DIAGNOSTICS
    diagnostics = Diagnostics()
    _LAST_DIAGNOSTICS = diagnostics
    try:
        config = {**DEFAULTS, **(context.get("config") or {})}
        # **明示的に True のときだけ動く**。壊れた config(型違い・欠落)で
        # 勝手に有効化されないよう、truthy 判定にしない(Step 1-13)。
        if config.get("enabled") is not True:
            diagnostics.fallback_reason = "disabled"
            return None
        if config.get("module") not in (None, "lethal_phase1"):
            # 別モジュール向けの config を渡された場合も動かない。
            diagnostics.fallback_reason = "module_mismatch"
            return None

        obs: Observation | None = context.get("observation")
        if obs is None or obs.select is None or obs.current is None:
            diagnostics.fallback_reason = "no_observation"
            return None
        if not _precheck(obs, config):
            diagnostics.fallback_reason = "precheck"
            return None

        hidden_state = _hidden_state(context)
        if hidden_state is None:
            diagnostics.fallback_reason = "no_hidden_state"
            return None

        diagnostics.started = True
        # 情報境界: デッキ情報は multiset へ落としてから探索へ渡す(順序を捨てる)
        deck_composition = coerce_deck(
            context.get("deck_composition") or context.get("full_deck")
        )
        result = _run(obs, hidden_state, deck_composition, config, diagnostics)
        if result is None:
            return None

        engine_action, current_obs = result
        selection = action_module.validate(
            engine_action,
            current_obs.select,
            current_obs.current,
            current_obs.current.yourIndex,
            same_observation=True,
        )
        if selection is None:
            diagnostics.fallback_reason = "action_validation_failed"
            return None
        diagnostics.executed = True
        return selection
    except Exception as exc:  # noqa: BLE001 - 例外を対戦へ伝播させない
        diagnostics.fallback_reason = f"exception:{type(exc).__name__}"
        return None


def _run(obs, hidden_state, deck_composition, config, diagnostics):
    """Phase 1 -> Phase 2 の順に確定探索する。戻り値は (EngineAction, obs)。"""
    hidden = HiddenState.from_stub(hidden_state)
    me = obs.current.yourIndex
    total_ms = float(config.get("phase12_ms", 100))
    with SearchSession(obs, hidden) as session:
        backend = CgBackend(session, obs, hidden, deck_composition=deck_composition)
        reasons: set[str] = set()
        nodes = 0
        elapsed = 0.0

        def finish(result=None):
            diagnostics.stop_reasons = tuple(sorted(reasons))
            diagnostics.nodes = nodes
            diagnostics.elapsed_ms = elapsed
            diagnostics.extra = dict(backend.refusals)
            if result is None:
                diagnostics.fallback_reason = "not_proven"
                return None
            return _to_engine_action(result.first_action, obs, me), obs

        if config.get("phase1_enabled", True):
            # Phase 1 と Phase 2 で予算を**分け合う**(合計が phase12_ms)。
            result = phase1.search(backend, _budget(config, total_ms))
            nodes += result.nodes
            elapsed += result.elapsed_ms
            reasons.update(r.name for r in result.stop_reasons)
            diagnostics.phase1_proof = result.proof.name
            if result.proof is Proof.PROVEN_WIN:
                return finish(result)

        if config.get("phase2_enabled", True):
            remaining = max(0.0, total_ms - elapsed)
            result = phase2.search(backend, _budget(config, remaining))
            nodes += result.nodes
            elapsed += result.elapsed_ms
            reasons.update(r.name for r in result.stop_reasons)
            diagnostics.phase2_proof = result.proof.name
            if result.proof is Proof.PROVEN_WIN:
                return finish(result)

        return finish()


def _to_engine_action(first_action, obs: Observation, me: int):
    """探索が返した抽象アクション(選択インデックス列)を ``EngineAction`` にする。"""
    engine_action = action_module.describe(list(first_action), obs.select, obs.current, me)
    if engine_action is None:
        raise ValueError("first action is not a legal selection")
    return engine_action


def _budget(config: dict, time_limit_ms: float) -> Budget:
    return Budget(
        time_limit_ms=time_limit_ms,
        max_nodes=int(config.get("max_nodes", 10_000)),
        max_depth=int(config.get("max_depth", 20)),
        max_chance_depth=int(config.get("max_chance_depth", 1)),
    )


def _precheck(obs: Observation, config: dict) -> bool:
    """起動条件(設計 §3.4)。Phase の定義とは分離する。

    サイド枚数だけで切ると、**サイド以外の勝ち筋**を取りこぼす。
    実測(Step 1-11): 保存盤面の Phase 1 `PROVEN_WIN` 6 件のうち 2 件は
    「相手のベンチが空でバトル場を倒すと場にポケモンがいなくなる」勝ちで、
    自分のサイドは 4〜5 枚残っていた。サイド条件だけの precheck は
    この 2 件(=33%)を起動前に捨てていた。
    """
    state = obs.current
    me = state.yourIndex
    if state.result != -1:
        return False
    if not _is_my_turn(state, me):
        return False
    if len(state.players[me].prize) <= int(config.get("max_remaining_prizes", 2)):
        return True
    opponent = state.players[1 - me]
    if not opponent.bench and opponent.active:
        # 相手のベンチが空: バトル場を 1 体きぜつさせるだけで勝てる可能性がある
        return True
    return False


def _is_my_turn(state: State, me: int) -> bool:
    if state.turn < 1 or state.firstPlayer < 0:
        return False
    starter_turn = state.turn % 2 == 1
    return starter_turn == (state.firstPlayer == me)


def _hidden_state(context: dict) -> dict | None:
    factory: Callable[[], dict | None] | None = context.get("hidden_state_factory")
    if factory is not None:
        try:
            return factory()
        except Exception:  # noqa: BLE001
            return None
    return context.get("hidden_state")


# 診断で使う定数(テストから参照する)
UNKNOWN_REASONS = {
    StopReason.UNSUPPORTED_EFFECT,
    StopReason.OUTCOMES_NOT_ENUMERABLE,
    StopReason.SHUFFLE_ENCOUNTERED,
    StopReason.DECK_REVEALED_AT_ROOT,
    StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL,
    StopReason.TIME_LIMIT,
    StopReason.NODE_LIMIT,
    StopReason.DEPTH_LIMIT,
    StopReason.CHANCE_DEPTH_LIMIT,
    StopReason.INCOMPLETE_ACTION_SET,
    StopReason.STATE_MISMATCH,
    StopReason.ENGINE_ERROR,
}
