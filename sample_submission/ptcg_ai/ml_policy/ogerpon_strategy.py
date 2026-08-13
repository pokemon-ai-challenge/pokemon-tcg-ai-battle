"""Strategy Window Builder(design.md §6、Phase2 item1)。

criticを毎選択で呼ぶのではなく、`EX_TEMPO` と `SINGLE_PRIZE_ROTATION` で選ぶ行動が
実際に分岐しうる局面(戦略ウィンドウ)だけを検出し、両Optionの `first_action` 候補
(`StrategyCandidate`)を構築する。Q-critic(Phase3)はまだ無いので、ここでは
「発火判定」と「両Optionの合法な最初の一手を用意する」ところまでを扱う。
どちらを選ぶかの判断はまだ実装しない。まだ ``ml_policy_agent`` には配線しない
(既定で本番の意思決定に一切影響しない)。

## 発火条件(design.md §6.1)

1. ``main_attach``: MAIN/ATTACHに、ブルルへの生産的な手貼り候補がある
   (かつ他に選べる選択肢が1つ以上ある)。
2. ``promote``: CARD/TO_ACTIVEまたはCARD/SWITCHに、攻撃可能なブルルとexの両方がある。
3. ``retreat``: MAIN/RETREATで、攻撃可能なブルルへの中継と、攻撃可能な現在のexの
   続投が競合する。
4. ``continue``: 既にSINGLE_PRIZE_ROTATION中(BUILD/READY/ACTIVE)。

優先順位は continue > retreat > promote > main_attach(既に中継中なら、その
継続判断を他の分類より優先する)。

## SINGLE_PRIZE_ROTATION側の合法手が構築できない場合

``main_attach``/``promote``/``retreat`` は、それぞれ対応する具体的な代替行動
(手貼り先・昇格先・retreat)を1つ選べる場合だけ候補を作る。``continue`` トリガーは
「今まさに中継を続けるかどうかを判断すべき局面」であることの検出だけを行い、
具体的に分岐する代替行動が無い(例: ACTIVE中の通常ATTACK)場合は
``build_candidates`` が ``None`` を返す(比較不能。安全側)。

## 安全ゲート(design.md §11.1)の位置づけ

ここで返す ``safety_flags`` は静的計算のみで、``search_step`` による1手仮実行
(design.md §11.1 が推奨する厳密な前後差確認)は行わない。Phase2はこの近似値を
データ収集・Strategy Window発火の参考に使うだけで、意思決定そのものにはまだ
使わない。厳密な安全ゲートの配線はPhase4のスコープ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cg.api import Observation, OptionType, SelectContext, SelectType

from ptcg_ai.ml_policy import ogerpon_option_state as option_state_mod
from ptcg_ai.ml_policy import ogerpon_planner as P

EX_TEMPO = "EX_TEMPO"
SINGLE_PRIZE_ROTATION = "SINGLE_PRIZE_ROTATION"

TRIGGER_MAIN_ATTACH = "main_attach"
TRIGGER_PROMOTE = "promote"
TRIGGER_RETREAT = "retreat"
TRIGGER_CONTINUE = "continue"

_MID_ROTATION_MODES = (option_state_mod.BUILD, option_state_mod.READY, option_state_mod.ACTIVE)


@dataclass
class StrategyCandidate:
    option_name: str
    first_action: list[int]
    trigger_kind: str
    target_serial: int | None = None
    target_card_id: int | None = None
    turns_until_ready: int = 0
    required_ko_gain: float = 0.0
    safety_flags: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 発火判定
# ---------------------------------------------------------------------------

def _is_single_select(select) -> bool:
    return select is not None and bool(select.option) and select.maxCount == 1


def _main_attach_trigger(state, me: int, select) -> int | None:
    """条件1。ブルルへの生産的な手貼り候補のoption indexを返す(無ければNone)。"""
    if int(select.type) != int(SelectType.MAIN):
        return None
    for i, opt in enumerate(select.option):
        if int(getattr(opt, "type", -1)) != int(OptionType.ATTACH):
            continue
        target = P._resolve_attach_target(state, me, opt)  # noqa: SLF001
        if target is not None and P.is_bulu(target) and P.energy_shortfall(target) > 0:
            # 他に選べる選択肢が無ければ「競合」ではないので対象外。
            if len(select.option) > 1:
                return i
    return None


def _promote_trigger(obs: Observation, me: int, select) -> int | None:
    """条件2。攻撃可能なブルルとexの両方がある場合、ブルル側のoption indexを返す。"""
    if int(select.type) != int(SelectType.CARD):
        return None
    ctx = int(select.context)
    if ctx not in (int(SelectContext.TO_ACTIVE), int(SelectContext.SWITCH)):
        return None
    bulu_idx = None
    has_ready_ex = False
    for i, opt in enumerate(select.option):
        target = P._resolve_option_pokemon(obs, opt, me)  # noqa: SLF001
        if target is None or not P.can_attack_now(target):
            continue
        if P.is_bulu(target):
            bulu_idx = i
        elif P.is_ex(target):
            has_ready_ex = True
    return bulu_idx if (bulu_idx is not None and has_ready_ex) else None


def _retreat_trigger(obs: Observation, me: int, state, select) -> int | None:
    """条件3。中継先の攻撃可能なブルルへ交代するRETREAT選択肢のindexを返す。"""
    if int(select.type) != int(SelectType.MAIN):
        return None
    mine = state.players[me]
    active = next((s for s in (mine.active or []) if s is not None), None)
    if active is None or not P.is_ex(active) or not P.can_attack_now(active):
        return None
    for i, opt in enumerate(select.option):
        if int(getattr(opt, "type", -1)) != int(OptionType.RETREAT):
            continue
        target = P._retreat_target_of(opt, obs, me)  # noqa: SLF001
        if target is not None and P.is_bulu(target) and P.can_attack_now(target):
            return i
    return None


def detect_trigger(obs: Observation, me: int, deck_ids, option_mode: str) -> str | None:
    """戦略ウィンドウの発火種別を返す。非該当・例外時はNone。

    オーガポンデッキ以外では常にNone(既存Plannerと同じゲート)。複数選択
    (maxCount>1)は対象外。
    """
    try:
        if not P.deck_is_ogerpon(deck_ids):
            return None
        select = obs.select
        state = obs.current
        if state is None or not _is_single_select(select):
            return None
        if option_mode in _MID_ROTATION_MODES:
            return TRIGGER_CONTINUE
        if _retreat_trigger(obs, me, state, select) is not None:
            return TRIGGER_RETREAT
        if _promote_trigger(obs, me, select) is not None:
            return TRIGGER_PROMOTE
        if _main_attach_trigger(state, me, select) is not None:
            return TRIGGER_MAIN_ATTACH
        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 候補構築
# ---------------------------------------------------------------------------

def _static_safety_flags(state, me: int, target, config: dict | None) -> dict[str, bool]:
    """design.md §11.1の安全ゲート名に対応する静的近似値(search_step仮実行はしない)。"""
    mine = state.players[me]
    opp = state.players[1 - me]
    active = next((s for s in (mine.active or []) if s is not None), None)
    active_ready = P.can_attack_now(active) if active is not None else False
    is_active_target = (active is not None
                        and getattr(target, "serial", None) == getattr(active, "serial", None))
    cfg = P.main_config(config)
    opp_active = next((s for s in (opp.active or []) if s is not None), None)
    risk, _ = P.ko_risk(target, opp_active, cfg)
    return {
        # activeが今まだ攻撃不能で、対象がactiveでないなら、今ターンの攻撃を
        # 諦める側の選択になる(design.md の enables_only_active_attack と対になる注意信号)。
        "single_prize_risks_losing_active_attack": (not active_ready) and not is_active_target,
        "single_prize_target_faces_certain_ko": risk == P.CERTAIN_KO,
    }


def _forced_attach_resolver_config(config: dict | None) -> dict:
    """``select_strategic_attach_candidate`` は旧Plannerのconfigゲート
    (``ogerpon_planner.enabled``/``attach_enabled``)に依存する。Strategy Window
    Builderはこの関数を「低位candidate resolver」として再利用するだけで、
    bonusは一切足さない(design.md §21で明示されている再利用方針)。呼び出し元の
    実運用config(旧bonus系がOFFかもしれない)に関わらず、resolver呼び出しの
    ためだけに内部で強制的に有効化する。
    """
    base = dict((config or {}).get("ogerpon_planner") or {})
    base["enabled"] = True
    base["attach_enabled"] = True
    return {**(config or {}), "ogerpon_planner": base}


def _resolve_single_prize_first_action(obs, me, config, deck_ids, trigger_kind, select, state):
    if trigger_kind == TRIGGER_MAIN_ATTACH:
        idx = P.select_strategic_attach_candidate(
            obs, me, _forced_attach_resolver_config(config), deck_ids)
        if idx is None:
            return None, None
        target = P._resolve_attach_target(state, me, select.option[idx])  # noqa: SLF001
        return idx, target
    if trigger_kind == TRIGGER_PROMOTE:
        idx = _promote_trigger(obs, me, select)
        if idx is None:
            return None, None
        target = P._resolve_option_pokemon(obs, select.option[idx], me)  # noqa: SLF001
        return idx, target
    if trigger_kind == TRIGGER_RETREAT:
        idx = _retreat_trigger(obs, me, state, select)
        if idx is None:
            return None, None
        target = P._retreat_target_of(select.option[idx], obs, me)  # noqa: SLF001
        return idx, target
    # TRIGGER_CONTINUE: この局面自体には具体的に分岐する代替行動が無い
    # (例: ACTIVE中の通常ATTACK)。比較不能として扱う。
    return None, None


def build_candidates(
    obs: Observation, me: int, config: dict | None, deck_ids,
    trigger_kind: str, baseline_action: list[int],
) -> tuple[StrategyCandidate, StrategyCandidate] | None:
    """発火した戦略ウィンドウについて、EX_TEMPO/SINGLE_PRIZE_ROTATION双方の候補を
    構築する。SINGLE_PRIZE_ROTATION側の合法な最初の一手が作れなければNone。

    ``baseline_action`` は呼び出し側が既存Policy/PIMCで既に決めた行動
    (EX_TEMPO側の first_action としてそのまま使う)。
    """
    try:
        select = obs.select
        state = obs.current
        if select is None or state is None or trigger_kind is None:
            return None
        idx, target = _resolve_single_prize_first_action(
            obs, me, config, deck_ids, trigger_kind, select, state)
        if idx is None or target is None or not (0 <= idx < len(select.option)):
            return None

        mine = state.players[me]
        opp = state.players[1 - me]
        turns_until_ready = max(0, P.energy_shortfall(target))
        required_ko_gain = P.required_ko_delta(mine, target, len(opp.prize or []))
        safety_flags = _static_safety_flags(state, me, target, config)
        safety_flags["legal"] = True

        ex_candidate = StrategyCandidate(
            option_name=EX_TEMPO, first_action=list(baseline_action),
            trigger_kind=trigger_kind,
        )
        single_candidate = StrategyCandidate(
            option_name=SINGLE_PRIZE_ROTATION, first_action=[idx],
            trigger_kind=trigger_kind,
            target_serial=getattr(target, "serial", None),
            target_card_id=getattr(target, "id", None),
            turns_until_ready=turns_until_ready,
            required_ko_gain=required_ko_gain,
            safety_flags=safety_flags,
        )
        return ex_candidate, single_candidate
    except Exception:  # noqa: BLE001
        return None
