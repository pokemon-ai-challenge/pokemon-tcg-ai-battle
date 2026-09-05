"""「テラスタル退避」ゲート(``ml_policy_agent._try_terastal_rotation``、r12 config-gated)専用の
KO整合性判定: にげる(RETREAT)へ差し替えても「今ターンのKO成否が変わらない」かを判定する。

既存の `ko_search` / `ability_draw_eval` / `boss_target_eval` と同じ「search_step →
ko_search.can_ko_from_node」の流儀を流用する(打点式を手書きせず、エンジンにやらせる)。
他の担当領域(`ml_policy_agent.py` は他エージェントが並行編集中)に手を入れず、専用の新規
モジュールとして独立させている(`boss_target_eval.py`/`ability_draw_eval.py` が
`ko_search.py` 本体を書き換えずに専用ロジックを別ファイルへ足したのと同じ流儀)。

判定は2段(呼び出し元 `_try_terastal_rotation` が「現アクティブで取れるKOが元々無い」場合を
条件3として自動的に満たす、という仕様に対応):

  1. 現在の select にある ATTACK 選択肢のどれかを**今すぐ**実行すればこのターンKOできるか
     (`current_ko`)。ATTACK選択肢が無ければ False(判定不能ではなく確定事実)。
  2. 1で KOが取れる場合のみ、``retreat_index``(RETREAT)→ にげるコストのエネルギー破棄
     (機械的に先頭から ``minCount`` 枚を選ぶ。どの1枚を破棄するかは退避先個体の状態に
     影響しないため任意でよい)→ ``target_bench_index``(退避先)の交代、を実際に
     ``search_step`` で辿った先の select にある ATTACK 選択肢のどれかを実行すれば同じく
     KOできるか(``after_retreat_ko``)。

1でKOが元々無ければ「守るべきKOが無い」ので2は評価せず ``parity=True`` を返す
(仕様: 「または元々KOが無い」)。想定外のスキーマ(にげるコストが0でENERGY選択が来ない等)は
判定不能として ``None`` を返し、呼び出し側は安全側で不介入にする。
"""
from __future__ import annotations

import time

from cg import api as cg_api
from cg.api import AreaType, OptionType, SelectContext, SelectType

from ptcg_ai.search import ko_search

DEFAULTS: dict = dict(ko_search.DEFAULTS)


def _attack_options_achieve_ko(
    node, me: int, root_my_prize: int, cfg: dict, deadline: float, budget: dict,
) -> tuple[bool, bool]:
    """``node`` の select にある ATTACK 選択肢のどれかを実行すればこのターンKOできるか。

    ``ability_draw_eval._any_attack...`` 相当だが、ここでは ATTACK 選択肢が1つも無い場合を
    「判定不能」ではなく**確定的な False**として扱う(呼び出し元の2箇所とも、ATTACK 選択肢の
    有無そのものが「攻撃できるか」という意味のある事実であり、探索予算切れとは区別したいため)。

    Returns:
        ``(achieves_ko, aborted)``。``aborted=True`` は予算切れ等で判定を打ち切った
        (=候補を網羅できていない)ことを示す。
    """
    select = node.observation.select
    if select is None or not select.option:
        return False, False
    attack_indices = [i for i, opt in enumerate(select.option) if opt.type == OptionType.ATTACK]
    if not attack_indices:
        return False, False  # 攻撃できない=確定的にKO無し(判定不能ではない)
    any_aborted = False
    for i in attack_indices:
        if time.perf_counter() > deadline:
            return False, True
        try:
            child = cg_api.search_step(node.searchId, [i])
        except (ValueError, RuntimeError):
            continue
        try:
            can_ko, aborted = ko_search.can_ko_from_node(
                child, me, root_my_prize, cfg, deadline, budget)
            any_aborted = any_aborted or aborted
            if can_ko:
                return True, False
        finally:
            try:
                cg_api.search_release(child.searchId)
            except Exception:  # noqa: BLE001
                pass
    return False, any_aborted


def _is_switch_select(select) -> bool:
    """RETREAT直後(にげるコスト破棄後)の「交代先を選ぶ」select か(実測スキーマ:
    CARD / SWITCH。``boss_target_eval._is_target_select`` と同じ形)。"""
    try:
        return (
            select is not None
            and select.type == SelectType.CARD
            and select.context == SelectContext.SWITCH
            and bool(select.option)
        )
    except Exception:  # noqa: BLE001
        return False


def evaluate(
    obs,
    hidden_state_factory,
    retreat_index: int,
    target_bench_index: int,
    config: dict | None = None,
    deadline: float | None = None,
) -> dict | None:
    """``retreat_index`` を選び ``target_bench_index``(自分のベンチの配列位置)へ交代しても、
    今ターンのKO成否が変わらないかを判定する。

    Args:
        obs: RETREATを選ぼうとしている MAIN 決定の Observation。
        hidden_state_factory: `search_begin` 用の0引数callable(lethal/ko_search と同じ)。
        retreat_index: ``obs.select.option`` 上の RETREAT のインデックス。
        target_bench_index: 交代させたいベンチの配列位置(``state.players[me].bench`` の
            インデックス。``switch_turn.py`` と同じ ``option.index`` との対応)。
        config: `DEFAULTS` を上書きする dict。
        deadline: ``time.perf_counter()`` 基準の締切(無ければ ``time_limit_ms`` から作る)。

    Returns:
        判定不能(隠れ状態が組めない/探索失敗/予算切れ/想定外スキーマ)なら ``None``
        (呼び出し側は介入しない)。成功時は
        ``{"current_ko": bool, "after_retreat_ko": bool | None, "parity": bool}``。
        ``current_ko`` が False なら ``after_retreat_ko`` は評価せず ``parity=True``
        (元々守るべきKOが無い)。
    """
    root = None
    try:
        if obs is None or obs.current is None or obs.select is None:
            return None
        state = obs.current
        me = state.yourIndex
        if state.result != -1 or hidden_state_factory is None:
            return None
        if not (0 <= retreat_index < len(obs.select.option)):
            return None
        if obs.select.option[retreat_index].type != OptionType.RETREAT:
            return None

        cfg = {**DEFAULTS, **(config or {})}
        hidden_state = hidden_state_factory()
        if hidden_state is None:
            return None
        if deadline is None:
            deadline = time.perf_counter() + float(cfg["time_limit_ms"]) / 1000.0
        root_my_prize = len(state.players[me].prize or [])

        root = cg_api.search_begin(
            obs,
            hidden_state["your_deck"],
            hidden_state["your_prize"],
            hidden_state["opponent_deck"],
            hidden_state["opponent_prize"],
            hidden_state["opponent_hand"],
            hidden_state["opponent_active"],
        )
        budget = {"nodes": 0}

        current_ko, aborted = _attack_options_achieve_ko(root, me, root_my_prize, cfg, deadline, budget)
        if aborted:
            return None
        if not current_ko:
            return {"current_ko": False, "after_retreat_ko": None, "parity": True}

        # にげる(エネ破棄→交代)を実際に search_step で辿る。
        nodes: list = []
        try:
            after_retreat = cg_api.search_step(root.searchId, [retreat_index])
        except (ValueError, RuntimeError):
            return None
        nodes.append(after_retreat)
        try:
            discard_select = after_retreat.observation.select
            if discard_select is None or discard_select.type != SelectType.ENERGY:
                return None  # 想定外のスキーマ(にげるコスト0等)=判定不能で安全側
            n_discard = max(int(discard_select.minCount), 0)
            if n_discard > len(discard_select.option):
                return None
            try:
                after_discard = cg_api.search_step(
                    after_retreat.searchId, list(range(n_discard)))
            except (ValueError, RuntimeError):
                return None
            nodes.append(after_discard)

            switch_select = after_discard.observation.select
            if not _is_switch_select(switch_select):
                return None
            switch_idx = next(
                (i for i, opt in enumerate(switch_select.option)
                 if opt.area == AreaType.BENCH and opt.index == target_bench_index
                 and opt.playerIndex == me),
                None,
            )
            if switch_idx is None:
                return None
            try:
                after_switch = cg_api.search_step(after_discard.searchId, [switch_idx])
            except (ValueError, RuntimeError):
                return None
            nodes.append(after_switch)

            if after_switch.observation.select is None \
                    or after_switch.observation.select.type != SelectType.MAIN:
                return None  # 想定外(効果解決待ち等)=判定不能

            after_ko, aborted2 = _attack_options_achieve_ko(
                after_switch, me, root_my_prize, cfg, deadline, budget)
            if aborted2:
                return None
            return {"current_ko": True, "after_retreat_ko": after_ko, "parity": bool(after_ko)}
        finally:
            for node in reversed(nodes):
                try:
                    cg_api.search_release(node.searchId)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return None
    finally:
        if root is not None:
            try:
                cg_api.search_release(root.searchId)
            except Exception:  # noqa: BLE001
                pass
