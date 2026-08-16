"""「みどりのまい」型の自己エネ加速+ドロー特性を止めてよいかの判定に使う専用KO評価。

`ml_policy_agent._try_ability_draw_brake`(``ability_draw_brake``、config-gated)専用。
背景・実例は同関数のdocstring参照。ここでは「現在の select に既に出ている ATTACK
選択肢のどれかを**今すぐ**実行すれば、これ以上エネルギーを足さなくてもこのターンKOできるか」
だけを判定する。

`ko_search.can_ko_this_turn` はこの decision で ABILITY(エネ加速+ドロー)を使う手も
候補に含めて全探索するため、「足す前から既にKOできていた(=足す必要が無かった)」と
「足したからKOできた」を区別できない。ここでは根の select にある ATTACK 選択肢だけを
`search_step` で1つずつ実行し、そこから `ko_search.can_ko_from_node` と同じDFS
(自分のサイドが減る=KOでプライズを取った)でKOに到達できるかを見る
(`boss_target_eval.evaluate_targets` の「search_step して can_ko_from_node」と同じ流儀)。
"""
from __future__ import annotations

import time

from cg import api as cg_api
from cg.api import OptionType

from ptcg_ai.search import ko_search

DEFAULTS: dict = dict(ko_search.DEFAULTS)


def already_ko_without_more_energy(
    obs,
    hidden_state_factory,
    config: dict | None = None,
    deadline: float | None = None,
) -> tuple[bool, bool]:
    """``(already_ko, aborted)`` を返す。

    ``already_ko``: 現在の ``obs.select.option`` にある ATTACK 選択肢のどれかを今すぐ
    実行すれば、このターンKO(サイド取得)できるなら True。
    ``aborted``: 判定不能(隠れ状態が組めない/ATTACK選択肢が無い/予算切れ/例外)なら True。
    呼び出し側は ``aborted`` が True のとき ``already_ko`` を veto の根拠に使ってはならない
    (``ko_search.can_ko_this_turn`` の False が「KOできない」と「判定不能」を区別しない
    のと同じ落とし穴を避けるため、この関数は最初から2値で返す)。
    """
    root = None
    try:
        if obs is None or obs.current is None or obs.select is None:
            return False, True
        state = obs.current
        me = state.yourIndex
        if state.result != -1 or hidden_state_factory is None:
            return False, True
        attack_indices = [
            i for i, opt in enumerate(obs.select.option) if opt.type == OptionType.ATTACK
        ]
        if not attack_indices:
            return False, True  # 呼び出し側の前提(ATTACK選択肢あり)が崩れている=判定不能

        cfg = {**DEFAULTS, **(config or {})}
        hidden_state = hidden_state_factory()
        if hidden_state is None:
            return False, True
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
        any_aborted = False
        for i in attack_indices:
            if time.perf_counter() > deadline:
                any_aborted = True
                break
            try:
                node = cg_api.search_step(root.searchId, [i])
            except (ValueError, RuntimeError):
                continue
            try:
                can_ko, aborted = ko_search.can_ko_from_node(
                    node, me, root_my_prize, cfg, deadline, budget)
                any_aborted = any_aborted or aborted
                if can_ko:
                    return True, False
            finally:
                try:
                    cg_api.search_release(node.searchId)
                except Exception:  # noqa: BLE001
                    pass
        return False, any_aborted
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return False, True
    finally:
        if root is not None:
            try:
                cg_api.search_release(root.searchId)
            except Exception:  # noqa: BLE001
                pass
