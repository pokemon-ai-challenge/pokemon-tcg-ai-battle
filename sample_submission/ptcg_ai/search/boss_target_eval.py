"""ボスの指令(1182)の**対象別**評価: 「その相手を引きずり出したら今ターンKOできるか」。

`ko_search.can_ko_this_turn` は「このターン誰かをKOできるか」しか答えないので、ボスの本質
(=どの相手を前に出すか)の良否は判定できない。実ラダー 93503044 T13 では、8エネのオーガポン
(まんようしぐれ)で届かない メガガルーラex(HP330)をわざわざ引きずり出し、60HP残しで逃げられ、
以降12ターン3サイド分をベンチに放置した。倒せない相手を出すボスは**盤面を悪化させる**。

ここでは打点式を手書きせず、エンジンにやらせる(汎用実装):

    search_begin(根) → search_step([ボスのPLAY]) → 対象選択(SelectType.CARD/SelectContext.SWITCH)
    → 各対象を search_step で1つずつ踏む → そこから `ko_search` と同じDFS
      (自分のサイドが減る = KOでプライズを取った)

弱点・特性・ロック・エネ加速→攻撃の手順もすべてエンジンの真実で解決されるため、特定カード
(オーガポン)専用の打点式には依存しない。

**KO判定の意味**: 「サイドが減った」= 何かをKOしたであって、厳密には「引きずり出した当人を
KOした」ではない(ベンチをKOする効果があれば理屈上ズレる)。ただしボス直後にアクティブへ出た
のはその対象で、攻撃で取れるプライズはアクティブのKO由来が実質すべてなので、この近似で扱う。

判定不能(隠れ状態が組めない/対象選択が来ない/予算切れ/例外)は呼び出し側が介入を見送れる
ように、返り値の ``aborted``/``searched`` で明示する(`ko_search` の False が「KOできない」と
「判定不能」を区別しないのと同じ落とし穴を避けるため。`_try_briar_gate` の実測メモ参照)。
"""
from __future__ import annotations

import time

from cg import api as cg_api
from cg.api import SelectContext, SelectType

from ptcg_ai.search import ko_search

DEFAULTS: dict = {
    # ボス後の残り手数(対象を出した後にエネ加速→攻撃まで届く深さ)。
    "max_depth": 4,
    # ボスPLAY決定1回あたりの総予算(対象すべての探索を合わせた値)。
    "time_limit_ms": 400,
    "max_nodes": 6000,
    "max_combinations_per_select": 64,
}


def evaluate_targets(
    obs,
    hidden_state_factory,
    play_index: int,
    config: dict | None = None,
    deadline: float | None = None,
) -> dict | None:
    """ボスのPLAY(``play_index``)を仮実行し、対象ごとのKO可否を返す。

    Args:
        obs: ボスをPLAYしようとしている MAIN 決定の Observation。
        hidden_state_factory: `search_begin` 用の0引数callable(lethal/ko_search と同じ)。
        play_index: ``obs.select.option`` 上のボスPLAYのインデックス。
        config: `DEFAULTS` を上書きする dict。
        deadline: ``time.perf_counter()`` 基準の締切(無ければ ``time_limit_ms`` から作る)。

    Returns:
        ``{"targets": [...], "any_aborted": bool}`` または判定不能なら ``None``。
        ``targets`` の各要素は
        ``{"option_index": int, "area": int|None, "index": int|None,
           "player_index": int|None, "can_ko": bool, "aborted": bool}``。
        ``option_index`` は**仮実行した対象選択**上の位置で、実際の対象選択 decision でも
        同じ並びで来る想定だが、呼び出し側は (area, index, player_index) で照合すること
        (順序に依存しない照合キー)。
    """
    cfg = {**DEFAULTS, **(config or {})}
    root = boss_child = None
    try:
        if obs is None or obs.current is None or obs.select is None:
            return None
        state = obs.current
        me = state.yourIndex
        if state.result != -1 or hidden_state_factory is None:
            return None
        if not (0 <= play_index < len(obs.select.option)):
            return None
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
        try:
            boss_child = cg_api.search_step(root.searchId, [play_index])
        except (ValueError, RuntimeError):
            return None
        target_select = boss_child.observation.select
        if not _is_target_select(target_select):
            return None  # 対象選択が来ない=想定外(効果が解決済み等)→判定不能

        budget = {"nodes": 0}
        targets: list[dict] = []
        any_aborted = False
        for i, opt in enumerate(target_select.option):
            entry = {
                "option_index": i,
                "area": getattr(opt, "area", None),
                "index": getattr(opt, "index", None),
                "player_index": getattr(opt, "playerIndex", None),
                "can_ko": False,
                "aborted": False,
            }
            if time.perf_counter() > deadline:
                entry["aborted"] = any_aborted = True
                targets.append(entry)
                continue
            try:
                node = cg_api.search_step(boss_child.searchId, [i])
            except (ValueError, RuntimeError):
                # この対象は踏めない(engineが拒否)=候補から外す扱い。判定不能ではない。
                targets.append(entry)
                continue
            try:
                can_ko, aborted = ko_search.can_ko_from_node(
                    node, me, root_my_prize, cfg, deadline, budget)
                entry["can_ko"] = bool(can_ko)
                entry["aborted"] = bool(aborted)
                any_aborted = any_aborted or bool(aborted)
            finally:
                try:
                    cg_api.search_release(node.searchId)
                except Exception:  # noqa: BLE001
                    pass
            targets.append(entry)
        return {"targets": targets, "any_aborted": any_aborted}
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return None
    finally:
        for node in (boss_child, root):
            if node is not None:
                try:
                    cg_api.search_release(node.searchId)
                except Exception:  # noqa: BLE001
                    pass


def _is_target_select(select) -> bool:
    """ボス直後の「引きずり出す相手を選ぶ」select か(実測スキーマ: CARD / SWITCH)。"""
    try:
        return (
            select is not None
            and select.type == SelectType.CARD
            and select.context == SelectContext.SWITCH
            and bool(select.option)
        )
    except Exception:  # noqa: BLE001
        return False
