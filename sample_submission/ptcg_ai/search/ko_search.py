"""ターン内KO探索: このターン中に(準備込みで)相手をKOできる=サイドを取れる手順があるかを
判定する。lethal_simple と同じ ``cg.api.search_begin``/``search_step`` の浅いDFSで、成功条件だけが
"勝ち(result==me)" ではなく "**このターンでサイドを取れる(=KO)**" になっている点が違う。

用途: 改造ハンマー浪費veto(倒せる相手のエネを剥がすのは無駄)。lethal は「攻撃が既に届く」瞬間
しか見ないが、こちらは「エネ装着→攻撃」等の**準備込みのターン先読み**でKO可能性を見る。

KO検出 = 自分のサイド枚数が根から減る(KOでプライズを取る=エンジンの真実)。lethal と違い
"確実性"の再検証はしない(veto は「温存が良さそう」を言えれば十分で、確定は不要)。判定不能/
時間切れ/例外はすべて False(安全側=vetoしない)。
"""
from __future__ import annotations

import itertools
import time

from cg import api as cg_api
from cg.api import OptionType, SelectType

DEFAULTS: dict = {
    "max_depth": 4,
    "time_limit_ms": 100,
    "max_nodes": 3000,
    "max_combinations_per_select": 64,
}

# KOに向かう順で探索: 攻撃→打点補助(特性/進化)→エネ加速→道具。END は自ターン内でKOに
# 寄与しないので除外(lethal_simple と同じ思想)。
_MAIN_OPTION_PRIORITY = {
    OptionType.ATTACK: 0,
    OptionType.ABILITY: 1,
    OptionType.EVOLVE: 2,
    OptionType.ATTACH: 3,
    OptionType.PLAY: 4,
    OptionType.RETREAT: 5,
}
_DEFAULT_PRIORITY = 50


class _Abort(Exception):
    pass


def can_ko_this_turn(
    obs,
    hidden_state_factory,
    config: dict | None = None,
    deadline: float | None = None,
    report: dict | None = None,
) -> bool:
    """このターン中にKO(サイド取得)できる手順があれば True。安全側で例外/不能は False。

    Args:
        obs: 現在の Observation(``current``/``select`` 必須)。
        hidden_state_factory: ``search_begin`` 用の0引数callable(lethal/attack_plan と同じ)。
        config: ``max_depth``/``time_limit_ms``/``max_nodes``/``max_combinations_per_select``。
        deadline: ``time.perf_counter()`` 基準の締切(無ければ ``time_limit_ms`` から作る)。
        report: 任意。渡すと ``{"searched": bool, "aborted": bool}`` を書き込む。
            戻り値の False は「KOできない」と「判定不能(隠れ状態を組めない/予算切れ/例外)」を
            区別できないため、**False を根拠に介入したい呼び出し側**はこれを見る
            (`ml_policy_agent._try_briar_gate`)。省略時(既定 None)は一切書き込まないので
            既存呼び出し(`_try_hammer_veto` 等)の挙動は完全に不変。
    """
    if report is not None:
        report["searched"] = False
        report["aborted"] = False
    try:
        cfg = {**DEFAULTS, **(config or {})}
        if obs is None or obs.current is None or obs.select is None:
            return False
        state = obs.current
        me = state.yourIndex
        if state.result != -1 or hidden_state_factory is None:
            return False
        hidden_state = hidden_state_factory()
        if hidden_state is None:
            return False
        if deadline is None:
            deadline = time.perf_counter() + cfg["time_limit_ms"] / 1000.0
        root_my_prize = len(state.players[me].prize or [])
        root = _begin(obs, hidden_state)
        if report is not None:
            report["searched"] = True
        budget = {"nodes": 0}
        try:
            return _dfs(root, me, root_my_prize, int(cfg["max_depth"]), cfg, deadline, budget, {})
        except _Abort:
            if report is not None:
                report["aborted"] = True
            return False
        finally:
            try:
                cg_api.search_release(root.searchId)
            except Exception:
                pass
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        if report is not None:
            report["searched"] = False
        return False


def can_ko_from_node(
    node,
    me: int,
    root_my_prize: int,
    config: dict | None = None,
    deadline: float | None = None,
    budget: dict | None = None,
    depth: int | None = None,
) -> tuple[bool, bool]:
    """**既に進めた**探索ノードから「このターンKO(サイド取得)できるか」を判定する。

    `can_ko_this_turn` は根の Observation から `search_begin` するため、「ボスの指令で相手Xを
    引きずり出した後にXをKOできるか」のような**特定の手を打った後**の判定ができない。この関数は
    呼び出し側が `search_step` で進めたノードを受け取り、同じDFS(自分のサイドが減る=KO)を回す。

    ノードの `search_release` は**呼び出し側の責任**(このAPIは解放しない)。`budget` を共有すると
    複数の候補(例: ボスの対象5体)でノード数の総予算を分け合える。

    Returns:
        ``(can_ko, aborted)``。``aborted=True`` は予算切れ(時間/ノード)で探索を打ち切った=
        「KOできない」ではなく**判定不能**であることを示す。介入(veto/差し替え)の根拠に
        False を使う側は必ず ``aborted`` を見ること(`ko_search` の False は両者を区別しない)。
    """
    cfg = {**DEFAULTS, **(config or {})}
    if deadline is None:
        deadline = time.perf_counter() + cfg["time_limit_ms"] / 1000.0
    if budget is None:
        budget = {"nodes": 0}
    depth_left = int(cfg["max_depth"]) if depth is None else int(depth)
    try:
        return _dfs(node, me, root_my_prize, depth_left, cfg, deadline, budget, {}), False
    except _Abort:
        return False, True
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return False, True


def _dfs(node, me, root_my_prize, depth_left, cfg, deadline, budget, visited) -> bool:
    obs = node.observation
    if obs.current is None or obs.select is None or not obs.select.option or depth_left <= 0:
        return False
    if obs.current.yourIndex != me:
        return False  # 相手ターンに移った=自分のKOチャンスは終わり
    key = f"{obs.current!r}|{obs.select!r}"
    if visited.get(key, -1) >= depth_left:
        return False
    visited[key] = depth_left

    for selection in _candidate_selections(obs.select, cfg):
        if time.perf_counter() > deadline:
            raise _Abort()
        if budget["nodes"] >= int(cfg["max_nodes"]):
            raise _Abort()
        budget["nodes"] += 1
        try:
            child = cg_api.search_step(node.searchId, selection)
        except ValueError:
            continue
        try:
            cs = child.observation.current
            if cs is None:
                continue
            if cs.result == me:
                return True  # 勝ち=当然KO到達
            if cs.result == -1 and len(cs.players[me].prize or []) < root_my_prize:
                return True  # サイドが減った=KOでプライズを取った
            if cs.result == -1 and cs.yourIndex == me:
                if _dfs(child, me, root_my_prize, depth_left - 1, cfg, deadline, budget, visited):
                    return True
        finally:
            try:
                cg_api.search_release(child.searchId)
            except Exception:
                pass
    return False


def _candidate_selections(select, cfg):
    order = list(range(len(select.option)))
    if select.type == SelectType.MAIN:
        order = [i for i in order if select.option[i].type != OptionType.END]
        order.sort(key=lambda i: _MAIN_OPTION_PRIORITY.get(select.option[i].type, _DEFAULT_PRIORITY))
    mn = max(select.minCount, 0)
    mx = min(select.maxCount, len(order))
    limit = int(cfg["max_combinations_per_select"])
    produced = 0
    for count in range(mn, mx + 1):
        for combo in itertools.combinations(order, count):
            yield list(combo)
            produced += 1
            if produced >= limit:
                return


def _begin(obs, hidden_state):
    return cg_api.search_begin(
        obs,
        hidden_state["your_deck"],
        hidden_state["your_prize"],
        hidden_state["opponent_deck"],
        hidden_state["opponent_prize"],
        hidden_state["opponent_hand"],
        hidden_state["opponent_active"],
    )
