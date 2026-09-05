#!/usr/bin/env python3
"""og_r7(リーサル修正済み)がなお「即勝ちを見逃した」とされる実ラダー局面の機械診断。

対象は battle-strategist がリプレイ突き合わせで挙げた4局面:
  - episode 93322802 T15: 相手アクティブのメガユキメノコexへの攻撃で勝ち確定だったとされる局面
  - episode 93324616 T9 / 93326434 T10 / 93335550 T15: 「勝ち」でなく「生存」局面
    (lethal が None を返すこと自体は正しいはずで、それ自体が別のガードの必要性を示す対照群)

読み取り専用の診断で、sample_submission/ cg/ data/ を一切変更しない。既存の
`_probe_missed_lethal.py`(G1で作成済み)の読み出し/表示/探索ユーティリティをそのまま
import して再利用し、重複実装しない。og_r7固有の点は:

  - lethal_search config を `configs/abl_5_full_og_r7.json` から読み込んで使う
    (400ms + priority_play_card_ids=[1182, 1201] + priority_play_max_remaining_prizes=2)。
    `ml_policy_agent._try_lethal` が呼ぶのと *同一* の config 値・同一の呼び出しシグネチャ
    (`lethal_simple.search(state, select.option, {"observation":..,"config":..,
    "hidden_state_factory":..})`) を使うことで、実戦の `_try_lethal` 呼び出しを
    1対1で再現する。

  - 対象ターン内で「自分が回答した select」を **型を問わず**(MAIN 以外の CARD
    sub-select も含めて)全部並べ、各時点で lethal_simple.search() が何を返すか・
    get_stats() の内訳を出す。「MAIN決定でだけ lethal が呼ばれるのでは」という仮説を
    確かめるため。コード上は `_try_lethal`(ml_policy_agent.py:312-359)が
    `obs.select.type` を一切参照せず毎 select で無条件に呼ばれることは読み取り済みだが、
    それを空撃ちせず実際に CARD select 上で `lethal_simple.search()` を叩いて裏取りする。

事前調査で判明した重要事実(このスクリプトを書く過程で確定させた、報告の前提):
  episode 93322802 T15 の「相手アクティブが残りHP70」という課題文の前提は、実際には
  **攻撃"後"の残量**だった。T15開始時点(row183)では相手のメガユキメノコexはT14中に
  「ミツルの思いやり」等で全回復しており HP310/310(満タン)。自分は実際には T15 の
  最終MAIN決定(row186)で ATTACK を選択しており(action=[0]、row187のaction経由で
  確認。row186 の観測に埋め込まれた累積ログにも T13 の 240 ダメ攻撃ログが見える)、
  まんようしぐれで240ダメージを与えたが 310 に対しては足りずKOできなかった(240ダメは
  T13 の同カード同エネ数の実測と一致)。つまり「攻撃を見逃した」のではなく「与えられる
  最大打点(240)が確定KOに届かなかった」局面。以下の診断はこれを実際に
  lethal_simple.search() を叩いて裏取りする。

使い方:
    python kaggle_replays/_probe_r7_misses.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# 既存の診断ハーネスをそのまま import して使う(sys.path 設定・cwd変更は
# _probe_missed_lethal 側で完結している。読み出し/表示/探索ユーティリティの
# 再実装を避けるため、ここでは import のみ行う)。
sys.path.insert(0, str(_HERE))
import _probe_missed_lethal as pm  # noqa: E402

from cg.api import SelectContext, SelectType, to_observation_class  # noqa: E402

# --------------------------------------------------------------------------
# og_r7 の lethal_search config(実戦の _try_lethal が読むのと同一節)
# --------------------------------------------------------------------------

_OG_R7_CONFIG_PATH = _ROOT / "sample_submission" / "configs" / "abl_5_full_og_r7.json"
OG_R7_LETHAL_CONFIG: dict[str, Any] = json.loads(
    _OG_R7_CONFIG_PATH.read_text(encoding="utf-8")
)["lethal_search"]

# 「予算があれば見つかるのか」を確かめるための緩和版(構造の問題か予算の問題かの切り分け)。
RELAXED_R7_CONFIG: dict[str, Any] = {
    **OG_R7_LETHAL_CONFIG,
    "time_limit_ms": 10000,
    "max_nodes": 2_000_000,
}

# (episode_id, 対象ターン, 一言ラベル)
TARGETS: list[tuple[int, int, str]] = [
    (93322802, 15, "勝ち確定と主張されている局面(検証対象)"),
    (93324616, 9, "生存の問題とされる局面(対照群1)"),
    (93326434, 10, "生存の問題とされる局面(対照群2)"),
    (93335550, 15, "生存の問題とされる局面(対照群3)"),
]


def mon_line(mon) -> str:
    if mon is None:
        return "伏せ"
    return f"{pm.card_name(mon.id)} HP{mon.hp}/{mon.maxHp} エネ{len(mon.energyCards)}"


def diagnose_decision(dec: dict[str, Any], deck: list[int]) -> dict[str, Any]:
    """1つの select 決定点に og_r7 の lethal 呼び出しをそのまま叩き、原因分類まで出す。"""
    obs = to_observation_class(dec["obs_dict"])
    state = obs.current
    me = state.yourIndex
    select = obs.select

    opp = state.players[1 - me]
    own_p = state.players[me]
    opp_act = opp.active[0] if opp.active else None

    print(
        f"\n  --- row={dec['row']:4d} T{state.turn:<2} "
        f"select={pm.enum_name(SelectType, select.type)}/"
        f"{pm.enum_name(SelectContext, select.context)} "
        f"opt={len(select.option)} 実戦の回答={dec['answer']}"
    )
    print(f"      自分アクティブ: {mon_line(own_p.active[0] if own_p.active else None)}"
          f" | 相手アクティブ: {mon_line(opp_act)}")
    print(f"      自分サイド残={len(own_p.prize)} 相手サイド残={len(opp.prize)}")
    for i, opt in enumerate(select.option):
        print("        " + pm.describe_option(i, opt, state, me))

    rec: dict[str, Any] = {
        "row": dec["row"], "turn": state.turn,
        "select_type": pm.enum_name(SelectType, select.type),
        "select_context": pm.enum_name(SelectContext, select.context),
    }

    # --- ゲート((a): result/手番/自サイド残の3条件。lethal_simple.search() 冒頭の
    # ガードと完全に同一の条件をここで手動確認する) ---
    result_ok = state.result == -1
    turn_ok = pm.lethal_simple._is_my_turn(state, me)
    prize_ok = len(own_p.prize) <= OG_R7_LETHAL_CONFIG["max_remaining_prizes"]
    print(f"      ゲート: result==-1={result_ok} is_my_turn={turn_ok} "
          f"自サイド残{len(own_p.prize)}<=上限{OG_R7_LETHAL_CONFIG['max_remaining_prizes']}={prize_ok}")
    if not (result_ok and turn_ok and prize_ok):
        print("      => (a) ゲート不通過。search() は探索すら始めずNoneを返す")
        rec["category"] = "a_gate"
        return rec

    # --- (b): 隠れ状態構築 ---
    hidden = pm.build_dummy_search_state(obs, deck)
    if hidden is None:
        print("      => (b) build_dummy_search_state -> None(隠れ状態を組めない)")
        rec["category"] = "b_hidden_state"
        return rec
    print(f"      P0 隠れ状態OK: 自山{len(hidden['your_deck'])}枚 相手山{len(hidden['opponent_deck'])}枚")

    # --- 実戦の _try_lethal と1対1のAPI呼び出し(config・context形も同一) ---
    pm.lethal_simple.reset_stats()
    t0 = time.perf_counter()
    context = {
        "observation": obs,
        "config": OG_R7_LETHAL_CONFIG,
        "hidden_state_factory": lambda: pm.build_dummy_search_state(obs, deck),
    }
    action = pm.lethal_simple.search(state, select.option, context)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = pm.lethal_simple.get_stats()
    rec["prod_action"] = action
    rec["prod_stats"] = stats
    print(f"      P1 og_r7 config lethal_simple.search() -> {action} ({elapsed_ms:.1f}ms)")
    print(f"         stats: searches={stats['searches']} found={stats['found']} "
          f"timeouts={stats['timeouts']} node_limit={stats['node_limit_hits']} "
          f"verify_rejects={stats['verify_rejects']}")

    if action is not None:
        print("      => リーサル発見。_try_lethal はこれを返し、_select_action はこれを"
              "そのまま採用するはず(lethalはpipelineより前に置かれている)")
        rec["category"] = "found"
        return rec

    if stats["verify_rejects"] > 0:
        print("      => (d) 見つかった列が verify_shuffles による再検証で棄却された"
              "(シャッフル依存=確定リーサルでない)")
        rec["category"] = "d_verify_reject"
        return rec

    # --- (c) vs 真の不在: 予算を大幅に緩めて再実行し、存在自体を確かめる ---
    p_relax = pm.run_id_dfs(obs, me, hidden, RELAXED_R7_CONFIG, RELAXED_R7_CONFIG["time_limit_ms"])
    found_relaxed = p_relax["path"] is not None
    print(f"      P3 緩和予算(10s/200万node)で再探索: found={found_relaxed} "
          f"nodes={p_relax['nodes']} depth={p_relax['depth_reached']} abort={p_relax['abort']} "
          f"{p_relax['elapsed_ms']:.1f}ms")
    if found_relaxed:
        print("      => (c) 本番予算(400ms/1万node)では未達だが、予算さえあれば確定リーサルは"
              "存在した(時間/ノード不足による見逃し)")
        print("      勝ち筋(緩和予算で発見):")
        for line in pm.explain_path(obs, me, hidden, p_relax["path"]):
            print("      " + line)
        rec["category"] = "c_budget"
        rec["relaxed_path"] = p_relax["path"]
    else:
        print("      => 緩和予算でも見つからない = この時点に確定リーサルは実在しない"
              "(lethal が None を返すのは正しい)")
        rec["category"] = "none_exists"
    return rec


def run_episode(episode_id: int, focus_turn: int, label: str) -> None:
    replay = pm.load_replay(episode_id)
    own = pm.own_index_of(replay)
    deck = pm.own_deck_of(replay, own)
    rewards = replay.get("rewards") or [None, None]
    print("=" * 100)
    print(f"episode {episode_id} | {label} | own_index={own} | rewards={rewards} | "
          f"対象ターン={focus_turn}")

    decisions = pm.own_decisions(replay, own)
    picked = [d for d in decisions if d["obs_dict"]["current"].get("turn") == focus_turn]
    print(f"対象ターンの自分の select 決定点(型を問わず全部): {len(picked)}件")

    for dec in picked:
        diagnose_decision(dec, deck)


def main() -> None:
    print(f"og_r7 lethal_search config (from {_OG_R7_CONFIG_PATH.relative_to(_ROOT)}):")
    print(f"  {OG_R7_LETHAL_CONFIG}")
    for episode_id, turn, label in TARGETS:
        run_episode(episode_id, turn, label)


if __name__ == "__main__":
    main()
