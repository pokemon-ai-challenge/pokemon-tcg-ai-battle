#!/usr/bin/env python3
"""実ラダーで見逃した「ボスの指令+攻撃=1手勝ち」局面を、本番リーサル探索に直接食わせて原因を切り分ける。

対象(battle-strategist がリプレイ突き合わせで確定させた2局面):
  - episode 92608360 T12: 自分サイド残り1、手札にボスの指令。相手ベンチに確1圏のポケモン多数。
  - episode 92588584 T16: サイド1-1、手札にボスの指令。相手ベンチ マシマシラ110/ドロンチ90。

このスクリプトは読み取り専用の診断で、sample_submission/ cg/ data/ を一切変更しない。
リプレイの observation dict はそのまま `to_observation_class` で Observation に戻せる
(`search_begin_input` も保存されている)ので、その試合の先頭から流し込む必要はなく、
該当行の observation を単体で探索に食わせられる。

原因の切り分けのために、同じ observation に対して以下を順に実行する:

  P0. `build_dummy_search_state` が None を返さないか (= 隠れ状態推定の失敗 (d) の検査)
  P1. 本番と同一 config での `lethal_simple.search()` (= 実戦の再現)
  P2. 本番 config の探索本体を自前ループで再実行し、消費ノード数 / 到達深さ /
      中断理由(time / nodes / 完了)を取得 (= (b) 予算不足の定量化)
  P3. 予算を大幅に緩めた config で同じ探索 (= 時間の問題か構造の問題かの切り分け)
  P4. 「最初の1手をボスの指令に固定」した強制探索 (= (a) そもそも列を展開しないのか、
      (c) ダメージ計算の都合で勝ちになっていないのかの切り分け。engine が勝ちを返すなら
      列自体は存在する)
  P5. P4 で見つかった列を `lethal_simple._replay_wins` で検証再生 (= verify_shuffles による
      棄却 (e') の検査)
  P6/P7. 予算は本番のまま「探索順序」「反復深化の有無」だけを変えた場合の到達可否
      (= 予算増額なしで直せるかの当たり。production は書き換えず、プロセス内で
      `_MAIN_OPTION_PRIORITY` を一時差し替えするだけ)

注意(この診断でハマった点):
  `build_dummy_search_state` は「観測に見えるカード ⊆ 渡した60枚デッキ」でないと None を
  返し、その場合リーサル探索は即 None になる。リポジトリの `deck.csv` は当時の提出と
  別デッキに変わっているため、deck.csv を読むと全局面が hidden_state=None になり
  「探索が走っていない」という誤診をする。ここでは steps[1] の60枚回答(その試合で実際に
  使ったデッキ)を使う。

使い方:
    python kaggle_replays/_probe_missed_lethal.py
    python kaggle_replays/_probe_missed_lethal.py --episode 92608360 --turn 12 --sweep --reorder
    python kaggle_replays/_probe_missed_lethal.py --scan --scan-ms 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))
# read_deck_csv() 等が相対パスを見るので cwd を sample_submission に合わせる(既存診断の流儀)。
os.chdir(_SUB)

from cg import api as cg_api  # noqa: E402
from cg.api import (  # noqa: E402
    AreaType,
    OptionType,
    SelectContext,
    SelectType,
    all_card_data,
    to_observation_class,
)
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state  # noqa: E402
from ptcg_ai.search import lethal_simple  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"

# configs/abl_5_full_og_routing.json の lethal_search 節と完全に同一(本番の再現)。
PROD_LETHAL_CONFIG: dict[str, Any] = {
    "enabled": True,
    "module": "lethal_simple",
    "max_remaining_prizes": 3,
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000,
    "max_combinations_per_select": 128,
    "verify_shuffles": 1,
}

# 研究用に予算だけ大幅に緩めた版(構造の問題か予算の問題かの切り分け用)。
# 本番との違いは時間とノード上限だけで、探索アルゴリズム自体は同一。
RELAXED_LETHAL_CONFIG: dict[str, Any] = {
    **PROD_LETHAL_CONFIG,
    "time_limit_ms": 10000,
    "max_nodes": 2000000,
}

TARGETS = [(92608360, 12), (92588584, 16)]


# --------------------------------------------------------------------------
# リプレイ読み出し
# --------------------------------------------------------------------------

def load_replay(episode_id: int) -> dict[str, Any]:
    path = _HERE / "replays" / f"episode-{episode_id}-replay.json"
    return json.loads(path.read_text(encoding="utf-8"))


def own_index_of(replay: dict[str, Any]) -> int:
    return replay["info"]["TeamNames"].index(TEAM_NAME)


def own_deck_of(replay: dict[str, Any], own_index: int) -> list[int]:
    """steps[1][own]['action'] が 60 枚のデッキ回答(= その試合で実際に使ったデッキ)。

    deck.csv を読むのではなくリプレイ実測のデッキを使う。リポジトリの deck.csv は
    その後の実験で書き換わっている可能性があり、build_dummy_search_state の
    整合チェック(観測に見えるカードがデッキに含まれるか)が誤って落ちるため。
    """
    deck = replay["steps"][1][own_index].get("action") or []
    if len(deck) != 60:
        raise RuntimeError(f"steps[1] のデッキ回答が60枚でない: {len(deck)}")
    return list(deck)


def own_decisions(replay: dict[str, Any], own_index: int) -> list[dict[str, Any]]:
    """自分が実際に回答した select 行を列挙する。

    スキーマ: steps[k][side]['action'] は steps[k-1][side]['observation']['select'] への
    回答(`_audit_hammer_replay.py` で検証済みの「1つ前行モデル」)。よって行 k の
    observation.select に対する回答は steps[k+1][own]['action'] にある。
    """
    steps = replay["steps"]
    out: list[dict[str, Any]] = []
    for k, row in enumerate(steps):
        cell = row[own_index]
        if cell.get("status") != "ACTIVE":
            continue
        obs_dict = cell.get("observation")
        if not obs_dict or not obs_dict.get("select") or not obs_dict.get("current"):
            continue
        answer = None
        if k + 1 < len(steps):
            answer = steps[k + 1][own_index].get("action")
        out.append({"row": k, "obs_dict": obs_dict, "answer": answer})
    return out


# --------------------------------------------------------------------------
# 表示ヘルパ
# --------------------------------------------------------------------------

_NAMES: dict[int, str] | None = None

# `all_card_data()` の name は英語("Boss's Orders")。日本語表記("ボスの指令")での照合も
# できるよう両方を許容する。cardId 直指定(1182)も冗長に持つ。
BOSS_ORDERS_ID = 1182


def _is_boss(name: str) -> bool:
    return "Boss" in name and "Orders" in name


def enum_name(enum_cls, value) -> str:
    """`cg.utils.to_dataclass` は Enum に変換せず素の int のまま入れるので、名前は自前で引く。"""
    if value is None:
        return "-"
    try:
        return enum_cls(value).name
    except Exception:  # noqa: BLE001 - 未知の値(コンペ期間中に追加されうる)
        return f"{enum_cls.__name__}({value})"


def card_name(card_id: int | None) -> str:
    global _NAMES
    if _NAMES is None:
        _NAMES = {c.cardId: c.name for c in all_card_data()}
    if card_id is None:
        return "-"
    return _NAMES.get(card_id, f"id={card_id}")


def option_card_name(opt, state, me: int) -> str:
    """選択肢が指しているカード名を解決する。

    重要: MAIN の選択肢は `cardId` が None で来る(実測)。PLAY/ATTACH は `index` が
    自分の手札インデックスを指すので、そこから名前を引く。ABILITY/RETREAT は
    area/index が場のポケモンを指す。
    """
    if opt.cardId is not None:
        return card_name(opt.cardId)
    if state is None:
        return "?"
    player = state.players[me]
    try:
        if opt.type in (OptionType.PLAY, OptionType.ATTACH) and opt.index is not None:
            hand = player.hand or []
            if 0 <= opt.index < len(hand):
                return card_name(hand[opt.index].id)
        if opt.type in (OptionType.ABILITY, OptionType.RETREAT, OptionType.EVOLVE):
            if opt.area == AreaType.ACTIVE and player.active:
                mon = player.active[0]
                return card_name(mon.id) if mon else "?"
            if opt.area == AreaType.BENCH and opt.index is not None:
                bench = player.bench
                if 0 <= opt.index < len(bench):
                    return card_name(bench[opt.index].id)
    except Exception:  # noqa: BLE001 - 表示用途なので失敗しても続行
        return "?"
    return "-"


def describe_option(i: int, opt, state=None, me: int | None = None) -> str:
    name = option_card_name(opt, state, me) if state is not None else card_name(opt.cardId)
    return (
        f"[{i:2d}] {enum_name(OptionType, opt.type):<9} card={name:<28}"
        f" area={enum_name(AreaType, opt.area):<10}"
        f" idx={opt.index} player={opt.playerIndex} attackId={opt.attackId}"
    )


def board_summary(state, me: int) -> str:
    lines = []
    for label, pi in (("自分", me), ("相手", 1 - me)):
        p = state.players[pi]
        mons = []
        for zone, lst in (("A", p.active), ("B", p.bench)):
            for mon in lst:
                if mon is None:
                    mons.append(f"{zone}:伏せ")
                    continue
                mons.append(
                    f"{zone}:{card_name(mon.id)}(HP{mon.hp}/{mon.maxHp},エネ{len(mon.energyCards)})"
                )
        lines.append(
            f"  {label}: サイド残{len(p.prize)} 山{p.deckCount} 手札{p.handCount} | " + ", ".join(mons)
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 探索プローブ
# --------------------------------------------------------------------------

def run_id_dfs(obs, me: int, hidden: dict, config: dict, time_limit_ms: float) -> dict[str, Any]:
    """`lethal_simple._find_winning_path` と同じ反復深化DFSを、計測値付きで実行する。

    本体をコピーせず `_dfs` をそのまま呼ぶので、探索の挙動は本番と同一。違いは
    budget dict をこちらが保持して消費ノード数 / 中断理由を観測できる点のみ。
    """
    root = lethal_simple._begin(obs, hidden)
    budget = {"nodes": 0, "depth_cutoff": False}
    deadline = time.perf_counter() + time_limit_ms / 1000.0
    started = time.perf_counter()
    depth_reached = 0
    abort = None
    path = None
    try:
        for depth_limit in range(1, int(config["max_depth"]) + 1):
            depth_reached = depth_limit
            visited: dict[str, int] = {}
            budget["depth_cutoff"] = False
            path = lethal_simple._dfs(root, me, depth_limit, config, deadline, budget, visited)
            if path is not None:
                break
            if not budget["depth_cutoff"]:
                abort = "exhausted"  # 到達可能な木を全部見たが勝ち筋なし
                break
    except lethal_simple._SearchAbort as e:
        abort = e.reason
    finally:
        try:
            cg_api.search_release(root.searchId)
        except Exception:
            pass
        try:
            cg_api.search_end()
        except Exception:
            pass
    return {
        "path": path,
        "nodes": budget["nodes"],
        "depth_reached": depth_reached,
        "abort": abort,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def forced_first_move_search(
    obs, me: int, hidden: dict, config: dict, first: list[int], time_limit_ms: float
) -> dict[str, Any]:
    """最初の1手を `first` に固定した上で、その先を反復深化DFSで探す。

    「探索順序さえ与えれば勝ち筋が存在するのか」を engine に直接聞くためのプローブ。
    見つかれば列は実在する(= (a)/(c) は否定され、原因は探索順序と予算 (b))。
    """
    root = lethal_simple._begin(obs, hidden)
    ids = [root.searchId]
    budget = {"nodes": 0, "depth_cutoff": False}
    deadline = time.perf_counter() + time_limit_ms / 1000.0
    started = time.perf_counter()
    path = None
    abort = None
    depth_reached = 0
    try:
        child = cg_api.search_step(root.searchId, first)
        ids.append(child.searchId)
        cur = child.observation.current
        if cur.result == me:
            path = [first]
        elif cur.result != -1 or cur.yourIndex != me:
            abort = "first_move_ends_turn"
        else:
            for depth_limit in range(1, int(config["max_depth"]) + 1):
                depth_reached = depth_limit
                visited: dict[str, int] = {}
                budget["depth_cutoff"] = False
                sub = lethal_simple._dfs(child, me, depth_limit, config, deadline, budget, visited)
                if sub is not None:
                    path = [first] + sub
                    break
                if not budget["depth_cutoff"]:
                    abort = "exhausted"
                    break
    except lethal_simple._SearchAbort as e:
        abort = e.reason
    except ValueError as e:
        abort = f"illegal_first_move: {e}"
    finally:
        for sid in ids:
            try:
                cg_api.search_release(sid)
            except Exception:
                pass
        try:
            cg_api.search_end()
        except Exception:
            pass
    return {
        "path": path,
        "nodes": budget["nodes"],
        "depth_reached": depth_reached,
        "abort": abort,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def measure_cost(obs, hidden: dict, n_steps: int = 40) -> dict[str, float]:
    """`search_begin` の固定費と `search_step` 1回あたりの単価を実測する。

    探索木の形に依存しないよう、root の選択肢を順に1手だけ踏んで(踏んだら release)
    平均を取る。`search_begin` は探索開始のたびに必ず1回払う固定費で、しかも
    `lethal_simple.search()` は deadline を search_begin より前に設定するので、
    この固定費はそのまま探索時間の食い込みになる。
    """
    t0 = time.perf_counter()
    root = lethal_simple._begin(obs, hidden)
    begin_ms = (time.perf_counter() - t0) * 1000.0

    n_opt = len(root.observation.select.option)
    steps = 0
    t1 = time.perf_counter()
    for i in range(min(n_steps, n_opt)):
        try:
            child = cg_api.search_step(root.searchId, [i])
        except ValueError:
            continue
        steps += 1
        try:
            cg_api.search_release(child.searchId)
        except Exception:
            pass
    step_ms = ((time.perf_counter() - t1) * 1000.0 / steps) if steps else float("nan")
    try:
        cg_api.search_release(root.searchId)
        cg_api.search_end()
    except Exception:
        pass
    return {
        "begin_ms": begin_ms,
        "step_ms": step_ms,
        "nodes_in_100ms": max(0.0, (100.0 - begin_ms)) / step_ms if step_ms else 0.0,
    }


_PLAY_FIRST_PRIORITY = {
    OptionType.ATTACK: 0,
    OptionType.PLAY: 1,       # 現行は 5(ATTACH/RETREAT より後ろ)。ボスの指令は PLAY。
    OptionType.ABILITY: 2,
    OptionType.EVOLVE: 3,
    OptionType.ATTACH: 4,
    OptionType.RETREAT: 5,
    OptionType.DISCARD: 6,
}


def with_priority(priority: dict):
    """`lethal_simple._MAIN_OPTION_PRIORITY` を一時的に差し替えるコンテキスト。

    production ファイルは書き換えず、プロセス内の module 変数だけを probe 中だけ変える。
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        original = dict(lethal_simple._MAIN_OPTION_PRIORITY)
        lethal_simple._MAIN_OPTION_PRIORITY.clear()
        lethal_simple._MAIN_OPTION_PRIORITY.update(priority)
        try:
            yield
        finally:
            lethal_simple._MAIN_OPTION_PRIORITY.clear()
            lethal_simple._MAIN_OPTION_PRIORITY.update(original)

    return _cm()


def run_fixed_depth_dfs(obs, me: int, hidden: dict, config: dict, depth: int,
                        time_limit_ms: float) -> dict[str, Any]:
    """反復深化をやめて depth 固定の1パスDFSを走らせる(反復深化の再展開コストの計測用)。"""
    root = lethal_simple._begin(obs, hidden)
    budget = {"nodes": 0, "depth_cutoff": False}
    deadline = time.perf_counter() + time_limit_ms / 1000.0
    started = time.perf_counter()
    path, abort = None, None
    try:
        path = lethal_simple._dfs(root, me, depth, config, deadline, budget, {})
        if path is None and not budget["depth_cutoff"]:
            abort = "exhausted"
    except lethal_simple._SearchAbort as e:
        abort = e.reason
    finally:
        try:
            cg_api.search_release(root.searchId)
        except Exception:
            pass
        try:
            cg_api.search_end()
        except Exception:
            pass
    return {"path": path, "nodes": budget["nodes"], "abort": abort,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0}


def candidate_rank(select, config: dict, target_selection: list[int]) -> int | None:
    """`_candidate_selections` の生成順で target_selection が何番目に出てくるか。"""
    for rank, sel in enumerate(lethal_simple._candidate_selections(select, config)):
        if sel == target_selection:
            return rank
    return None


def explain_path(obs, me: int, hidden: dict, path: list[list[int]]) -> list[str]:
    """見つかった勝ち筋を、各手で何を選んだか読める形に展開する。"""
    out: list[str] = []
    root = lethal_simple._begin(obs, hidden)
    ids = [root.searchId]
    node = root
    try:
        for step_no, sel in enumerate(path):
            select = node.observation.select
            cur_before = node.observation.current
            picked = [
                describe_option(i, select.option[i], cur_before, cur_before.yourIndex)
                for i in sel
            ]
            out.append(
                f"    手{step_no + 1}: select={enum_name(SelectType, select.type)}"
                f"/{enum_name(SelectContext, select.context)} 選択={sel}"
            )
            for line in picked:
                out.append(f"           {line}")
            node = cg_api.search_step(node.searchId, sel)
            ids.append(node.searchId)
            cur = node.observation.current
            opp = cur.players[1 - me]
            act = opp.active[0] if opp.active else None
            act_s = (
                f"{card_name(act.id)} HP{act.hp}/{act.maxHp}" if act is not None else "-"
            )
            out.append(
                f"           -> result={cur.result} 相手アクティブ={act_s} "
                f"相手サイド残={len(opp.prize)} 自分サイド残={len(cur.players[me].prize)}"
            )
    except Exception as e:  # noqa: BLE001 - 診断用途、失敗しても継続
        out.append(f"    (展開中に例外: {e!r})")
    finally:
        for sid in ids:
            try:
                cg_api.search_release(sid)
            except Exception:
                pass
        try:
            cg_api.search_end()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# 1決定点の診断
# --------------------------------------------------------------------------

def probe_decision(dec: dict[str, Any], full_deck: list[int], deep: bool, sweep: bool = False,
                   reorder: bool = False) -> dict[str, Any]:
    obs = to_observation_class(dec["obs_dict"])
    state = obs.current
    me = state.yourIndex
    select = obs.select

    rec: dict[str, Any] = {
        "row": dec["row"],
        "turn": state.turn,
        "select_type": enum_name(SelectType, select.type),
        "select_context": enum_name(SelectContext, select.context),
        "n_options": len(select.option),
        "answer": dec["answer"],
    }

    print(f"\n--- row={dec['row']} turn={state.turn} "
          f"select={enum_name(SelectType, select.type)}/{enum_name(SelectContext, select.context)} "
          f"選択肢{len(select.option)}件 実戦の回答={dec['answer']} ---")
    print(board_summary(state, me))
    hand = state.players[me].hand or []
    print("  手札: " + ", ".join(f"{i}:{card_name(c.id)}" for i, c in enumerate(hand)))
    for i, opt in enumerate(select.option):
        print("   " + describe_option(i, opt, state, me))

    # P0: 隠れ状態推定 (d)
    hidden = build_dummy_search_state(obs, full_deck)
    rec["hidden_state_ok"] = hidden is not None
    if hidden is None:
        print("  P0 build_dummy_search_state -> None (隠れ状態を組めない = 探索は即 None)")
        return rec
    print(f"  P0 build_dummy_search_state -> OK (自山{len(hidden['your_deck'])}枚 "
          f"サイド{len(hidden['your_prize'])}枚 相手山{len(hidden['opponent_deck'])}枚)")

    # P1: 本番と同一 config での実 API 呼び出し(実戦の再現)
    lethal_simple.reset_stats()
    t0 = time.perf_counter()
    prod_action = lethal_simple.search(
        state,
        select.option,
        {
            "observation": obs,
            "config": PROD_LETHAL_CONFIG,
            "hidden_state_factory": lambda: build_dummy_search_state(obs, full_deck),
        },
    )
    prod_ms = (time.perf_counter() - t0) * 1000.0
    prod_stats = lethal_simple.get_stats()
    rec["prod_action"] = prod_action
    rec["prod_stats"] = prod_stats
    print(f"  P1 本番config lethal_simple.search -> {prod_action} ({prod_ms:.1f}ms) "
          f"stats: searches={prod_stats['searches']} found={prod_stats['found']} "
          f"timeouts={prod_stats['timeouts']} node_limit={prod_stats['node_limit_hits']} "
          f"verify_rejects={prod_stats['verify_rejects']}")

    # P1b: 予算の内訳(search_begin の固定費 と 1ノード(search_step)あたりの単価)。
    # 「100ms で何ノード踏めるのか」の内訳を出して、予算不足の中身を分解する。
    timing = measure_cost(obs, hidden, n_steps=40)
    rec["timing"] = timing
    print(f"  P1b コスト内訳: search_begin={timing['begin_ms']:.1f}ms / "
          f"search_step 1回={timing['step_ms']:.2f}ms "
          f"→ 100ms 予算で踏めるノード数の目安={timing['nodes_in_100ms']:.0f}")

    # P2: 本番予算での探索本体を計測付きで再実行 (b の定量化)
    p2 = run_id_dfs(obs, me, hidden, PROD_LETHAL_CONFIG, PROD_LETHAL_CONFIG["time_limit_ms"])
    rec["p2"] = {k: v for k, v in p2.items() if k != "path"}
    rec["p2"]["found"] = p2["path"] is not None
    print(f"  P2 本番予算DFS: found={p2['path'] is not None} nodes={p2['nodes']} "
          f"depth_reached={p2['depth_reached']} abort={p2['abort']} {p2['elapsed_ms']:.1f}ms")

    if select.type != SelectType.MAIN:
        return rec

    # ボスの指令(相手のベンチを呼ぶサポート)候補を洗い出す。MAIN の選択肢は cardId=None で
    # 来るため、option.index -> 手札インデックス -> カード名 で同定する。
    boss_options = [
        i for i, opt in enumerate(select.option)
        if opt.type == OptionType.PLAY and _is_boss(option_card_name(opt, state, me))
    ]
    rec["boss_option_indices"] = boss_options
    if boss_options:
        ranks = {i: candidate_rank(select, PROD_LETHAL_CONFIG, [i]) for i in boss_options}
        rec["boss_candidate_rank"] = ranks
        print(f"  ボスの指令の選択肢index={boss_options} / _candidate_selections 生成順位={ranks} "
              f"(depth_limit=1 の全候補数={len(select.option) - sum(1 for o in select.option if o.type == OptionType.END)})")
    else:
        print("  ボスの指令は今の選択肢に存在しない")

    # P2b: 時間予算スイープ(「何msあれば届くのか」= 修正の当たりを付けるための実測)
    if sweep:
        rec["sweep"] = []
        for ms in (100, 200, 400, 800, 1600, 3200):
            r = run_id_dfs(obs, me, hidden, PROD_LETHAL_CONFIG, ms)
            rec["sweep"].append({
                "time_limit_ms": ms, "found": r["path"] is not None,
                "nodes": r["nodes"], "depth_reached": r["depth_reached"],
                "abort": r["abort"], "elapsed_ms": round(r["elapsed_ms"], 1),
            })
            print(f"  P2b sweep {ms:5d}ms: found={r['path'] is not None} nodes={r['nodes']:6d} "
                  f"depth={r['depth_reached']} abort={r['abort']}")
            if r["path"] is not None:
                break

    # P6: 探索順序だけを変えて本番予算(100ms)で再実行する。
    # 現行 `_MAIN_OPTION_PRIORITY` は PLAY を 5(ATTACH や RETREAT より後ろ)に置いており、
    # ボスの指令は PLAY なので後回しになる。PLAY を ATTACK の直後に上げるだけで
    # 本番予算内に収まるかを確かめる(= 予算増額なしで直せるかの当たり)。実装はせず計測のみ。
    if reorder and select.type == SelectType.MAIN:
        original = dict(lethal_simple._MAIN_OPTION_PRIORITY)
        promoted = {
            OptionType.ATTACK: 0,
            OptionType.PLAY: 1,       # ← ここだけ変更(5 -> 1)
            OptionType.ABILITY: 2,
            OptionType.EVOLVE: 3,
            OptionType.ATTACH: 4,
            OptionType.RETREAT: 5,
            OptionType.DISCARD: 6,
        }
        try:
            lethal_simple._MAIN_OPTION_PRIORITY.clear()
            lethal_simple._MAIN_OPTION_PRIORITY.update(promoted)
            r = run_id_dfs(obs, me, hidden, PROD_LETHAL_CONFIG, PROD_LETHAL_CONFIG["time_limit_ms"])
        finally:
            lethal_simple._MAIN_OPTION_PRIORITY.clear()
            lethal_simple._MAIN_OPTION_PRIORITY.update(original)
        rec["p6_reorder_play_first"] = {
            "found": r["path"] is not None, "nodes": r["nodes"],
            "depth_reached": r["depth_reached"], "abort": r["abort"],
        }
        print(f"  P6 PLAY優先に並べ替え + 本番予算100ms: found={r['path'] is not None} "
              f"nodes={r['nodes']} depth={r['depth_reached']} abort={r['abort']}")

        # P7: 反復深化をやめて深さ4の1パスDFSにした場合(順序 現行 / PLAY優先 の2通り)。
        # 反復深化の再展開コストと、順序の寄与を分離して測る。
        rec["p7_fixed_depth4"] = {}
        for label, prio in (("現行順序", dict(original)), ("PLAY優先", _PLAY_FIRST_PRIORITY)):
            with with_priority(prio):
                r7 = run_fixed_depth_dfs(
                    obs, me, hidden, PROD_LETHAL_CONFIG, 4, PROD_LETHAL_CONFIG["time_limit_ms"]
                )
            rec["p7_fixed_depth4"][label] = {
                "found": r7["path"] is not None, "nodes": r7["nodes"], "abort": r7["abort"],
            }
            print(f"  P7 深さ4固定1パスDFS({label}) + 本番予算100ms: "
                  f"found={r7['path'] is not None} nodes={r7['nodes']} abort={r7['abort']}")

    if not deep:
        return rec

    # P3: 予算を緩めた探索 (時間の問題か構造の問題か)
    p3 = run_id_dfs(obs, me, hidden, RELAXED_LETHAL_CONFIG, RELAXED_LETHAL_CONFIG["time_limit_ms"])
    rec["p3"] = {k: v for k, v in p3.items() if k != "path"}
    rec["p3"]["found"] = p3["path"] is not None
    rec["p3"]["path"] = p3["path"]
    print(f"  P3 緩和予算DFS(60s/200万node): found={p3['path'] is not None} nodes={p3['nodes']} "
          f"depth_reached={p3['depth_reached']} abort={p3['abort']} {p3['elapsed_ms'] / 1000:.1f}s")
    if p3["path"] is not None:
        print("     勝ち筋:")
        for line in explain_path(obs, me, hidden, p3["path"]):
            print(line)

    # P4: 最初の1手をボスの指令に固定した強制探索
    rec["p4"] = []
    for boss_idx in boss_options:
        p4 = forced_first_move_search(
            obs, me, hidden, RELAXED_LETHAL_CONFIG, [boss_idx], RELAXED_LETHAL_CONFIG["time_limit_ms"]
        )
        entry = {"first": [boss_idx], **{k: v for k, v in p4.items() if k != "path"},
                 "found": p4["path"] is not None, "path": p4["path"]}
        rec["p4"].append(entry)
        print(f"  P4 ボス固定探索 first=[{boss_idx}]: found={p4['path'] is not None} "
              f"nodes={p4['nodes']} depth_reached={p4['depth_reached']} abort={p4['abort']} "
              f"{p4['elapsed_ms'] / 1000:.1f}s")
        if p4["path"] is not None:
            print("     勝ち筋:")
            for line in explain_path(obs, me, hidden, p4["path"]):
                print(line)
            # P5: 別シャッフルでの検証再生(verify_shuffles による棄却の検査)
            ok = lethal_simple._replay_wins(
                obs, me, build_dummy_search_state(obs, full_deck), p4["path"],
                time.perf_counter() + 10.0,
            )
            try:
                cg_api.search_end()
            except Exception:
                pass
            entry["verify_replay_wins"] = ok
            print(f"  P5 別シャッフルでの検証再生: {'勝ちのまま(採用される)' if ok else '棄却される'}")

    return rec


def scan_game(replay: dict[str, Any], own: int, deck: list[int], relax_ms: float) -> dict[str, Any]:
    """1試合の全 select 行について、本番予算と緩和予算の結果を比較する。

    「本番予算では見つからないが、予算さえあれば確定リーサルだった」決定点を数えて
    見逃しの規模を出す。lethal のゲート(自分サイド残 <= max_remaining_prizes)を
    通る行だけが探索対象になる点も本番と同じ。
    """
    rows: list[dict[str, Any]] = []
    for dec in own_decisions(replay, own):
        obs = to_observation_class(dec["obs_dict"])
        state, select = obs.current, obs.select
        me = state.yourIndex
        if len(state.players[me].prize) > PROD_LETHAL_CONFIG["max_remaining_prizes"]:
            continue
        if not lethal_simple._is_my_turn(state, me) or state.result != -1:
            continue
        hidden = build_dummy_search_state(obs, deck)
        if hidden is None:
            rows.append({"row": dec["row"], "turn": state.turn, "hidden_state_ok": False})
            continue
        prod = run_id_dfs(obs, me, hidden, PROD_LETHAL_CONFIG, PROD_LETHAL_CONFIG["time_limit_ms"])
        relax = run_id_dfs(obs, me, hidden, RELAXED_LETHAL_CONFIG, relax_ms)
        rows.append({
            "row": dec["row"], "turn": state.turn, "hidden_state_ok": True,
            "select_type": enum_name(SelectType, select.type),
            "n_options": len(select.option),
            "prod_found": prod["path"] is not None, "prod_nodes": prod["nodes"],
            "prod_abort": prod["abort"], "prod_depth": prod["depth_reached"],
            "relax_found": relax["path"] is not None, "relax_nodes": relax["nodes"],
            "relax_abort": relax["abort"], "relax_ms": round(relax["elapsed_ms"], 1),
            "relax_path": relax["path"],
        })
        flag = ""
        if rows[-1].get("relax_found") and not rows[-1].get("prod_found"):
            flag = "  <== 予算があれば勝てた(本番は見逃し)"
        print(f"  row={dec['row']:4d} T{state.turn:<3} {enum_name(SelectType, select.type):<6} "
              f"opt={len(select.option):3d} prod={rows[-1]['prod_found']}({prod['nodes']:5d}n,{prod['abort']}) "
              f"relax={rows[-1]['relax_found']}({relax['nodes']:6d}n,{relax['abort']},{relax['elapsed_ms'] / 1000:.1f}s)"
              f"{flag}")
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="見逃した確定リーサル局面を本番探索に直接食わせて原因を切り分ける")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--turn", type=int, default=None)
    ap.add_argument("--all-turns", action="store_true", help="指定ターンだけでなく全ターンの MAIN 決定を走査する")
    ap.add_argument("--rows", default=None, help="この行(steps index)だけを対象にする。カンマ区切り")
    ap.add_argument("--main-only", action="store_true", help="MAIN select の決定点だけを対象にする")
    ap.add_argument("--relax-ms", type=float, default=None, help="緩和探索の時間上限(ms)を上書き")
    ap.add_argument("--sweep", action="store_true", help="時間予算スイープ(100..3200ms)を実行する")
    ap.add_argument("--reorder", action="store_true", help="PLAY優先の並べ替え版を本番予算で計測する(P6)")
    ap.add_argument("--no-deep", action="store_true", help="P3/P4/P5(重い緩和探索)をスキップする")
    ap.add_argument("--scan", action="store_true",
                    help="ターンを絞らず、lethal ゲートを通る全決定点で本番予算 vs 緩和予算を比較する")
    ap.add_argument("--scan-ms", type=float, default=5000.0, help="--scan での緩和予算(ms)")
    ap.add_argument("--out", default=str(_HERE / "_probe_missed_lethal_results.json"))
    args = ap.parse_args()

    targets = TARGETS
    if args.episode is not None:
        targets = [(args.episode, args.turn)]
    if args.relax_ms is not None:
        RELAXED_LETHAL_CONFIG["time_limit_ms"] = args.relax_ms
    row_filter = {int(x) for x in args.rows.split(",")} if args.rows else None

    results = []
    for episode_id, turn in targets:
        replay = load_replay(episode_id)
        own = own_index_of(replay)
        deck = own_deck_of(replay, own)
        rewards = replay.get("rewards") or [None, None]
        print("=" * 100)
        print(f"episode {episode_id} | own_index={own} | rewards={rewards} | 対象ターン={turn}")
        print(f"デッキ(steps[1]の回答) 先頭10枚: {[card_name(c) for c in deck[:10]]}")

        if args.scan:
            scan = scan_game(replay, own, deck, args.scan_ms)
            good = [r for r in scan["rows"] if r.get("relax_found") and not r.get("prod_found")]
            print(f"  -> lethal ゲート通過={len(scan['rows'])}件 / "
                  f"本番予算で発見={sum(1 for r in scan['rows'] if r.get('prod_found'))}件 / "
                  f"緩和予算で発見={sum(1 for r in scan['rows'] if r.get('relax_found'))}件 / "
                  f"見逃し={len(good)}件")
            results.append({"episode": episode_id, "scan": scan})
            continue

        decisions = own_decisions(replay, own)
        picked = []
        for dec in decisions:
            t = dec["obs_dict"]["current"].get("turn")
            if not (args.all_turns or turn is None or t == turn):
                continue
            if row_filter is not None and dec["row"] not in row_filter:
                continue
            if args.main_only and (dec["obs_dict"]["select"] or {}).get("type") != int(SelectType.MAIN):
                continue
            picked.append(dec)
        print(f"自分の select 行 総数={len(decisions)} / 対象={len(picked)}")

        ep_rec = {"episode": episode_id, "turn": turn, "own_index": own, "decisions": []}
        for dec in picked:
            ep_rec["decisions"].append(probe_decision(dec, deck, deep=not args.no_deep, sweep=args.sweep, reorder=args.reorder))
        results.append(ep_rec)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
