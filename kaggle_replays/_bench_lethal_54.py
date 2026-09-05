#!/usr/bin/env python3
"""G1 で確定した「見逃した確定リーサル」54決定点に対して、指定 config のリーサル探索を回すベンチ。

背景(G1調査、`_probe_missed_lethal.py` / `_probe_missed_lethal_results.json`):
実ラダーの2試合(92608360 T12 / 92588584 T16)で「ボスの指令→入替→攻撃→勝ち」の確定リーサルが
存在したのに、本番の `lethal_simple` は time_limit_ms=100 で ~110-150 ノードしか踏めず1件も
発見できなかった。R7 の是正は (1) 優先展開(対象カードの手札PLAYを ATTACK 直後へ)と
(2) 予算 100→400ms の2点で、どちらも config-gated。

このベンチは局面抽出を `_probe_missed_lethal.py` から丸ごと流用し、**同じ54決定点**
(2試合の全 lethal ゲート通過点)に対して:

  - 参照パス: 緩和予算(既定5000ms・現行順序)の素DFS = 「予算さえあれば見つかっていた」集合 R
    (G1 の `--scan` と同じ定義。verify は掛けない)
  - 各 config パス: production と同じ `lethal_simple.search()`(= verify_shuffles 込み)を実行。
    非 None が返る = **検証再生を通った**新規発見。あわせて計測用の素DFSでノード数・中断理由も取る。

判定基準(R7 合意):
  - `abl_5_full`(現行)は 0/54 を再現すること
  - `abl_5_full_og_r7` は **確定2局面で 2/2 必須**、R(緩和予算で見つかっていた集合)で 10/10 目標
  - 未達の局面はノード数・中断理由を個別に出力する

production(`sample_submission/`)・`cg/`・`data/` は一切変更しない読み取り専用の診断。

使い方(cwd はどこでもよい。内部で sample_submission に chdir する):
    python kaggle_replays/_bench_lethal_54.py
    python kaggle_replays/_bench_lethal_54.py --config abl_5_full --config abl_5_full_og_r7
    python kaggle_replays/_bench_lethal_54.py --relax-ms 0        # 参照パスを省略(高速)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# `_probe_missed_lethal` は import 時に sys.path / cwd を sample_submission に合わせる。
# 局面抽出・表示・計測用DFSをそのまま再利用する(同じ抽出ロジック=同じ54点を保証する)。
import _probe_missed_lethal as PROBE  # noqa: E402

from cg.api import SelectType, to_observation_class  # noqa: E402
from ptcg_ai.core import config as ptcg_config  # noqa: E402
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state  # noqa: E402
from ptcg_ai.search import lethal_simple  # noqa: E402

# 対象試合(G1 と同じ2試合)。ターンは絞らず、lethal ゲートを通る全決定点を対象にする。
EPISODES = [92608360, 92588584]

# lethal ゲート(自分の残りサイド)。本番 `abl_5_full` の lethal_search.max_remaining_prizes=3。
# 決定点の集合を config によってブレさせないため、抽出は常にこの値で固定する。
GATE_MAX_PRIZES = 3

# G1 が「確定リーサルが実在し、別シャッフルでの検証再生も通る」と個別に確認した2局面
# (`_probe_missed_lethal_results.json` の P4/P5: p4.found=True かつ verify_replay_wins=True)。
# ここは og_r7 で必ず 2/2 になっていなければならない。
CONFIRMED = [(92608360, 157), (92588584, 201)]

DEFAULT_CONFIGS = ["abl_5_full", "abl_5_full_og_r7"]


# --------------------------------------------------------------------------
# 決定点の抽出(_probe_missed_lethal.scan_game と同じ条件)
# --------------------------------------------------------------------------

def collect_decision_points() -> list[dict[str, Any]]:
    """2試合から lethal ゲートを通る全決定点を集める(= G1 の54点)。"""
    points: list[dict[str, Any]] = []
    for episode_id in EPISODES:
        replay = PROBE.load_replay(episode_id)
        own = PROBE.own_index_of(replay)
        deck = PROBE.own_deck_of(replay, own)
        for dec in PROBE.own_decisions(replay, own):
            obs = to_observation_class(dec["obs_dict"])
            state = obs.current
            me = state.yourIndex
            if len(state.players[me].prize) > GATE_MAX_PRIZES:
                continue
            if not lethal_simple._is_my_turn(state, me) or state.result != -1:
                continue
            hidden = build_dummy_search_state(obs, deck)
            points.append({
                "episode": episode_id,
                "row": dec["row"],
                "turn": state.turn,
                "me": me,
                "deck": deck,
                "obs": obs,
                "select_type": PROBE.enum_name(SelectType, obs.select.type),
                "n_options": len(obs.select.option),
                "hidden_state_ok": hidden is not None,
                "hidden": hidden,
            })
    return points


# --------------------------------------------------------------------------
# 1決定点の実行
# --------------------------------------------------------------------------

def run_production_search(point: dict[str, Any], lethal_cfg: dict) -> dict[str, Any]:
    """production と同じ `lethal_simple.search()` を1回だけ回す(verify_shuffles 込み)。

    非 None が返る = 検証再生を通った採用可能な手。stats は呼び出し前に reset するので
    この1回分だけの内訳(timeouts / node_limit_hits / verify_rejects)になる。
    """
    obs = point["obs"]
    deck = point["deck"]
    lethal_simple.reset_stats()
    started = time.perf_counter()
    action = lethal_simple.search(
        obs.current,
        obs.select.option,
        {
            "observation": obs,
            "config": lethal_cfg,
            "hidden_state_factory": lambda: build_dummy_search_state(obs, deck),
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stats = lethal_simple.get_stats()
    return {
        "action": action,
        "found": action is not None,
        "elapsed_ms": elapsed_ms,
        "searches": stats["searches"],
        "timeouts": stats["timeouts"],
        "node_limit_hits": stats["node_limit_hits"],
        "verify_rejects": stats["verify_rejects"],
    }


def run_instrumented_dfs(point: dict[str, Any], lethal_cfg: dict, time_limit_ms: float) -> dict[str, Any]:
    """同じ config の探索本体を計測付きで回す(消費ノード数 / 到達深さ / 中断理由)。

    `_probe_missed_lethal.run_id_dfs` は `lethal_simple._dfs` をそのまま呼ぶので挙動は本番と同一。
    verify は掛からないので「探索が到達できたか」だけを見る指標。
    """
    if point["hidden"] is None:
        return {"found": False, "nodes": 0, "depth_reached": 0, "abort": "no_hidden_state", "elapsed_ms": 0.0}
    result = PROBE.run_id_dfs(point["obs"], point["me"], point["hidden"], lethal_cfg, time_limit_ms)
    return {
        "found": result["path"] is not None,
        "path": result["path"],
        "nodes": result["nodes"],
        "depth_reached": result["depth_reached"],
        "abort": result["abort"],
        "elapsed_ms": result["elapsed_ms"],
    }


# --------------------------------------------------------------------------
# config 読み出し
# --------------------------------------------------------------------------

def load_lethal_config(name: str) -> dict:
    """`sample_submission/configs/<name>.json` の lethal_search 節を DEFAULTS 込みで返す。"""
    cfg = ptcg_config.load_config(name)
    section = cfg.get("lethal_search")
    if not section:
        raise SystemExit(f"config '{name}' に lethal_search 節がありません(名前が違う可能性)")
    return {**lethal_simple.DEFAULTS, **section}


def describe_priority(lethal_cfg: dict) -> str:
    ids = lethal_cfg.get("priority_play_card_ids") or []
    if not ids:
        return "優先展開なし(既存順序)"
    names = ", ".join(f"{i}:{PROBE.card_name(i)}" for i in ids)
    return f"優先展開=[{names}] / 発動サイド<= {lethal_cfg.get('priority_play_max_remaining_prizes')}"


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


# --------------------------------------------------------------------------
# 本体
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="G1の54決定点に対するリーサル探索ベンチ(config別)")
    ap.add_argument("--config", action="append", default=None,
                    help="評価する config 名(複数指定可)。既定: " + " / ".join(DEFAULT_CONFIGS))
    ap.add_argument("--relax-ms", type=float, default=5000.0,
                    help="参照パス(緩和予算・現行順序)の時間上限ms。0 で参照パスを省略")
    ap.add_argument("--relax-nodes", type=int, default=2_000_000, help="参照パスのノード上限")
    ap.add_argument("--repeats", type=int, default=1,
                    help="1決定点あたりの試行回数。隠れ状態のシャッフルと壁時計のゆらぎで"
                         "境界局面は発見が安定しないため、2以上にすると安定度が測れる")
    ap.add_argument("--seed", type=int, default=20260815,
                    help="`build_dummy_search_state` が使う global random のシード。"
                         "config 間で同じ隠れ状態列を踏ませて比較を公平にする")
    ap.add_argument("--out", default=str(_HERE / "_bench_lethal_54_results.json"))
    args = ap.parse_args()
    config_names = args.config or DEFAULT_CONFIGS
    repeats = max(1, args.repeats)

    points = collect_decision_points()
    print("=" * 100)
    print(f"決定点(lethalゲート通過, 残りサイド<= {GATE_MAX_PRIZES}) = {len(points)}件 "
          f"/ 隠れ状態を組めた点 = {sum(1 for p in points if p['hidden_state_ok'])}件")
    for episode_id in EPISODES:
        rows = [p for p in points if p["episode"] == episode_id]
        print(f"  episode {episode_id}: {len(rows)}件 (rows {rows[0]['row']}..{rows[-1]['row']})")

    # ---- 参照パス: 緩和予算・現行順序の素DFS(= G1 の「予算があれば見つかっていた」集合 R)
    reference: dict[tuple[int, int], dict] = {}
    if args.relax_ms > 0:
        random.seed(args.seed)
        relax_cfg = {**lethal_simple.DEFAULTS, "max_remaining_prizes": GATE_MAX_PRIZES,
                     "time_limit_ms": args.relax_ms, "max_nodes": args.relax_nodes}
        print(f"\n--- 参照パス: 緩和予算 {args.relax_ms:.0f}ms / {args.relax_nodes} nodes / 現行順序 ---")
        for point in points:
            res = run_instrumented_dfs(point, relax_cfg, args.relax_ms)
            reference[(point["episode"], point["row"])] = res
            if res["found"]:
                print(f"  R ep{point['episode']} row={point['row']:4d} T{point['turn']:<3} "
                      f"{point['select_type']:<6} nodes={res['nodes']:6d} {res['elapsed_ms'] / 1000:.1f}s "
                      f"path={res.get('path')}")
        print(f"  -> 緩和予算で到達できた決定点 R = {sum(1 for r in reference.values() if r['found'])}件")

    ref_keys = [k for k, v in reference.items() if v["found"]]

    # ---- config 別パス
    results: dict[str, Any] = {
        "n_points": len(points),
        "reference_found": len(ref_keys),
        "reference_keys": [list(k) for k in ref_keys],
        "confirmed_keys": [list(k) for k in CONFIRMED],
        "configs": {},
    }
    for name in config_names:
        lethal_cfg = load_lethal_config(name)
        print("\n" + "=" * 100)
        print(f"config = {name} | time_limit_ms={lethal_cfg['time_limit_ms']} "
              f"max_nodes={lethal_cfg['max_nodes']} verify_shuffles={lethal_cfg['verify_shuffles']}")
        print(f"  {describe_priority(lethal_cfg)}")

        random.seed(args.seed)
        rows: list[dict[str, Any]] = []
        times: list[float] = []
        for point in points:
            trials = [run_production_search(point, lethal_cfg) for _ in range(repeats)]
            dfs = run_instrumented_dfs(point, lethal_cfg, lethal_cfg["time_limit_ms"])
            times.extend(t["elapsed_ms"] for t in trials)
            key = (point["episode"], point["row"])
            found_count = sum(1 for t in trials if t["found"])
            first_hit = next((t for t in trials if t["found"]), trials[-1])
            rows.append({
                "episode": point["episode"], "row": point["row"], "turn": point["turn"],
                "select_type": point["select_type"], "n_options": point["n_options"],
                "hidden_state_ok": point["hidden_state_ok"],
                "in_reference": key in reference and reference[key]["found"],
                "confirmed": key in CONFIRMED,
                "repeats": repeats,
                "found_count": found_count,
                "prod_found": found_count > 0,
                "prod_found_all": found_count == repeats,
                "prod_action": first_hit["action"],
                "prod_elapsed_ms": round(first_hit["elapsed_ms"], 1),
                "prod_timeouts": sum(int(t["timeouts"]) for t in trials),
                "prod_node_limit_hits": sum(int(t["node_limit_hits"]) for t in trials),
                "prod_verify_rejects": sum(int(t["verify_rejects"]) for t in trials),
                "prod_searches": sum(int(t["searches"]) for t in trials),
                "dfs_found": dfs["found"], "dfs_nodes": dfs["nodes"],
                "dfs_depth": dfs["depth_reached"], "dfs_abort": dfs["abort"],
            })

        found_rows = [r for r in rows if r["prod_found"]]
        summary = {
            "config": name,
            "time_limit_ms": lethal_cfg["time_limit_ms"],
            "priority_play_card_ids": list(lethal_cfg.get("priority_play_card_ids") or []),
            "repeats": repeats,
            "n_points": len(rows),
            # found = 1回でも見つかった決定点数 / found_all = 全試行で見つかった決定点数
            "found": len(found_rows),
            "found_all": sum(1 for r in rows if r["prod_found_all"]),
            "timeouts": sum(int(r["prod_timeouts"]) for r in rows),
            "node_limit_hits": sum(int(r["prod_node_limit_hits"]) for r in rows),
            "verify_rejects": sum(int(r["prod_verify_rejects"]) for r in rows),
            "searches": sum(int(r["prod_searches"]) for r in rows),
            "ms_p50": _pct(times, 0.50),
            "ms_p95": _pct(times, 0.95),
            "ms_max": round(max(times), 1) if times else None,
            "ms_total": round(sum(times), 1),
            "confirmed_found": sum(1 for r in rows if r["confirmed"] and r["prod_found"]),
            "confirmed_found_all": sum(1 for r in rows if r["confirmed"] and r["prod_found_all"]),
            "confirmed_total": len(CONFIRMED),
            "reference_found": sum(1 for r in rows if r["in_reference"] and r["prod_found"]),
            "reference_found_all": sum(1 for r in rows if r["in_reference"] and r["prod_found_all"]),
            "reference_total": len(ref_keys),
        }
        results["configs"][name] = {"summary": summary, "rows": rows}

        print(f"  発見 {summary['found']}/{summary['n_points']}"
              f"(全{repeats}試行で安定: {summary['found_all']}) "
              f"| 確定2局面 {summary['confirmed_found']}/{summary['confirmed_total']}"
              f"(安定 {summary['confirmed_found_all']}) "
              f"| 参照R {summary['reference_found']}/{summary['reference_total']}"
              f"(安定 {summary['reference_found_all']}) | "
              f"timeouts={summary['timeouts']} node_limit={summary['node_limit_hits']} "
              f"verify_rejects={summary['verify_rejects']} | "
              f"壁時計 p50={summary['ms_p50']}ms p95={summary['ms_p95']}ms max={summary['ms_max']}ms")

        for row in found_rows:
            tags = []
            if row["confirmed"]:
                tags.append("確定")
            if row["in_reference"]:
                tags.append("R")
            print(f"    [発見] ep{row['episode']} row={row['row']:4d} T{row['turn']:<3} "
                  f"{row['select_type']:<6} action={row['prod_action']} "
                  f"{row['found_count']}/{repeats}試行 "
                  f"({row['prod_elapsed_ms']:.0f}ms, DFS nodes={row['dfs_nodes']}) "
                  f"{'/'.join(tags)} verify=通過")

        # 未達の内訳(確定2局面 と 参照R に入っているのに見つからなかった点だけ個別に出す)
        misses = [
            r for r in rows
            if not r["prod_found_all"] and (r["confirmed"] or r["in_reference"])
        ]
        for row in misses:
            why = "timeout" if row["prod_timeouts"] else (
                "node_limit" if row["prod_node_limit_hits"] else (
                    "verify_reject" if row["prod_verify_rejects"] else (
                        "no_hidden_state" if not row["hidden_state_ok"] else "探索完走(勝ち筋なし)")))
            print(f"    [未達] ep{row['episode']} row={row['row']:4d} T{row['turn']:<3} "
                  f"{row['select_type']:<6} 発見{row['found_count']}/{repeats}試行 理由={why} "
                  f"DFS nodes={row['dfs_nodes']} depth={row['dfs_depth']} abort={row['dfs_abort']} "
                  f"{'確定' if row['confirmed'] else 'R'}")

    # ---- 判定
    print("\n" + "=" * 100)
    print("判定:")
    for name, block in results["configs"].items():
        s = block["summary"]
        if not s["priority_play_card_ids"] and s["time_limit_ms"] <= 100:
            # 現行(abl_5_full)は「確定2局面を1件も見つけられない」ことが再現条件。
            # 境界局面(予算ちょうどで届く点)はシャッフル・壁時計のゆらぎで揺れるので、
            # 全体の発見数ではなく確定2局面での 0 発見を判定に使う。
            verdict = ("OK(確定2局面=0発見を再現)" if s["confirmed_found"] == 0
                       else f"NG(現行なのに確定{s['confirmed_found']}件発見)")
        else:
            ok_confirmed = s["confirmed_found_all"] == s["confirmed_total"]
            ok_ref = s["reference_total"] == 0 or s["reference_found"] == s["reference_total"]
            verdict = ("OK(確定2/2 かつ R 全達成)" if ok_confirmed and ok_ref
                       else ("必須のみOK(確定2/2、R未達あり)" if ok_confirmed else "NG(確定2局面に未達)"))
        print(f"  {name}: 発見{s['found']}/{s['n_points']} 確定{s['confirmed_found']}/{s['confirmed_total']} "
              f"R{s['reference_found']}/{s['reference_total']} -> {verdict}")

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
