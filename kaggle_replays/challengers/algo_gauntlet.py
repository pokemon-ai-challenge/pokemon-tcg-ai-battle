"""Cross-Eval Phase E/F — algorithm-only gauntlet(Old Full PIMC vs Current ISMCTS、同一デッキ・対フィールド)。

診断専用。新規アルゴリズムなし。2 セルとも **ismcts_v1_agent.agent** を使い ismcts.enabled を false/true で
切替えるだけ(F6 no-search equivalence 保証 → 唯一差は MAIN を浅 PIMC で解くか深 ISMCTS で解くか):
  old_full        = ismcts.enabled=false → Champion(abl_5_full: lethal→PIMC pipeline→policy)
  current_ismcts  = ismcts.enabled=true  → 同 Champion 土台 + ISMCTS(v2.11 batched, FULL rollout, 1350ms, adaptive OFF)
両者とも同じ Policy(構成C)/Belief N=8/lethal/相手推定。contender デッキは固定(既定 sample_submission/deck.csv)。
相手 = round_robin.ARCHS(alakazam mirror + 7 メタアーキ)を v0only + アーキ別重みで pilot(両セル共通=delta で相殺)。

play_match(run_match)を直接使い、独自 ProcessPoolExecutor で並列化(run_league 非改変)。side 交互。
使用: python algo_gauntlet.py --games 100 --workers 15 [--deck cur|old] [--archs a,b,...] [--cells old_full,current_ismcts]
出力: 各セル×相手の winrate/CI/errors + Phase F delta table(current-old)+ macro winrate。JSON 保存。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_LEAGUE = _ROOT / "league"
for _p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE), str(_LEAGUE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ISMCTS_CFG = _HERE / "configs" / "ismcts_v2_11_h4batch_t1350.json"  # current_ismcts(enabled=true)
_CONTENDER_DECKS = {"cur": _SUB / "deck.csv", "old": _ROOT / "experimental_decks" / "alakazam_xerosic" / "deck.csv"}
_OPP_CONFIG = "ml_lethal_attackplan_v0only"

# ---- worker globals(initializer で 1 回だけ構築)----
_G: dict = {}


def _load_deck(path) -> list[int]:
    return [int(x) for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip().lstrip("-").isdigit()][:60]


def _winit(deck_key: str):
    """worker 初期化: sys.path/cwd、config/deck/相手定義を globals へ。"""
    os.chdir(_SUB)
    import ismcts_v1_agent as A
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    from ptcg_ai.core.config import load_config
    from round_robin import ARCHS
    import run_match
    from ptcg_ai.hidden_information import match_context
    cfg_on = A.load_config(_ISMCTS_CFG)
    cfg_off = json.loads(json.dumps(cfg_on)); cfg_off["ismcts"]["enabled"] = False       # old_full
    cfg_gate = json.loads(json.dumps(cfg_on))                                             # v2.10b Search-Dynamics Gate
    cfg_gate["ismcts"]["dynamics_gate"] = {"enabled": True, "share": 0.5, "gap": 0.2}
    opp_base = load_config(_OPP_CONFIG)
    _G.update(A=A, mpa=mpa, match_context=match_context, play_match=run_match.play_match,
              cfg={"current_ismcts": cfg_on, "old_full": cfg_off, "current_ismcts_gate": cfg_gate},
              our_deck=_load_deck(_CONTENDER_DECKS[deck_key]),
              archs={n: {"deck": _load_deck(_ROOT / d), "weights": w, "share": s} for (n, d, w, s) in ARCHS},
              opp_base=opp_base)


def _opp_cfg(weights):
    import copy
    c = copy.deepcopy(_G["opp_base"])
    if weights:
        c["policy_weights_path"] = str((_ROOT / weights) if not Path(weights).is_absolute() else Path(weights))
    return c


def _play(task):
    """1 試合。task=(game_id, cell, opp_name, seed, our_p0)。返: (game_id, cell, opp_name, our_win|None, error)。"""
    game_id, cell, opp_name, seed, our_p0 = task
    A = _G["A"]; mpa = _G["mpa"]; mc = _G["match_context"]
    our_cfg = _G["cfg"][cell]; arch = _G["archs"][opp_name]
    opp_cfg = _opp_cfg(arch["weights"])
    our_agent = lambda obs: A.agent(obs, our_cfg)
    opp_agent = lambda obs: mpa.agent(obs, opp_cfg)
    mc.reset()
    if our_p0:
        r = _G["play_match"](our_agent, opp_agent, _G["our_deck"], arch["deck"], seed)
        win = None if r.winner is None else (r.winner == 0)
    else:
        r = _G["play_match"](opp_agent, our_agent, arch["deck"], _G["our_deck"], seed)
        win = None if r.winner is None else (r.winner == 1)
    return (game_id, cell, opp_name, win, r.error)


def _wilson(w, n):
    if n == 0:
        return (float("nan"), (float("nan"), float("nan")))
    p = w / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, (c - h, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--deck", choices=list(_CONTENDER_DECKS), default="cur")
    ap.add_argument("--archs", default="alakazam,mega_lucario_ex,archaludon_ex,crustle,dragapult_ex,marnie_grimmsnarl_ex,rocket_mewtwo_ex,shirona_garchomp_ex")
    ap.add_argument("--cells", default="old_full,current_ismcts")
    ap.add_argument("--out", default=str(_HERE / "results" / "algo_gauntlet"))
    ap.add_argument("--resume-dir", default=None,
                    help="指定すると Measurement v2.1 auto-resume（manifest 照合・completed skip・durable per-game）を有効化")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # ARCHS share を後で使うため一度ロード(親プロセス)
    from round_robin import ARCHS
    share = {n: s for (n, _d, _w, s) in ARCHS}

    agg = {c: {o: {"w": 0, "n": 0, "err": 0} for o in archs} for c in cells}
    rr = None
    if args.resume_dir:
        # Measurement v2.1 auto-resume: semantic manifest 照合 → completed skip → remaining のみ実行。
        import hashlib
        sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
        import resume as R

        def _sha(rel):
            p = _ROOT / rel
            return hashlib.sha1(p.read_bytes()).hexdigest()[:16] if p.exists() else None
        our_deck_bytes = _CONTENDER_DECKS[args.deck].read_bytes()
        manifest = {
            "experiment_name": Path(args.resume_dir).name, "parent": "algo_gauntlet_field",
            "candidate": ",".join(cells), "unique_difference": "cells vs frozen field",
            "candidate_config_sha1": _sha("kaggle_replays/challengers/configs/ismcts_v2_11_h4batch_t1350.json"),
            "parent_config_sha1": _sha("sample_submission/configs/abl_5_full.json"),
            "opponent_config_sha1": _sha("sample_submission/configs/ml_lethal_attackplan_v0only.json"),
            "policy_weights_sha1": _sha("sample_submission/ptcg_ai/learning/policy_weights.json"),
            "deck_sha1": hashlib.sha1(our_deck_bytes).hexdigest()[:16],
            "gauntlet_manifest_sha1": hashlib.sha1((",".join(sorted(archs))).encode()).hexdigest()[:16],
            "evaluation_mode": "fast_gauntlet", "primary_metric": "binary_win_loss",
            "planned_n": args.games * len(archs) * len(cells),
            "target_effect_pp": None, "alpha": 0.05, "power": 0.80,
            "allocation": {c: {o: args.games for o in archs} for c in cells},
            "macro_definition": "equal_weight", "search_budget_ms": 1350, "machine_class": R.machine_class(),
        }
        exp_h = R.manifest_hash(manifest)
        sched = R.build_schedule(exp_h, cells, archs, args.games)
        rr = R.ResumableRun(args.resume_dir, manifest, sched)
        mode, remaining = rr.start()
        print(f"  Experiment: {manifest['experiment_name']}  Manifest hash: {exp_h[:12]}", flush=True)
        print(f"  Planned games: {len(sched)}  Completed: {len(rr.completed)}  Remaining: {len(remaining)}  "
              f"Resume validation: {mode.upper()}  Machine: {manifest['machine_class']}", flush=True)
        for gid, rec in rr.completed.items():                 # 既存 completed を agg 復元
            p = R.parse_game_id(gid); a = agg[p["cell"]][p["opp"]]
            if rec.get("error"):
                a["err"] += 1
            elif rec.get("result") in ("win", "loss"):
                a["n"] += 1; a["w"] += int(rec["result"] == "win")
        tasks = [(gid, p["cell"], p["opp"], 1000 + p["game_index"], p["side"] == "p0")
                 for gid in remaining for p in (R.parse_game_id(gid),)]
    else:
        tasks = [(f"{c}|{o}|{g}|{'p0' if g % 2 == 0 else 'p1'}", c, o, 1000 + g, g % 2 == 0)
                 for c in cells for o in archs for g in range(args.games)]
    print(f"=== algo_gauntlet deck={args.deck} cells={cells} archs={len(archs)} games/pair={args.games} "
          f"to_run={len(tasks)} workers={args.workers} resume={'ON' if rr else 'OFF'} ===", flush=True)

    partial = outdir / f"_algo_gauntlet_{args.deck}_partial.json"
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit, initargs=(args.deck,)) as ex:
        futs = [ex.submit(_play, t) for t in tasks]
        for f in as_completed(futs):
            game_id, cell, opp, win, err = f.result()
            if rr is not None:                                # durable per-game 記録(保存後のみ completed)
                rr.record(game_id, {"result": ("win" if win else "loss") if win is not None else None,
                                    "opp": opp, "cell": cell, "error": err})
            a = agg[cell][opp]
            if err:
                a["err"] += 1
            elif win is not None:
                a["n"] += 1; a["w"] += int(win)
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)
                if rr is None:
                    partial.write_text(json.dumps({"deck": args.deck, "done": done, "total": len(tasks),
                                                   "agg": agg, "elapsed_s": time.time() - t0}, ensure_ascii=False), encoding="utf-8")
    if rr is not None:
        rr.close()

    # 出力
    rows = {}
    print(f"\n=== Per-cell winrate (deck={args.deck}) [{time.time()-t0:.0f}s] ===")
    for cell in cells:
        print(f"-- {cell} --")
        rows[cell] = {}
        for opp in archs:
            a = agg[cell][opp]; p, (lo, hi) = _wilson(a["w"], a["n"])
            rows[cell][opp] = {"w": a["w"], "n": a["n"], "wr": p, "ci": [lo, hi], "err": a["err"], "share": share.get(opp, 0)}
            print(f"   {opp:22s} {a['w']:>3}/{a['n']:<3} wr={p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] err={a['err']}")

    def macro(cell):
        ws = [rows[cell][o]["wr"] for o in archs if rows[cell][o]["n"] > 0]
        return sum(ws) / len(ws) if ws else float("nan")

    def weighted(cell):
        num = den = 0.0
        for o in archs:
            r = rows[cell][o]
            if r["n"] > 0:
                num += r["share"] * r["wr"]; den += r["share"]
        return num / den if den else float("nan")

    if len(cells) == 2:                       # 任意の 2 セル比較(cell1 - cell0)
        c0, c1 = cells
        print(f"\n=== F. Archetype Delta Table ({c1} - {c0}), deck={args.deck} ===")
        print(f"  {'opponent':22s} {c0[:10]:>10} {c1[:10]:>10} {'delta':>8} {'games':>6}")
        for opp in archs:
            a0 = rows[c0][opp]; a1 = rows[c1][opp]
            d = (a1["wr"] - a0["wr"]) * 100 if (a0["n"] and a1["n"]) else float("nan")
            print(f"  {opp:22s} {a0['wr']*100:>9.1f}% {a1['wr']*100:>9.1f}% {d:>+7.1f} {min(a0['n'],a1['n']):>6}")
        m0, m1 = macro(c0), macro(c1); w0, w1 = weighted(c0), weighted(c1)
        print(f"  {'MACRO (equal wt)':22s} {m0*100:>9.1f}% {m1*100:>9.1f}% {(m1-m0)*100:>+7.1f}")
        print(f"  {'WEIGHTED (metashare)':22s} {w0*100:>9.1f}% {w1*100:>9.1f}% {(w1-w0)*100:>+7.1f}")

    # 標準 result schema(Measurement Protocol v2 Phase J): machine/throughput/totals も記録。
    import os, platform, socket
    elapsed = time.time() - t0
    total_games = sum(agg[c][o]["n"] + agg[c][o]["err"] for c in cells for o in archs)
    per_cell = {}
    for cell in cells:
        tot_n = sum(agg[cell][o]["n"] for o in archs); tot_w = sum(agg[cell][o]["w"] for o in archs)
        per_cell[cell] = {"wins": tot_w, "losses": tot_n - tot_w, "completed_n": tot_n,
                          "errors": sum(agg[cell][o]["err"] for o in archs),
                          "macro": macro(cell), "weighted": weighted(cell)}
    out = outdir / f"_algo_gauntlet_{args.deck}.json"
    out.write_text(json.dumps({
        "schema": "measurement_v2.result", "mode": "fast_gauntlet", "primary_metric": "binary_win_loss",
        "deck": args.deck, "games_per_pair": args.games, "cells": cells, "archs": archs,
        "planned_n_per_cell": args.games * len(archs), "total_games": total_games,
        "per_cell": per_cell, "rows": rows,
        "machine": {"host": socket.gethostname(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "workers": args.workers, "wall_time_s": round(elapsed, 1),
        "games_per_min": round(total_games / (elapsed / 60), 1) if elapsed else None,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  results -> {out}  ({per_cell})")


if __name__ == "__main__":
    main()
