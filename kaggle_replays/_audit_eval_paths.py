"""Phase1: 全ローカル評価経路の測定忠実度監査。

「両陣営が別デッキでも `search_begin` が失敗せず、両方で探索が実際に走るか」を
**経路ごとに実測**する。`pipeline.search` は例外を握りつぶすため、失敗しても勝率は出てしまう。
勝率比較の前に、経路ごとに失敗率0を確認するのが目的。

対象経路:
  - measurement.runner.play_game      (driver.run_sprt_ab が使う)
  - league.run_match.play_match       (run_league の1試合)
  - league.run_league.run_league      workers=1 (逐次)
  - league.run_league.run_league      workers=2 (並列 = worker 再利用経路)
  - kaggle_replays.rl.eval_field      (run_league 経由の field 評価)

デッキ組み合わせは2種類以上で検証する(alakazam vs mega_lucario / alakazam vs crustle)。

出力: _audit_eval_paths_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

# 各経路は「別プロセスで」走らせる。プローブは atexit でカウンタを吐くため、
# 経路ごとにプロセスを分けると集計が混ざらない(並列 worker も同じ dir に吐く)。
_CHILD = r'''
import json, os, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(ROOT / "league"))
sys.path.insert(0, str(ROOT / "sample_submission"))
sys.path.insert(0, str(ROOT))
import agents
agents.ensure_production_cwd()
import _probe_search_begin as probe
probe.install()

path = "{path}"
opp = "{opp}"
games = {games}
CLIMB = r"{climb}"
WDIR = ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
deck_a = str(ROOT / "sample_submission" / "deck.csv")
deck_b = str(DECKDIR / opp / "01.csv")
w_b = str(WDIR / ("policy_weights_%s.json" % opp))

if path in ("runner", "run_match"):
    import runner
    cfg_a = agents.load_config_copy("climb_baseline"); cfg_a["policy_weights_path"] = CLIMB
    cfg_b = agents.load_config_copy("climb_baseline"); cfg_b["policy_weights_path"] = w_b
    A = agents.make_ml_policy_agent(cfg_a); B = agents.make_ml_policy_agent(cfg_b)
    d0 = runner.load_deck(deck_a); d1 = runner.load_deck(deck_b)
    if path == "runner":
        for g in range(games):
            runner.play_game(A, B, d0, d1) if g % 2 == 0 else runner.play_game(B, A, d1, d0)
    else:
        from run_match import play_match
        for g in range(games):
            play_match(A, B, d0, d1, seed=g) if g % 2 == 0 else play_match(B, A, d1, d0, seed=g)
elif path.startswith("run_league"):
    import run_league
    workers = int(path.split("_w")[1])
    if workers > 1:
        # worker 側にもプローブを仕込む(モジュール参照なので spawn でも pickle 可能)。
        run_league._worker_init = probe.worker_init_with_probe
    run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=deck_a, deck_b_path=deck_b, seed_start=0, progress_every=0,
        weights_a_path=CLIMB, weights_b_path=w_b,
        config_base="climb_baseline", workers=workers, log=lambda m: None)
elif path == "eval_field":
    sys.path.insert(0, str(ROOT / "kaggle_replays" / "rl"))
    import eval_field
    eval_field.alakazam_vs(opp, CLIMB, games, 1, "climb_baseline")
probe.flush()
'''


def run_path(path: str, opp: str, games: int, python: str) -> dict:
    probe_dir = Path(tempfile.mkdtemp(prefix=f"probe_{path}_"))
    code = _CHILD.format(root=str(_ROOT), path=path, opp=opp, games=games, climb=CLIMB)
    env = dict(os.environ, PTCG_PROBE_DIR=str(probe_dir), PYTHONIOENCODING="utf-8")
    proc = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True,
                          timeout=3600)
    sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
    import _probe_search_begin as probe

    out = probe.collect(probe_dir)
    out["path"] = path
    out["opponent"] = opp
    out["games"] = games
    out["returncode"] = proc.returncode
    if proc.returncode != 0:
        out["stderr_tail"] = proc.stderr[-1500:]
    shutil.rmtree(probe_dir, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--opponents", default="mega_lucario_ex,crustle")
    ap.add_argument("--paths", default="runner,run_match,run_league_w1,run_league_w2,eval_field")
    ap.add_argument("--out", default=str(_HERE / "_audit_eval_paths_results.json"))
    args = ap.parse_args()

    results = []
    for opp in args.opponents.split(","):
        for path in args.paths.split(","):
            print(f"[run] {path} vs {opp} ({args.games} games) ...", file=sys.stderr, flush=True)
            r = run_path(path, opp, args.games, sys.executable)
            results.append(r)
            fr = r.get("begin_fail_rate")
            print(f"      begin_called={r['begin_called']} fail_rate={fr} "
                  f"rc={r['returncode']} procs={r['processes']}", file=sys.stderr, flush=True)
            if r.get("stderr_tail"):
                print(f"      STDERR: {r['stderr_tail'][-400:]}", file=sys.stderr, flush=True)

    verdict = []
    for r in results:
        both_sides = all(
            (r["by_side"][f"player{i}"]["begin_called"] or 0) > 0 for i in (0, 1))
        ok = (r["returncode"] == 0 and r["begin_called"] > 0
              and r["begin_failed"] == 0 and both_sides)
        verdict.append({
            "path": r["path"], "opponent": r["opponent"],
            "begin_called": r["begin_called"], "begin_fail_rate": r["begin_fail_rate"],
            "both_sides_searched": both_sides, "PASS": ok,
            "errors": r["errors"],
        })

    out = {"games_per_path": args.games, "results": results, "verdict": verdict,
           "ALL_PASS": all(v["PASS"] for v in verdict)}
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("ALL_PASS =", out["ALL_PASS"])
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
