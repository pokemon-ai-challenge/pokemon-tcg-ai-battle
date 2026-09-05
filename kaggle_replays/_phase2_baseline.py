"""Phase2: 修正済み評価基盤での climb ベースライン再測定(7アーキタイプ)。

過去の FIELD 勝率は「相手側の探索が無効化された条件」の値なので、正式な比較値として使わない。
本スクリプトは修正後の条件で `climb_baseline` の基準値を取り直す。

各アーキタイプについて `run_league` で対戦させ、以下を記録する:
  勝率 / Wilson 95%CI / 先攻・後攻別 / games/min / エラー率
  **search_begin 失敗率**(0でなければその結果は勝率比較に使ってはならない)
再現メタ(git commit / config / 重みSHA / deck SHA / seed)も併せて保存する。

使い方:
  python kaggle_replays/_phase2_baseline.py --stage A --games 20 --workers 7
  python kaggle_replays/_phase2_baseline.py --stage B --games 100 --workers 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

# 過去 FIELD と同じ7アーキタイプ・同じメタシェア重み(分布は維持、勝率値は接続しない)。
FIELD = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]

_CHILD = r'''
import json, os, sys
from pathlib import Path
ROOT = Path(r"{root}")
for p in (ROOT / "kaggle_replays" / "measurement", ROOT / "league",
          ROOT / "sample_submission", ROOT):
    sys.path.insert(0, str(p))
import agents
agents.ensure_production_cwd()
import _probe_search_begin as probe
probe.install()
import run_league
workers = {workers}
if workers > 1:
    run_league._worker_init = probe.worker_init_with_probe
s = run_league.run_league(
    agent_a_name="ml_policy", agent_b_name="ml_policy", games={games},
    deck_a_path=r"{deck_a}", deck_b_path=r"{deck_b}",
    seed_start={seed}, progress_every=0,
    weights_a_path=r"{w_a}", weights_b_path=r"{w_b}",
    config_base="{config}", workers=workers, log=lambda m: None)
s.pop("games", None)   # 個別レコードは重いので落とす
Path(r"{out}").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
probe.flush()
'''


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (max(0.0, c - m), min(1.0, c + m))


def run_arch(arch: str, games: int, workers: int, seed: int, config: str) -> dict:
    probe_dir = Path(tempfile.mkdtemp(prefix=f"p2_{arch}_"))
    out_json = probe_dir / "summary.json"
    code = _CHILD.format(
        root=str(_ROOT), games=games, workers=workers, seed=seed, config=config,
        deck_a=str(_SUB / "deck.csv"), deck_b=str(_DECKDIR / arch / "01.csv"),
        w_a=CLIMB, w_b=str(_WDIR / f"policy_weights_{arch}.json"), out=str(out_json))
    env = dict(os.environ, PTCG_PROBE_DIR=str(probe_dir), PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - t0

    sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
    import _probe_search_begin as probe

    pstats = probe.collect(probe_dir)
    summary = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    shutil.rmtree(probe_dir, ignore_errors=True)

    ov = summary.get("overall", {})
    n = ov.get("games", 0) or 0
    wins = ov.get("wins", 0) or 0
    lo, hi = _wilson(wins, n)
    to = summary.get("by_turn_order", {})
    return {
        "archetype": arch,
        "games_run": summary.get("games_run"),
        "valid_games": n,
        "climb_wins": wins,
        "win_rate": round(wins / n, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "first": to.get("a_player0_first", {}),
        "second": to.get("a_player1_second", {}),
        "errors": summary.get("errors", {}),
        "elapsed_min": round(elapsed / 60, 2),
        "games_per_min": round((summary.get("games_run") or 0) / (elapsed / 60), 3)
        if elapsed > 0 else None,
        "search": {
            "begin_called": pstats["begin_called"],
            "begin_failed": pstats["begin_failed"],
            "begin_fail_rate": pstats["begin_fail_rate"],
            "by_side": pstats["by_side"],
            "errors": pstats["errors"],
        },
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-800:] if proc.returncode != 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--config", default="climb_baseline")
    ap.add_argument("--archetypes", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    archs = ([a for a in args.archetypes.split(",") if a] or [a for a, _ in FIELD])
    shares = dict(FIELD)

    meta = {
        "stage": args.stage,
        "config": args.config,
        "games_per_archetype": args.games,
        "workers": args.workers,
        "seed_start": args.seed,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                                         capture_output=True, text=True).stdout.strip()),
        "policy_weights_sha": _sha(CLIMB),
        "value_weights_sha": _sha(_WDIR / "value_weights.json"),
        "deck_sha": _sha(_SUB / "deck.csv"),
        "config_sha": _sha(_SUB / "configs" / f"{args.config}.json"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results = []
    for arch in archs:
        print(f"[{args.stage}] {arch} n={args.games} ...", file=sys.stderr, flush=True)
        r = run_arch(arch, args.games, args.workers, args.seed, args.config)
        results.append(r)
        print(f"    win={r['win_rate']} CI={r['wilson95']} "
              f"fail_rate={r['search']['begin_fail_rate']} "
              f"g/min={r['games_per_min']} rc={r['returncode']}",
              file=sys.stderr, flush=True)
        if r["stderr_tail"]:
            print(f"    STDERR {r['stderr_tail'][-400:]}", file=sys.stderr, flush=True)
        time.sleep(5)   # プール解放待ち(Windows のプロセスハンドル枯渇対策)

    valid = [r for r in results if r["valid_games"]]
    tot_share = sum(shares.get(r["archetype"], 0) for r in valid)
    weighted = (sum(shares.get(r["archetype"], 0) * r["win_rate"] for r in valid) / tot_share
                if tot_share else None)
    total_n = sum(r["valid_games"] for r in valid)
    total_w = sum(r["climb_wins"] for r in valid)
    lo, hi = _wilson(total_w, total_n)

    all_clean = all((r["search"]["begin_fail_rate"] in (0, 0.0)) and r["returncode"] == 0
                    for r in results)
    out = {
        "meta": meta,
        "per_archetype": results,
        "field_weighted_win_rate": round(weighted, 4) if weighted is not None else None,
        "unweighted_pooled": {
            "n": total_n, "wins": total_w,
            "win_rate": round(total_w / total_n, 4) if total_n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
        },
        "ALL_SEARCH_CLEAN": all_clean,
        "USABLE_AS_BASELINE": all_clean,
    }
    dest = Path(args.out) if args.out else _HERE / f"_phase2_baseline_stage{args.stage}.json"
    if not dest.is_absolute():
        dest = _HERE / dest.name
    print(json.dumps({k: v for k, v in out.items() if k != "per_archetype"},
                     ensure_ascii=False, indent=2))
    for r in results:
        print(f"  {r['archetype']:24s} n={r['valid_games']:4d} win={r['win_rate']} "
              f"CI={r['wilson95']} fail={r['search']['begin_fail_rate']}")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
