"""Phase3: `climb_v15_leaf_a05` vs `climb_baseline` の head-to-head SPRT。

差分は **leaf evaluator だけ**(dynamic top-k を混ぜない)。Valueブレンド単体の効果を測る。
`driver.run_sprt_ab` を使い、探索失敗率を同時に記録する
(失敗率が0でない結果は勝率比較として無効 = §2.2)。

  A(control)  = climb_baseline      leaf_eval = handcrafted
  B(candidate)= climb_v15_leaf_a05  leaf_eval = 0.5*learned + 0.5*handcrafted

SPRT: p0=0.50 / delta_min / alpha=0.05 / beta=0.10、先後交互(alternate_sides)、resume 対応。

使い方:
  python kaggle_replays/_phase3_sprt_leaf.py --max-games 400 --workers 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import _probe_search_begin as probe  # noqa: E402
import agents  # noqa: E402
import driver  # noqa: E402
import runner  # noqa: E402

CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="climb_v15_leaf_a05")
    ap.add_argument("--control", default="climb_baseline")
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--delta-min", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--n-max", type=int, default=3000)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    agents.ensure_production_cwd()

    # 探索失敗率の観測(親 + 並列 worker の両方)。
    probe_dir = Path(tempfile.mkdtemp(prefix="p3_sprt_"))
    os.environ["PTCG_PROBE_DIR"] = str(probe_dir)
    probe.install()
    if args.workers > 1:
        driver._par_worker_init = probe.driver_worker_init_with_probe

    cand = driver.AgentSpec(config_name=args.candidate, policy_weights_path=CLIMB,
                            label=args.candidate)
    ctrl = driver.AgentSpec(config_name=args.control, policy_weights_path=CLIMB,
                            label=args.control)
    deck = runner.load_deck(_SUB / "deck.csv")

    # `agents.ensure_production_cwd()` で CWD が sample_submission/ へ移るため、相対指定は
    # リポジトリルート基準へ明示解決する(そうしないと sample_submission/ 配下に掘られる)。
    run_dir = Path(args.run_dir) if args.run_dir else (_HERE / "runs" / "phase3_leaf_a05")
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    report = driver.run_sprt_ab(
        cand=cand, ctrl=ctrl, deck=deck,
        delta_min=args.delta_min, alpha=args.alpha, beta=args.beta, n_max=args.n_max,
        alternate_sides=True, run_dir=run_dir, resume=args.resume,
        max_games=args.max_games, progress_every=20, workers=args.workers,
        note="Phase3: leaf blend alpha=0.5 vs handcrafted (single-factor)",
    )
    elapsed = time.perf_counter() - t0
    probe.flush()
    pstats = probe.collect(probe_dir)

    out = {
        "report": report,
        "search_integrity": {
            "begin_called": pstats["begin_called"],
            "begin_failed": pstats["begin_failed"],
            "begin_fail_rate": pstats["begin_fail_rate"],
            "by_side": pstats["by_side"],
            "errors": pstats["errors"],
            "processes_reporting": pstats["processes"],
        },
        "VALID_FOR_WINRATE_COMPARISON": pstats["begin_failed"] == 0 and pstats["begin_called"] > 0,
        "repro": {
            "git_commit": report.get("meta", {}).get("git_ref"),
            "candidate_config_sha": _sha(_SUB / "configs" / f"{args.candidate}.json"),
            "control_config_sha": _sha(_SUB / "configs" / f"{args.control}.json"),
            "policy_weights_sha": _sha(CLIMB),
            "value_weights_sha": _sha(_WDIR / "value_weights.json"),
            "deck_sha": _sha(_SUB / "deck.csv"),
            "elapsed_min": round(elapsed / 60, 2),
            "games_per_min": round(report["sprt"]["n"] / (elapsed / 60), 3) if elapsed else None,
            "run_dir": str(run_dir),
        },
    }
    dest = Path(args.out) if args.out else (_HERE / "_phase3_sprt_leaf_a05.json")
    if not dest.is_absolute():
        dest = _HERE / dest.name
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
