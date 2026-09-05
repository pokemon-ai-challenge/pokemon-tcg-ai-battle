"""ISMCTS v1 vs Current Champion(abl_5_full)mirror H2H — small screen / Formal Gate 1。

measurement/driver.run_sprt_ab を再利用(mirror、同 deck、手番交互)。Candidate=ismcts_v1(agent_fn=
ismcts_v1_agent.default_agent=picklable)、Control=abl_5_full。ISMCTS=large change → δ_min=0.05。
**module 冒頭で sys.path を設定**(spawn worker が __main__=本ファイルを再 import した際に agent を解決可能に)。
__main__ ガード必須。ローカル専用・production/frozen 評価資産 非改変。

使用:
  screen: python gate1_ismcts.py --mode screen --games 60 --workers 15 --run-dir results/ismcts_screen
  gate1 : python gate1_ismcts.py --mode gate1  --workers 15 --run-dir results/ismcts_gate1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_MEAS = _ROOT / "kaggle_replays" / "measurement"
_SEARCH = _ROOT / "kaggle_replays" / "search"
for _p in (str(_SUB), str(_MEAS), str(_SEARCH), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agents  # noqa: E402  (measurement/agents.py)
import driver  # noqa: E402
import ismcts_v1_agent  # noqa: E402


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "gate1"], default="screen")
    ap.add_argument("--games", type=int, default=60, help="screen: 対戦数上限")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--n-max", type=int, default=3000)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cand = driver.AgentSpec("ismcts_v1", agent_fn=ismcts_v1_agent.default_agent, label="ismcts_v1")
    ctrl = driver.AgentSpec("abl_5_full", label="abl_5_full")
    deck = _load_deck()
    run_dir = Path(args.run_dir) if args.run_dir else (_HERE / "results" / f"ismcts_{args.mode}")

    if args.mode == "screen":
        print(f"=== ISMCTS v1 small screen(health + preliminary): {args.games} games, workers={args.workers} ===")
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.05, alpha=0.05, beta=0.10,
                                 n_max=10_000_000, max_games=args.games, alternate_sides=True,
                                 run_dir=run_dir, resume=args.resume, workers=args.workers,
                                 progress_every=10, note="ismcts_v1 vs abl_5_full screen")
    else:
        print(f"=== ISMCTS v1 FORMAL Gate 1(large change δ_min=0.05): n_max={args.n_max}, workers={args.workers} ===")
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.05, alpha=0.05, beta=0.10,
                                 n_max=args.n_max, alternate_sides=True, run_dir=run_dir,
                                 resume=args.resume, workers=args.workers, progress_every=20,
                                 note="ismcts_v1 vs abl_5_full FORMAL Gate1")

    r = rep["result"]
    print("\n--- result ---")
    print(f"  decision={rep['decision']}  N={rep['sprt']['n']}  candidate(ismcts) wins={r['candidate_wins']} "
          f"winrate={r['candidate_winrate']:.4f} CI{r['wilson95_ci']} llr={rep['sprt']['llr']:.2f}")
    print(f"  errors: cand={r['cand_errors']} ctrl={r['ctrl_errors']}  elapsed={rep['elapsed_sec']}s "
          f"throughput={round(rep['sprt']['n']/(rep['elapsed_sec']/60),1) if rep['elapsed_sec'] else '?'} g/min")
    print(f"  channel: {rep['channel_breakdown']}")
    print(f"  report: {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
