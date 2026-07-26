"""ISMCTS strength scaling H2H(compute-unconstrained)。

budget(iterations)別の ISMCTS(Research Challenger、時間 cap 無し)を Current Champion(abl_5_full)へ H2H し、
**compute → strength の scaling curve** を取る。budget は env `ISMCTS_CONFIG`(configs/ismcts_hc{N}.json)で切替
(spawn worker が継承)。measurement/driver.run_sprt_ab 再利用(mirror、同 deck、手番交互)。
production/frozen 非改変。__main__ ガード必須。

使用: python scaling_h2h.py --budgets 8,16,32,64 --games 60 --workers 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_MEAS = _ROOT / "kaggle_replays" / "measurement"
_SEARCH = _ROOT / "kaggle_replays" / "search"
for _p in (str(_SUB), str(_MEAS), str(_SEARCH), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agents  # noqa: E402
import driver  # noqa: E402
import ismcts_v1_agent  # noqa: E402


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "scaling"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    budgets = [int(b) for b in args.budgets.split(",")]
    deck = _load_deck()
    ctrl = driver.AgentSpec("abl_5_full", label="abl_5_full")
    run_root = Path(args.run_dir)
    print(f"=== ISMCTS strength scaling H2H(compute-unconstrained): budgets={budgets} games={args.games} workers={args.workers} ===")
    curve = []
    for N in budgets:
        cfg_path = _HERE / "configs" / f"ismcts_hc{N}.json"
        if not cfg_path.exists():
            print(f"[skip] config 不在: {cfg_path}"); continue
        os.environ["ISMCTS_CONFIG"] = str(cfg_path)   # ← worker(spawn)が継承
        cand = driver.AgentSpec("ismcts_hc", agent_fn=ismcts_v1_agent.default_agent, label=f"ismcts_hc{N}")
        t0 = time.perf_counter()
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.05, alpha=0.05, beta=0.10,
                                 n_max=10_000_000, max_games=args.games, alternate_sides=True,
                                 run_dir=run_root / f"hc{N}", resume=False, workers=args.workers,
                                 progress_every=20, note=f"ismcts_hc{N} vs abl_5_full scaling")
        r = rep["result"]; el = time.perf_counter() - t0
        curve.append((N, rep["sprt"]["n"], r["candidate_wins"], r["candidate_winrate"], r["wilson95_ci"],
                      r["cand_errors"], r["ctrl_errors"], el))
        print(f"  hc{N}: {r['candidate_wins']}/{rep['sprt']['n']} wr={r['candidate_winrate']:.4f} "
              f"CI{r['wilson95_ci']} err={r['cand_errors']}/{r['ctrl_errors']} "
              f"[{el:.0f}s, {rep['sprt']['n']/(el/60):.1f} g/min]")

    print("\n=== strength scaling curve(ISMCTS vs Current Champion abl_5_full)===")
    print(f"  {'iters':>6} {'games':>6} {'winrate':>8} {'Wilson95_CI':>20} {'errors':>8}")
    for (N, n, w, wr, ci, ce, te, el) in curve:
        print(f"  {N:>6} {n:>6} {wr:>8.4f} [{ci[0]:.3f},{ci[1]:.3f}]   c{ce}/t{te}")
    print("\n判定: 8→16→32→64 で winrate が一貫して上昇なら ISMCTS 有効(compute efficiency 問題)。"
          "横ばいなら compute を増やしても伸びず→ leaf/belief/branching を疑う。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
