"""ISMCTS equal-wall-time H2H(v1 rollout vs v2.1 leaf-at-node、同 wall-time budget)vs Current Champion。

各 (variant, T) config を env ISMCTS_CONFIG で切替、abl_5_full と mirror H2H。同 wall-time で v2.1 が v1 より
多く探索でき strength を再現/改善するか(H3、Primary)を測る。measurement/driver.run_sprt_ab 再利用。__main__ ガード必須。

使用: python walltime_h2h.py --budgets 1350,3000,5500 --games 100 --workers 15
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
for _p in (str(_SUB), str(_ROOT / "kaggle_replays" / "measurement"), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
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
    ap.add_argument("--budgets", default="1350,3000,5500")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "walltime"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    budgets = [int(b) for b in args.budgets.split(",")]
    deck = _load_deck()
    ctrl = driver.AgentSpec("abl_5_full", label="abl_5_full")
    run_root = Path(args.run_dir)
    print(f"=== ISMCTS equal-wall-time H2H vs abl_5_full: budgets={budgets}ms games={args.games} workers={args.workers} ===")
    curve = []
    for T in budgets:
        for tag in ("v1", "v2_1"):
            cfg = _HERE / "configs" / f"ismcts_{tag}_t{T}.json"
            if not cfg.exists():
                print(f"[skip] {cfg}"); continue
            os.environ["ISMCTS_CONFIG"] = str(cfg)
            cand = driver.AgentSpec("ismcts_wt", agent_fn=ismcts_v1_agent.default_agent, label=f"{tag}_t{T}")
            t0 = time.perf_counter()
            rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.05, alpha=0.05, beta=0.10,
                                     n_max=10_000_000, max_games=args.games, alternate_sides=True,
                                     run_dir=run_root / f"{tag}_t{T}", resume=False, workers=args.workers,
                                     progress_every=25, note=f"{tag}_t{T} vs abl_5_full")
            r = rep["result"]; el = time.perf_counter() - t0
            curve.append((T, tag, rep["sprt"]["n"], r["candidate_winrate"], r["wilson95_ci"],
                          r["cand_errors"], r["ctrl_errors"], rep["decision"]))
            print(f"  {tag}@{T}ms: {r['candidate_wins']}/{rep['sprt']['n']} wr={r['candidate_winrate']:.4f} "
                  f"CI{r['wilson95_ci']} dec={rep['decision']} err={r['cand_errors']}/{r['ctrl_errors']} "
                  f"[{el:.0f}s, {rep['sprt']['n']/(el/60):.1f} g/min]")

    print("\n=== equal-wall-time strength(vs Champion abl_5_full)===")
    print(f"  {'budget':>8} {'variant':>7} {'games':>6} {'winrate':>8} {'Wilson95':>18} {'SPRT':>10}")
    for (T, tag, n, wr, ci, ce, te, dec) in curve:
        print(f"  {T:>6}ms {tag:>7} {n:>6} {wr:>8.4f} [{ci[0]:.3f},{ci[1]:.3f}] {str(dec):>10}  err c{ce}/t{te}")
    print("\nH3: 同 budget で v2.1 winrate > v1 winrate なら leaf-at-node efficiency 成功(Case A)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
