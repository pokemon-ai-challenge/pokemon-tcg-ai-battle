"""ISMCTS v2.4 Phase N — direct v1 comparison(student vs teacher v1 FULL、equal-wall-time 直接 H2H)。

Primary parent-ablation: 同じ wall-time で Small rollout Policy(student)は Full rollout(v1)より強いか。
cand = student(ISMCTS_CONFIG=student config)、ctrl = v1(ISMCTS_CONFIG_CTRL=v1 config)。両者 ISMCTS。
driver.run_sprt_ab 再利用。__main__ ガード必須。

使用: python direct_v1_h2h.py --tags v2_4_h16,v2_4_h4 --budget 1350 --games 100 --workers 15
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

import driver  # noqa: E402
import ismcts_v1_agent  # noqa: E402


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v2_4_h16,v2_4_h4", help="student variant tags(config ismcts_<tag>_t<budget>.json)")
    ap.add_argument("--control", default="v1", help="control tag(config ismcts_<control>_t<budget>.json)。既定 v1。")
    ap.add_argument("--budgets", default="1350", help="comma-separated budgets ms")
    ap.add_argument("--delta-min", type=float, default=0.05, help="SPRT delta_min(小さいほど早期停止せず N 増=CI 締まる)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "direct_v1"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    budgets = [int(b) for b in args.budgets.split(",")]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    deck = _load_deck()
    run_root = Path(args.run_dir)
    print(f"=== ISMCTS v2.4 direct H2H: student vs v1 FULL  budgets={budgets}ms games={args.games} workers={args.workers} ===")
    curve = []
    for T in budgets:
        ctrl_cfg = _HERE / "configs" / f"ismcts_{args.control}_t{T}.json"
        os.environ["ISMCTS_CONFIG_CTRL"] = str(ctrl_cfg)   # control(既定 v1、--control で変更)
        for tag in tags:
            cfg = _HERE / "configs" / f"ismcts_{tag}_t{T}.json"
            if not cfg.exists():
                print(f"[skip] {cfg}"); continue
            os.environ["ISMCTS_CONFIG"] = str(cfg)          # candidate = student
            cand = driver.AgentSpec("ismcts_student", agent_fn=ismcts_v1_agent.default_agent, label=tag)
            ctrl = driver.AgentSpec("ismcts_v1", agent_fn=ismcts_v1_agent.default_agent_ctrl, label=f"v1_t{T}")
            t0 = time.perf_counter()
            rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=args.delta_min, alpha=0.05, beta=0.10,
                                     n_max=10_000_000, max_games=args.games, alternate_sides=True,
                                     run_dir=run_root / f"{tag}_vs_v1_t{T}", resume=False, workers=args.workers,
                                     progress_every=25, note=f"{tag} vs v1 @ {T}ms")
            r = rep["result"]; el = time.perf_counter() - t0
            curve.append((T, tag, rep["sprt"]["n"], r["candidate_winrate"], r["wilson95_ci"],
                          r["cand_errors"], r["ctrl_errors"], rep["decision"]))
            print(f"  {tag} vs v1@{T}ms: {r['candidate_wins']}/{rep['sprt']['n']} wr={r['candidate_winrate']:.4f} "
                  f"CI{r['wilson95_ci']} dec={rep['decision']} err={r['cand_errors']}/{r['ctrl_errors']} "
                  f"[{el:.0f}s]")

    print("\n=== direct v1 H2H(student winrate vs v1 FULL rollout, equal-wall-time)===")
    print(f"  {'budget':>7} {'student':>9} {'games':>6} {'wr vs v1':>9} {'Wilson95':>18} {'SPRT':>10}")
    for (T, tag, n, wr, ci, ce, te, dec) in curve:
        print(f"  {T:>6}ms {tag:>9} {n:>6} {wr:>9.4f} [{ci[0]:.3f},{ci[1]:.3f}] {str(dec):>10}  err c{ce}/t{te}")
    print("\n>0.50 = student が v1 より強い(同 wall-time)。~0.50 = competitive(Case B)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
