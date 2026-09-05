"""ISMCTS v2.13 Phase J — Hard-Root Extension formal H2H(candidate vs parent、直接 ISMCTS-vs-ISMCTS)。

cand = v2.13 hard-ext(ISMCTS_CONFIG)、ctrl = v2.12 adaptive max-1350(ISMCTS_CONFIG_CTRL)。両者 ISMCTS。
唯一差分は「1350ms で未収束な hard root だけ same tree を最大 3000ms 継続」。driver.run_sprt_ab を source of
truth に SPRT で PROMOTE/FUTILITY 判定。今回は non-inferiority ではなく positive(strength gain)狙い。__main__ 必須。

使用例(desktop 15 workers 目安):
  python hardext_h2h.py --games 400 --workers 15
  python hardext_h2h.py --cand-config configs/ismcts_v2_13_hardext_t3000.json \
                        --ctrl-config configs/ismcts_v2_12_adaptive_t1350.json --games 800 --delta-min 0.03
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
    ap.add_argument("--cand-config", default=str(_HERE / "configs" / "ismcts_v2_13_hardext_t3000.json"))
    ap.add_argument("--ctrl-config", default=str(_HERE / "configs" / "ismcts_v2_12_adaptive_t1350.json"))
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--delta-min", type=float, default=0.03, help="SPRT delta_min(小=早期停止しにくく N 増=CI 締まる)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "hardext_h2h"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cand_cfg = Path(args.cand_config); ctrl_cfg = Path(args.ctrl_config)
    for p in (cand_cfg, ctrl_cfg):
        if not p.exists():
            print(f"[error] config 不在: {p}"); return 2
    os.environ["ISMCTS_CONFIG"] = str(cand_cfg)        # candidate = v2.13 hard-ext(default_agent)
    os.environ["ISMCTS_CONFIG_CTRL"] = str(ctrl_cfg)   # control   = v2.12 adaptive(default_agent_ctrl)
    deck = _load_deck()
    print(f"=== v2.13 Hard-Root Extension formal H2H ===")
    print(f"  cand = {cand_cfg.name}")
    print(f"  ctrl = {ctrl_cfg.name}")
    print(f"  games<= {args.games}  workers={args.workers}  delta_min={args.delta_min}  alpha={args.alpha} beta={args.beta}")
    cand = driver.AgentSpec("v2_13_hardext", agent_fn=ismcts_v1_agent.default_agent, label="v2_13_hardext_t3000")
    ctrl = driver.AgentSpec("v2_12_adaptive", agent_fn=ismcts_v1_agent.default_agent_ctrl, label="v2_12_adaptive_t1350")
    t0 = time.perf_counter()
    rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=args.delta_min, alpha=args.alpha, beta=args.beta,
                             n_max=10_000_000, max_games=args.games, alternate_sides=True,
                             run_dir=Path(args.run_dir), resume=args.resume, workers=args.workers,
                             progress_every=25, note="v2.13 hard-ext vs v2.12 adaptive")
    r = rep["result"]; el = time.perf_counter() - t0
    print(f"\n=== Phase J result [{el:.0f}s, {rep['sprt']['n']/(el/60):.1f} g/min] ===")
    print(f"  n={rep['sprt']['n']}  cand wins={r['candidate_wins']}  winrate={r['candidate_winrate']:.4f}")
    print(f"  Wilson95 CI = [{r['wilson95_ci'][0]:.3f}, {r['wilson95_ci'][1]:.3f}]")
    print(f"  SPRT decision = {rep['decision']}   errors cand/ctrl = {r['cand_errors']}/{r['ctrl_errors']}")
    print("\n  Primary は winrate(strength gain)。PROMOTE=Case A、CI が 0.5 跨ぎ広い=Case B、")
    print("  0.5 近傍で締まる=Case C、有意に <0.5=Case D。err>0 は Case F(runtime/健全性)要調査。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
