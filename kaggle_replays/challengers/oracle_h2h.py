"""ISMCTS v2.9 Oracle H2H — 各 arm(ALL/V/Q)vs Original Policy-only(control)の勝率診断。

candidate = oracle_agent(ORACLE_ARM env、MAIN root で H4@128 search→gate→Search or Original action)。
control = abl_2_policy_only(純 Original Policy、search なし。search は side-effect 無し=v1 F7 済ゆえ winrate 等価、
compute 半減)。gate 発火率は別途 --measure-gates(instrumented 単プロセス)で計測。
driver.run_sprt_ab 再利用。__main__ ガード必須。

使用: python oracle_h2h.py --arms ALL,V,Q --games 800 --workers 15
      python oracle_h2h.py --measure-gates --games 6   # gate 発火率のみ(単プロセス)
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
import oracle_agent  # noqa: E402


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _measure_gates(n_games):
    """単プロセスで各 arm の gate 発火率 / fires-per-game を計測(ORACLE_LOG)。"""
    from cg.api import to_observation_class, SelectType
    from cg.game import battle_start, battle_select, battle_finish
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    from ptcg_ai.hidden_information import match_context
    os.chdir(_SUB); os.environ["ORACLE_LOG"] = "1"
    deck = _load_deck()
    for arm in ("ALL", "V", "Q"):
        os.environ["ORACLE_ARM"] = arm
        oracle_agent._GATE_LOG.clear()
        main_roots = 0; fires = 0; games_with_fire = 0
        for g in range(n_games):
            match_context.reset()
            od, sd = battle_start(deck, deck)
            if sd.errorType != 0:
                continue
            before = len(oracle_agent._GATE_LOG)
            for _ in range(400):
                obs = to_observation_class(od); cur = obs.current
                if cur is None or cur.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                act = oracle_agent.oracle_agent(obs)
                od = battle_select(act)
            battle_finish()
            logs = oracle_agent._GATE_LOG[before:]
            gf = sum(1 for r in logs if r["arm"] == arm and r["fires"])
            if gf > 0:
                games_with_fire += 1
        logs = [r for r in oracle_agent._GATE_LOG if r["arm"] == arm]
        main_roots = len(logs); fires = sum(1 for r in logs if r["fires"])
        changed = sum(1 for r in logs if r["changed"])
        print(f"  {arm}: MAIN roots={main_roots} CHANGED={changed}({changed/max(main_roots,1):.1%}) "
              f"gate_fires={fires}({fires/max(main_roots,1):.1%}) fires/game={fires/max(n_games,1):.2f} "
              f"games_with_fire={games_with_fire}/{n_games}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ALL,V,Q")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--measure-gates", action="store_true")
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "oracle"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.measure_gates:
        print(f"=== gate 発火率計測({args.games} games/arm, 単プロセス)===")
        _measure_gates(args.games)
        return 0
    deck = _load_deck()
    ctrl = driver.AgentSpec("abl_2_policy_only", label="original_policy")   # 純 Original(search なし)
    run_root = Path(args.run_dir)
    print(f"=== v2.9 Oracle H2H vs Original Policy-only: arms={args.arms} games={args.games} workers={args.workers} ===")
    print(f"    Teacher = H4@128 fixed iters (search side-effect 無し=winrate 等価, control search 省略)")
    curve = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        os.environ["ORACLE_ARM"] = arm
        cand = driver.AgentSpec("oracle", agent_fn=oracle_agent.oracle_agent, label=f"oracle_{arm}")
        t0 = time.perf_counter()
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.03, alpha=0.05, beta=0.10,
                                 n_max=10_000_000, max_games=args.games, alternate_sides=True,
                                 run_dir=run_root / f"arm_{arm}", resume=False, workers=args.workers,
                                 progress_every=50, note=f"oracle {arm} vs original")
        r = rep["result"]; el = time.perf_counter() - t0
        curve.append((arm, rep["sprt"]["n"], r["candidate_winrate"], r["wilson95_ci"],
                      r["cand_errors"], r["ctrl_errors"], rep["decision"]))
        print(f"  {arm}: {r['candidate_wins']}/{rep['sprt']['n']} wr={r['candidate_winrate']:.4f} "
              f"CI{r['wilson95_ci']} dec={rep['decision']} err={r['cand_errors']}/{r['ctrl_errors']} [{el:.0f}s]")

    print("\n=== Oracle upper-bound(gate 発火時 Search 100%採用 vs Original Policy-only)===")
    print(f"  {'arm':>5} {'games':>6} {'winrate':>8} {'Wilson95':>18} {'Δ vs 0.50':>10} {'SPRT':>10}")
    for (arm, n, wr, ci, ce, te, dec) in curve:
        print(f"  {arm:>5} {n:>6} {wr:>8.4f} [{ci[0]:.3f},{ci[1]:.3f}] {(wr-0.5)*100:>+8.1f}pt {str(dec):>10} err c{ce}/t{te}")
    print("\nALL≈0.50 → correction subset 自体に online 価値なし(Case A)。ALL>0.50 → 価値あり(gate/learning が律速)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
