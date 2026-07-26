"""Lucario Opponent Promotion Gate v1 — online mirror orchestrator(Phase D1/D2)。

frozen contract(lucario_opponent_gate_v1.json, hash a058f5a05e9fb0ce)に従い、5 recipe で
Candidate(Lucario BC)vs Control(generic)を **per-recipe SPRT**(measurement driver.run_sprt_ab 再利用)。
recipe-uniform + stratified aggregate・collapse guard・PROMOTE/FAIL/REVIEW を frozen 契約のみで判定。
D0 で全 hash(gate/weights/config/recipe/cg.dll)を検証。Windows spawn 対策で __main__ ガード必須。
ローカル専用・git 未追跡・production/cg/shared/opponents 非改変(import して呼ぶだけ)。

使用: python lucario_mirror.py --workers 15 [--n-max 600] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MEASUREMENT = _HERE.parent / "measurement"
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_MEASUREMENT))

import agents  # noqa: E402  (measurement/agents.py)
import driver  # noqa: E402
import lucario_gate as LG  # noqa: E402

_LEARN = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
_ARCH = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "mega_lucario_ex"
_LUCARIO_BC = _LEARN / "policy_weights_mega_lucario_ex.json"
_GENERIC = _LEARN / "policy_weights.json"
EXPECTED_CG = "c7c87eb76513784b"
OFFLINE_MET = True   # contract.decision.offline_precondition.status = MET(+26.4pt), read-only 検証済


def _load_deck(path) -> list[int]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _p0p1(run_dir: Path) -> tuple[float | None, float | None]:
    """outcomes.jsonl から candidate の P0/P1 別勝率(診断)。"""
    p = run_dir / "outcomes.jsonl"
    if not p.exists():
        return None, None
    p0 = p1 = None
    w0 = n0 = w1 = n1 = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("cand_side") == 0:
            n0 += 1; w0 += int(r.get("cand_won", 0))
        else:
            n1 += 1; w1 += int(r.get("cand_won", 0))
    if n0:
        p0 = round(w0 / n0, 4)
    if n1:
        p1 = round(w1 / n1, 4)
    return p0, p1


def _d0_integrity(contract: dict) -> list[str]:
    errs = LG.verify_frozen()
    if LG.file_hash(_LUCARIO_BC) != contract["candidate"]["weights_hash"]:
        errs.append("candidate(Lucario BC) weights hash 不一致")
    if LG.file_hash(_GENERIC) != contract["control"]["weights_hash"]:
        errs.append("control(generic) weights hash 不一致")
    cfg = agents._SUB / "configs" / "abl_2_policy_only.json"
    if LG.file_hash(cfg) != contract["candidate"]["config_hash"]:
        errs.append("abl_2_policy_only config hash 不一致")
    cg = agents._SUB / "cg" / "cg.dll"
    if not cg.exists() or LG.file_hash(cg) != EXPECTED_CG:
        errs.append(f"cg.dll hash 不一致({LG.file_hash(cg) if cg.exists() else 'missing'})")
    for r in contract["recipes"]["list"]:
        path = _ARCH / f"{r['recipe_id']}.csv"
        if not path.exists() or LG.file_hash(path) != r["deck_hash"]:
            errs.append(f"recipe {r['recipe_id']} deck hash 不一致")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--n-max", type=int, default=None, help="既定は contract の sprt.n_max")
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "lucario_mirror_v1"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    contract = LG.load_contract()
    d0 = _d0_integrity(contract)
    if d0:
        print(f"[ABORT] D0 integrity 失敗: {d0}")
        return 1
    agents.ensure_production_cwd()

    sp = contract["sprt"]
    n_max = args.n_max or sp["n_max"]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Lucario Opponent Mirror v1  gate=a058f5a0  workers={args.workers}  n_max/recipe={n_max} ===")
    print(f"    Candidate=Lucario BC(c4ca3599)  Control=generic(735dd38a)  config=abl_2_policy_only")

    per_recipe = []
    t0 = time.perf_counter()
    for r in contract["recipes"]["list"]:
        rid = r["recipe_id"]
        deck = _load_deck(_ARCH / f"{rid}.csv")
        cand = driver.AgentSpec("abl_2_policy_only", policy_weights_path=str(_LUCARIO_BC), label="lucario_bc")
        ctrl = driver.AgentSpec("abl_2_policy_only", policy_weights_path=str(_GENERIC), label="generic")
        rdir = run_dir / f"recipe_{rid}"
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=sp["delta_min"], alpha=sp["alpha"],
                                 beta=sp["beta"], n_max=n_max, alternate_sides=True, run_dir=rdir,
                                 resume=args.resume, workers=args.workers, progress_every=0,
                                 note=f"lucario_opponent_promotion recipe={rid}")
        res = rep["result"]
        p0, p1 = _p0p1(rdir)
        entry = {"recipe_id": rid, "deck_hash": r["deck_hash"],
                 "n": rep["sprt"]["n"], "wins": rep["sprt"]["s"], "losses": rep["sprt"]["n"] - rep["sprt"]["s"],
                 "winrate": (res["candidate_winrate"] or 0.0), "wilson95_ci": res["wilson95_ci"],
                 "sprt_decision": rep["decision"], "llr": rep["sprt"]["llr"],
                 "cand_P0_winrate": p0, "cand_P1_winrate": p1,
                 "cand_errors": res["cand_errors"], "ctrl_errors": res["ctrl_errors"],
                 "terminal": rep["channel_breakdown"]}
        per_recipe.append(entry)
        print(f"  recipe {rid}: {entry['wins']}/{entry['n']} wr={entry['winrate']:.3f} "
              f"CI{entry['wilson95_ci']} {entry['sprt_decision']} llr={entry['llr']:.2f} "
              f"P0/P1={p0}/{p1} err={entry['cand_errors']}/{entry['ctrl_errors']} "
              f"[{time.perf_counter()-t0:.0f}s]")

    # aggregate + collapse + decision(frozen contract)
    agg = LG.aggregate(per_recipe)
    collapse = LG.collapse_check(per_recipe, contract)
    total_games = sum(e["n"] for e in per_recipe)
    total_err = sum(e["cand_errors"] + e["ctrl_errors"] for e in per_recipe)
    error_rate = total_err / max(1, total_games)
    integrity_ok = error_rate <= 0.01
    decision = LG.decide(OFFLINE_MET, per_recipe, agg, collapse, integrity_ok, contract)

    report = {
        "kind": "lucario_opponent_mirror_v1", "run_type": contract["run_type"],
        "gate_hash": LG.EXPECTED_GATE_HASH, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "candidate": contract["candidate"], "control": contract["control"],
        "offline_precondition": contract["decision"]["offline_precondition"],
        "per_recipe": per_recipe, "aggregate": agg,
        "collapse": collapse, "total_games": total_games, "errors": total_err,
        "error_rate": round(error_rate, 5), "integrity_ok": integrity_ok,
        "decision": decision,
    }
    (run_dir / "lucario_mirror_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- aggregate ---")
    print(f"  recipe-uniform point={agg['point']:.4f} se={agg['se']:.4f} lower95={agg['one_sided_95_lower']:.4f}")
    print(f"  point>=0.53: {decision['aggregate_point_ge_053']}  lower>0.50: {decision['aggregate_lower_gt_050']}")
    print(f"  PROMOTE recipes: {decision['recipe_requirement']}  collapse: {collapse}  integrity_ok: {integrity_ok}")
    print(f"\n  DECISION: {decision['decision']}  ({decision['reason']})")
    print(f"  report: {run_dir / 'lucario_mirror_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
