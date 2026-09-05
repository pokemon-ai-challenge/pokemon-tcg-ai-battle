"""汎用 Opponent Promotion online mirror orchestrator(archetype 非依存・Phase B)。

--gate <contract.json> を source of truth に、archetype-specific BC vs generic を per-recipe SPRT
(measurement driver.run_sprt_ab 再利用)。recipe-uniform + stratified aggregate・collapse・
PROMOTE/FAIL/REVIEW を opponent_gate で判定。D0 で contract/weights/config/recipe/cg.dll hash 検証。
Windows spawn 対策で __main__ ガード必須。ローカル専用・production/cg/shared 非改変。

使用: python opponent_mirror.py --gate contracts/archaludon_ex_opponent_gate_v1.json \
        --expected-hash 2d087f8f074e9c96 --workers 15
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

import agents  # noqa: E402
import driver  # noqa: E402
import opponent_gate as OG  # noqa: E402

EXPECTED_CG = "c7c87eb76513784b"


def _load_deck(path) -> list[int]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _p0p1(run_dir: Path) -> tuple[float | None, float | None]:
    p = run_dir / "outcomes.jsonl"
    if not p.exists():
        return None, None
    w0 = n0 = w1 = n1 = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("cand_side") == 0:
            n0 += 1; w0 += int(r.get("cand_won", 0))
        else:
            n1 += 1; w1 += int(r.get("cand_won", 0))
    return (round(w0 / n0, 4) if n0 else None), (round(w1 / n1, 4) if n1 else None)


def _d0(contract: dict) -> list[str]:
    errs = []
    cand_w = _ROOT / contract["candidate"]["weights_path"]
    ctrl_w = _ROOT / contract["control"]["weights_path"]
    if OG.file_hash(cand_w) != contract["candidate"]["weights_hash"]:
        errs.append("candidate weights hash 不一致")
    if OG.file_hash(ctrl_w) != contract["control"]["weights_hash"]:
        errs.append("control weights hash 不一致")
    cfg = agents._SUB / "configs" / f"{contract['candidate']['config']}.json"
    if OG.file_hash(cfg) != contract["candidate"]["config_hash"]:
        errs.append("config hash 不一致")
    cg = agents._SUB / "cg" / "cg.dll"
    if not cg.exists() or OG.file_hash(cg) != EXPECTED_CG:
        errs.append("cg.dll hash 不一致")
    for r in contract["recipes"]["list"]:
        p = _ROOT / r["path"]
        if not p.exists() or OG.file_hash(p) != r["deck_hash"]:
            errs.append(f"recipe {r['recipe_id']} hash 不一致")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="contract JSON path")
    ap.add_argument("--expected-hash", default=None, help="frozen gate hash(freeze 検証)")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gate_path = Path(args.gate).resolve()   # chdir 前に絶対化(ensure_production_cwd 対策)
    contract = OG.load_contract(gate_path)
    arch = contract["archetype"]
    if args.expected_hash:
        fe = OG.verify_frozen(gate_path, args.expected_hash)
        if fe:
            print(f"[ABORT] {fe}")
            return 1
    d0 = _d0(contract)
    if d0:
        print(f"[ABORT] D0 integrity 失敗: {d0}")
        return 1
    agents.ensure_production_cwd()

    sp = contract["sprt"]
    n_max = args.n_max or sp["n_max"]
    run_dir = Path(args.run_dir or (_HERE / "results" / arch / "mirror_v1"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cand_w = str(_ROOT / contract["candidate"]["weights_path"])
    ctrl_w = str(_ROOT / contract["control"]["weights_path"])
    offline_met = str(contract["decision"]["offline_precondition"].get("status", "")).startswith("MET")

    print(f"=== {arch} Opponent Mirror v1  gate={OG.file_hash(gate_path)}  workers={args.workers}  n_max={n_max} ===")
    print(f"    Candidate={contract['candidate']['weights_hash']}  Control={contract['control']['weights_hash']}  offline_met={offline_met}")

    per_recipe = []
    t0 = time.perf_counter()
    for r in contract["recipes"]["list"]:
        rid = r["recipe_id"]
        deck = _load_deck(_ROOT / r["path"])
        cand = driver.AgentSpec(contract["candidate"]["config"], policy_weights_path=cand_w, label=f"{arch}_bc")
        ctrl = driver.AgentSpec(contract["control"]["config"], policy_weights_path=ctrl_w, label="generic")
        rdir = run_dir / f"recipe_{rid}"
        rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=sp["delta_min"], alpha=sp["alpha"],
                                 beta=sp["beta"], n_max=n_max, alternate_sides=True, run_dir=rdir,
                                 resume=args.resume, workers=args.workers, progress_every=0,
                                 note=f"{contract['run_type']} recipe={rid}")
        res = rep["result"]
        p0, p1 = _p0p1(rdir)
        e = {"recipe_id": rid, "deck_hash": r["deck_hash"], "n": rep["sprt"]["n"], "wins": rep["sprt"]["s"],
             "losses": rep["sprt"]["n"] - rep["sprt"]["s"], "winrate": (res["candidate_winrate"] or 0.0),
             "wilson95_ci": res["wilson95_ci"], "sprt_decision": rep["decision"], "llr": rep["sprt"]["llr"],
             "cand_P0_winrate": p0, "cand_P1_winrate": p1, "cand_errors": res["cand_errors"],
             "ctrl_errors": res["ctrl_errors"], "terminal": rep["channel_breakdown"]}
        per_recipe.append(e)
        print(f"  recipe {rid}: {e['wins']}/{e['n']} wr={e['winrate']:.3f} CI{e['wilson95_ci']} "
              f"{e['sprt_decision']} llr={e['llr']:.2f} P0/P1={p0}/{p1} err={e['cand_errors']}/{e['ctrl_errors']} "
              f"[{time.perf_counter()-t0:.0f}s]")

    agg = OG.aggregate(per_recipe)
    collapse = OG.collapse_check(per_recipe, contract)
    total_games = sum(e["n"] for e in per_recipe)
    total_err = sum(e["cand_errors"] + e["ctrl_errors"] for e in per_recipe)
    integrity_ok = (total_err / max(1, total_games)) <= 0.01
    decision = OG.decide(offline_met, per_recipe, agg, collapse, integrity_ok, contract)

    report = {"kind": "opponent_mirror_v1", "archetype": arch, "run_type": contract["run_type"],
              "gate_hash": OG.file_hash(gate_path), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "elapsed_sec": round(time.perf_counter() - t0, 1),
              "candidate": contract["candidate"], "control": contract["control"],
              "offline_precondition": contract["decision"]["offline_precondition"],
              "per_recipe": per_recipe, "aggregate": agg, "collapse": collapse,
              "total_games": total_games, "errors": total_err,
              "error_rate": round(total_err / max(1, total_games), 5), "integrity_ok": integrity_ok,
              "decision": decision}
    (run_dir / f"{arch}_mirror_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- aggregate ---")
    print(f"  recipe-uniform point={agg['point']:.4f} se={agg['se']:.4f} lower95={agg['one_sided_95_lower']:.4f}")
    print(f"  point>=0.53: {decision['aggregate_point_ge_practical']}  lower>0.50: {decision['aggregate_lower_gt_050']}")
    print(f"  PROMOTE recipes: {decision['recipe_requirement']}  collapse: {collapse}  integrity: {integrity_ok}")
    print(f"\n  DECISION: {decision['decision']}  ({decision['reason']})")
    print(f"  report: {run_dir / f'{arch}_mirror_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
