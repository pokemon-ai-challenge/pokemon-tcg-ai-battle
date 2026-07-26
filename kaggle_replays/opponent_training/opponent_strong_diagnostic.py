"""汎用 Strong-Opponent Diagnostic(archetype 非依存・Phase B/D3)。

--gate <contract.json> に従い abl_5_full(Alakazam Champion)vs archetype Champion v1(BC)を各 recipe で
fixed-N 対戦し **Alakazam 勝率**を測る(単一 pool SPRT にしない=fixed-N diagnostic)。frozen baseline
(vs generic)は不変=別保存の診断。field_eval の flat single-pool 再利用。deck0≠deck1・手番交互・per-game reset。
ローカル専用・production/cg/shared 非改変。

使用: python opponent_strong_diagnostic.py --gate contracts/archaludon_ex_opponent_gate_v1.json \
        --expected-hash 2d087f8f074e9c96 --workers 15 --games 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MEASUREMENT = _HERE.parent / "measurement"
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_MEASUREMENT))

import agents  # noqa: E402
import field_eval  # noqa: E402
import opponent_gate as OG  # noqa: E402
from driver import resolve_workers  # noqa: E402
from sprt import wilson_interval  # noqa: E402

EXPECTED_CG = "c7c87eb76513784b"


def _load_deck(path) -> list[int]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--expected-hash", default=None)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gate_path = Path(args.gate).resolve()   # chdir 前に絶対化(ensure_production_cwd 対策)
    contract = OG.load_contract(gate_path)
    arch = contract["archetype"]
    if args.expected_hash and OG.verify_frozen(gate_path, args.expected_hash):
        print(f"[ABORT] gate hash 不一致")
        return 1
    diag = contract["strong_opponent_diagnostic"]
    baseline = diag["baseline_vs_generic"]
    bc_weights = _ROOT / contract["candidate"]["weights_path"]

    # D0
    cg = agents._SUB / "cg" / "cg.dll"
    errs = []
    if not cg.exists() or OG.file_hash(cg) != EXPECTED_CG:
        errs.append("cg.dll hash 不一致")
    if OG.file_hash(bc_weights) != contract["candidate"]["weights_hash"]:
        errs.append("BC weights hash 不一致")
    for r in contract["recipes"]["list"]:
        p = _ROOT / r["path"]
        if not p.exists() or OG.file_hash(p) != r["deck_hash"]:
            errs.append(f"recipe {r['recipe_id']} hash 不一致")
    if errs:
        print(f"[ABORT] D0: {errs}")
        return 1
    agents.ensure_production_cwd()

    own_deck = _load_deck(agents._SUB / "deck.csv")
    member = {"opponent_id": f"{arch}_champion_v1", "archetype": arch, "agent_type": "policy_only",
              "config_path": str(agents._SUB / "configs" / f"{contract['candidate']['config']}.json"),
              "weights_path": str(bc_weights)}
    recipes = contract["recipes"]["list"]
    tasks = []
    for r in recipes:
        path = str(_ROOT / r["path"])
        for gi in range(args.games):
            tasks.append(("abl_5_full", "candidate", member["opponent_id"], r["recipe_id"], path, gi, gi % 2))

    run_dir = Path(args.run_dir or (_HERE / "results" / arch / "strong_diagnostic_v1"))
    run_dir.mkdir(parents=True, exist_ok=True)
    nw = resolve_workers(args.workers)
    print(f"=== {arch} Strong-Opponent Diagnostic: abl_5_full(Alakazam) vs {arch} Champion v1 ===")
    print(f"    games/recipe={args.games} total={len(tasks)} workers={nw}  baseline(vs generic, frozen)={baseline}")

    records = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nw, initializer=field_eval._flat_worker_init,
                             initargs=(own_deck, [member])) as ex:
        seen = 0
        for rec in ex.map(field_eval._flat_play, tasks, chunksize=1):
            records.append(rec); seen += 1
            if seen % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {seen}/{len(tasks)} [{el:.0f}s, {seen/(el/60):.1f} g/min]")

    per_recipe = []
    for r in recipes:
        rid = r["recipe_id"]
        rr = [x for x in records if x["recipe_id"] == rid]
        nat = [x for x in rr if not x.get("error")]
        wins = sum(x["evaluated_won"] for x in nat); n = len(nat)
        lo, hi = wilson_interval(wins, n) if n else (0.0, 1.0)
        p0 = [x for x in nat if x["evaluated_side"] == 0]
        p1 = [x for x in nat if x["evaluated_side"] == 1]
        per_recipe.append({"recipe_id": rid, "n": n, "alakazam_wins": wins,
                           "alakazam_winrate": (wins / n if n else None), "wilson95_ci": [round(lo, 4), round(hi, 4)],
                           "alakazam_P0_winrate": (round(sum(x["evaluated_won"] for x in p0) / len(p0), 4) if p0 else None),
                           "alakazam_P1_winrate": (round(sum(x["evaluated_won"] for x in p1) / len(p1), 4) if p1 else None),
                           "errors": sum(1 for x in rr if x.get("error")),
                           "terminal": dict(Counter(x.get("primary", "unknown") for x in nat))})

    k = len(per_recipe)
    point = sum(e["alakazam_winrate"] for e in per_recipe) / k
    se = math.sqrt(sum((1.0 / k) ** 2 * (e["alakazam_winrate"] * (1 - e["alakazam_winrate"]) / e["n"]) for e in per_recipe))
    total_games = sum(e["n"] for e in per_recipe)
    total_err = sum(e["errors"] for e in per_recipe)

    report = {"kind": "opponent_strong_diagnostic_v1", "archetype": arch, "run_type": "strong_opponent_diagnostic",
              "gate_hash": OG.file_hash(gate_path), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "matchup": f"abl_5_full(Alakazam) vs {arch} Champion v1(BC)",
              "games_per_recipe": args.games, "total_games": total_games, "errors": total_err,
              "error_rate": round(total_err / max(1, total_games), 5), "per_recipe": per_recipe,
              "aggregate_alakazam_winrate": round(point, 4), "aggregate_stratified_se": round(se, 4),
              "baseline_vs_generic_frozen": baseline, "delta_vs_baseline": round(point - baseline, 4),
              "interpretation": "低下は opponent が強くなった証拠(Alakazam 弱化ではない)。baseline は不変。"}
    (run_dir / f"{arch}_strong_diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- per-recipe(Alakazam winrate)---")
    for e in per_recipe:
        print(f"  {e['recipe_id']}: {e['alakazam_wins']}/{e['n']} wr={e['alakazam_winrate']:.3f} "
              f"CI{e['wilson95_ci']} P0/P1={e['alakazam_P0_winrate']}/{e['alakazam_P1_winrate']} "
              f"term={e['terminal']} err={e['errors']}")
    print(f"\n  Alakazam vs {arch} Champion v1: {point:.4f} (±SE {se:.4f})")
    print(f"  vs frozen baseline {baseline}: Δ = {point-baseline:+.4f}   errors={total_err}/{total_games}")
    print(f"  report: {run_dir / f'{arch}_strong_diagnostic_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
