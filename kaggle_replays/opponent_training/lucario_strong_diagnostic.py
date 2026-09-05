"""Phase D3: Strong-Opponent Diagnostic — abl_5_full(Alakazam Champion)vs Lucario Champion v1(BC)。

frozen contract の strong_opponent_diagnostic に従い、各 mega_lucario recipe で fixed-N(既定200)対戦し
**Alakazam の対 strong-Lucario 勝率**を測る(**単一 pool SPRT にしない=fixed-N diagnostic**)。
Reference Pool v1 の 0.820(vs generic Lucario)は不変=これは別保存の診断。低下は「opponent が強くなった」証拠。

field_eval の flat single-pool(model load 償却)を再利用。deck0(Alakazam deck.csv)≠deck1(Lucario recipe)、
手番交互、per-game match_context.reset()。ローカル専用・git 未追跡・production/cg/shared 非改変。

使用: python lucario_strong_diagnostic.py --workers 15 [--games 200]
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
import lucario_gate as LG  # noqa: E402
from driver import resolve_workers  # noqa: E402
from sprt import wilson_interval  # noqa: E402

_LEARN = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
_ARCH = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "mega_lucario_ex"
_LUCARIO_BC = _LEARN / "policy_weights_mega_lucario_ex.json"
_CONFIG = agents._SUB / "configs" / "abl_2_policy_only.json"
Z95_1S = 1.6448536269514722
BASELINE_VS_GENERIC = 0.820   # Reference Pool v1(frozen, 不変)


def _load_deck(path) -> list[int]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _d0(contract) -> list[str]:
    errs = LG.verify_frozen()
    cg = agents._SUB / "cg" / "cg.dll"
    if not cg.exists() or LG.file_hash(cg) != "c7c87eb76513784b":
        errs.append("cg.dll hash 不一致")
    if LG.file_hash(_LUCARIO_BC) != contract["candidate"]["weights_hash"]:
        errs.append("Lucario BC weights hash 不一致")
    for r in contract["recipes"]["list"]:
        p = _ARCH / f"{r['recipe_id']}.csv"
        if not p.exists() or LG.file_hash(p) != r["deck_hash"]:
            errs.append(f"recipe {r['recipe_id']} hash 不一致")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--games", type=int, default=200, help="recipe あたり games(diagnostic)")
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "lucario_strong_diagnostic_v1"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    contract = LG.load_contract()
    d0 = _d0(contract)
    if d0:
        print(f"[ABORT] D0 integrity 失敗: {d0}")
        return 1
    agents.ensure_production_cwd()
    own_deck = _load_deck(agents._SUB / "deck.csv")   # Alakazam Champion のデッキ

    # Lucario Champion v1 を opponent(policy-only + BC 重み)として member 化。
    member = {"opponent_id": "lucario_champion_v1", "archetype": "mega_lucario_ex",
              "agent_type": "policy_only", "config_path": str(_CONFIG), "weights_path": str(_LUCARIO_BC)}
    members = [member]
    recipes = contract["recipes"]["list"]

    # flat tasks: evaluated=abl_5_full(Alakazam), opponent=Lucario Champion v1、各 recipe args.games。
    tasks = []
    for r in recipes:
        path = str(_ARCH / f"{r['recipe_id']}.csv")
        for gi in range(args.games):
            tasks.append(("abl_5_full", "candidate", "lucario_champion_v1", r["recipe_id"], path, gi, gi % 2))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    nw = resolve_workers(args.workers)
    print(f"=== Strong-Opponent Diagnostic: abl_5_full(Alakazam) vs Lucario Champion v1 ===")
    print(f"    games/recipe={args.games}  recipes={len(recipes)}  total={len(tasks)}  workers={nw}")
    print(f"    baseline(vs generic Lucario, frozen)= {BASELINE_VS_GENERIC}")

    records = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nw, initializer=field_eval._flat_worker_init,
                             initargs=(own_deck, members)) as ex:
        seen = 0
        for rec in ex.map(field_eval._flat_play, tasks, chunksize=1):
            records.append(rec); seen += 1
            if seen % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {seen}/{len(tasks)} [{el:.0f}s, {seen/(el/60):.1f} g/min]")

    # per-recipe Alakazam(evaluated)勝率
    per_recipe = []
    for r in recipes:
        rid = r["recipe_id"]
        rr = [x for x in records if x["recipe_id"] == rid]
        nat = [x for x in rr if not x.get("error")]
        wins = sum(x["evaluated_won"] for x in nat)
        n = len(nat)
        wr = wins / n if n else None
        lo, hi = wilson_interval(wins, n) if n else (0.0, 1.0)
        p0 = [x for x in nat if x["evaluated_side"] == 0]
        p1 = [x for x in nat if x["evaluated_side"] == 1]
        per_recipe.append({
            "recipe_id": rid, "n": n, "alakazam_wins": wins, "alakazam_winrate": wr,
            "wilson95_ci": [round(lo, 4), round(hi, 4)],
            "alakazam_P0_winrate": (round(sum(x["evaluated_won"] for x in p0) / len(p0), 4) if p0 else None),
            "alakazam_P1_winrate": (round(sum(x["evaluated_won"] for x in p1) / len(p1), 4) if p1 else None),
            "errors": sum(1 for x in rr if x.get("error")),
            "terminal": dict(Counter(x.get("primary", "unknown") for x in nat)),
        })

    # aggregate: recipe-uniform Alakazam 勝率 + stratified CI
    k = len(per_recipe)
    point = sum(e["alakazam_winrate"] for e in per_recipe) / k
    var = sum((1.0 / k) ** 2 * (e["alakazam_winrate"] * (1 - e["alakazam_winrate"]) / e["n"]) for e in per_recipe)
    se = math.sqrt(var)
    total_games = sum(e["n"] for e in per_recipe)
    total_err = sum(e["errors"] for e in per_recipe)

    report = {
        "kind": "lucario_strong_opponent_diagnostic_v1", "run_type": "strong_opponent_diagnostic",
        "gate_hash": LG.EXPECTED_GATE_HASH, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "matchup": "abl_5_full(Alakazam Champion) vs Lucario Champion v1(BC, policy-only)",
        "games_per_recipe": args.games, "total_games": total_games, "errors": total_err,
        "error_rate": round(total_err / max(1, total_games), 5),
        "per_recipe": per_recipe,
        "aggregate_alakazam_winrate": round(point, 4),
        "aggregate_stratified_se": round(se, 4),
        "aggregate_wilson_like_ci95": [round(point - 1.959964 * se, 4), round(point + 1.959964 * se, 4)],
        "baseline_vs_generic_lucario_frozen": BASELINE_VS_GENERIC,
        "delta_vs_baseline": round(point - BASELINE_VS_GENERIC, 4),
        "interpretation": "低下は Lucario opponent が強くなった証拠(Alakazam の弱化ではない)。Reference Pool v1 の 0.820 は不変。",
    }
    (run_dir / "strong_diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- per-recipe(Alakazam winrate)---")
    for e in per_recipe:
        print(f"  {e['recipe_id']}: {e['alakazam_wins']}/{e['n']} wr={e['alakazam_winrate']:.3f} "
              f"CI{e['wilson95_ci']} P0/P1={e['alakazam_P0_winrate']}/{e['alakazam_P1_winrate']} "
              f"term={e['terminal']} err={e['errors']}")
    print("\n--- aggregate ---")
    print(f"  Alakazam vs Lucario Champion v1: {point:.4f}  (±SE {se:.4f})")
    print(f"  vs frozen baseline (generic Lucario) {BASELINE_VS_GENERIC}: Δ = {point-BASELINE_VS_GENERIC:+.4f}")
    print(f"  errors={total_err}/{total_games} error_rate={report['error_rate']}")
    print(f"  report: {run_dir / 'strong_diagnostic_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
