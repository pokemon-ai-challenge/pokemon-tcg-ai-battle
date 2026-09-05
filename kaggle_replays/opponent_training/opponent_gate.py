"""汎用 Opponent Promotion Gate 判定ロジック(archetype 非依存・cg 不使用)。

lucario_gate.py の判定数式を archetype 非依存へ一般化した再利用部品(Phase B)。閾値は contract JSON
から読む(hard-code しない)。**Lucario の frozen artifact(lucario_gate.py/lucario_opponent_gate_v1.json)は
変更しない**=historical 保持。本モジュールは archaludon 以降が使用。ローカル専用・git 未追跡。

契約の標準(v1、結果前 freeze):
  per-recipe SPRT δ_min0.03/α.05/β.10/n_max600 / aggregate recipe-uniform: point≥0.53 ∧ 片側95%下限>0.50 /
  recipe_requirement ≥4/5 PROMOTE / collapse: 勝率<0.45 ∧ Wilson95上端<0.50 / integrity error≤1%。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MEASUREMENT = _HERE.parent / "measurement"
sys.path.insert(0, str(_MEASUREMENT))
from sprt import wilson_interval  # noqa: E402

Z95_1S = 1.6448536269514722


def file_hash(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load_contract(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen(path, expected_hash: str) -> list[str]:
    h = file_hash(path)
    return [] if h == expected_hash else [f"gate hash 不一致: {h} != {expected_hash}"]


def aggregate(per_recipe: list[dict]) -> dict:
    """recipe-uniform mean winrate + stratified Wald 片側95%下限。Var=Σ(1/k)² p_r(1−p_r)/n_r。"""
    k = len(per_recipe)
    point = sum(r["winrate"] for r in per_recipe) / k
    var = sum((1.0 / k) ** 2 * (r["winrate"] * (1 - r["winrate"]) / r["n"] if r["n"] else 0.0)
              for r in per_recipe)
    se = math.sqrt(var)
    return {"weighting": "recipe_uniform", "point": point, "se": se,
            "one_sided_95_lower": point - Z95_1S * se, "k": k}


def collapse_check(per_recipe: list[dict], contract: dict) -> list[str]:
    cg = contract["collapse_guard"]
    pt_lt, up_lt = cg["point_winrate_lt"], cg["wilson95_upper_lt"]
    fired = []
    for r in per_recipe:
        lo, hi = wilson_interval(r["wins"], r["n"]) if r["n"] else (0.0, 1.0)
        if r["winrate"] < pt_lt and hi < up_lt:
            fired.append(r["recipe_id"])
    return fired


def decide(offline_met: bool, per_recipe: list[dict], agg: dict, collapse: list[str],
           integrity_ok: bool, contract: dict) -> dict:
    total = contract["recipe_requirement"]["total"]
    need = contract["recipe_requirement"]["promote_count_min"]
    pt_thr = contract["aggregate"]["practical_threshold"]
    promote_n = sum(1 for r in per_recipe if r.get("sprt_decision") == "PROMOTE")
    futility_n = sum(1 for r in per_recipe if r.get("sprt_decision") == "FUTILITY")
    point_ok = agg["point"] >= pt_thr
    stat_ok = agg["one_sided_95_lower"] > 0.50

    if not integrity_ok:
        decision, reason = "FAIL", "integrity failure"
    elif collapse:
        decision, reason = "FAIL", f"collapse guard fired: {collapse}"
    elif futility_n > total // 2:
        decision, reason = "FAIL", f"過半 recipe が FUTILITY({futility_n}/{total})"
    elif offline_met and promote_n >= need and point_ok and stat_ok:
        decision, reason = "PROMOTE", "全条件 MET"
    else:
        decision, reason = "REVIEW", "borderline(PROMOTE 条件未達だが明確な FAIL でもない)"

    return {"decision": decision, "reason": reason, "offline_met": offline_met,
            "promote_recipes": promote_n, "futility_recipes": futility_n,
            "recipe_requirement": f"{promote_n}/{total} (need >={need})",
            "aggregate_point": agg["point"], "aggregate_point_ge_practical": point_ok,
            "aggregate_lower": agg["one_sided_95_lower"], "aggregate_lower_gt_050": stat_ok,
            "collapse": collapse, "integrity_ok": integrity_ok}
