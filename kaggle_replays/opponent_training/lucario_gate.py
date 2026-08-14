"""Lucario Opponent Promotion Gate v1 — 判定ロジック(cg 非使用・contract 駆動)。

frozen contract = lucario_opponent_gate_v1.json(hash a058f5a05e9fb0ce)を source of truth に、
per-recipe SPRT 結果から aggregate(recipe-uniform + stratified CI)・collapse guard・
PROMOTE/FAIL/REVIEW を出す。閾値は contract から読む(hard-code しない)。ローカル専用・git 未追跡。
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
from sprt import wilson_interval  # noqa: E402  (measurement 再利用)

CONTRACT_PATH = _HERE / "lucario_opponent_gate_v1.json"
EXPECTED_GATE_HASH = "a058f5a05e9fb0ce"
Z95_1S = 1.6448536269514722


def file_hash(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load_contract(path=None) -> dict:
    return json.loads(Path(path or CONTRACT_PATH).read_text(encoding="utf-8"))


def verify_frozen(path=None) -> list[str]:
    h = file_hash(path or CONTRACT_PATH)
    return [] if h == EXPECTED_GATE_HASH else [f"gate hash 不一致: {h} != {EXPECTED_GATE_HASH}"]


def aggregate(per_recipe: list[dict]) -> dict:
    """recipe-uniform mean winrate + stratified Wald 片側95%下限。

    per_recipe: [{recipe_id, n, wins, winrate}, ...]。Var=Σ(1/k)² p_r(1−p_r)/n_r。
    """
    k = len(per_recipe)
    point = sum(r["winrate"] for r in per_recipe) / k
    var = 0.0
    for r in per_recipe:
        p, n = r["winrate"], r["n"]
        var += (1.0 / k) ** 2 * (p * (1 - p) / n if n else 0.0)
    se = math.sqrt(var)
    lower = point - Z95_1S * se
    return {"weighting": "recipe_uniform", "point": point, "se": se,
            "one_sided_95_lower": lower, "k": k}


def collapse_check(per_recipe: list[dict], contract: dict) -> list[str]:
    """各 recipe: 勝率 < point_lt かつ Wilson95 上端 < upper_lt → collapse。"""
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
    """frozen contract で PROMOTE/FAIL/REVIEW。"""
    total = contract["recipe_requirement"]["total"]
    need = contract["recipe_requirement"]["promote_count_min"]
    pt_thr = contract["aggregate"]["practical_threshold"]
    promote_n = sum(1 for r in per_recipe if r.get("sprt_decision") == "PROMOTE")
    futility_n = sum(1 for r in per_recipe if r.get("sprt_decision") == "FUTILITY")
    point_ok = agg["point"] >= pt_thr
    stat_ok = agg["one_sided_95_lower"] > 0.50

    if not integrity_ok:
        decision = "FAIL"; reason = "integrity failure"
    elif collapse:
        decision = "FAIL"; reason = f"collapse guard fired: {collapse}"
    elif futility_n > total // 2:
        decision = "FAIL"; reason = f"過半 recipe が FUTILITY({futility_n}/{total})"
    elif offline_met and promote_n >= need and point_ok and stat_ok:
        decision = "PROMOTE"; reason = "全条件 MET"
    else:
        decision = "REVIEW"; reason = "borderline(PROMOTE 条件未達だが明確な FAIL でもない)"

    return {"decision": decision, "reason": reason,
            "offline_met": offline_met, "promote_recipes": promote_n, "futility_recipes": futility_n,
            "recipe_requirement": f"{promote_n}/{total} (need >={need})",
            "aggregate_point": agg["point"], "aggregate_point_ge_053": point_ok,
            "aggregate_lower": agg["one_sided_95_lower"], "aggregate_lower_gt_050": stat_ok,
            "collapse": collapse, "integrity_ok": integrity_ok}
