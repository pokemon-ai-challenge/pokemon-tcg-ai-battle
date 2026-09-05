"""Phase19.13 §55-B/G/H: decision census / shortcut disagreement / tie_eps 分布。"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

PATHS = ("D0_lethal", "D1_searched", "D2_shortcut", "D3_ineligible", "D4_fallback")
SELT = {"0": "MAIN", "1": "CARD", "2": "EVOLVE", "3": "ENERGY", "4": "COUNT",
        "5": "YES_NO", "6": "ORDER", "7": "MULLIGAN", "8": "ATTACK", "9": "SETUP"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_p1913_c*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1913_census.json")
    args = ap.parse_args()
    R = [json.loads(l) for f in sorted(glob.glob(args.data))
         for l in gzip.open(f, "rt", encoding="utf-8")]
    n = len(R)
    rep = {"decisions": n, "games": len({r["game"] for r in R}),
           "arch": dict(Counter(r["arch"] for r in R))}

    # ---- B. census ----
    tot_ms = sum(r["ms"] or 0.0 for r in R)
    cen = {}
    for p in PATHS:
        s = [r for r in R if r["path"] == p]
        ms = sum(r["ms"] or 0.0 for r in s)
        cen[p] = {"n": len(s), "rate": round(len(s) / n, 4),
                  "runtime_share": round(ms / tot_ms, 4) if tot_ms else None,
                  "mean_ms": round(st.mean([r["ms"] or 0 for r in s]), 2) if s else None}
    rep["census"] = cen
    rep["total_ms_per_game"] = round(tot_ms / max(1, len({r["game"] for r in R})), 1)
    rep["decisions_per_game"] = round(n / max(1, len({r["game"] for r in R})), 1)

    # ineligible の内訳(coverage C-B の中身)
    inel = [r for r in R if r["path"] == "D3_ineligible"]
    rep["ineligible_breakdown"] = {
        "by_select_type": dict(Counter(SELT.get(r["select_type"], r["select_type"])
                                       for r in inel).most_common()),
        "main_but_multiselect": sum(1 for r in inel if r["select_type"] == "0")}
    rep["coverage_class"] = dict(Counter(r["coverage_class"] for r in R if r["coverage_class"]))

    # searched decisions のうち予算(400ms)に張り付いた割合
    srch = [r for r in R if r["path"] == "D1_searched"]
    rep["searched"] = {
        "n": len(srch),
        "budget_saturated_rate": round(sum(1 for r in srch if (r["pipe_ms"] or 0) >= 395)
                                       / max(1, len(srch)), 4),
        "mean_pipe_ms": round(st.mean([r["pipe_ms"] or 0 for r in srch]), 1) if srch else None,
        "mean_n_cands": round(st.mean([r["n_cands"] for r in srch]), 2) if srch else None}

    # ---- G. shortcut ----
    sc = [r for r in R if r["path"] == "D2_shortcut"]
    cf = [r for r in sc if r.get("cf_action") is not None]
    dis = [r for r in cf if r.get("cf_disagree")]
    saved = st.mean([r.get("cf_ms", 0) - (r["ms"] or 0) for r in cf]) if cf else None
    rep["shortcut"] = {
        "activation_rate": round(len(sc) / n, 4),
        "cf_evaluated": len(cf), "cf_none": len(sc) - len(cf),
        "disagreement_rate": round(len(dis) / max(1, len(cf)), 4),
        "disagreement_ci95": wilson(len(dis), len(cf)),
        "mean_ms_shortcut": round(st.mean([r["ms"] or 0 for r in sc]), 2) if sc else None,
        "mean_ms_full_search": round(st.mean([r.get("cf_ms", 0) for r in cf]), 1) if cf else None,
        "runtime_saved_ms_per_shortcut": round(saved, 1) if saved else None,
        "top1_prob": {
            "mean": round(st.mean([r["policy_top1_prob"] for r in sc
                                   if r["policy_top1_prob"] is not None]), 4),
            "p10": round(sorted(r["policy_top1_prob"] for r in sc
                                if r["policy_top1_prob"] is not None)[len(sc) // 10], 4)}}
    # §35 disagreement が top1 confidence のどこに集中するか
    bands = [(0.90, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.01)]
    rep["shortcut"]["by_top1_prob"] = {}
    for lo, hi in bands:
        s = [r for r in cf if lo <= (r["policy_top1_prob"] or 0) < hi]
        if len(s) < 10:
            continue
        d = sum(1 for r in s if r.get("cf_disagree"))
        rep["shortcut"]["by_top1_prob"]["%.3f-%.3f" % (lo, hi)] = {
            "n": len(s), "disagree": d, "rate": round(d / len(s), 4)}
    rep["shortcut"]["by_n_opt"] = {}
    for lo, hi, nm in ((2, 3, "2"), (3, 5, "3-4"), (5, 8, "5-7"), (8, 999, "8+")):
        s = [r for r in cf if lo <= r["n_opt"] < hi]
        if len(s) < 10:
            continue
        d = sum(1 for r in s if r.get("cf_disagree"))
        rep["shortcut"]["by_n_opt"][nm] = {"n": len(s), "rate": round(d / len(s), 4)}

    # ---- H. tie_eps 分布(searched decisions の候補平均スコア差)----
    gaps = [r for r in srch if r.get("gap") is not None]
    rep["tie"] = {"n_with_gap": len(gaps),
                  "tie_changed_rate": round(sum(1 for r in gaps if r.get("tie_changed"))
                                            / max(1, len(gaps)), 4),
                  "by_gap_band": {}}
    for lo, hi, nm in ((0, .01, "<.01"), (.01, .02, ".01-.02"), (.02, .05, ".02-.05"),
                       (.05, .10, ".05-.10"), (.10, 9, ">=.10")):
        s = [r for r in gaps if lo <= r["gap"] < hi]
        if not s:
            continue
        rep["tie"]["by_gap_band"][nm] = {
            "n": len(s), "rate": round(len(s) / len(gaps), 4),
            "tie_changed": sum(1 for r in s if r.get("tie_changed"))}
    rep["tie"]["gap_le_tie_eps_rate"] = round(
        sum(1 for r in gaps if r["gap"] <= 0.02) / max(1, len(gaps)), 4)

    # ---- turn 別 ----
    rep["by_turn_band"] = {}
    for nm, lo, hi in (("early", 0, 6), ("middle", 6, 11), ("late", 11, 999)):
        s = [r for r in R if lo <= r["turn"] < hi]
        if not s:
            continue
        rep["by_turn_band"][nm] = {"n": len(s), **{
            p: round(sum(1 for r in s if r["path"] == p) / len(s), 4) for p in PATHS}}

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
