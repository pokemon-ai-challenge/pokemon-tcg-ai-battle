"""Phase9B: teacher A/B による教師再現性評価と信頼度層の確定。

循環評価を避けるため、役割を厳密に分ける(§8.3):
  teacher_A : 信頼度層の**定義のみ**(A 内部の 3対3 split-half)
  teacher_B : 教師再現性(A-B)と、後段のモデル評価に使う**正式教師**

A で層を定義して A で性能を主張する、という Phase9A の誤りを構造的に防ぐ。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _pairwise(a, b):
    tot = ok = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if b[i] == b[j]:
                continue
            tot += 1
            d = a[i] - a[j]
            ok += 0.5 if d == 0 else (1.0 if d * (b[i] - b[j]) > 0 else 0.0)
    return (ok / tot) if tot else None


def _rank(v):
    n = len(v); o = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and v[o[j+1]] == v[o[i]]:
            j += 1
        avg = (i+j)/2.0+1
        for k in range(i, j+1):
            r[o[k]] = avg
        i = j+1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    da = sum((x-ma)**2 for x in ra)**0.5; db = sum((y-mb)**2 for y in rb)**0.5
    return (num/(da*db)) if da > 0 and db > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE / "_actionq_v2.jsonl.gz"))
    ap.add_argument("--out", default=str(_HERE / "_actionq_v2_reliability.json"))
    args = ap.parse_args()

    rows = []
    with gzip.open(args.data, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    test = [r for r in rows if r["split"] == 2
            and r["candidates"][0].get("teacher_B") is not None]

    recs = []
    for r in test:
        A = [c["teacher_A"]["mean"] for c in r["candidates"]]
        B = [c["teacher_B"]["mean"] for c in r["candidates"]]
        # 信頼度は **teacher_A の内部 split-half のみ**から定義する
        a1 = [c["teacher_A"]["half_a"] for c in r["candidates"]]
        a2 = [c["teacher_A"]["half_b"] for c in r["candidates"]]
        stdA = statistics.mean([c["teacher_A"]["std"] for c in r["candidates"]])
        spreadA = max(A) - min(A)
        recs.append({
            "group_id": r["group_id"], "turn_band": r["turn_band"],
            "cand_band": r["cand_band"], "arch": r["opponent_archetype"],
            "n_cands": len(r["candidates"]),
            "confA_pairwise": _pairwise(a1, a2),          # 層定義用(Aのみ)
            "confA_spread_over_std": (spreadA / stdA) if stdA > 0 else None,
            "AB_pairwise": _pairwise(A, B),               # 教師再現性(独立)
            "AB_spearman": _spearman(A, B),
            "AB_top1": 1.0 if A.index(max(A)) == B.index(max(B)) else 0.0,
            "spread": spreadA, "std": stdA,
        })

    # 信頼度層の条件は **モデル評価前に固定**(SHA を残す)
    TIER = {"high": {"confA_pairwise_min": 0.70, "spread_over_std_min": 1.0}}
    tier_sha = hashlib.sha256(json.dumps(TIER, sort_keys=True).encode()).hexdigest()[:16]
    for x in recs:
        hi = ((x["confA_pairwise"] or 0) >= TIER["high"]["confA_pairwise_min"]
              and (x["confA_spread_over_std"] or 0) >= TIER["high"]["spread_over_std_min"])
        x["tier"] = "high" if hi else "other"

    def agg(sel):
        if not sel:
            return None
        f = lambda k: [x[k] for x in sel if x.get(k) is not None]  # noqa: E731
        return {
            "n": len(sel),
            "coverage": round(len(sel) / len(recs), 4),
            "AB_pairwise": round(statistics.mean(f("AB_pairwise")), 4),
            "AB_spearman": round(statistics.mean(f("AB_spearman")), 4),
            "AB_top1": round(statistics.mean(f("AB_top1")), 4),
            "spread": round(statistics.mean(f("spread")), 5),
            "std": round(statistics.mean(f("std")), 5),
            "spread_over_std": round(statistics.mean(f("confA_spread_over_std")), 3),
        }

    out = {"data": args.data, "tier_definition": TIER, "tier_definition_sha": tier_sha,
           "note": "層は teacher_A のみで定義し、再現性・性能は teacher_B で測る(循環回避)。",
           "test_groups": len(recs),
           "all": agg(recs),
           "high": agg([x for x in recs if x["tier"] == "high"]),
           "other": agg([x for x in recs if x["tier"] == "other"]),
           "by_turn": {b: agg([x for x in recs if x["turn_band"] == b])
                       for b in ("early", "middle", "late")},
           "by_cand": {b: agg([x for x in recs if x["cand_band"] == b])
                       for b in ("small", "medium", "large")},
           "by_arch": {a: agg([x for x in recs if x["arch"] == a])
                       for a in sorted({x["arch"] for x in recs})},
           "records": recs}

    hi = out["high"] or {"n": 0, "coverage": 0, "AB_pairwise": 0}
    allv = out["all"]
    out["GATE"] = {
        "test_groups>=250": len(recs) >= 250,
        "middle+late>=40%": round(sum(1 for x in recs if x["turn_band"] != "early")
                                  / len(recs), 4) >= 0.40,
        "GateA_all_pairwise>=0.70": (allv["AB_pairwise"] or 0) >= 0.70,
        "GateB_high_coverage>=30%": (hi["coverage"] or 0) >= 0.30,
        "GateB_high_pairwise>=0.80": (hi["AB_pairwise"] or 0) >= 0.80,
        "GateB_high_groups>=75": (hi["n"] or 0) >= 75,
    }
    g = out["GATE"]
    out["GATE"]["GateA_PASS"] = bool(g["test_groups>=250"] and g["GateA_all_pairwise>=0.70"])
    out["GATE"]["GateB_PASS"] = bool(
        g["test_groups>=250"] and g["GateB_high_coverage>=30%"]
        and g["GateB_high_pairwise>=0.80"] and g["GateB_high_groups>=75"])
    out["GATE"]["VERDICT"] = ("PASS" if (out["GATE"]["GateA_PASS"] or out["GATE"]["GateB_PASS"])
                              else "FAIL")
    print(json.dumps({k: v for k, v in out.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
