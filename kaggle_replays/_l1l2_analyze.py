"""Phase13 §16-§21: L1/L2 の集計・paired bootstrap・教師整合の後解析。

epsilon は結果を見る前に固定する(§13/§25):
    EPS_L1 = EPS_L2 = 0.05
group 値は「seed(continuation)内で平均してから」bootstrap する(§17)。
個々の continuation を独立 group として扱わない。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EPS_L1 = 0.05
EPS_L2 = 0.05
B = 10000


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def winner(delta, eps):
    if delta > eps:
        return "S3"
    if delta < -eps:
        return "Q0"
    return "tie"


def boot(vals, b=B, seed=0):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return None
    rs = np.random.RandomState(seed)
    m = a[rs.randint(0, len(a), size=(b, len(a)))].mean(axis=1)
    return {"mean": round(float(a.mean()), 4),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)], "n": len(a)}


def prep(rows):
    out = []
    for r in rows:
        q, s = r["Q0"], r["S3"]
        if q["l1_mean"] is None or s["l1_mean"] is None:
            continue
        e = dict(r)
        e["l1_q0"], e["l1_s3"] = q["l1_mean"], s["l1_mean"]
        e["l1_delta"] = s["l1_mean"] - q["l1_mean"]
        e["l2_q0"], e["l2_s3"] = q["l2_mean"], s["l2_mean"]
        e["l2_delta"] = (None if q["l2_mean"] is None or s["l2_mean"] is None
                         else s["l2_mean"] - q["l2_mean"])
        e["pd_q0"], e["pd_s3"] = q["prize_diff_mean"], s["prize_diff_mean"]
        e["len_q0"], e["len_s3"] = q["steps_mean"], s["steps_mean"]
        e["term_q0"], e["term_s3"] = q["terminal_rate"], s["terminal_rate"]
        e["w_l1"] = winner(e["l1_delta"], EPS_L1)
        e["w_l2"] = winner(e["l2_delta"], EPS_L2) if e["l2_delta"] is not None else None
        out.append(e)
    return out


def subset_table(G, keyfn, l2_only=False):
    tab = {}
    for k in sorted({keyfn(g) for g in G}):
        sub = [g for g in G if keyfn(g) == k]
        l2 = [g["l2_delta"] for g in sub if g["l2_delta"] is not None]
        tab[str(k)] = {
            "groups": len(sub),
            "l1_delta": round(st.mean([g["l1_delta"] for g in sub]), 4),
            "l2_delta": round(st.mean(l2), 4) if l2 else None,
            "Q0_support_l2": sum(1 for g in sub if g["w_l2"] == "Q0"),
            "S3_support_l2": sum(1 for g in sub if g["w_l2"] == "S3"),
            "tie_l2": sum(1 for g in sub if g["w_l2"] == "tie"),
        }
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_l1l2_main_w*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase13_report.json")
    args = ap.parse_args()

    rows = load(args.data)
    G = prep(rows)
    if not G:
        print(json.dumps({"error": "no groups"}))
        return
    rep = {"eps_l1": EPS_L1, "eps_l2": EPS_L2, "bootstrap_resamples": B,
           "n_groups": len(G),
           "n_cont": len(G[0]["Q0"]["l1_values"]),
           "set": {"turn_band": dict(Counter(g["turn_band"] for g in G)),
                   "cand_band": dict(Counter(g["cand_band"] for g in G)),
                   "arch": dict(Counter(g["arch"] for g in G)),
                   "me_first": dict(Counter(str(g["me_first"]) for g in G)),
                   "action_type_q0": dict(Counter(g["action_type_q0"] for g in G)),
                   "cands_mean": round(st.mean([g["n_cands"] for g in G]), 2)}}

    # ---- D. L1 ----
    l2G = [g for g in G if g["l2_delta"] is not None]
    rep["L1"] = {
        "primary_Q0": round(st.mean([g["l1_q0"] for g in G]), 4),
        "primary_S3": round(st.mean([g["l1_s3"] for g in G]), 4),
        "delta": boot([g["l1_delta"] for g in G]),
        "prize_diff_Q0": round(st.mean([g["pd_q0"] for g in G if g["pd_q0"] is not None]), 4),
        "prize_diff_S3": round(st.mean([g["pd_s3"] for g in G if g["pd_s3"] is not None]), 4),
        "prize_diff_delta": boot([g["pd_s3"] - g["pd_q0"] for g in G
                                  if g["pd_s3"] is not None and g["pd_q0"] is not None]),
        "terminal_rate_Q0": round(st.mean([g["term_q0"] for g in G]), 4),
        "terminal_rate_S3": round(st.mean([g["term_s3"] for g in G]), 4),
        "support": dict(Counter(g["w_l1"] for g in G)),
    }
    # ---- E. L2 ----
    def rate(g, tag, val):
        oc = g[tag]["l2_outcomes"]
        return sum(1 for o in oc if o == val) / len(oc) if oc else None

    rep["L2"] = {
        "outcome_Q0": round(st.mean([g["l2_q0"] for g in l2G]), 4),
        "outcome_S3": round(st.mean([g["l2_s3"] for g in l2G]), 4),
        "delta": boot([g["l2_delta"] for g in l2G]),
        "win_rate_Q0": round(st.mean([rate(g, "Q0", 1.0) for g in l2G]), 4),
        "win_rate_S3": round(st.mean([rate(g, "S3", 1.0) for g in l2G]), 4),
        "loss_rate_Q0": round(st.mean([rate(g, "Q0", 0.0) for g in l2G]), 4),
        "loss_rate_S3": round(st.mean([rate(g, "S3", 0.0) for g in l2G]), 4),
        "draw_rate_Q0": round(st.mean([rate(g, "Q0", 0.5) for g in l2G]), 4),
        "draw_rate_S3": round(st.mean([rate(g, "S3", 0.5) for g in l2G]), 4),
        "game_length_Q0": round(st.mean([g["len_q0"] for g in l2G]), 2),
        "game_length_S3": round(st.mean([g["len_s3"] for g in l2G]), 2),
        "game_length_delta": boot([g["len_s3"] - g["len_q0"] for g in l2G]),
        "support": dict(Counter(g["w_l2"] for g in l2G)),
    }

    # ---- null control(noise floor)----
    nl = [r for r in rows if r.get("NULL") and r["NULL"].get("l1_mean") is not None]
    if nl:
        n1 = [r["NULL"]["l1_mean"] - r["Q0"]["l1_mean"] for r in nl]
        n2 = [r["NULL"]["l2_mean"] - r["Q0"]["l2_mean"] for r in nl
              if r["NULL"]["l2_mean"] is not None and r["Q0"]["l2_mean"] is not None]
        rep["null_control"] = {
            "n_groups": len(nl),
            "l1_delta": boot(n1), "l1_abs_mean": round(st.mean([abs(x) for x in n1]), 4),
            "l2_delta": boot(n2) if n2 else None,
            "l2_abs_mean": round(st.mean([abs(x) for x in n2]), 4) if n2 else None,
            "note": "同じ行動・同じ決定化をもう一度打った差。0 に近いほど CRN が効いている。"
                    "これが実効的な分解能の下限になる。",
        }

    # ---- F. L1/L2 整合 ----
    rep["L1_L2_agreement"] = dict(Counter(
        "L1={} / L2={}".format(g["w_l1"], g["w_l2"]) for g in l2G))

    # ---- G/H. 教師との整合(後解析)----
    T = [g for g in G if g.get("teacher")]
    rep["teacher"] = {"groups_with_teacher": len(T),
                      "cls": dict(Counter(g["teacher"]["cls"] for g in T))}
    if T:
        def tsup(g):
            s = g["teacher"]["support_q0"]
            return "supports_Q0" if s > 0.5 else "supports_S3" if s < 0.5 else "near_tie"
        rep["teacher"]["vs_L1_L2"] = {}
        for k in ("supports_Q0", "supports_S3", "near_tie"):
            sub = [g for g in T if tsup(g) == k]
            l2s = [g for g in sub if g["l2_delta"] is not None]
            rep["teacher"]["vs_L1_L2"][k] = {
                "groups": len(sub),
                "L1_Q0": sum(1 for g in sub if g["w_l1"] == "Q0"),
                "L1_S3": sum(1 for g in sub if g["w_l1"] == "S3"),
                "L1_tie": sum(1 for g in sub if g["w_l1"] == "tie"),
                "L2_Q0": sum(1 for g in l2s if g["w_l2"] == "Q0"),
                "L2_S3": sum(1 for g in l2s if g["w_l2"] == "S3"),
                "L2_tie": sum(1 for g in l2s if g["w_l2"] == "tie"),
                "mean_l1_delta": round(st.mean([g["l1_delta"] for g in sub]), 4) if sub else None,
                "mean_l2_delta": round(st.mean([g["l2_delta"] for g in l2s]), 4) if l2s else None,
            }
        rep["teacher"]["by_stability"] = subset_table(T, lambda g: g["teacher"]["cls"])
        # 教師が Q0 を支持した度合いと L2 delta の相関(転移しているか)
        pairs = [(g["teacher"]["support_q0"], g["l2_delta"]) for g in T
                 if g["l2_delta"] is not None]
        if len(pairs) > 3:
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            ma, mb = st.mean(a), st.mean(b)
            da = sum((x - ma) ** 2 for x in a) ** 0.5
            db = sum((x - mb) ** 2 for x in b) ** 0.5
            rep["teacher"]["corr_support_q0_vs_l2_delta"] = (
                round(sum((x - ma) * (y - mb) for x, y in pairs) / (da * db), 4)
                if da > 0 and db > 0 else None)

    # ---- I/J. subset ----
    rep["by_turn"] = subset_table(G, lambda g: g["turn_band"])
    rep["by_cand"] = subset_table(G, lambda g: g["cand_band"])
    rep["by_arch"] = subset_table(G, lambda g: g["arch"])
    rep["by_first"] = subset_table(G, lambda g: "first" if g["me_first"] else "second")
    rep["by_action_type"] = subset_table(G, lambda g: g["action_type_q0"])

    # ---- 21. model margin との関係 ----
    def corr(xs, ys):
        if len(xs) < 4:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return (round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (dx * dy), 4)
                if dx > 0 and dy > 0 else None)

    lg = [g for g in G if g["l2_delta"] is not None]
    rep["margins"] = {
        "corr_s3_margin_vs_l2_delta": corr([g["s3_margin"] for g in lg],
                                           [g["l2_delta"] for g in lg]),
        "corr_q0_margin_vs_l2_delta": corr([g["q0_margin"] for g in lg],
                                           [g["l2_delta"] for g in lg]),
        "corr_s3_margin_vs_l1_delta": corr([g["s3_margin"] for g in G],
                                           [g["l1_delta"] for g in G]),
        "mean_q0_margin": round(st.mean([g["q0_margin"] for g in G]), 4),
        "mean_s3_margin": round(st.mean([g["s3_margin"] for g in G]), 4),
    }

    # ---- K. 注目局面 ----
    ex = []
    for g in sorted(l2G, key=lambda x: -abs(x["l2_delta"]))[:40]:
        t = g.get("teacher")
        tag = None
        if t:
            s = t["support_q0"]
            if s > 0.5 and g["w_l2"] == "S3":
                tag = "teacher=Q0 but L2=S3"
            elif s < 0.5 and g["w_l2"] == "Q0":
                tag = "teacher=S3 but L2=Q0"
            elif s == 0.5 and g["w_l2"] in ("Q0", "S3"):
                tag = "teacher near-tie but L2 decisive"
        if tag and len(ex) < 10:
            ex.append({"group_id": g["group_id"], "tag": tag, "turn": g["turn"],
                       "arch": g["arch"], "n_cands": g["n_cands"],
                       "action_type_q0": g["action_type_q0"],
                       "action_type_s3": g["action_type_s3"],
                       "teacher_support_q0": t["support_q0"], "teacher_cls": t["cls"],
                       "teacher_margin": t["mean_margin"],
                       "q0_margin": g["q0_margin"], "s3_margin": g["s3_margin"],
                       "l1_delta": round(g["l1_delta"], 4),
                       "l2_delta": round(g["l2_delta"], 4),
                       "l2_q0": g["l2_q0"], "l2_s3": g["l2_s3"]})
    rep["examples"] = ex

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_groups", "n_cont", "L1", "L2",
                                          "null_control", "L1_L2_agreement")
                      if k in rep}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
