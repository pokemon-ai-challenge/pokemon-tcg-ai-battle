"""Phase19.14 解析: CARD census(§56-A/B)と terminal 価値監査(§56-D〜L)。

oracle 選択バイアスを最初から除く(§20-§22): M本を A/B 半分に割り、
**Aで選び B で採点**(逆向きも取って平均)。同一標本 oracle は上方バイアス参照として併記のみ。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

CARD_RATE_ALL = 0.2972      # Phase19.13 実測(2529/8510)


def load(p):
    return [json.loads(l) for f in sorted(glob.glob(p))
            for l in gzip.open(f, "rt", encoding="utf-8")]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def boot(v, B=10000, seed=0):
    if not v:
        return None
    a = np.asarray(v, float)
    rs = np.random.RandomState(seed)
    m = a[rs.randint(0, len(a), size=(B, len(a)))].mean(1)
    return {"mean": round(float(a.mean()), 4), "n": len(a),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)]}


def crossfit(g):
    """A半分で選び B半分で採点(両方向)。返り値は held-out outcome の各種。"""
    cs = g["cands"]
    if len(cs) < 2:
        return None
    half = min(len(c["j"]) for c in cs) // 2
    if half < 2:
        return None
    A = {c["idx"]: st.mean(c["j"][:half]) for c in cs}
    B = {c["idx"]: st.mean(c["j"][half:2 * half]) for c in cs}
    pp = {c["idx"]: (c["policy_prob"] or 0.0) for c in cs}
    idx = list(A)
    out = {}
    for sel_src, score_src, tag in ((A, B, "ab"), (B, A, "ba")):
        s = max(idx, key=lambda i: sel_src[i])
        out.setdefault("strong", []).append(score_src[s])
        p = max(idx, key=lambda i: pp[i])
        out.setdefault("policy", []).append(score_src[p])
        out.setdefault("random", []).append(st.mean([score_src[i] for i in idx]))
        out.setdefault("oracle_biased", []).append(max(score_src[i] for i in idx))
    full = {c["idx"]: st.mean(c["j"]) for c in cs}
    # 不偏 spread: A半分で best/worst を決め、**B半分で採点**した差。
    # raw max-min は推定ノイズの上に max を取るので必ず上方バイアスする(§32)。
    unb = st.mean([
        B[max(idx, key=lambda i: A[i])] - B[min(idx, key=lambda i: A[i])],
        A[max(idx, key=lambda i: B[i])] - A[min(idx, key=lambda i: B[i])]])
    return {"strong": st.mean(out["strong"]), "policy": st.mean(out["policy"]),
            "random": st.mean(out["random"]),
            "oracle_biased": st.mean(out["oracle_biased"]),
            "raw_spread": max(full.values()) - min(full.values()),
            "unbiased_spread": unb,
            "cf_spread": st.mean([max(B.values()) - min(B.values()),
                                  max(A.values()) - min(A.values())]),
            "n_cand": len(cs), "M": g["M"]}


def ref_reliability(G, m_req):
    """M/2 vs M/2 の候補順位一致(参照値そのものの信頼性、§23/§26)。"""
    ok = tot = 0.0
    t1 = []
    sp = []
    for g in G:
        if g["M"] < m_req:
            continue
        cs = g["cands"]
        h = min(len(c["j"]) for c in cs) // 2
        if h < 2 or len(cs) < 2:
            continue
        A = {c["idx"]: st.mean(c["j"][:h]) for c in cs}
        B = {c["idx"]: st.mean(c["j"][h:2 * h]) for c in cs}
        for a, b in itertools.combinations(list(A), 2):
            if B[a] == B[b]:
                continue
            tot += 1
            d = A[a] - A[b]
            ok += 0.5 if d == 0 else (1.0 if d * (B[a] - B[b]) > 0 else 0.0)
        best_a = max(A, key=lambda i: A[i])
        t1.append(1.0 if B[best_a] == max(B.values()) else 0.0)
        sp.append(abs((max(A.values()) - min(A.values()))
                      - (max(B.values()) - min(B.values()))))
    return {"pairs": int(tot), "pairwise": round(ok / tot, 4) if tot else None,
            "best_action_agreement": round(st.mean(t1), 4) if t1 else None,
            "spread_mean_abs_diff": round(st.mean(sp), 4) if sp else None,
            "roots": len(t1)}


def block(G, label):
    cf = [(g, crossfit(g)) for g in G]
    cf = [(g, c) for g, c in cf if c]
    if not cf:
        return None
    S = [c["strong"] for _, c in cf]
    Pl = [c["policy"] for _, c in cf]
    R = [c["random"] for _, c in cf]
    O = [c["oracle_biased"] for _, c in cf]
    out = {"label": label, "roots": len(cf),
           "held_out": {"strong_selector": boot(S), "policy": boot(Pl),
                        "random_candidate": boot(R),
                        "oracle_same_sample_BIASED": boot(O)},
           "primary_delta_CARD_CF": boot([a - b for a, b in zip(S, Pl)]),
           "strong_minus_random": boot([a - b for a, b in zip(S, R)]),
           "policy_minus_random": boot([a - b for a, b in zip(Pl, R)])}
    # spread 分布(raw と cross-fit)
    for key, nm in (("raw_spread", "raw"), ("unbiased_spread", "unbiased")):
        d = {}
        v = [c[key] for _, c in cf]
        for lo, hi, t in ((0, .02, "<.02"), (.02, .05, ".02-.05"), (.05, .10, ".05-.10"),
                          (.10, .20, ".10-.20"), (.20, 9, ">=.20")):
            k = sum(1 for x in v if lo <= x < hi)
            d[t] = {"n": k, "rate": round(k / len(v), 4)}
        d["mean"] = round(st.mean(v), 4)
        d["P(>=.10)"] = round(sum(1 for x in v if x >= .10) / len(v), 4)
        d["P(>=.20)"] = round(sum(1 for x in v if x >= .20) / len(v), 4)
        out["spread_" + nm] = d
    # Policy ranking 品質(全Jに対する pairwise / top1)
    ok = tot = 0.0
    t1 = []
    for g, _ in cf:
        cs = g["cands"]
        full = {c["idx"]: st.mean(c["j"]) for c in cs}
        pp = {c["idx"]: (c["policy_prob"] or 0.0) for c in cs}
        for a, b in itertools.combinations(list(full), 2):
            if full[a] == full[b]:
                continue
            tot += 1
            d = pp[a] - pp[b]
            ok += 0.5 if d == 0 else (1.0 if d * (full[a] - full[b]) > 0 else 0.0)
        p = max(full, key=lambda i: pp[i])
        t1.append(1.0 if full[p] == max(full.values()) else 0.0)
    out["policy_ranking_vs_full_J"] = {
        "pairwise": round(ok / tot, 4) if tot else None, "pairs": int(tot),
        "top1_agreement": round(st.mean(t1), 4) if t1 else None,
        "note": "同一標本参照なので上方バイアスあり。primary は cross-fit の差"}
    return out


def strata(G, keyfn, minn=12):
    out = {}
    for k in sorted({str(keyfn(g)) for g in G}):
        sub = [g for g in G if str(keyfn(g)) == k]
        if len(sub) < minn:
            continue
        cf = [c for c in (crossfit(g) for g in sub) if c]
        if len(cf) < minn:
            continue
        out[k] = {"n": len(cf),
                  "cf_spread": round(st.mean([c["cf_spread"] for c in cf]), 4),
                  "policy_loss": round(st.mean([c["strong"] - c["policy"] for c in cf]), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="kaggle_replays/_p1914_a*.jsonl.gz")
    ap.add_argument("--quality", default="kaggle_replays/_p1914q_p*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1914.json")
    args = ap.parse_args()
    rep = {}

    # ---------- A/B census ----------
    C = load(args.census)
    if C:
        n = len(C)
        cls = Counter(r["cls"] for r in C)
        rep["census"] = {
            "card_decisions": n, "games": len({r["game"] for r in C}),
            "by_class": {k: {"n": v, "rate": round(v / n, 4)} for k, v in cls.most_common()},
            "card_rate_of_all_decisions": CARD_RATE_ALL,
            "genuine_observable_rate_of_all": round(
                CARD_RATE_ALL * cls["C2_genuine"] / n, 4),
            "unobservable_rate_of_all": round(
                CARD_RATE_ALL * cls["C3_unobservable"] / n, 4),
            "mean_ms": round(st.mean([r["ms"] for r in C]), 2),
            "card_per_game": round(n / max(1, len({r["game"] for r in C})), 1)}
        by = {}
        for r in C:
            d = by.setdefault(r["context_name"], {"n": 0, "opts": [], "cls": Counter()})
            d["n"] += 1
            d["opts"].append(r["n_opt"])
            d["cls"][r["cls"]] += 1
        rep["census"]["by_subtype"] = {
            k: {"n": v["n"], "rate": round(v["n"] / n, 4),
                "mean_options": round(st.mean(v["opts"]), 2),
                **{c: v["cls"][c] for c in ("C0_trivial", "C1_dup_equivalent",
                                            "C2_genuine", "C3_unobservable")}}
            for k, v in sorted(by.items(), key=lambda kv: -kv[1]["n"])}
        rep["census"]["chain_pos"] = dict(Counter(min(r["chain_pos"], 4) for r in C))
        rep["census"]["multiselect_rate"] = round(
            sum(1 for r in C if not (r["min"] <= 1 <= r["max"])) / n, 4)

    # ---------- quality ----------
    G = load(args.quality)
    if not G:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return
    rep["quality"] = {
        "roots": len(G), "games": len({g["game"] for g in G}),
        "by_cls": dict(Counter(g["cls"] for g in G)),
        "by_M": dict(Counter(g["M"] for g in G)),
        "turn_band": dict(Counter(g["turn_band"] for g in G)),
        "arch": dict(Counter(g["arch"] for g in G)),
        "capped_rate": round(sum(1 for g in G if g["capped"]) / len(G), 4),
        "mean_effective_cands": round(st.mean([g["n_effective"] for g in G]), 2),
        "terminal_rate": round(st.mean([c["terminal_rate"] for g in G
                                        for c in g["cands"]]), 4),
        "mean_audit_ms": round(st.mean([g["audit_ms"] for g in G]), 1)}
    rep["reference_reliability"] = {
        "M8_4v4": ref_reliability([g for g in G if g["M"] == 8], 8),
        "M32_16v16": ref_reliability([g for g in G if g["M"] >= 32], 32)}
    rep["all"] = block(G, "all")
    for c in ("C2_genuine", "C3_unobservable"):
        b = block([g for g in G if g["cls"] == c], c)
        if b:
            rep[c] = b
    b32 = block([g for g in G if g["M"] >= 32], "M32_subset")
    if b32:
        rep["M32_subset"] = b32
    rep["strata"] = {
        "by_subtype": strata(G, lambda g: g["context_name"]),
        "by_turn": strata(G, lambda g: g["turn_band"]),
        "by_n_cand": strata(G, lambda g: "2" if g["n_effective"] == 2
                            else ("3-4" if g["n_effective"] <= 4 else "5+")),
        "by_chain_pos": strata(G, lambda g: min(g["chain_pos"], 3))}

    # §43-§45 horizon ladder(M32 のみ、secondary)
    lad = {}
    for ck in ("C1", "C1b", "C4"):
        ok = tot = 0.0
        for g in G:
            if g["M"] < 32:
                continue
            cs = g["cands"]
            vv, jj = {}, {}
            for c in cs:
                v = [x.get(ck) for x in (c["ckpt"] or []) if x and x.get(ck) is not None]
                if len(v) >= 4:
                    vv[c["idx"]] = st.mean(v)
                    jj[c["idx"]] = st.mean(c["j"])
            for a, b in itertools.combinations(list(vv), 2):
                if jj[a] == jj[b]:
                    continue
                tot += 1
                d = vv[a] - vv[b]
                ok += 0.5 if d == 0 else (1.0 if d * (jj[a] - jj[b]) > 0 else 0.0)
        lad[ck] = {"pairs": int(tot), "agreement_with_terminal":
                   round(ok / tot, 4) if tot else None}
    rep["horizon_ladder"] = lad

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
