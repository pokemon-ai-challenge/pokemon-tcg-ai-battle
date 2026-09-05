"""Phase19 §14-§18: long-horizon target の reliability / null control / short-vs-long 監査。

モデルを学習する前に、**target 自体が再現可能か**を確認する(Gate O1)。
audit subset は同一 root・同一候補について独立な2ブロック(各 M=8)を持つので、
  block A vs block B = target reliability かつ null control(同じ行動を2回評価した差)
になる。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TIE = 0.005


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def _pairwise(a, b, tie=0.0):
    tot = ok = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(b[i] - b[j]) <= tie:
                continue
            tot += 1
            d = a[i] - a[j]
            ok += 0.5 if d == 0 else (1.0 if d * (b[i] - b[j]) > 0 else 0.0)
    return (ok / tot) if tot else None


def _rank(v):
    n = len(v)
    o = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


def _kendall(a, b):
    con = dis = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    return (con - dis) / (con + dis) if (con + dis) else None


def short_mean(c):
    b = c.get("short_blocks")
    return st.mean(b) if b else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_lh_w*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase19_audit.json")
    args = ap.parse_args()
    G = load(args.data)
    rep = {"n_groups": len(G), "n_candidates": sum(len(g["candidates"]) for g in G),
           "m": G[0]["m"] if G else None,
           "games": len({g["game"] for g in G})}

    # ---------- §14/§15 target reliability(audit subset) ----------
    A = [g for g in G if g.get("is_audit") and all(c["lh_b"] for c in g["candidates"])]
    pw, sp, kd, t1, mad = [], [], [], [], []
    marg = {"<0.05": [], "0.05-0.10": [], "0.10-0.20": [], ">=0.20": []}
    for g in A:
        a = [c["lh_a"]["mean"] for c in g["candidates"]]
        b = [c["lh_b"]["mean"] for c in g["candidates"]]
        v = _pairwise(a, b)
        if v is not None:
            pw.append(v)
        v = _spearman(a, b)
        if v is not None:
            sp.append(v)
        v = _kendall(a, b)
        if v is not None:
            kd.append(v)
        t1.append(1.0 if a.index(max(a)) == b.index(max(b)) else 0.0)
        mad += [abs(x - y) for x, y in zip(a, b)]
        # margin 別: block A の候補間 margin ごとに B が同じ向きか
        for i, j in itertools.combinations(range(len(a)), 2):
            m = abs(a[i] - a[j])
            key = ("<0.05" if m < 0.05 else "0.05-0.10" if m < 0.10
                   else "0.10-0.20" if m < 0.20 else ">=0.20")
            if a[i] == a[j]:
                continue
            agree = 1.0 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else (
                0.5 if b[i] == b[j] else 0.0)
            marg[key].append(agree)
    rep["reliability"] = {
        "audit_groups": len(A),
        "pairwise_agreement": round(st.mean(pw), 4) if pw else None,
        "spearman": round(st.mean(sp), 4) if sp else None,
        "kendall": round(st.mean(kd), 4) if kd else None,
        "top1_agreement": round(st.mean(t1), 4) if t1 else None,
        "mean_abs_target_diff": round(st.mean(mad), 4) if mad else None,
        "by_margin": {k: {"pairs": len(v), "agreement": round(st.mean(v), 4)}
                      for k, v in marg.items() if v},
    }

    # ---------- §16 null control(同一候補を2回評価した差) ----------
    d = [c["lh_a"]["mean"] - c["lh_b"]["mean"] for g in A for c in g["candidates"]]
    if d:
        rs = np.random.RandomState(0)
        arr = np.asarray(d)
        bs = arr[rs.randint(0, len(arr), size=(10000, len(arr)))].mean(1)
        rep["null_control"] = {
            "candidates": len(d), "groups": len(A),
            "mean_abs_delta": round(float(np.abs(arr).mean()), 4),
            "aggregate_mean": round(float(arr.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                     round(float(np.percentile(bs, 97.5)), 4)],
            "note": "同じ候補・同じ continuation policy を独立 seed で2回評価した差。"
                    "long-horizon target のノイズ床。"}

    # ---------- §17 short teacher vs long target ----------
    sl, t1a, both = [], [], []
    stable_agree = []
    for g in G:
        s = [short_mean(c) for c in g["candidates"]]
        if any(x is None for x in s):
            continue
        l = [c["lh_a"]["mean"] for c in g["candidates"]]
        v = _pairwise(s, l)
        if v is not None:
            sl.append(v)
        t1a.append(1.0 if s.index(max(s)) == l.index(max(l)) else 0.0)
        both += list(zip(s, l))
        # short teacher が stable(4block 全一致)なペアだけ
        for i, j in itertools.combinations(range(len(s)), 2):
            bl = [c["short_blocks"] for c in g["candidates"]]
            diffs = [bl[i][b] - bl[j][b] for b in range(len(bl[i]))]
            sg = [0 if abs(x) < TIE else (1 if x > 0 else -1) for x in diffs]
            nz = [x for x in sg if x != 0]
            if not nz or max(nz.count(1), nz.count(-1)) != len(diffs):
                continue
            if l[i] == l[j]:
                continue
            stable_agree.append(1.0 if (s[i] - s[j]) * (l[i] - l[j]) > 0 else 0.0)
    if both:
        xs = [x for x, _ in both]
        ys = [y for _, y in both]
        mx, my = st.mean(xs), st.mean(ys)
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        corr = (sum((x - mx) * (y - my) for x, y in both) / (dx * dy)
                if dx > 0 and dy > 0 else None)
    else:
        corr = None

    def wilson(k, n, z=1.96):
        if n == 0:
            return None
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return [round(c - h, 4), round(c + h, 4)]

    rep["short_vs_long"] = {
        "groups": len(sl),
        "pairwise_agreement": round(st.mean(sl), 4) if sl else None,
        "top1_agreement": round(st.mean(t1a), 4) if t1a else None,
        "correlation": round(corr, 4) if corr is not None else None,
        "stable_short_pairs": len(stable_agree),
        "stable_short_agreement": round(st.mean(stable_agree), 4) if stable_agree else None,
        "stable_short_ci": wilson(sum(stable_agree), len(stable_agree)) if stable_agree else None,
    }

    # ---------- target 分布 / turn 別 variance ----------
    allv = [c["lh_a"]["var"] for g in G for c in g["candidates"]]
    allm = [c["lh_a"]["mean"] for g in G for c in g["candidates"]]
    rep["target_distribution"] = {
        "mean": round(st.mean(allm), 4), "sd": round(st.pstdev(allm), 4),
        "mean_within_candidate_variance": round(st.mean(allv), 4),
        "spread_within_group": round(st.mean(
            [max(c["lh_a"]["mean"] for c in g["candidates"])
             - min(c["lh_a"]["mean"] for c in g["candidates"]) for g in G]), 4),
    }
    rep["by_turn_variance"] = {
        b: {"groups": sum(1 for g in G if g["turn_band"] == b),
            "mean_target_var": round(st.mean(
                [c["lh_a"]["var"] for g in G if g["turn_band"] == b
                 for c in g["candidates"]]), 4)}
        for b in ("early", "middle", "late")
        if any(g["turn_band"] == b for g in G)}
    rep["terminal_reasons"] = dict(Counter(
        r for g in G for c in g["candidates"] for r in c["lh_a"]["reasons"]))

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
