"""Phase19.5 §15-§27: multi-horizon target の reliability / terminal relevance / Value-copy 監査。

各 horizon について:
  self reliability   : block A vs block B(独立 seed family)
  action sensitivity : group 内 spread と tie 率(§25)
  terminal relevance : High-M terminal reference との一致(§21)
  Value-copy degree  : root V(s) / immediate V(s') との相関(§24)

§22: cross-agreement は参照側の self-reliability で上限が決まるので、必ず三つ組で並べる。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HZ = ("H0", "H1", "H2", "H4", "HT")
TIE = 1e-9


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def vals(g, blk, h):
    """group の候補ごとの target 値(horizon h、block blk)。"""
    out = []
    for c in g["candidates"]:
        b = c.get("lh_a" if blk == "A" else "lh_b")
        if b is None:
            return None
        if h == "HT":
            out.append(b["mean"])
        else:
            z = b.get("hz", {}).get(h)
            if z is None:
                return None
            out.append(z["mean"])
    return out


def half_vals(g, blk, h, half):
    """M 本を前半/後半に割った推定(High-M の自己一致用)。"""
    out = []
    for c in g["candidates"]:
        b = c.get("lh_a" if blk == "A" else "lh_b")
        if b is None:
            return None
        seq = b["outcomes"] if h == "HT" else b.get("hz", {}).get(h, {}).get("values")
        if not seq:
            return None
        n = len(seq) // 2
        s = seq[:n] if half == 0 else seq[n:]
        if not s:
            return None
        out.append(st.mean(s))
    return out


def pairwise(a, b):
    ok = tot = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(b[i] - b[j]) <= TIE:
                continue
            tot += 1
            d = a[i] - a[j]
            ok += 0.5 if abs(d) <= TIE else (1.0 if d * (b[i] - b[j]) > 0 else 0.0)
    return ok, tot


def spearman(a, b):
    if len(a) < 3:
        return None

    def rk(v):
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
    ra, rb = rk(a), rk(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (dx * dy)
            if dx > 0 and dy > 0 else None)


def agree_table(G, get_a, get_b, margin_from="a"):
    """2つの推定量の一致(全体 + margin 別)。"""
    ok = tot = 0.0
    t1, sp, mad = [], [], []
    marg = {"<0.05": [0.0, 0.0], "0.05-0.10": [0.0, 0.0],
            "0.10-0.20": [0.0, 0.0], ">=0.20": [0.0, 0.0]}
    for g in G:
        a, b = get_a(g), get_b(g)
        if a is None or b is None or len(a) != len(b) or len(a) < 2:
            continue
        o, t = pairwise(a, b)
        ok += o
        tot += t
        t1.append(1.0 if a.index(max(a)) == b.index(max(b)) else 0.0)
        v = spearman(a, b)
        if v is not None:
            sp.append(v)
        mad += [abs(x - y) for x, y in zip(a, b)]
        ref = a if margin_from == "a" else b
        for i, j in itertools.combinations(range(len(a)), 2):
            m = abs(ref[i] - ref[j])
            key = ("<0.05" if m < 0.05 else "0.05-0.10" if m < 0.10
                   else "0.10-0.20" if m < 0.20 else ">=0.20")
            if abs(b[i] - b[j]) <= TIE:
                continue
            d = a[i] - a[j]
            marg[key][0] += 0.5 if abs(d) <= TIE else (
                1.0 if d * (b[i] - b[j]) > 0 else 0.0)
            marg[key][1] += 1
    return {"pairwise": round(ok / tot, 4) if tot else None,
            "pairs": int(tot),
            "top1": round(st.mean(t1), 4) if t1 else None,
            "spearman": round(st.mean(sp), 4) if sp else None,
            "mean_abs_diff": round(st.mean(mad), 4) if mad else None,
            "by_margin": {k: {"pairs": int(v[1]), "agreement": round(v[0] / v[1], 4)}
                          for k, v in marg.items() if v[1] > 0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="kaggle_replays/_mh_a*.jsonl.gz")
    ap.add_argument("--highm", default="kaggle_replays/_mh_r*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase195_audit.json")
    args = ap.parse_args()
    A = [g for g in load(args.audit) if all(c.get("lh_b") for c in g["candidates"])]
    rep = {"audit_groups": len(A),
           "candidates": sum(len(g["candidates"]) for g in A),
           "m": A[0]["m"] if A else None}

    # ---- §15/§16 self reliability(block A vs B)----
    rep["self_reliability"] = {
        h: agree_table(A, lambda g, h=h: vals(g, "A", h), lambda g, h=h: vals(g, "B", h))
        for h in HZ}

    # ---- §25 action sensitivity ----
    sens = {}
    for h in HZ:
        spread, tie = [], []
        for g in A:
            a = vals(g, "A", h)
            if not a:
                continue
            spread.append(max(a) - min(a))
            n = len(a)
            npair = n * (n - 1) / 2
            nt = sum(1 for i, j in itertools.combinations(range(n), 2)
                     if abs(a[i] - a[j]) <= 1e-6)
            tie.append(nt / npair if npair else 0.0)
        sens[h] = {"mean_group_spread": round(st.mean(spread), 4) if spread else None,
                   "tie_rate": round(st.mean(tie), 4) if tie else None,
                   "target_sd": round(st.pstdev(
                       [x for g in A for x in (vals(g, "A", h) or [])]), 4)}
    rep["action_sensitivity"] = sens

    # ---- §24 Value-copy audit ----
    vc = {}
    for h in HZ:
        xs, ys, zs = [], [], []
        for g in A:
            a = vals(g, "A", h)
            h0 = vals(g, "A", "H0")
            if not a or not h0:
                continue
            xs += a
            ys += h0                       # immediate V(s') = H0
            zs += [st.mean(h0)] * len(a)   # root 近傍(group 平均)
        vc[h] = {"corr_immediate_V": round(corr(xs, ys), 4) if xs else None,
                 "corr_group_mean_V": round(corr(xs, zs), 4) if xs else None}
    rep["value_copy"] = vc

    # ---- 短期 teacher(参考)----
    def short_vals(g):
        out = []
        for c in g["candidates"]:
            b = c.get("short_blocks")
            if not b:
                return None
            out.append(st.mean(b))
        return out
    rep["short_teacher_vs_HT_M8"] = agree_table(A, short_vals, lambda g: vals(g, "A", "HT"))

    # ---- §19-§21 High-M terminal reference ----
    R = load(args.highm)
    if R:
        rep["high_m"] = {"groups": len(R), "m_ref": R[0]["m"] if R else None}
        rep["high_m"]["self_reliability_half_split"] = {
            h: agree_table(R, lambda g, h=h: half_vals(g, "A", h, 0),
                           lambda g, h=h: half_vals(g, "A", h, 1)) for h in HZ}
        rep["terminal_relevance"] = {
            h: agree_table(R, lambda g, h=h: vals(g, "A", h),
                           lambda g: vals(g, "A", "HT"), margin_from="b")
            for h in ("H0", "H1", "H2", "H4")}
        rep["terminal_relevance"]["short_teacher"] = agree_table(
            R, short_vals, lambda g: vals(g, "A", "HT"), margin_from="b")
        # terminal regret(各 target の top1 が terminal でどれだけ損か)
        reg = {}
        for name, fn in [(h, (lambda g, h=h: vals(g, "A", h))) for h in
                         ("H0", "H1", "H2", "H4")] + [("short_teacher", short_vals)]:
            r = []
            for g in R:
                a = fn(g)
                t = vals(g, "A", "HT")
                if not a or not t:
                    continue
                r.append(max(t) - t[a.index(max(a))])
            reg[name] = round(st.mean(r), 4) if r else None
        rep["terminal_regret"] = reg
    else:
        rep["high_m"] = None

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
