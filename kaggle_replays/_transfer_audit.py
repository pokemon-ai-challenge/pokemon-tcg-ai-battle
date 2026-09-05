"""Phase19.11 Stage B/C: high-M ΔJ 妥当性 と Frozen Value の絶対妥当性。

Stage B(§19-§21): ΔL1 と高精度 ΔJ の対応。
  **ΔL1 と ΔJ は独立な trajectory 半分ずつから作る**(同一 trajectory から両方を取ると
  「良い C4 に着いた試合は勝ちやすい」という機械的相関が入るため)。

Stage C(§27-§29): 各 checkpoint の Frozen Value V_t が、その trajectory の
  実 terminal outcome をどれだけ予測するか。こちらは同一 trajectory 対応でよい。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

CK = ("C1b", "C2", "C4")


def load(p):
    R = []
    for f in sorted(glob.glob(p)):
        R += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return R


def corr(a, b, kind="p"):
    if len(a) < 3:
        return None
    if kind == "s":
        def rk(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[o[k]] = (i + j) / 2.0 + 1
                i = j + 1
            return r
        a, b = rk(a), rk(b)
    ma, mb = st.mean(a), st.mean(b)
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db)
            if da > 0 and db > 0 else None)


def auc(y, s):
    p = sorted(zip(s, y))
    r = {}
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1][0] == p[i][0]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2.0 + 1
        i = j + 1
    pos = sum(1 for _, yy in p if yy == 1)
    neg = len(p) - pos
    if pos == 0 or neg == 0:
        return None
    return (sum(r[k] for k, (_, yy) in enumerate(p) if yy == 1)
            - pos * (pos + 1) / 2) / (pos * neg)


def boot(v, B=10000, seed=0):
    a = np.asarray(v, float)
    rs = np.random.RandomState(seed)
    m = a[rs.randint(0, len(a), size=(B, len(a)))].mean(1)
    return {"mean": round(float(a.mean()), 4),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)], "n": len(a)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_p1911_h*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1911_transfer.json")
    args = ap.parse_args()
    G = load(args.data)
    rep = {"groups": len(G), "games": len({g["game"] for g in G}),
           "turn_band": dict(Counter(g["turn_band"] for g in G)),
           "arch": dict(Counter(g["arch"] for g in G))}

    # ---------- Stage B: ΔL1(前半)と ΔJ(後半)を独立に ----------
    items = []
    for g in G:
        E, T = g["Q0"], g["S3"]          # Q0キー=EntityQ枝 / S3キー=TES枝
        te, tt = E.get("trajs"), T.get("trajs")
        if not te or not tt:
            continue
        h = min(len(te), len(tt)) // 2
        if h < 4:
            continue

        def c4(trs):
            v = [t["ckpt"].get("C4") for t in trs if t["ckpt"].get("C4") is not None]
            return st.mean(v) if v else None

        def out(trs):
            # L2 は **実 terminal のみ**(非終局の Frozen Value 代用は使わない)
            v = [t["outcome"] for t in trs if t["terminal"] is not None]
            return st.mean(v) if len(v) >= 4 else None
        l1e, l1t = c4(te[:h]), c4(tt[:h])          # 前半 -> ΔL1
        je, jt = out(te[h:]), out(tt[h:])          # 後半 -> ΔJ
        ja, jb = out(te), out(tt)                  # 全32 -> 高精度 ΔJ(報告用)
        qe, qt = out(te[:h]), out(tt[:h])          # 前半 -> ΔJ(16A/16B 信頼性用)
        if None in (l1e, l1t, je, jt, ja, jb):
            continue
        items.append({"gid": g["group_id"], "turn": g["turn_band"], "cand": g["cand_band"],
                      "dL1": l1t - l1e, "dJ_half": jt - je, "dJ_full": jb - ja,
                      "dJ_halfA": (qt - qe) if None not in (qe, qt) else None,
                      "trate": st.mean([1.0 if t["terminal"] is not None else 0.0
                                        for t in te + tt])})
    rep["stage_b"] = {"n": len(items)}
    if items:
        A = [x["dL1"] for x in items]
        B = [x["dJ_half"] for x in items]
        F = [x["dJ_full"] for x in items]
        sa = [x for x in items if x["dL1"] != 0 and x["dJ_half"] != 0]
        rep["stage_b"].update({
            "mean_dL1": round(st.mean(A), 4), "mean_dJ_full": round(st.mean(F), 4),
            "dJ_full_ci": boot(F)["ci95"],
            "pearson_dL1_dJ": round(corr(A, B), 4),
            "spearman_dL1_dJ": round(corr(A, B, "s"), 4),
            "sign_agreement": round(sum(1 for x in sa if x["dL1"] * x["dJ_half"] > 0)
                                    / len(sa), 4) if sa else None,
            "n_sign": len(sa),
            "terminal_rate": round(st.mean([x["trate"] for x in items]), 4)})
        # 16A vs 16B: ΔJ 参照値そのものの信頼性(§21)
        rl = [x for x in items if x["dJ_halfA"] is not None]
        rs2 = [x for x in rl if x["dJ_halfA"] != 0 and x["dJ_half"] != 0]
        rep["stage_b"]["dJ_reference_reliability_16A_16B"] = {
            "n": len(rl),
            "pearson": round(corr([x["dJ_halfA"] for x in rl],
                                  [x["dJ_half"] for x in rl]) or 0, 4),
            "sign_agreement": round(sum(1 for x in rs2 if x["dJ_halfA"] * x["dJ_half"] > 0)
                                    / len(rs2), 4) if rs2 else None,
            "mean_abs_diff": round(st.mean([abs(x["dJ_halfA"] - x["dJ_half"])
                                            for x in rl]), 4)}
        # ΔL1 bin 別 ΔJ(§20)
        bins = [("<.01", 0, .01), (".01-.03", .01, .03), (".03-.05", .03, .05),
                (".05-.10", .05, .10), (">=.10", .10, 9)]
        rep["stage_b"]["by_dL1_bin"] = {}
        for nm, lo, hi in bins:
            s = [x for x in items if lo <= x["dL1"] < hi]      # TES が L1 で優位
            if len(s) < 5:
                continue
            rep["stage_b"]["by_dL1_bin"][nm] = {
                "n": len(s), "mean_dL1": round(st.mean([x["dL1"] for x in s]), 4),
                "mean_dJ": round(st.mean([x["dJ_half"] for x in s]), 4),
                "mean_dJ_full": round(st.mean([x["dJ_full"] for x in s]), 4)}
        # §40 transfer slope
        if len(A) > 10:
            X = np.vstack([np.asarray(A), np.ones(len(A))]).T
            beta, *_ = np.linalg.lstsq(X, np.asarray(B), rcond=None)
            pred = X @ beta
            r2 = 1 - ((np.asarray(B) - pred) ** 2).sum() / max(
                1e-12, ((np.asarray(B) - st.mean(B)) ** 2).sum())
            rs = np.random.RandomState(0)
            bs = []
            for _ in range(2000):
                idx = rs.randint(0, len(A), len(A))
                Xi = np.vstack([np.asarray(A)[idx], np.ones(len(A))]).T
                bi, *_ = np.linalg.lstsq(Xi, np.asarray(B)[idx], rcond=None)
                bs.append(bi[0])
            rep["stage_b"]["transfer_slope"] = {
                "beta": round(float(beta[0]), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                         round(float(np.percentile(bs, 97.5)), 4)],
                "r2": round(float(r2), 4),
                "note": "診断用の単純回帰。因果証明ではない(§42)"}

    # ---------- Stage C: V_t vs 実 terminal(同一 trajectory)----------
    sc = {}
    for ck in CK:
        vs, ys = [], []
        for g in G:
            for key in ("Q0", "S3"):
                for t in (g[key].get("trajs") or []):
                    v = t["ckpt"].get(ck)
                    o = t["outcome"]
                    if v is None or o is None or t["terminal"] is None:
                        continue
                    vs.append(v)
                    ys.append(o)
        if len(vs) < 50:
            continue
        yb = [1 if y == 1.0 else 0 for y in ys]
        mae = st.mean([abs(v - y) for v, y in zip(vs, ys)])
        brier = st.mean([(v - y) ** 2 for v, y in zip(vs, ys)])
        cal = {}
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            s = [(v, y) for v, y in zip(vs, ys) if lo <= v < hi]
            if len(s) >= 20:
                cal["%.1f-%.1f" % (lo, hi)] = {
                    "n": len(s), "mean_V": round(st.mean([v for v, _ in s]), 4),
                    "terminal": round(st.mean([y for _, y in s]), 4)}
        sc[ck] = {"n": len(vs), "pearson": round(corr(vs, ys), 4),
                  "spearman": round(corr(vs, ys, "s"), 4),
                  "auc": round(auc(yb, vs), 4) if auc(yb, vs) else None,
                  "mae": round(mae, 4), "brier": round(brier, 4), "calibration": cal}
    rep["stage_c"] = sc

    # ---------- Stage D: proxy-validity ladder(checkpoint 別)----------
    # 半分A の ΔV_ck と 半分B の ΔJ(独立)。どの地平が terminal の代理として最良か。
    lad = {}
    for ck in CK:
        xs, ys = [], []
        for g in G:
            te, tt = g["Q0"].get("trajs"), g["S3"].get("trajs")
            if not te or not tt:
                continue
            h = min(len(te), len(tt)) // 2
            if h < 4:
                continue

            def mv(trs, k=ck):
                v = [t["ckpt"].get(k) for t in trs if t["ckpt"].get(k) is not None]
                return st.mean(v) if len(v) >= 4 else None

            def mo(trs):
                v = [t["outcome"] for t in trs if t["terminal"] is not None]
                return st.mean(v) if len(v) >= 4 else None
            a, b = mv(te[:h]), mv(tt[:h])
            c, d = mo(te[h:]), mo(tt[h:])
            if None in (a, b, c, d):
                continue
            xs.append(b - a)
            ys.append(d - c)
        if len(xs) < 30:
            continue
        sg = [(x, y) for x, y in zip(xs, ys) if x != 0 and y != 0]
        lad[ck] = {"n": len(xs), "pearson": round(corr(xs, ys), 4),
                   "spearman": round(corr(xs, ys, "s"), 4),
                   "sign_agreement": round(sum(1 for x, y in sg if x * y > 0)
                                           / len(sg), 4) if sg else None,
                   "mean_dV": round(st.mean(xs), 4)}
    rep["stage_d_proxy_ladder"] = lad

    # ---------- Stage E: 減衰補正と分散分解 ----------
    # ΔL1 の split-half 信頼性(前半16を8+8に割る)-> 減衰補正 β
    q1, q2, ref = [], [], []
    for g in G:
        te, tt = g["Q0"].get("trajs"), g["S3"].get("trajs")
        if not te or not tt or min(len(te), len(tt)) < 16:
            continue
        h = min(len(te), len(tt)) // 2
        q = h // 2

        def c4q(trs, lo, hi):
            v = [t["ckpt"].get("C4") for t in trs[lo:hi] if t["ckpt"].get("C4") is not None]
            return st.mean(v) if len(v) >= 3 else None
        a1, b1 = c4q(te, 0, q), c4q(tt, 0, q)
        a2, b2 = c4q(te, q, h), c4q(tt, q, h)
        if None in (a1, b1, a2, b2):
            continue
        q1.append(b1 - a1)
        q2.append(b2 - a2)
    se = {}
    if len(q1) >= 30:
        r_half = corr(q1, q2) or 0.0
        # Spearman-Brown: 8+8 -> 16 continuations 相当の信頼性
        r16 = 2 * r_half / (1 + r_half) if r_half > -1 else 0.0
        b_obs = rep["stage_b"].get("transfer_slope", {}).get("beta")
        se = {"dL1_split_half_r_8v8": round(r_half, 4),
              "dL1_reliability_at_16": round(r16, 4),
              "beta_observed": b_obs,
              "beta_disattenuated": round(b_obs / r16, 4) if b_obs and r16 > 0.05 else None,
              "n": len(q1)}
        # 観測 ΔL1(+0.0097, Phase19.10)に適用した予測 ΔL2
        for nm, bb in (("observed", b_obs),
                       ("disattenuated", se.get("beta_disattenuated"))):
            if bb:
                se["predicted_dL2_at_dL1_0.0097_%s" % nm] = round(0.0097 * bb, 5)
    # sign agreement の天井補正
    ceil = rep["stage_b"].get("dJ_reference_reliability_16A_16B", {}).get("sign_agreement")
    obs = rep["stage_b"].get("sign_agreement")
    if ceil and obs and ceil > 0.5:
        se["sign_agreement_vs_reference_ceiling"] = {
            "observed": obs, "reference_ceiling": ceil,
            "normalized": round((obs - 0.5) / (ceil - 0.5), 4)}
    # base rate
    allo = [t["outcome"] for g in G for k in ("Q0", "S3")
            for t in (g[k].get("trajs") or []) if t["terminal"] is not None]
    se["terminal_base_rate"] = round(st.mean(allo), 4) if allo else None
    se["terminal_n"] = len(allo)
    rep["stage_e"] = se

    # ---------- 層別(§11/§12)----------
    strat = {}
    for key, fn in (("turn", lambda x: x["turn"]), ("cand", lambda x: x["cand"])):
        strat[key] = {}
        for k in sorted({fn(x) for x in items}):
            s = [x for x in items if fn(x) == k]
            if len(s) < 10:
                continue
            strat[key][k] = {"n": len(s),
                             "mean_dL1": round(st.mean([x["dL1"] for x in s]), 4),
                             "mean_dJ": round(st.mean([x["dJ_full"] for x in s]), 4),
                             "corr": round(corr([x["dL1"] for x in s],
                                                [x["dJ_half"] for x in s]) or 0, 4)}
    rep["strata"] = strat

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("groups", "stage_b", "stage_c", "strata")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
