"""Phase19.13 §14-§21 / §26-§34 / §37-§45: leaf ranking・shortcut・tie の terminal 品質。

terminal reference は **決定化された世界**の中の結果。候補間は同一決定化で paired なので
順位付けの評価に使えるが、絶対勝率としては読まない。
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

TIE_EPS = 0.02


def load(p):
    return [json.loads(l) for f in sorted(glob.glob(p))
            for l in gzip.open(f, "rt", encoding="utf-8")]


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
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db)
            if da > 0 and db > 0 else None)


def auc(y, s):
    p = sorted(zip(s, y))
    r, i = {}, 0
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


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def boot(v, B=10000, seed=0):
    a = np.asarray(v, float)
    rs = np.random.RandomState(seed)
    m = a[rs.randint(0, len(a), size=(B, len(a)))].mean(1)
    return {"mean": round(float(a.mean()), 4), "n": len(a),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)]}


def cand_means(g, dets=None):
    """候補ごとの (mean_h, mean_j)。dets 指定時はその決定化だけで再計算(split-half 用)。"""
    if dets is None:
        return {c["idx"]: (c["mean_h"], c["mean_j"], c["policy_prob"]) for c in g["cands"]}
    acc, full = {}, {}
    for lf in g["leaves"]:
        if lf["j"] is None:
            continue
        full.setdefault(lf["cand"], []).append(lf["j"])
        if lf["det"] in dets:
            acc.setdefault(lf["cand"], [[], []])
            acc[lf["cand"]][0].append(lf["h"])
            acc[lf["cand"]][1].append(lf["j"])
    pp = {c["idx"]: c["policy_prob"] for c in g["cands"]}
    # 順位付け側(h)だけ dets を絞り、参照 J は常に全決定化平均を使う。
    return {i: (st.mean(h), st.mean(full[i]), pp.get(i, 0.0))
            for i, (h, j) in acc.items() if len(full.get(i, [])) >= 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_p1913q_q*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1913_quality.json")
    args = ap.parse_args()
    G = load(args.data)
    S = [g for g in G if g["kind"] == "searched"]
    C = [g for g in G if g["kind"] == "shortcut"]
    rep = {"groups": len(G), "searched": len(S), "shortcut": len(C),
           "games": len({g["game"] for g in G}),
           "turn_band": dict(Counter(g["turn_band"] for g in G)),
           "arch": dict(Counter(g["arch"] for g in G))}

    # ================= C. leaf absolute validity(葉単位)=================
    hs, js = [], []
    tsteps = []
    for g in S:
        for lf in g["leaves"]:
            if lf["j"] is None:
                continue
            hs.append(lf["h"])
            js.append(lf["j"])
            tsteps.append(lf["steps"])
    yb = [1 if y == 1.0 else 0 for y in js]
    rep["leaf_absolute"] = {
        "leaf_states": len(hs),
        "pearson": round(corr(hs, js), 4), "spearman": round(corr(hs, js, "s"), 4),
        "auc": round(auc(yb, hs), 4) if auc(yb, hs) else None,
        "mae": round(st.mean([abs(a - b) for a, b in zip(hs, js)]), 4),
        "brier": round(st.mean([(a - b) ** 2 for a, b in zip(hs, js)]), 4),
        "terminal_base_rate": round(st.mean(js), 4),
        "mean_continuation_steps": round(st.mean(tsteps), 1),
        "calibration": {}}
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        s = [(a, b) for a, b in zip(hs, js) if lo <= a < hi]
        if len(s) >= 20:
            rep["leaf_absolute"]["calibration"]["%.1f-%.1f" % (lo, hi)] = {
                "n": len(s), "mean_h": round(st.mean([a for a, _ in s]), 4),
                "terminal": round(st.mean([b for _, b in s]), 4)}

    # ================= D. within-root candidate ranking =================
    def rank_metrics(groups, dets=None):
        ok = tot = 0.0
        top1, sps, regret, oracle_gap = [], [], [], []
        for g in groups:
            cm = cand_means(g, dets)
            if len(cm) < 2:
                continue
            idx = list(cm)
            for a, b in itertools.combinations(idx, 2):
                ja, jb = cm[a][1], cm[b][1]
                if ja == jb:
                    continue
                tot += 1
                d = cm[a][0] - cm[b][0]
                ok += 0.5 if d == 0 else (1.0 if d * (ja - jb) > 0 else 0.0)
            bi = max(idx, key=lambda i: cm[i][0])
            best_j = max(cm[i][1] for i in idx)
            top1.append(1.0 if cm[bi][1] == best_j else 0.0)
            regret.append(best_j - cm[bi][1])
            oracle_gap.append(best_j - min(cm[i][1] for i in idx))
            sp = corr([cm[i][0] for i in idx], [cm[i][1] for i in idx], "s")
            if sp is not None:
                sps.append(sp)
        return {"groups": len(regret),
                "pairwise": round(ok / tot, 4) if tot else None, "pairs": int(tot),
                "top1_agreement": round(st.mean(top1), 4) if top1 else None,
                "spearman": round(st.mean(sps), 4) if sps else None,
                "terminal_regret": round(st.mean(regret), 4) if regret else None,
                "regret_ci": boot(regret)["ci95"] if regret else None,
                "oracle_spread": round(st.mean(oracle_gap), 4) if oracle_gap else None}
    rep["leaf_ranking"] = rank_metrics(S)
    # production は 400ms 予算で平均 1.7 決定化しか完了しない(実測)。
    # 同じ handcrafted 評価器を「production が実際に見ている本数」で回した場合。
    rep["leaf_ranking_at_production_budget"] = {
        "det1": rank_metrics(S, {0}), "det2": rank_metrics(S, {0, 1}),
        "det4": rank_metrics(S, {0, 1, 2, 3}),
        "note": "J は常に全8決定化の平均を参照(順位付け側だけ本数を絞る)"}

    # 参照値ノイズ床: 決定化 0-3 の J で順位付け -> 4-7 の J を予測(handcrafted と同じ土俵)
    ok = tot = 0.0
    t1, rg = [], []
    def half_j(g, dets):
        acc = {}
        for lf in g["leaves"]:
            if lf["j"] is not None and lf["det"] in dets:
                acc.setdefault(lf["cand"], []).append(lf["j"])
        return {i: (0.0, st.mean(v), 0.0) for i, v in acc.items() if len(v) >= 2}
    for g in S:
        A, B = half_j(g, {0, 1, 2, 3}), half_j(g, {4, 5, 6, 7})
        idx = [i for i in A if i in B]
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(idx, 2):
            if B[a][1] == B[b][1]:
                continue
            tot += 1
            d = A[a][1] - A[b][1]
            ok += 0.5 if d == 0 else (1.0 if d * (B[a][1] - B[b][1]) > 0 else 0.0)
        bi = max(idx, key=lambda i: A[i][1])
        t1.append(1.0 if B[bi][1] == max(B[i][1] for i in idx) else 0.0)
        rg.append(max(B[i][1] for i in idx) - B[bi][1])
    rep["reference_noise_floor"] = {
        "groups": len(rg), "pairwise_self": round(ok / tot, 4) if tot else None,
        "top1_self": round(st.mean(t1), 4) if t1 else None,
        "regret_self": round(st.mean(rg), 4) if rg else None,
        "note": "M=4 vs M=4 の自己一致。handcrafted(M=8)の上限ではないが順位ノイズの目安"}

    # Policy 単体を同じ土俵で(対照)
    okp = totp = 0.0
    rp = []
    for g in S:
        cm = cand_means(g)
        idx = list(cm)
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(idx, 2):
            if cm[a][1] == cm[b][1]:
                continue
            totp += 1
            d = cm[a][2] - cm[b][2]
            okp += 0.5 if d == 0 else (1.0 if d * (cm[a][1] - cm[b][1]) > 0 else 0.0)
        bi = max(idx, key=lambda i: cm[i][2])
        rp.append(max(cm[i][1] for i in idx) - cm[bi][1])
    rep["policy_only_control"] = {
        "pairwise": round(okp / totp, 4) if totp else None,
        "terminal_regret": round(st.mean(rp), 4) if rp else None, "groups": len(rp)}

    # ================= E. margin bins(§17)=================
    bins = [(0, .01, "<.01"), (.01, .02, ".01-.02"), (.02, .05, ".02-.05"),
            (.05, .10, ".05-.10"), (.10, 9, ">=.10")]
    mb = {}
    for lo, hi, nm in bins:
        ok = tot = 0.0
        dj = []
        for g in S:
            cm = cand_means(g)
            for a, b in itertools.combinations(list(cm), 2):
                dh = cm[a][0] - cm[b][0]
                if not (lo <= abs(dh) < hi):
                    continue
                dJ = cm[a][1] - cm[b][1]
                tot += 1
                ok += 1.0 if dh * dJ > 0 else (0.5 if dJ == 0 else 0.0)
                dj.append(abs(dJ))
                # handcrafted が優位とする側の terminal 差(符号付き)
                mb.setdefault("_signed", []).append((nm, dJ if dh > 0 else -dJ))
        if tot:
            mb[nm] = {"pairs": int(tot), "terminal_agreement": round(ok / tot, 4),
                      "agreement_ci": wilson(round(ok), int(tot)),
                      "mean_abs_terminal_gap": round(st.mean(dj), 4)}
    sg = mb.pop("_signed", [])
    for nm in list(mb):
        v = [x for k, x in sg if k == nm]
        if v:
            mb[nm]["mean_signed_terminal_gap"] = round(st.mean(v), 4)
    rep["margin_bins"] = mb

    # ================= tie_eps 妥当性(§18/§37-§42)=================
    tie = {"tie_eps": TIE_EPS}
    near = [x for k, x in sg if k in ("<.01", ".01-.02")]
    far = [x for k, x in sg if k in (".02-.05", ".05-.10", ">=.10")]
    tie["mean_signed_terminal_gap_within_eps"] = round(st.mean(near), 4) if near else None
    tie["mean_signed_terminal_gap_beyond_eps"] = round(st.mean(far), 4) if far else None
    tie["abs_terminal_gap_within_eps"] = round(st.mean([abs(x) for x in near]), 4) if near else None
    tie["abs_terminal_gap_beyond_eps"] = round(st.mean([abs(x) for x in far]), 4) if far else None
    # production の tie-break(Policy 確率上位)vs 平均最大 の terminal 比較
    d_tb, n_tie, n_changed = [], 0, 0
    for g in S:
        cm = cand_means(g)
        if len(cm) < 2:
            continue
        idx = list(cm)
        bi = max(idx, key=lambda i: cm[i][0])
        best = cm[bi][0]
        cont = [i for i in idx if best - cm[i][0] <= TIE_EPS]
        if len(cont) < 2:
            continue
        n_tie += 1
        pick = max(cont, key=lambda i: cm[i][2])          # production の tie-break
        if pick != bi:
            n_changed += 1
            d_tb.append(cm[pick][1] - cm[bi][1])
    tie["roots_with_tie_set"] = n_tie
    tie["tie_break_changed"] = n_changed
    tie["tie_break_minus_argmax_terminal"] = boot(d_tb) if d_tb else None
    rep["tie"] = tie

    # ================= F. leaf error set(§20/§21)=================
    err = []
    for g in S:
        cm = cand_means(g)
        for a, b in itertools.combinations(list(cm), 2):
            dh, dJ = cm[a][0] - cm[b][0], cm[a][1] - cm[b][1]
            if abs(dh) >= 0.05 and abs(dJ) >= 0.10 and dh * dJ < 0:
                wa, lo_ = (a, b) if dh > 0 else (b, a)     # handcrafted が推した方 / 実は良い方
                err.append({"gid": g["group_id"], "turn": g["turn"], "arch": g["arch"],
                            "dh": round(abs(dh), 4), "dJ": round(abs(dJ), 4),
                            "pref": wa, "better": lo_,
                            "terms_pref": _mean_terms(g, wa), "terms_better": _mean_terms(g, lo_)})
    err.sort(key=lambda x: -(x["dh"] * x["dJ"]))
    rep["error_set"] = {"n": len(err),
                        "rate_of_high_margin_pairs": None,
                        "examples": err[:12]}
    # §21 taxonomy: handcrafted が見ている4項のどれが誤選択を駆動したか
    tax: Counter = Counter()
    for e in err[:200]:
        p, b = e["terms_pref"], e["terms_better"]
        if not p or not b:
            tax["unknown"] += 1
            continue
        c = {"prize": 1.20 * (p["prize"] - b["prize"]),
             "board": 0.05 * (p["board"] - b["board"]),
             "energy": 0.15 * (p["energy"] - b["energy"]),
             "bench": 0.10 * (p["bench"] - b["bench"])}
        drv = max(c, key=lambda k: c[k])
        tax[drv if c[drv] > 0 else "no_positive_driver"] += 1
        # 補助シグナル: 誤選択側の方が手札/山が細っていないか
        if p["hand"] < b["hand"] - 1:
            tax["+hand_resource_loss"] += 1
        if 0 <= p["deck_me"] < b["deck_me"] - 1:
            tax["+deck_burn"] += 1
    rep["error_taxonomy"] = dict(tax.most_common())

    # ================= G. shortcut(§26-§34)=================
    dis = [g for g in C if g.get("disagree")]
    ev = [g for g in dis if g.get("j_prod") and g.get("j_cf")
          and len(g["j_prod"]) >= 4 and len(g["j_cf"]) >= 4]
    dj = [st.mean(g["j_cf"]) - st.mean(g["j_prod"]) for g in ev]
    rep["shortcut"] = {
        "roots": len(C), "disagree": len(dis),
        "disagreement_rate": round(len(dis) / max(1, len(C)), 4),
        "disagreement_ci": wilson(len(dis), len(C)),
        "evaluated": len(ev),
        "delta_J_fullsearch_minus_shortcut": boot(dj) if dj else None,
        "false_skip_rate_of_disagreements": round(
            sum(1 for x in dj if x > 0) / len(dj), 4) if dj else None,
        "mean_cf_ms": round(st.mean([g["cf_ms"] for g in C if g.get("cf_ms")]), 1) if C else None}

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {k: rep[k] for k in ("groups", "searched", "shortcut", "leaf_absolute",
                               "leaf_ranking", "reference_noise_floor",
                               "policy_only_control", "margin_bins", "tie",
                               "error_taxonomy", "shortcut")}
    out["leaf_absolute"] = {k: v for k, v in rep["leaf_absolute"].items() if k != "calibration"}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _mean_terms(g, cand):
    t = [lf["terms"] for lf in g["leaves"] if lf["cand"] == cand and lf.get("terms")]
    if not t:
        return None
    return {k: round(st.mean([x[k] for x in t]), 3) for k in t[0]}


if __name__ == "__main__":
    main()
