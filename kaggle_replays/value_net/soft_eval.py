"""Phase12: soft distillation の評価指標(§14/§15)。

**主指標は stable pair accuracy**。全 pair 等重み pairwise は near-tie(42.6%)に支配されるため
参考値に降格する(§4)。

各指標は group ごとの (numerator, denominator) を保持し、group 単位 paired bootstrap
(§17)でそのまま再集計できるようにする。
"""
from __future__ import annotations

import math
import statistics

import soft_target as ST


def _dir_support(p):
    """block-support から向きを取る。0.5 ちょうど(=情報なし)は None。"""
    return None if p == 0.5 else (1 if p > 0.5 else -1)


def _dir_margin(m):
    return None if m == 0 else (1 if m > 0 else -1)


def _correct(pred_diff, direction):
    if pred_diff == 0:
        return 0.5
    return 1.0 if pred_diff * direction > 0 else 0.0


def _spearman(a, b):
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
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


# 指標名 -> (pair を採るか, 重み, 向きの取り方)
_METRICS = {
    "all_pairwise":     (lambda p: p["margin_band"] != "__none__", lambda p: 1.0, "margin"),
    "stable":           (lambda p: p["cls"] == "stable", lambda p: 1.0, "support"),
    "mostly":           (lambda p: p["cls"] == "mostly", lambda p: 1.0, "support"),
    "support_weighted": (lambda p: p["confidence"] > 0, lambda p: p["confidence"], "support"),
    "margin_weighted":  (lambda p: p["margin_weight"] > 0, lambda p: p["margin_weight"], "margin"),
    "medium_margin":    (lambda p: p["margin_band"] == "medium", lambda p: 1.0, "margin"),
    "large_margin":     (lambda p: p["margin_band"] == "large", lambda p: 1.0, "margin"),
    "near_tie":         (lambda p: p["margin_band"] == "very_small", lambda p: 1.0, "margin"),
}


def group_metrics(pairs, pred, teacher_mean):
    """1 group 分の評価。各指標について (num, den) と、regret / spearman / 予測差を返す。"""
    out = {}
    for name, (keep, wfn, dsrc) in _METRICS.items():
        num = den = 0.0
        for p in pairs:
            if not keep(p):
                continue
            d = _dir_support(p["support"]) if dsrc == "support" else _dir_margin(p["mean_margin"])
            if d is None:
                continue
            w = wfn(p)
            if w <= 0:
                continue
            num += w * _correct(pred[p["i"]] - pred[p["j"]], d)
            den += w
        out[name] = (num, den)

    bi = max(range(len(pred)), key=lambda i: pred[i])
    ti = max(range(len(teacher_mean)), key=lambda i: teacher_mean[i])
    out["_regret"] = teacher_mean[ti] - teacher_mean[bi]
    out["_top1"] = 1.0 if bi == ti else 0.0
    out["_spearman"] = _spearman(pred, teacher_mean)
    # §19.1 near-tie で |Q_i-Q_j| が小さくなるか / §19.3 model confidence
    out["_absq"] = {b: [] for b in ("very_small", "small", "medium", "large")}
    out["_calib"] = []
    out["_modelconf"] = []
    for p in pairs:
        dq = pred[p["i"]] - pred[p["j"]]
        out["_absq"][p["margin_band"]].append(abs(dq))
        sg = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, dq))))
        out["_calib"].append((p["support"], sg))
        out["_modelconf"].append((abs(sg - 0.5), p["confidence"], p["abs_margin"]))
    return out


def aggregate(per_group):
    """group ごとの結果 -> 全体指標。"""
    agg = {}
    for name in _METRICS:
        num = sum(g[name][0] for g in per_group)
        den = sum(g[name][1] for g in per_group)
        agg[name] = round(num / den, 4) if den > 0 else None
        agg[name + "_n"] = int(round(den))
    rg = [g["_regret"] for g in per_group]
    agg["regret"] = round(statistics.mean(rg), 4) if rg else None
    agg["regret_median"] = round(statistics.median(rg), 4) if rg else None
    agg["regret_p90"] = round(sorted(rg)[int(0.9 * (len(rg) - 1))], 4) if rg else None
    agg["regret_large_rate"] = round(sum(1 for r in rg if r >= 0.07) / len(rg), 4) if rg else None
    agg["top1"] = round(statistics.mean([g["_top1"] for g in per_group]), 4) if per_group else None
    sp = [g["_spearman"] for g in per_group if g["_spearman"] is not None]
    agg["spearman"] = round(statistics.mean(sp), 4) if sp else None

    absq = {}
    for b in ("very_small", "small", "medium", "large"):
        v = [x for g in per_group for x in g["_absq"][b]]
        absq[b] = round(statistics.mean(v), 4) if v else None
    agg["abs_q_diff_by_margin"] = absq
    # BT loss はスケール不変なので、絶対値でなく near-tie / large の**比**で判定する
    agg["abs_q_ratio_neartie_over_large"] = (
        round(absq["very_small"] / absq["large"], 4)
        if absq["very_small"] is not None and absq["large"] else None)

    bins = {}
    ce_num = ce_den = 0.0
    for g in per_group:
        for s, pr in g["_calib"]:
            k = round(min(1.0, max(0.0, s)) * 4) / 4.0
            bins.setdefault(k, []).append(pr)
            ce_num += abs(pr - s)
            ce_den += 1
    agg["calibration"] = {str(k): {"pairs": len(v), "pred_mean": round(statistics.mean(v), 4)}
                          for k, v in sorted(bins.items())}
    agg["calibration_error"] = round(ce_num / ce_den, 4) if ce_den else None

    mc = [x for g in per_group for x in g["_modelconf"]]
    if len(mc) > 2:
        agg["corr_modelconf_supportconf"] = round(_corr([a for a, _, _ in mc],
                                                        [b for _, b, _ in mc]), 4)
        agg["corr_modelconf_margin"] = round(_corr([a for a, _, _ in mc],
                                                   [c for _, _, c in mc]), 4)
    return agg


def _corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else 0.0


def evaluate(groups, score_fn, n_blocks=ST.NB):
    per = []
    for g in groups:
        pairs = ST.group_pairs(g, n_blocks=n_blocks)
        if not pairs:
            continue
        pred = score_fn(g)
        if pred is None or len(pred) != len(g["candidates"]):
            continue
        tm = [statistics.mean(c["blocks"][:n_blocks]) for c in g["candidates"]]
        per.append(group_metrics(pairs, pred, tm))
    out = aggregate(per)
    out["n_groups"] = len(per)
    out["_per_group"] = per
    return out
