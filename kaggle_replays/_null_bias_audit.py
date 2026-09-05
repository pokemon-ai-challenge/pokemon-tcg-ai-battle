"""Phase19.5 §3-§6: Phase19 の null systematic bias 監査(N0 gate)。

Phase19 で同一候補の独立2ブロック差が aggregate +0.0168 CI[+0.0034,+0.0298] となり、
対称なら 0 のはずの量が 0 を外れた。target 比較へ進む前にこれを調べる。

やること:
  1. 観測された signed delta の permutation test(A/B ラベル交換可能性の検定)
  2. group 単位 / candidate 単位のランダム swap
  3. ブロック平均を除去(centering)しても pairwise agreement が変わるか
     -> 定数オフセットは**候補の順位**を変えないので、reliability 低下の説明にはならない
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


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def _pairwise(a, b):
    tot = ok = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if b[i] == b[j]:
                continue
            tot += 1
            d = a[i] - a[j]
            ok += 0.5 if d == 0 else (1.0 if d * (b[i] - b[j]) > 0 else 0.0)
    return ok, tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_lh_w*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase195_null.json")
    args = ap.parse_args()
    G = [g for g in load(args.data)
         if g.get("is_audit") and all(c.get("lh_b") for c in g["candidates"])]
    rep = {"audit_groups": len(G),
           "candidates": sum(len(g["candidates"]) for g in G)}

    d = np.asarray([c["lh_a"]["mean"] - c["lh_b"]["mean"]
                    for g in G for c in g["candidates"]])
    ga = np.asarray([st.mean([c["lh_a"]["mean"] for c in g["candidates"]]) for g in G])
    gb = np.asarray([st.mean([c["lh_b"]["mean"] for c in g["candidates"]]) for g in G])
    rs = np.random.RandomState(0)

    def ci(x, B=10000):
        m = x[rs.randint(0, len(x), size=(B, len(x)))].mean(1)
        return [round(float(np.percentile(m, 2.5)), 4),
                round(float(np.percentile(m, 97.5)), 4)]

    rep["observed"] = {
        "mean_A": round(float(np.mean([c["lh_a"]["mean"] for g in G
                                       for c in g["candidates"]])), 4),
        "mean_B": round(float(np.mean([c["lh_b"]["mean"] for g in G
                                       for c in g["candidates"]])), 4),
        "candidate_level_signed_delta": round(float(d.mean()), 4),
        "candidate_level_ci95": ci(d),
        "group_level_signed_delta": round(float((ga - gb).mean()), 4),
        "group_level_ci95": ci(ga - gb),
        "mean_abs_delta": round(float(np.abs(d).mean()), 4)}

    # ---- §5 permutation test: A/B ラベルをランダムに入れ替える ----
    B = 20000
    flips = rs.choice([-1.0, 1.0], size=(B, len(d)))
    perm = (d[None, :] * flips).mean(1)
    obs = float(d.mean())
    p_two = float((np.abs(perm) >= abs(obs)).mean())
    gf = rs.choice([-1.0, 1.0], size=(B, len(G)))
    permg = ((ga - gb)[None, :] * gf).mean(1)
    p_g = float((np.abs(permg) >= abs(float((ga - gb).mean()))).mean())
    rep["permutation_test"] = {
        "resamples": B,
        "candidate_level_p_two_sided": round(p_two, 4),
        "group_level_p_two_sided": round(p_g, 4),
        "null_sd_candidate": round(float(perm.std()), 4),
        "note": "A/B ラベルが交換可能なら observed は null 分布内に収まるはず。"}

    # ---- §5 group / candidate 単位ランダム swap 後の signed delta ----
    sw_g = float(((ga - gb) * rs.choice([-1.0, 1.0], size=len(G))).mean())
    sw_c = float((d * rs.choice([-1.0, 1.0], size=len(d))).mean())
    rep["random_swap"] = {"group_swap_delta": round(sw_g, 4),
                          "candidate_swap_delta": round(sw_c, 4)}

    # ---- §22 の核心: 定数オフセットを除いても pairwise agreement は変わるか ----
    raw_ok = raw_tot = cen_ok = cen_tot = 0.0
    for g in G:
        a = [c["lh_a"]["mean"] for c in g["candidates"]]
        b = [c["lh_b"]["mean"] for c in g["candidates"]]
        o, t = _pairwise(a, b)
        raw_ok += o
        raw_tot += t
        ma, mb = st.mean(a), st.mean(b)
        o, t = _pairwise([x - ma for x in a], [x - mb for x in b])
        cen_ok += o
        cen_tot += t
    rep["centering_check"] = {
        "pairwise_raw": round(raw_ok / raw_tot, 4) if raw_tot else None,
        "pairwise_block_centered": round(cen_ok / cen_tot, 4) if cen_tot else None,
        "note": "block ごとの平均を引いても pairwise は変わらない = 定数オフセットは"
                "候補順位に影響しない。したがって Phase19 の低 reliability を"
                "systematic bias では説明できない。"}

    # ---- 効果量の文脈 ----
    rep["context"] = {
        "signed_delta_vs_mean_abs_delta": round(abs(obs) / float(np.abs(d).mean()), 4),
        "signed_delta_vs_group_spread": round(
            abs(obs) / st.mean([max(c["lh_a"]["mean"] for c in g["candidates"])
                                - min(c["lh_a"]["mean"] for c in g["candidates"])
                                for g in G]), 4)}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
