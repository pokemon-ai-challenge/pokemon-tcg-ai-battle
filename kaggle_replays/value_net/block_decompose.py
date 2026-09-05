"""Phase11D: 教師4ブロックの再現性・stable pair 分類・分散分解。

目的は「teacher_mean のうち、独立試行で再現する成分はどれだけか」を定量化し、
Search Distillation の target を T0〜T5 のどれにすべきか決めること。

用語は厳密に扱う: reliability は **teacher block reliability estimate** と呼び、
「真の説明可能割合」「理論上限」「Bayes 上限」とは呼ばない。
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train_action_q as T  # noqa: E402
from action_q import ActionQNet, CandidateSetQNet  # noqa: E402

TIE = 0.005          # 同点閾値(モデル結果を見る前に固定)
NB = 4


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]


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
    n = len(v)
    o = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[o[j + 1]] == v[o[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[o[k]] = avg
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
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


def block_scores(g, b):
    return [c["blocks"][b] for c in g["candidates"]]


def pair_class(g):
    """group 内の各候補ペアを stable / mostly / unstable / always_tie に分類。"""
    n = len(g["candidates"])
    out = []
    for i, j in itertools.combinations(range(n), 2):
        signs = []
        diffs = []
        for b in range(NB):
            d = g["candidates"][i]["blocks"][b] - g["candidates"][j]["blocks"][b]
            diffs.append(d)
            signs.append(0 if abs(d) < TIE else (1 if d > 0 else -1))
        nz = [s for s in signs if s != 0]
        mu = statistics.mean(diffs)
        if not nz:
            cls = "always_tie"
        else:
            pos, neg = nz.count(1), nz.count(-1)
            agree = max(pos, neg)
            cls = ("stable" if agree == NB else
                   "mostly" if agree == NB - 1 else "unstable")
        out.append({"i": i, "j": j, "cls": cls, "mean_margin": mu,
                    "abs_margin": abs(mu), "diffs": diffs,
                    "support": max(signs.count(1), signs.count(-1)) / NB})
    return out


def margin_band(m):
    a = abs(m)
    return ("very_small" if a < 0.01 else "small" if a < 0.03 else
            "medium" if a < 0.07 else "large")


def build_model(ck):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    kind = d["kind"]
    m = (ActionQNet(d["state_dim"], d["option_dim"], use_cards=False) if kind == "pool"
         else CandidateSetQNet(d["state_dim"], d["option_dim"], mode=kind.split(":", 1)[1]))
    m.load_state_dict(d["state_dict"])
    m.eval()
    return m, np.asarray(d["mean"]), np.asarray(d["std"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_teacher_blocks_all.jsonl.gz"))
    ap.add_argument("--frozen", default=str(_HERE / "frozen"))
    ap.add_argument("--out", default=str(_HERE / "block_decompose_report.json"))
    args = ap.parse_args()
    G = load(args.data)

    # ---- C. block ペア6組の再現性 ----
    def repro(rows):
        pw, sp, kd, t1, t2 = [], [], [], [], []
        for g in rows:
            for a, b in itertools.combinations(range(NB), 2):
                A, B = block_scores(g, a), block_scores(g, b)
                if max(A) == min(A) or max(B) == min(B):
                    continue
                v = _pairwise(A, B)
                if v is not None:
                    pw.append(v)
                v = _spearman(A, B)
                if v is not None:
                    sp.append(v)
                v = _kendall(A, B)
                if v is not None:
                    kd.append(v)
                t1.append(1.0 if A.index(max(A)) == B.index(max(B)) else 0.0)
                s = lambda V: set(sorted(range(len(V)), key=lambda i: V[i],  # noqa: E731
                                         reverse=True)[:2])
                t2.append(1.0 if s(A) == s(B) else 0.0)
        m = lambda x: round(statistics.mean(x), 4) if x else None  # noqa: E731
        sd = lambda x: round(statistics.pstdev(x), 4) if len(x) > 1 else None  # noqa: E731
        return {"n_blockpairs": len(pw), "pairwise": m(pw), "pairwise_sd": sd(pw),
                "spearman": m(sp), "kendall": m(kd), "top1": m(t1), "top2": m(t2)}

    # ---- D/E. stable pair 分類 ----
    allpairs = []
    for g in G:
        for p in pair_class(g):
            p.update({"turn_band": g["turn_band"], "cand_band": g["cand_band"],
                      "arch": g["arch"], "stratum": g["stratum"], "gid": g["group_id"]})
            allpairs.append(p)

    def cls_frac(ps):
        c = Counter(p["cls"] for p in ps)
        n = max(1, len(ps))
        return {"pairs": len(ps),
                **{k: round(c[k] / n, 4) for k in
                   ("stable", "mostly", "unstable", "always_tie")},
                "stable_plus_mostly": round((c["stable"] + c["mostly"]) / n, 4)}

    # ---- F. 分散分解 ----
    def decompose(ps):
        if not ps:
            return None
        mu = np.array([p["mean_margin"] for p in ps])
        # noise = 同一 pair の block 間分散(不偏)の平均
        noise = np.mean([np.var(np.array(p["diffs"]), ddof=1) for p in ps])
        total = np.var(mu, ddof=1)
        # mu は K=6×4 の平均なので、mu の分散は signal + noise/NB を含む
        signal = total - noise / NB
        rel = signal / (signal + noise) if (signal + noise) > 0 else None
        return {"pairs": len(ps),
                "var_of_mean_margins": round(float(total), 6),
                "within_pair_block_variance": round(float(noise), 6),
                "signal_variance_estimate": round(float(signal), 6),
                "signal_over_noise": round(float(signal / noise), 4) if noise > 0 else None,
                "reliability_estimate": round(float(rel), 4) if rel is not None else None,
                "negative_signal_flag": bool(signal < 0)}

    report = {
        "note": ("reliability は teacher block reliability estimate。"
                 "真の説明可能割合・理論上限とは呼ばない。"),
        "tie_threshold": TIE, "blocks": NB,
        "groups": len(G), "candidates": sum(len(g["candidates"]) for g in G),
        "block_reproducibility_all": repro(G),
        "by_stratum": {str(s): repro([g for g in G if g["stratum"] == s])
                       for s in (0, 1, 2)},
        "by_turn": {b: repro([g for g in G if g["turn_band"] == b])
                    for b in ("early", "middle", "late")},
        "by_cand": {b: repro([g for g in G if g["cand_band"] == b])
                    for b in ("small", "medium", "large")},
        "stable_pairs_all": cls_frac(allpairs),
        "stable_by_stratum": {str(s): cls_frac([p for p in allpairs if p["stratum"] == s])
                              for s in (0, 1, 2)},
        "stable_by_turn": {b: cls_frac([p for p in allpairs if p["turn_band"] == b])
                           for b in ("early", "middle", "late")},
        "stable_by_cand": {b: cls_frac([p for p in allpairs if p["cand_band"] == b])
                           for b in ("small", "medium", "large")},
        "stable_by_margin": {b: cls_frac([p for p in allpairs
                                          if margin_band(p["mean_margin"]) == b])
                             for b in ("very_small", "small", "medium", "large")},
        "variance_all": decompose(allpairs),
        "variance_by_turn": {b: decompose([p for p in allpairs if p["turn_band"] == b])
                             for b in ("early", "middle", "late")},
        "variance_by_cand": {b: decompose([p for p in allpairs if p["cand_band"] == b])
                             for b in ("small", "medium", "large")},
    }

    # ---- I. 凍結モデルを安定性別に再評価(教師は4ブロック平均)----
    F = Path(args.frozen)
    models = {}
    for n in ("Q0-old", "Q0-expanded", "QT", "QT-identity"):
        p = F / (n + ".pt")
        if p.exists():
            models[n] = build_model(p)

    def model_pair_acc(name, sel):
        """指定 pair 集合での正答率(教師 = 4ブロック平均の符号)。"""
        tot = ok = 0.0
        for g in G:
            ps = [p for p in pair_class(g) if sel(p)]
            if not ps:
                continue
            if name == "Policy":
                sc = [c["policy_score"] for c in g["candidates"]]
            else:
                m, sm, ss = models[name]
                gg = dict(g)
                gg["candidates"] = [dict(c, teacher_mean=0.0, teacher_std=0.0)
                                    for c in g["candidates"]]
                gg["outcome"] = 0
                gg["selected_option"] = None
                # Q0/QT は use_cards=False でカード集合を使わないが、make_batch は
                # zone テンソルを必ず組むので空リストを渡す(出力には影響しない)。
                gg.setdefault("hand_ids", [])
                gg.setdefault("discard_ids", [])
                gg.setdefault("opp_visible_ids", [])
                sc = T.score_groups(m, [gg], sm, ss)[0]
            for p in ps:
                d = sc[p["i"]] - sc[p["j"]]
                tot += 1
                ok += 0.5 if d == 0 else (1.0 if d * p["mean_margin"] > 0 else 0.0)
        return round(ok / tot, 4) if tot else None

    subsets = {
        "stable": lambda p: p["cls"] == "stable",
        "mostly": lambda p: p["cls"] == "mostly",
        "unstable": lambda p: p["cls"] == "unstable",
        "large_margin": lambda p: margin_band(p["mean_margin"]) == "large",
        "small_margin": lambda p: margin_band(p["mean_margin"]) in ("very_small", "small"),
        "all_nontie": lambda p: p["cls"] != "always_tie",
    }
    report["model_by_stability"] = {
        n: {k: model_pair_acc(n, f) for k, f in subsets.items()}
        for n in ["Policy"] + list(models)}

    # ---- H. target 候補の coverage ----
    n_all = len(allpairs)
    c = Counter(p["cls"] for p in allpairs)
    report["target_candidates"] = {
        "T0_raw_K6": {"coverage": 1.0, "note": "単一ブロック。ノイズをそのまま学習"},
        "T1_4block_mean": {"coverage": 1.0, "note": "分散1/4。生成コスト4倍"},
        "T2_majority": {"coverage": round((c["stable"] + c["mostly"]) / n_all, 4),
                        "note": "2対2はtieとして除外"},
        "T3_stable_only": {"coverage": round(c["stable"] / n_all, 4)},
        "T4_weighted_stable": {"coverage": round((c["stable"] + c["mostly"]) / n_all, 4),
                               "note": "stable=1.0 / mostly=0.5"},
        "T5_soft_confidence": {"coverage": round(1 - c["always_tie"] / n_all, 4),
                               "note": "block支持率をsoft target化"},
    }
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("by_turn", "by_cand", "stable_by_turn",
                                   "stable_by_cand", "variance_by_turn",
                                   "variance_by_cand", "by_stratum",
                                   "stable_by_stratum")},
                     ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
