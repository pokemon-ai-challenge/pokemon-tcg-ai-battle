"""Phase9C: Q0 再学習(old / balanced-small / expanded)と expanded test 評価。

比較の狙いは切り分け(§10.3):
    balanced-small > old  → **局面構成**の改善が効いた
    expanded > balanced-small → **データ量**の増加が効いた

条件を揃える:
  - モデル構造は Phase8 の Q0 と同一(state166 + option65 + action card ID)
  - Loss は 1.0*L_return + 1.0*L_rank + 0.2*L_policy_aux(固定)
  - **同一総update数**にそろえる(データ量差で学習量が変わらないように)
  - 正式教師は expanded test の **teacher_B**(層定義に使った teacher_A では評価しない)
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, statistics, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train_action_q as T  # noqa: E402


def load(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stratified_subsample(groups, n, seed=0):
    """turn_band × cand_band で比例配分して n 件抽出(局面構成を保つ)。"""
    rs = np.random.RandomState(seed)
    buckets = {}
    for g in groups:
        buckets.setdefault((g["turn_band"], g["cand_band"]), []).append(g)
    out, total = [], len(groups)
    for k, v in buckets.items():
        take = max(1, round(n * len(v) / total))
        idx = rs.choice(len(v), min(take, len(v)), replace=False)
        out += [v[i] for i in idx]
    rs.shuffle(out)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(_HERE.parent / "_actionq_v2.jsonl.gz"))
    ap.add_argument("--old", default=str(_HERE.parent / "_actionq_dataset_main.jsonl.gz"))
    ap.add_argument("--rel", default=str(_HERE.parent / "_actionq_v2_reliability.json"))
    ap.add_argument("--updates", type=int, default=760, help="全armで揃える総update数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--arms", default="", help="カンマ区切りで対象armを限定")
    ap.add_argument("--out", default=str(_HERE / "phase9c_report.json"))
    args = ap.parse_args()

    v2 = load(args.v2)
    tr_new = [g for g in v2 if g["split"] == 0]
    va_new = [g for g in v2 if g["split"] == 1]
    te_new = [g for g in v2 if g["split"] == 2]
    old = load(args.old)
    for g in old:
        g["_s"] = T.split_of(g)
    tr_old = [g for g in old if g["_s"] == 0]
    va_old = [g for g in old if g["_s"] == 1]

    rel = json.loads(Path(args.rel).read_text(encoding="utf-8"))
    tier = {r["group_id"]: r["tier"] for r in rel["records"]}
    for g in te_new:
        g["_tier"] = tier.get(g["group_id"], "other")

    small = stratified_subsample(tr_new, len(tr_old), seed=0)
    print(f"train: old={len(tr_old)} balanced_small={len(small)} expanded={len(tr_new)} "
          f"| test={len(te_new)}", file=sys.stderr)

    class A:
        batch = args.batch
        lr = args.lr
        epochs = 1
    KIND = {"Q0-old": "pool", "Q0-expanded": "pool",
            "QC-capacity": "cset:capacity", "QG-mean": "cset:mean",
            "QT-candidate": "cset:attn", "QT-identity": "cset:identity",
            "QT-no-ranking": "cset:attn"}
    arms = {"Q0-old": (tr_old, va_old), "Q0-expanded": (tr_new, va_new),
            "QC-capacity": (tr_new, va_new), "QG-mean": (tr_new, va_new),
            "QT-candidate": (tr_new, va_new), "QT-identity": (tr_new, va_new),
            "QT-no-ranking": (tr_new, va_new)}
    report = {"updates_per_arm": args.updates, "batch": args.batch,
              "seeds": args.seeds, "bootstrap": args.boot,
              "teacher_ceiling_AB_pairwise": rel["all"]["AB_pairwise"],
              "teacher_ceiling_high": rel["high"]["AB_pairwise"],
              "train_sizes": {k: len(v[0]) for k, v in arms.items()},
              "test_groups": len(te_new),
              "arms": {}, "per_group": {}}

    # Policy anchor
    pol = T.eval_ranker(te_new, lambda g: [c["policy_score"] for c in g["candidates"]])
    report["arms"]["Policy"] = {"all": {k: v for k, v in pol.items() if k != "_per_group"}}
    report["per_group"]["Policy"] = pol["_per_group"]["pairwise"]
    for tname in ("high", "other"):
        sub = [g for g in te_new if g["_tier"] == tname]
        r = T.eval_ranker(sub, lambda g: [c["policy_score"] for c in g["candidates"]])
        report["arms"]["Policy"][tname] = {k: v for k, v in r.items() if k != "_per_group"}

    only = [x for x in args.arms.split(",") if x]
    for name, (tr, va) in arms.items():
        if only and name not in only:
            continue
        A.epochs = max(1, round(args.updates / max(1, len(tr) / args.batch)))
        per, pg = [], []
        t0 = time.time()
        for seed in [int(s) for s in args.seeds.split(",")]:
            X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
            sm, ss = X.mean(0), X.std(0)
            ss[ss == 0] = 1.0
            m = T.train(tr, va, KIND[name], False, True,
                        name != "QT-no-ranking", sm, ss, A, seed)
            res = {"all": T.eval_ranker(te_new, lambda g: T.score_groups(m, [g], sm, ss)[0])}
            pg.append(res["all"]["_per_group"]["pairwise"])
            for tname in ("high", "other"):
                sub = [g for g in te_new if g["_tier"] == tname]
                res[tname] = T.eval_ranker(sub, lambda g: T.score_groups(m, [g], sm, ss)[0])
            for band in ("early", "middle", "late"):
                sub = [g for g in te_new if g["turn_band"] == band]
                res[band] = T.eval_ranker(sub, lambda g: T.score_groups(m, [g], sm, ss)[0])
            for cb in ("small", "medium", "large"):
                sub = [g for g in te_new if g["cand_band"] == cb]
                res[cb] = T.eval_ranker(sub, lambda g: T.score_groups(m, [g], sm, ss)[0])
            per.append(res)
        agg = {"epochs": A.epochs, "train_sec": round(time.time() - t0, 1)}
        for scope in ("all", "high", "other", "early", "middle", "late",
                      "small", "medium", "large"):
            agg[scope] = {}
            for k in ("pairwise", "spearman", "kendall", "top1", "top2_recall", "ndcg", "regret"):
                vals = [p[scope][k] for p in per if p[scope].get(k) is not None]
                agg[scope][k] = round(statistics.mean(vals), 4) if vals else None
            agg[scope]["n_groups"] = per[0][scope]["n_groups"]
        agg["per_seed_pairwise"] = [p["all"]["pairwise"] for p in per]
        report["arms"][name] = agg
        n = min(len(x) for x in pg)
        report["per_group"][name] = [statistics.mean([x[i] for x in pg]) for i in range(n)]
        print(f"  [{name}] epochs={A.epochs} test pairwise={agg['all']['pairwise']} "
              f"spearman={agg['all']['spearman']}", file=sys.stderr, flush=True)

    rs = np.random.RandomState(0)

    def boot(a, b):
        x, y = report["per_group"].get(a), report["per_group"].get(b)
        if not x or not y:
            return None
        n = min(len(x), len(y))
        d = np.asarray(x[:n]) - np.asarray(y[:n])
        idx = rs.randint(0, n, size=(args.boot, n))
        mm = d[idx].mean(axis=1)
        return {"mean_diff": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(mm, 2.5)), 4),
                         round(float(np.percentile(mm, 97.5)), 4)], "n": n}

    report["bootstrap"] = {
        "QT - Q0-old": boot("QT-candidate", "Q0-old"),
        "QT - Q0-expanded": boot("QT-candidate", "Q0-expanded"),
        "QT - QC": boot("QT-candidate", "QC-capacity"),
        "QT - QG": boot("QT-candidate", "QG-mean"),
        "QT - QT-identity": boot("QT-candidate", "QT-identity"),
        "QT - Policy": boot("QT-candidate", "Policy"),
        "expanded - old": boot("Q0-expanded", "Q0-old"),
    }
    report.pop("per_group")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
