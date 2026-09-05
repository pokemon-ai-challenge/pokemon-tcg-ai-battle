"""Phase11 §19-1,2: モデルを凍結(checkpoint保存)し、train/val/test の underfit/overfit を診断。

Phase10 までは学習後にメモリ上で評価して捨てていたため checkpoint が無い。
実ゲームA/B と独立評価には**固定された重み**が要るので、ここで保存する。
保存後は **test 結果を見て選び直さない**(model selection は validation pairwise のみ)。
"""
from __future__ import annotations

import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train_action_q as T  # noqa: E402
from eval_action_q_v2 import load  # noqa: E402


def sha_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(_HERE.parent / "_actionq_v2.jsonl.gz"))
    ap.add_argument("--old", default=str(_HERE.parent / "_actionq_dataset_main.jsonl.gz"))
    ap.add_argument("--rel", default=str(_HERE.parent / "_actionq_v2_reliability.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=760)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default=str(_HERE / "frozen"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

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

    class A:
        batch = args.batch
        lr = args.lr
        epochs = 1

    specs = {
        "Q0-old":      ("pool",        tr_old, va_old, True),
        "Q0-expanded": ("pool",        tr_new, va_new, True),
        "QT":          ("cset:attn",   tr_new, va_new, True),
        "QT-identity": ("cset:identity", tr_new, va_new, True),
    }
    report = {"seed": args.seed, "updates": args.updates,
              "teacher_ceiling": rel["all"]["AB_pairwise"],
              "models": {}}

    for name, (kind, tr, va, rank) in specs.items():
        A.epochs = max(1, round(args.updates / max(1, len(tr) / args.batch)))
        X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
        sm, ss = X.mean(0), X.std(0)
        ss[ss == 0] = 1.0
        t0 = time.time()
        m = T.train(tr, va, kind, False, True, rank, sm, ss, A, args.seed)
        ck = out / f"{name}.pt"
        torch.save({"state_dict": m.state_dict(), "kind": kind, "mean": sm, "std": ss,
                    "option_dim": len(tr[0]["candidates"][0]["option_feat"]),
                    "state_dim": len(sm), "seed": args.seed, "epochs": A.epochs}, ck)
        sc = lambda g: T.score_groups(m, [g], sm, ss)[0]  # noqa: E731
        d = {"checkpoint": str(ck), "checkpoint_sha": sha_of(ck),
             "params": sum(p.numel() for p in m.parameters()),
             "kind": kind, "epochs": A.epochs, "train_sec": round(time.time() - t0, 1),
             "train_groups": len(tr),
             "train": T.eval_ranker(tr, sc), "validation": T.eval_ranker(va, sc),
             "test": T.eval_ranker(te_new, sc),
             "high_conf": T.eval_ranker([g for g in te_new if g["_tier"] == "high"], sc)}
        for k in ("train", "validation", "test", "high_conf"):
            d[k].pop("_per_group", None)
        report["models"][name] = d
        print(f"  [{name}] train={d['train']['pairwise']} val={d['validation']['pairwise']} "
              f"test={d['test']['pairwise']} high={d['high_conf']['pairwise']}",
              file=sys.stderr, flush=True)

    # Policy anchor(学習不要)
    pol = lambda g: [c["policy_score"] for c in g["candidates"]]  # noqa: E731
    report["models"]["Policy"] = {
        "checkpoint": None, "params": 0, "kind": "anchor",
        "train": {k: v for k, v in T.eval_ranker(tr_new, pol).items() if k != "_per_group"},
        "validation": {k: v for k, v in T.eval_ranker(va_new, pol).items() if k != "_per_group"},
        "test": {k: v for k, v in T.eval_ranker(te_new, pol).items() if k != "_per_group"},
        "high_conf": {k: v for k, v in T.eval_ranker(
            [g for g in te_new if g["_tier"] == "high"], pol).items() if k != "_per_group"}}

    # underfit / overfit 判定
    for n, d in report["models"].items():
        tr_p, te_p = d["train"]["pairwise"], d["test"]["pairwise"]
        if tr_p is None or te_p is None:
            d["fit_verdict"] = "n/a"
            continue
        gap = tr_p - te_p
        d["train_test_gap"] = round(gap, 4)
        d["fit_verdict"] = ("overfit" if gap >= 0.10 else
                            "underfit" if tr_p < 0.72 else "mixed")
    report["data_sha"] = sha_of(args.v2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (out / "freeze_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
