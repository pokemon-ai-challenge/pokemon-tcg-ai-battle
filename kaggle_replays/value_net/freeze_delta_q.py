"""Phase15 §3/§28: Phase16 で使う Delta-Q を1つに固定して checkpoint 化する。

学習条件は正式比較と完全に同一。再現できたことを phase15_delta.json の
per-seed 記録との一致で検証してから保存する(不一致なら異常終了)。
seed は「最小 seed = 0」の事前規則。test を見て選ばない。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import soft_eval as SE  # noqa: E402
import train_delta_q as TD  # noqa: E402
import train_soft as TS  # noqa: E402

KEYS = ("stable", "support_weighted", "margin_weighted", "large_margin",
        "all_pairwise", "regret", "spearman")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_dlt_probes_*.jsonl.gz"))
    ap.add_argument("--report", default=str(_HERE / "phase15_delta.json"))
    ap.add_argument("--arm", default="D1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default=str(_HERE / "frozen" / "DeltaQ_D1.pt"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    groups = []
    for f in sorted(glob.glob(args.data)):
        groups += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    TS.stratified_split(groups)
    TS.attach_pairs(groups)
    TD.featurize(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    model = TD.train_arm(tr, va, args.arm, smean, sstd, args, args.seed)
    m = SE.evaluate(te, lambda g: TD.score_group(model, g, smean, sstd, args.arm))
    got = {k: m[k] for k in KEYS}

    ref = json.loads(Path(args.report).read_text(encoding="utf-8"))["arms"][args.arm]
    want = {k: ref["per_seed_" + k][args.seed] for k in
            ("stable", "support_weighted", "margin_weighted", "regret")}
    bad = {k: (got[k], want[k]) for k in want if abs(got[k] - want[k]) > 1e-6}
    if bad:
        print(json.dumps({"error": "reproduction mismatch", "mismatch": bad},
                         ensure_ascii=False, indent=2))
        sys.exit(1)

    D = len(tr[0]["candidates"][0]["option_feat"])
    ck = {"arch": "DeltaQNet", "arm": args.arm, "delta_kind": TD.ARMS[args.arm][0],
          "delta_dim": TD.delta_width(args.arm), "state_dim": len(smean), "option_dim": D,
          "hidden": args.hidden, "seed": args.seed, "epochs": args.epochs,
          "batch": args.batch, "lr": args.lr,
          "state_dict": model.state_dict(), "mean": smean.tolist(), "std": sstd.tolist(),
          "target": "Phase12 S3 (block-support soft + confidence)",
          "split": f"train/val/test={len(tr)}/{len(va)}/{len(te)} game-level stratified",
          "test_metrics": got,
          "note": "Phase15 正式比較と同一条件で再現した Delta-Q。production 未採用。"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(json.dumps({"saved": args.out, "arm": args.arm, "seed": args.seed,
                      "sha256_16": hashlib.sha256(Path(args.out).read_bytes()).hexdigest()[:16],
                      "test_metrics": got, "n_params": sum(p.numel() for p in model.parameters())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
