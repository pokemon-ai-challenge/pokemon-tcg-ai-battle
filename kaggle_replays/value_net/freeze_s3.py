"""Phase13 §28-1: Phase12 の S3(seed 0)を**決定的に再現**して checkpoint 化する。

Phase12 の `train_soft.py` はモデルを保存しなかったため、L1/L2 で使う S3 の実体が無い。
ここでは学習条件を一切変えず(同一データ・同一 split・同一 seed・同一 epoch/lr/batch)、
同じ経路をもう一度たどって重みを取り出す。**再学習ではなく再現**であることを、
Phase12 が記録した test 指標との完全一致で検証する(一致しなければ異常終了)。

seed は「最小 seed = 0」を事前規則として採用する。3 seed の test 結果を見て選ばない(§25)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import soft_eval as SE  # noqa: E402
import train_soft as TS  # noqa: E402

CHECK_KEYS = ("stable", "mostly", "support_weighted", "margin_weighted",
              "large_margin", "all_pairwise", "regret", "spearman")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_teacher_blocks_p12.jsonl.gz"))
    ap.add_argument("--report", default=str(_HERE / "phase12_report.json"))
    ap.add_argument("--arm", default="S3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(_HERE / "frozen" / "S3.pt"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    groups = TS.load(Path(args.data))
    TS.stratified_split(groups)
    TS.attach_pairs(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    model, _hist = TS.train_arm(tr, va, args.arm, smean, sstd, args, args.seed)
    m = SE.evaluate(te, lambda g: TS.score_group(model, g, smean, sstd))
    got = {k: m[k] for k in CHECK_KEYS}

    ref = json.loads(Path(args.report).read_text(encoding="utf-8"))
    want = ref["per_seed"][args.arm][args.seed]
    mismatch = {k: (got[k], want[k]) for k in CHECK_KEYS
                if got[k] is None or abs(got[k] - want[k]) > 1e-6}
    if mismatch:
        print(json.dumps({"error": "reproduction mismatch — 再現できていないので凍結しない",
                          "mismatch": mismatch}, ensure_ascii=False, indent=2))
        sys.exit(1)

    D = len(tr[0]["candidates"][0]["option_feat"])
    ck = {"kind": "pool", "state_dim": len(smean), "option_dim": D,
          "state_dict": model.state_dict(), "mean": smean.tolist(), "std": sstd.tolist(),
          "arm": args.arm, "seed": args.seed, "epochs": args.epochs,
          "batch": args.batch, "lr": args.lr,
          "use_cards": False, "use_action_card": True,
          "dataset_sha": hashlib.sha256(Path(args.data).read_bytes()).hexdigest()[:16],
          "split": "train/val/test = %d/%d/%d (game-level stratified, seed 0)"
                   % (len(tr), len(va), len(te)),
          "target": "block-support soft + confidence weight (Phase12 S3)",
          "note": "Phase12 の学習条件を変更せず決定的に再現したもの。"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()[:16]
    print(json.dumps({"reproduced": True, "arm": args.arm, "seed": args.seed,
                      "test_metrics": got, "checkpoint": args.out, "sha256_16": sha,
                      "dataset_sha": ck["dataset_sha"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
