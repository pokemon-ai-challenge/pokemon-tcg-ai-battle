"""Phase12A §13: soft target の暗記・calibration 診断。

soft target なので pairwise 1.0 は要求しない。確認するのは、

  * 4/4 pair  -> sigmoid(Q_i-Q_j) が 1 / 0 方向へ寄るか
  * 3/4 pair  -> 0.75 付近へ寄るか
  * 2/2 pair  -> 0.5 付近に留まるか(無理に割らないか)

合成データ(真の順位が既知)と、実データ 64 group の両方で見る。
これは最適化バグの除外テストであり、性能判定ではない。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)          # 生成workerと同時に走らせるため

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
import train_soft as TS  # noqa: E402


def synth(n_groups=64, n_cand=5, sdim=12, odim=8, seed=0):
    """真スコア = w·option + v·state。block ノイズ量を候補ごとに変えて 4/4〜2/2 を作る。"""
    rs = np.random.RandomState(seed)
    w, v = rs.randn(odim), rs.randn(sdim) * 0.3
    out = []
    for g in range(n_groups):
        sf = rs.randn(sdim).astype(np.float32)
        cands = []
        for c in range(n_cand):
            of = rs.randn(odim).astype(np.float32)
            true = float(of @ w + sf @ v) * 0.02
            noise = 0.002 if c % 2 == 0 else 0.03   # 半分はノイズ大 = 2/2 が出る
            cands.append({"option_index": c, "option_feat": of.tolist(),
                          "action_card_id": -1, "option_type": 0,
                          "policy_score": 0.0, "policy_rank": c,
                          "blocks": [round(true + float(rs.randn()) * noise, 6)
                                     for _ in range(4)]})
        out.append({"group_id": f"s{g}", "game": g, "turn": 6, "turn_band": "middle",
                    "cand_band": "medium", "arch": "synth", "stratum": 0,
                    "state_feat": sf.tolist(), "candidates": cands})
    return out


def run(groups, arm, epochs, lr=3e-3, seed=0):
    TS.attach_pairs(groups)
    X = np.asarray([g["state_feat"] for g in groups], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0
    args = argparse.Namespace(epochs=epochs, batch=16, lr=lr)
    model, hist = TS.train_arm(groups, groups, arm, smean, sstd, args, seed, log=False)
    m = SE.evaluate(groups, lambda g: TS.score_group(model, g, smean, sstd))
    m.pop("_per_group")
    m["final_train_loss"] = hist[-1]["loss"]
    return m


def summarize(m):
    return {"stable": m["stable"], "mostly": m["mostly"], "near_tie": m["near_tie"],
            "all_pairwise": m["all_pairwise"], "support_weighted": m["support_weighted"],
            "calibration": m["calibration"], "calibration_error": m["calibration_error"],
            "abs_q_diff_by_margin": m["abs_q_diff_by_margin"],
            "final_train_loss": m["final_train_loss"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_teacher_blocks_all.jsonl.gz"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default=str(_HERE / "phase12a_memorize.json"))
    args = ap.parse_args()

    rep = {"n_groups": args.n, "epochs": args.epochs, "synthetic": {}, "real": {}}

    for arm in ("S0", "S2", "S3"):
        rep["synthetic"][arm] = summarize(run(synth(args.n), arm, args.epochs))
        print(f"  synth [{arm}] stable={rep['synthetic'][arm]['stable']} "
              f"calib_err={rep['synthetic'][arm]['calibration_error']}",
              file=sys.stderr, flush=True)

    real = TS.load(Path(args.data))[: args.n]
    for arm in ("S0", "S2", "S3"):
        rep["real"][arm] = summarize(run([dict(g) for g in real], arm, args.epochs))
        print(f"  real  [{arm}] stable={rep['real'][arm]['stable']} "
              f"calib_err={rep['real'][arm]['calibration_error']}",
              file=sys.stderr, flush=True)

    # 期待方向の自動判定(§13)
    ok = {}
    for scope in ("synthetic", "real"):
        c = rep[scope]["S3"]["calibration"]
        hi = c.get("1.0", {}).get("pred_mean")
        mid = c.get("0.5", {}).get("pred_mean")
        lo = c.get("0.0", {}).get("pred_mean")
        ok[scope] = {
            "support1.0_above_0.5": (hi is not None and hi > 0.5),
            "support0.0_below_0.5": (lo is not None and lo < 0.5),
            "support0.5_near_0.5": (mid is not None and abs(mid - 0.5) < 0.15),
            "monotone": (hi is not None and lo is not None and mid is not None
                         and lo < mid < hi),
            "stable_high": (rep[scope]["S3"]["stable"] or 0) >= 0.85,
        }
    rep["gates"] = ok
    # near-tie の Q 差が S0 より小さくなるか(§19.1 の先取り確認)
    rep["near_tie_absq"] = {
        s: {a: rep[s][a]["abs_q_diff_by_margin"]["very_small"] for a in ("S0", "S2", "S3")}
        for s in ("synthetic", "real")}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gates": rep["gates"], "near_tie_absq": rep["near_tie_absq"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
