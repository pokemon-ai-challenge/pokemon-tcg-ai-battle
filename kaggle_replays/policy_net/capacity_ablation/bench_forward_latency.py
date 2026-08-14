#!/usr/bin/env python3
"""PolicyModel の pure-Python forward レイテンシを hidden size 別に実測する(容量ablation §11)。

features_configC.npz の生特徴(raw state/option/card_id、標準化前)を決定点ごとに取り出し、
各重み(M32/M64/M128)の PolicyModel._forward を全選択肢について呼んで時間を測る。
本番の score_options が回すのと同一の計算経路(標準化 → 連結 → 各層 forward)。

使い方(repo root から):
  PYTHONIOENCODING=utf-8 python kaggle_replays/policy_net/capacity_ablation/bench_forward_latency.py \
      --n 2000 --out kaggle_replays/policy_net/capacity_ablation/latency.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_POLICY_NET = _HERE.parent
_REPO = _POLICY_NET.parent.parent
sys.path.insert(0, str(_REPO / "sample_submission"))

from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402

_WEIGHTS = {
    32: _REPO / "sample_submission/ptcg_ai/learning/policy_weights_m32.json",
    64: _REPO / "sample_submission/ptcg_ai/learning/policy_weights_m64.json",
    128: _REPO / "sample_submission/ptcg_ai/learning/policy_weights_m128.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(_POLICY_NET / "features_configC.npz"))
    ap.add_argument("--n", type=int, default=2000, help="計測する決定点数(先頭N件)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    state = data["state_features"].astype(np.float64)
    option = data["option_features"]  # object array
    card_ids = data["option_card_ids"]
    n = min(args.n, len(state))
    # 生特徴を Python list 化して事前準備(計測ループに numpy 変換コストを含めない)。
    decisions = []
    for i in range(n):
        srow = state[i].tolist()
        orows = [r.tolist() for r in np.asarray(option[i], dtype=np.float64)]
        cids = [int(c) for c in np.asarray(card_ids[i])]
        decisions.append((srow, orows, cids))
    total_options = sum(len(orows) for _, orows, _ in decisions)
    print(f"決定点={n}  総選択肢={total_options}  平均選択肢={total_options / n:.2f}")

    results = {}
    for h, path in _WEIGHTS.items():
        if not path.exists():
            print(f"skip M{h}: {path} なし")
            continue
        model = PolicyModel(path)
        assert model.is_ready
        # ウォームアップ
        for srow, orows, cids in decisions[:50]:
            for orow, cid in zip(orows, cids):
                model._forward(srow, orow, cid)
        t0 = time.perf_counter()
        for srow, orows, cids in decisions:
            for orow, cid in zip(orows, cids):
                model._forward(srow, orow, cid)
        elapsed = time.perf_counter() - t0
        per_decision_ms = elapsed / n * 1000
        per_option_us = elapsed / total_options * 1e6
        results[f"M{h}"] = {
            "hidden_size": h,
            "total_seconds": elapsed,
            "per_decision_ms": per_decision_ms,
            "per_option_us": per_option_us,
        }
        print(f"M{h:<3}: {per_decision_ms:.4f} ms/決定  {per_option_us:.2f} us/選択肢  (計 {elapsed:.3f}s)")

    summary = {"n_decisions": n, "total_options": total_options, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
