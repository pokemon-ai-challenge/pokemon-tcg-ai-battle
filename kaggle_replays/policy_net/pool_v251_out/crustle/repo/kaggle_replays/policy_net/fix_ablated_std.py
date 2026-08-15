#!/usr/bin/env python3
"""--ablate-features で潰した列の std を 0 に書き換える。

潰した列は学習中ずっと 0 なので、標準化の std が **0 ではなく 1e-6 にクリップ**される。
この重みをそのまま推論に回すと `(実データ - 0) / 1e-6` で 100万倍に増幅され、
対照群だけ数値が壊れる（実際にこれで「実験群が +26.8pt で圧勝」という
真逆の結果が出た）。std を 0 にすると policy_model._forward の
「std が偽なら 0」経路へ落ちて、学習時と同じ入力になる。

使い方:
    python fix_ablated_std.py --weights <in.json> --out <out.json> \
        --ablate-features opt_role_draw,opt_role_search
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "sample_submission"))
from ptcg_ai.learning.encoder import FEATURE_NAMES, OPTION_FEATURE_NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablate-features", required=True,
                    help="train.py に渡したものと同じ指定")
    args = ap.parse_args()

    names = [t.strip() for t in args.ablate_features.split(",") if t.strip()]
    st_idx = [i for i, f in enumerate(FEATURE_NAMES)
              if any(f == n or f.endswith("_" + n) for n in names)]
    op_idx = [i for i, f in enumerate(OPTION_FEATURE_NAMES)
              if any(f == n or f.endswith("_" + n) for n in names)]
    if not st_idx and not op_idx:
        raise SystemExit(f"一致する特徴名がありません: {names}")

    w = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    std = w["standardization"]
    for i in st_idx:
        std["state_std"][i] = 0.0
    for i in op_idx:
        std["option_std"][i] = 0.0
    Path(args.out).write_text(json.dumps(w), encoding="utf-8")
    print(f"std を 0 に修正: {args.out}  (状態 {len(st_idx)} / 選択肢 {len(op_idx)} 列)")


if __name__ == "__main__":
    main()
