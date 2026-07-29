"""npz に `matchup` 列(learner vs opponent)を足して別ファイルに保存する。

狙い: 「序盤の critic は局面ではなく、どのデッキ同士の対戦かを当てているだけ」という疑いを
直接測る。`early_game_signal.py --extra-baseline-col matchup` に渡すと、
**盤面を一切見ずマッチアップだけを見るモデル**の AUC が出る。

これが局面ベースのモデルと同程度なら、そのモデルは局面を評価していない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None, help="既定: <入力>_matchup.npz")
    args = ap.parse_args()

    src = Path(args.npz)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_matchup.npz")

    data = np.load(src, allow_pickle=False)
    cols = {k: data[k] for k in data.files}
    if "learner" not in cols or "opponent" not in cols:
        raise SystemExit(f"learner / opponent 列が無い: {sorted(cols)}")

    cols["matchup"] = np.char.add(np.char.add(cols["learner"].astype("U24"), "_vs_"),
                                  cols["opponent"].astype("U24")).astype("<U56")
    np.savez_compressed(out, **cols)

    uniq, counts = np.unique(cols["matchup"], return_counts=True)
    print(f"{src.name} -> {out.name}  rows={len(cols['matchup'])}  matchup種類={len(uniq)}")
    for u, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
        print(f"  {u:<52} {c}")


if __name__ == "__main__":
    main()
