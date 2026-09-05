#!/usr/bin/env python3
"""方策の上位 k 手に正解が含まれる率（recall@k）を測る。

探索の幅を絞る（方策の上位 k 手だけを展開する）設計が成立するかの前提条件。
recall@k が低ければ、**幅の絞り込みが探索の到達点を先に決めてしまう**。
どれだけ深く読んでも、候補に正解が入っていなければ届かない。

Top-1 が 0.74 でも recall@5 が 0.95 なら「候補には入っているが順位を付けられていない」
であり、探索で決着させられる。逆に recall@10 が 0.6 なら、方策自体を先に直す必要がある。

選択肢数の帯ごとに出す。誤答は 9択以上に集中している（決定点の16.7%に誤答の31.8%）ので、
**その帯での recall がこの設計の成否を決める**。

使い方:
    python recall_at_k.py --features features_marnie_grimmsnarl_ex.npz \
        --weights ../../sample_submission/ptcg_ai/learning/policy_weights_marnie_grimmsnarl_ex_bc2.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "sample_submission"))

_TEST = 2
_KS = (1, 2, 3, 5, 10, 15)


def score_all(w: dict, state_raw, option_raw, card_ids) -> np.ndarray:
    std = w["standardization"]
    s_mean = np.asarray(std["state_mean"], dtype=np.float64)
    s_std = np.asarray(std["state_std"], dtype=np.float64)
    o_mean = np.asarray(std["option_mean"], dtype=np.float64)
    o_std = np.asarray(std["option_std"], dtype=np.float64)
    s = np.where(s_std != 0, (state_raw - s_mean) / np.where(s_std != 0, s_std, 1.0), 0.0)
    o = np.where(o_std != 0, (option_raw - o_mean) / np.where(o_std != 0, o_std, 1.0), 0.0)
    emb = w["card_embedding"]
    table = np.asarray(emb["table"], dtype=np.float64)
    idx = np.where((card_ids >= 0) & (card_ids <= emb["card_id_max"]), card_ids, 0)
    h = np.concatenate([np.repeat(s[None, :], o.shape[0], axis=0), o, table[idx]], axis=1)
    for i, layer in enumerate(w["layers"]):
        h = h @ np.asarray(layer["weight"], dtype=np.float64).T + np.asarray(layer["bias"], dtype=np.float64)
        if i != len(w["layers"]) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0]


def bucket_of(n_opt: int) -> str:
    if n_opt <= 2:
        return "2"
    if n_opt <= 4:
        return "3-4"
    if n_opt <= 8:
        return "5-8"
    if n_opt <= 16:
        return "9-16"
    return "17+"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    w = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    split = d["split"].astype(np.int64)
    test_idx = np.where(split == _TEST)[0]
    if args.limit:
        test_idx = test_idx[: args.limit]

    state, opts, cids = d["state_features"], d["option_features"], d["option_card_ids"]
    chosen = d["chosen_index"].astype(np.int64)

    # bucket -> k -> hit数 / bucket -> 件数
    hits = collections.defaultdict(lambda: collections.Counter())
    n_by = collections.Counter()
    # 展開コスト: 上位k手に絞ったときに実際に展開する候補数の合計
    expand = collections.Counter()

    for pos in test_idx:
        o = np.asarray(opts[pos], dtype=np.float64)
        n_opt = o.shape[0]
        sc = score_all(w, np.asarray(state[pos], dtype=np.float64), o,
                       np.asarray(cids[pos], dtype=np.int64))
        order = np.argsort(-sc)
        gold = int(chosen[pos])
        rank = int(np.where(order == gold)[0][0])
        b = bucket_of(n_opt)
        n_by[b] += 1
        for k in _KS:
            if rank < k:
                hits[b][k] += 1
            expand[k] += min(k, n_opt)

    print(f"features: {args.features}")
    print(f"weights : {args.weights}")
    print(f"test    : {sum(n_by.values())} 決定点\n")

    order_b = ["2", "3-4", "5-8", "9-16", "17+"]
    header = f"{'選択肢数':<8}{'n':>8}" + "".join(f"{'@'+str(k):>9}" for k in _KS)
    print(header)
    for b in order_b:
        if b not in n_by:
            continue
        n = n_by[b]
        row = f"{b:<8}{n:>8}" + "".join(f"{hits[b][k]/n:>9.4f}" for k in _KS)
        print(row)

    tot = sum(n_by.values())
    agg = {k: sum(hits[b][k] for b in n_by) / tot for k in _KS}
    print(f"{'全体':<8}{tot:>8}" + "".join(f"{agg[k]:>9.4f}" for k in _KS))

    print("\n=== 展開コスト（上位k手に絞ったときの平均候補数）===")
    base = sum(np.asarray(opts[p]).shape[0] for p in test_idx) / tot
    print(f"  絞らない場合の平均選択肢数: {base:.2f}")
    for k in _KS:
        print(f"  上位{k:<3}に絞る: 平均 {expand[k]/tot:.2f} 候補  "
              f"(削減 {base/(expand[k]/tot):.2f} 分の1、recall {agg[k]:.4f})")


if __name__ == "__main__":
    main()
