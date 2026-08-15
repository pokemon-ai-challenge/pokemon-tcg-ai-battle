#!/usr/bin/env python3
"""同一 test split で任意の policy_weights JSON の Top-1 一致率を測る。

train.py の pure_python_forward と同じ計算を numpy でベクトル化したもの。
新旧の重みを公平に比べるためだけの評価用スクリプト(提出物ではない)。
"""
from __future__ import annotations

import json
import sys

import numpy as np

_TEST = 2


def load_weights(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_all(w: dict, state_raw, option_raw, card_ids) -> np.ndarray:
    """1決定点ぶんの全選択肢スコア。state_raw は (S,)、option_raw は (K, O)。"""
    std = w["standardization"]
    s_mean = np.asarray(std["state_mean"], dtype=np.float64)
    s_std = np.asarray(std["state_std"], dtype=np.float64)
    o_mean = np.asarray(std["option_mean"], dtype=np.float64)
    o_std = np.asarray(std["option_std"], dtype=np.float64)

    # std==0 の次元は 0 埋め(pure_python_forward と同じ)。
    s = np.where(s_std != 0, (state_raw - s_mean) / np.where(s_std != 0, s_std, 1.0), 0.0)
    o = np.where(o_std != 0, (option_raw - o_mean) / np.where(o_std != 0, o_std, 1.0), 0.0)

    emb = w["card_embedding"]
    table = np.asarray(emb["table"], dtype=np.float64)
    cid_max = emb["card_id_max"]
    idx = np.where((card_ids >= 0) & (card_ids <= cid_max), card_ids, 0)
    e = table[idx]

    k = o.shape[0]
    h = np.concatenate([np.repeat(s[None, :], k, axis=0), o, e], axis=1)

    layers = w["layers"]
    for i, layer in enumerate(layers):
        W = np.asarray(layer["weight"], dtype=np.float64)
        b = np.asarray(layer["bias"], dtype=np.float64)
        h = h @ W.T + b
        if i != len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0]


def main() -> None:
    features_path, *weight_paths = sys.argv[1:]
    d = np.load(features_path, allow_pickle=True)
    split = d["split"].astype(np.int64)
    test_idx = np.where(split == _TEST)[0]

    state = d["state_features"]
    opts = d["option_features"]
    cids = d["option_card_ids"]
    chosen = d["chosen_index"].astype(np.int64)

    print(f"features: {features_path}")
    print(f"test split: {len(test_idx)} 決定点\n")

    for wp in weight_paths:
        w = load_weights(wp)
        hit = 0
        for pos in test_idx:
            sc = score_all(w, np.asarray(state[pos], dtype=np.float64),
                           np.asarray(opts[pos], dtype=np.float64),
                           np.asarray(cids[pos], dtype=np.int64))
            if int(np.argmax(sc)) == int(chosen[pos]):
                hit += 1
        n = len(test_idx)
        p = hit / n
        se = (p * (1 - p) / n) ** 0.5
        print(f"{wp}\n  Top-1 = {hit}/{n} = {p:.4f}  (±{1.96*se:.4f})")


if __name__ == "__main__":
    main()
