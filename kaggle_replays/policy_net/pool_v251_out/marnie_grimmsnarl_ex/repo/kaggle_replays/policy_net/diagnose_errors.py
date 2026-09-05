#!/usr/bin/env python3
"""学習済み方策が test split の**どこで**間違えるかを分解する。

Top-1 一致率が 0.74 だとして、残り 0.26 の中身を知らないと次に何を足すべきか決められない。
容量を増やしても改善しなかった（hidden 32/64/128 で差なし）以上、残差の内訳を見て
「表現力不足」なのか「入力情報の不足」なのか「そもそも教師が一貫していない」のかを
切り分ける。

出力:
  1. select_type 別の一致率（どの判断種別が弱いか）
  2. MAIN(select_type=0) を、実際に選ばれた選択肢の opttype 別に分解
  3. 誤答時の混同（正解 opttype -> 予測 opttype）
  4. 選択肢数 n_options 別の一致率
  5. ターン帯別の一致率
  6. 教師の一貫性の上限推定（同一の状態特徴で異なる選択がされている割合）

使い方:
    python diagnose_errors.py --features features_marnie_grimmsnarl_ex.npz \
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
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "sample_submission"))
from ptcg_ai.learning.encoder import FEATURE_NAMES, OPTION_FEATURE_NAMES  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_TEST = 2

_OPTTYPE_IDX = [i for i, n in enumerate(OPTION_FEATURE_NAMES) if n.startswith("opttype_")]
_OPTTYPE_NAME = [OPTION_FEATURE_NAMES[i][len("opttype_"):] for i in _OPTTYPE_IDX]
_N_OPTIONS_IDX = OPTION_FEATURE_NAMES.index("n_options")


def opttype_of(option_row: np.ndarray) -> str:
    """選択肢1件の opttype ワンホットから型名を返す。"""
    sub = option_row[_OPTTYPE_IDX]
    if sub.max() <= 0:
        return "(none)"
    return _OPTTYPE_NAME[int(sub.argmax())]


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


def _rate(hit: int, n: int) -> str:
    if not n:
        return "     -    "
    p = hit / n
    return f"{p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=None, help="test split 先頭N件のみ")
    ap.add_argument(
        "--ablate-features", default=None,
        help="train.py --ablate-features と同じ指定。潰して学習した重みは、評価時も同じ列を"
             "潰さないと桁が壊れる（潰した列の std は 0 ではなく 1e-6 にクリップされるため、"
             "生値が 100万倍に増幅されて入る）。",
    )
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    w = json.loads(Path(args.weights).read_text(encoding="utf-8"))

    split = d["split"].astype(np.int64)
    test_idx = np.where(split == _TEST)[0]
    if args.limit:
        test_idx = test_idx[: args.limit]

    state = d["state_features"]
    opts = d["option_features"]
    cids = d["option_card_ids"]
    chosen = d["chosen_index"].astype(np.int64)

    if args.ablate_features:
        names = [t.strip() for t in args.ablate_features.split(",") if t.strip()]
        st_idx = [i for i, f in enumerate(FEATURE_NAMES)
                  if any(f == n or f.endswith("_" + n) for n in names)]
        op_idx = [i for i, f in enumerate(OPTION_FEATURE_NAMES)
                  if any(f == n or f.endswith("_" + n) for n in names)]
        state = np.asarray(state, dtype=np.float64).copy()
        state[:, st_idx] = 0.0
        opts = [np.asarray(o, dtype=np.float64).copy() for o in opts]
        for o in opts:
            o[:, op_idx] = 0.0
        print(f"  ablate: 状態側 {len(st_idx)} / 選択肢側 {len(op_idx)} 次元を 0 で潰した")
    sel_type = d["select_type"].astype(np.int64) if "select_type" in d else None
    turn = d["turn"].astype(np.int64) if "turn" in d else None

    by_sel = collections.defaultdict(lambda: [0, 0])
    by_opttype = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    by_nopt = collections.defaultdict(lambda: [0, 0])
    by_turn = collections.defaultdict(lambda: [0, 0])
    state_key_choices = collections.defaultdict(collections.Counter)

    for pos in test_idx:
        o = np.asarray(opts[pos], dtype=np.float64)
        sc = score_all(w, np.asarray(state[pos], dtype=np.float64), o,
                       np.asarray(cids[pos], dtype=np.int64))
        pred = int(np.argmax(sc))
        gold = int(chosen[pos])
        ok = int(pred == gold)

        st = int(sel_type[pos]) if sel_type is not None else -1
        by_sel[st][0] += ok
        by_sel[st][1] += 1

        gold_t = opttype_of(o[gold])
        if st == 0:  # MAIN のみ opttype 分解する
            by_opttype[gold_t][0] += ok
            by_opttype[gold_t][1] += 1
            if not ok:
                confusion[(gold_t, opttype_of(o[pred]))] += 1

        n_opt = int(round(float(o[gold][_N_OPTIONS_IDX]))) if o.shape[0] else 0
        bucket = ("2" if n_opt <= 2 else "3-4" if n_opt <= 4 else "5-8" if n_opt <= 8
                  else "9-16" if n_opt <= 16 else "17+")
        by_nopt[bucket][0] += ok
        by_nopt[bucket][1] += 1

        if turn is not None:
            t = int(turn[pos])
            tb = "1-2" if t <= 2 else "3-5" if t <= 5 else "6-10" if t <= 10 else "11+"
            by_turn[tb][0] += ok
            by_turn[tb][1] += 1

        # 教師の一貫性: 状態特徴を丸めたキーで、同一状態に対する選択の割れ方を見る。
        key = (hash(np.round(np.asarray(state[pos], dtype=np.float64), 3).tobytes()), o.shape[0])
        state_key_choices[key][gold] += 1

    n = len(test_idx)
    hit = sum(v[0] for v in by_sel.values())
    print(f"features: {args.features}")
    print(f"weights : {args.weights}")
    print(f"test    : {n} 決定点   全体 Top-1 = {hit}/{n} = {hit/n:.4f}\n")

    print("=== 1. select_type 別 ===")
    print(f"{'select_type':>12}{'n':>9}{'Top-1':>9}{'誤答数':>9}")
    for k in sorted(by_sel):
        h, t = by_sel[k]
        print(f"{k:>12}{t:>9}{_rate(h,t):>9}{t-h:>9}")

    print("\n=== 2. MAIN(select_type=0) を、正解の opttype 別に分解 ===")
    print(f"{'opttype':>16}{'n':>9}{'Top-1':>9}{'誤答数':>9}")
    for k in sorted(by_opttype, key=lambda x: -by_opttype[x][1]):
        h, t = by_opttype[k]
        print(f"{k:>16}{t:>9}{_rate(h,t):>9}{t-h:>9}")

    print("\n=== 3. MAIN の誤答: 正解 opttype -> 予測 opttype（上位15）===")
    for (g, p), c in confusion.most_common(15):
        print(f"  {g:>14} -> {p:<14} {c:>6}")

    print("\n=== 4. 選択肢数別 ===")
    print(f"{'n_options':>12}{'n':>9}{'Top-1':>9}")
    for k in ("2", "3-4", "5-8", "9-16", "17+"):
        if k in by_nopt:
            h, t = by_nopt[k]
            print(f"{k:>12}{t:>9}{_rate(h,t):>9}")

    if by_turn:
        print("\n=== 5. ターン帯別 ===")
        print(f"{'turn':>12}{'n':>9}{'Top-1':>9}")
        for k in ("1-2", "3-5", "6-10", "11+"):
            if k in by_turn:
                h, t = by_turn[k]
                print(f"{k:>12}{t:>9}{_rate(h,t):>9}")

    # 同一状態で選択が割れている割合 = 教師の非一貫性。ここは原理的に当てられない。
    dup = [c for c in state_key_choices.values() if sum(c.values()) > 1]
    if dup:
        total = sum(sum(c.values()) for c in dup)
        major = sum(max(c.values()) for c in dup)
        print(f"\n=== 6. 教師の一貫性（同一状態が複数回出現した {len(dup)} 状態・{total} 決定点）===")
        print(f"  多数派を必ず当てたときの上限 Top-1 = {major}/{total} = {major/total:.4f}")
        print("  （この範囲では、どんなモデルでもこれ以上は当てられない）")


if __name__ == "__main__":
    main()
