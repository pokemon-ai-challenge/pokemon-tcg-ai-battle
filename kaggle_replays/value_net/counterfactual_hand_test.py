"""Phase6C §9.3: 反実仮想テスト「手札のカードIDだけ差し替えたら Value は変わるか」。

盤面特徴(166次元)は**固定したまま**、手札のカードID集合だけを差し替えて予測差を見る。
これにより「手札枚数」ではなく **カードの中身** に反応しているかを分離できる。

  A) identity swap : 手札の枚数は同じまま、別のカードIDへ入れ替える
                     → V1 は動くべき / V0p は原理的に動かない(入力に無い)
  B) count change  : 手札を1枚減らす(枚数の効果。166次元にも入っているので V0p も動きうる)

`A` で差が出て、かつ `B` の差だけでは説明できないなら、
「カード枚数ではなくカード内容を見ている」と言える。

test split の局面のみ使用(学習に使っていない試合)。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from train_hand_value import ValueNet  # noqa: E402

MAX_HAND, MAX_DISC, MAX_OPP = 20, 30, 30


class _Scorer:
    def __init__(self, ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.use_cards = bool(ck["use_cards"])
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.model = ValueNet(len(self.mean), self.use_cards)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def _pad(self, batch_ids, cap):
        L = max(1, min(cap, max((len(v) for v in batch_ids), default=1)))
        a = np.zeros((len(batch_ids), L), dtype=np.int64)
        for i, v in enumerate(batch_ids):
            for j, c in enumerate(v[:L]):
                a[i, j] = c + 1
        return torch.from_numpy(a)

    @torch.no_grad()
    def score(self, X, hands, discs, opps):
        x = torch.from_numpy(((X - self.mean) / self.std).astype(np.float32))
        kw = {}
        if self.use_cards:
            kw = {"hand": self._pad(hands, MAX_HAND),
                  "disc": self._pad(discs, MAX_DISC),
                  "opp": self._pad(opps, MAX_OPP)}
        return torch.sigmoid(self.model(x, **kw)).numpy()


def _get(flat, off, row, cap):
    return flat[off[row]:off[row + 1]][:cap].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0p", default=str(_HERE / "hand_value_v0p.pt"))
    ap.add_argument("--v1", default=str(_HERE / "hand_value_v1.pt"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(_HERE / "counterfactual_report.json"))
    args = ap.parse_args()

    d = np.load(_HERE / "features.npz", allow_pickle=True)
    c = np.load(_HERE / "card_features.npz")
    X, split = d["X"], d["split"]
    hf, ho = c["hand_flat"], c["hand_off"]
    df, do = c["discard_flat"], c["discard_off"]
    of, oo = c["opp_flat"], c["opp_off"]

    te = np.where(split == 2)[0]
    rs = np.random.RandomState(0)
    # 手札が3枚以上ある局面だけ(差し替えの意味がある)
    cand = [r for r in te if (ho[r + 1] - ho[r]) >= 3]
    rows = rs.choice(cand, min(args.n, len(cand)), replace=False)

    # 差し替え用のカードIDプール(test split の手札に実際に現れたID)
    pool = np.unique(hf[ho[te[0]]:ho[te[-1] + 1]])
    pool = pool[pool > 0]

    Xb = X[rows]
    hands = [_get(hf, ho, r, MAX_HAND) for r in rows]
    discs = [_get(df, do, r, MAX_DISC) for r in rows]
    opps = [_get(of, oo, r, MAX_OPP) for r in rows]

    # A) identity swap: 枚数そのまま、全カードIDを別IDへ
    hands_swap = []
    for h in hands:
        repl = rs.choice(pool, len(h), replace=True).tolist()
        hands_swap.append([int(v) for v in repl])
    # B) count change: 1枚減らす
    hands_drop = [h[:-1] if len(h) > 1 else h for h in hands]

    out = {"n_positions": len(rows), "arms": {}}
    for name, path in (("V0p_state_only", args.v0p), ("V1_hand_aware", args.v1)):
        if not Path(path).exists():
            out["arms"][name] = {"error": "checkpoint not found"}
            continue
        s = _Scorer(path)
        base = s.score(Xb, hands, discs, opps)
        swap = s.score(Xb, hands_swap, discs, opps)
        drop = s.score(Xb, hands_drop, discs, opps)
        da = np.abs(swap - base)
        db = np.abs(drop - base)
        out["arms"][name] = {
            "identity_swap_mean_abs_delta": round(float(da.mean()), 5),
            "identity_swap_median_abs_delta": round(float(np.median(da)), 5),
            "identity_swap_max_abs_delta": round(float(da.max()), 5),
            "identity_swap_frac_changed_gt_0.01": round(float((da > 0.01).mean()), 4),
            "count_change_mean_abs_delta": round(float(db.mean()), 5),
            "base_mean_pred": round(float(base.mean()), 4),
        }

    a = out["arms"].get("V0p_state_only", {})
    b = out["arms"].get("V1_hand_aware", {})
    if "identity_swap_mean_abs_delta" in a and "identity_swap_mean_abs_delta" in b:
        out["verdict"] = {
            "V1_reacts_to_card_identity": b["identity_swap_mean_abs_delta"] > 0.005,
            "V0p_is_blind_to_card_identity": a["identity_swap_mean_abs_delta"] < 1e-6,
            "V1_identity_effect_vs_count_effect_ratio": (
                round(b["identity_swap_mean_abs_delta"]
                      / max(b["count_change_mean_abs_delta"], 1e-9), 2)),
            "note": ("V0p は手札IDを入力に持たないので identity swap では厳密に0が期待値。"
                     "V1 の identity 効果が count 効果と同程度以上なら『枚数ではなく中身』を"
                     "見ていると言える。"),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
