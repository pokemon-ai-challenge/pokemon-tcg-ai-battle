"""Phase6B/6C: Hand-aware Value v1(DeepSets 型 Set Encoder)の学習と V0 比較。

比較は **同一 split・同一 trunk・同一学習レシピ**で行い、差分を「カード集合入力の有無」だけにする:

  V0p (control) : state166 のみ                      → trunk → value
  V1  (candidate): state166 + {hand, discard, opp_visible} のカード集合 → trunk → value

既存のデプロイ済み V0(`value_weights.json`)も参考として test で評価するが、
学習レシピが違うため **因果比較には V0p を使う**(V0 との差はレシピ差を含む)。

隠れ情報は入れない: 相手手札・山札は `build_card_features.py` の時点で除外済み(リーク検査0件)。
split は `features.npz` の `split` 列(md5(episode_id)%100、試合単位)をそのまま使う。

出力: hand_value_v1.pt / hand_value_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
PAD = 0                      # card_id は +1 シフトして 0 を padding に使う
N_CARD = 1269 + 1            # max_card_id=1267 -> +1 シフトで 1268、余裕を持たせる


class ZoneEncoder(nn.Module):
    """カード集合 -> zone ベクトル(順序不変な sum+mean pooling)。"""

    def __init__(self, emb: nn.Embedding, zone_id: int, n_zones: int,
                 emb_dim: int, hidden: int, out: int):
        super().__init__()
        self.emb = emb
        self.zone = nn.Embedding(n_zones, 8)
        self.zone_id = zone_id
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 8, hidden), nn.ReLU(), nn.Linear(hidden, out), nn.ReLU())
        self.out = out

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L] (0 = padding)
        mask = (ids != PAD).float().unsqueeze(-1)             # [B,L,1]
        e = self.emb(ids)                                      # [B,L,E]
        z = self.zone(torch.full_like(ids, self.zone_id))      # [B,L,8]
        h = self.mlp(torch.cat([e, z], dim=-1)) * mask         # padding を0に
        s = h.sum(dim=1)
        cnt = mask.sum(dim=1).clamp(min=1.0)
        return torch.cat([s, s / cnt], dim=-1)                 # sum と mean の両方


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, use_cards: bool, emb_dim=32, zone_out=64):
        super().__init__()
        self.use_cards = use_cards
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        fuse_in = 64
        if use_cards:
            self.emb = nn.Embedding(N_CARD, emb_dim, padding_idx=PAD)
            self.zones = nn.ModuleList([
                ZoneEncoder(self.emb, i, 3, emb_dim, 48, zone_out) for i in range(3)])
            fuse_in += 3 * zone_out * 2
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, hand=None, disc=None, opp=None):
        parts = [self.state_mlp(x)]
        if self.use_cards:
            parts += [self.zones[0](hand), self.zones[1](disc), self.zones[2](opp)]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _ragged_batch(flat, off, rows, cap, device):
    """ragged(flat+offsets)から [B, L] のパディング済みテンソルを作る。"""
    lens = np.minimum(off[rows + 1] - off[rows], cap)
    L = int(max(1, lens.max()))
    out = np.zeros((len(rows), L), dtype=np.int64)
    for i, (r, ln) in enumerate(zip(rows, lens)):
        if ln > 0:
            s = off[r]
            out[i, :ln] = flat[s:s + ln] + 1        # +1 シフト(0=padding)
    return torch.from_numpy(out).to(device)


def _metrics(p, y, w=None):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    auc = ((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
           if npos and nneg else float("nan"))
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    # calibration error (10 bins, equal width)
    ece = 0.0
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10 if b < 9 else p <= 1.0)
        if m.sum():
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return {"auc": round(float(auc), 4), "logloss": round(ll, 4),
            "brier": round(brier, 4), "calibration_error": round(float(ece), 4),
            "mean_pred": round(float(p.mean()), 4), "base_rate": round(float(y.mean()), 4),
            "n": int(len(p))}


def train_model(use_cards, X, y, W, split, cards, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    model = ValueNet(X.shape[1], use_cards).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    Wt = torch.from_numpy(W.astype(np.float32))

    best = {"auc": -1}
    best_state = None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            rows = perm[i:i + args.batch]
            xb = Xt[rows].to(device)
            yb = yt[rows].to(device)
            wb = Wt[rows].to(device)
            kw = {}
            if use_cards:
                kw = {
                    "hand": _ragged_batch(cards["hand_flat"], cards["hand_off"], rows, 20, device),
                    "disc": _ragged_batch(cards["discard_flat"], cards["discard_off"], rows, 30, device),
                    "opp": _ragged_batch(cards["opp_flat"], cards["opp_off"], rows, 30, device),
                }
            opt.zero_grad()
            out = model(xb, **kw)
            loss = (lossf(out, yb) * wb).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(rows)
        pv = predict(model, use_cards, Xt, cards, va, device, args.batch)
        m = _metrics(pv, y[va])
        print(f"    [{'V1' if use_cards else 'V0p'} seed{seed}] epoch {ep+1}/{args.epochs} "
              f"loss={tot/len(perm):.4f} val_auc={m['auc']} val_ll={m['logloss']}",
              file=sys.stderr, flush=True)
        if m["auc"] > best["auc"]:
            best = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, use_cards, Xt, cards, rows, device, batch=8192):
    model.eval()
    out = np.zeros(len(rows), dtype=np.float64)
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        kw = {}
        if use_cards:
            kw = {
                "hand": _ragged_batch(cards["hand_flat"], cards["hand_off"], r, 20, device),
                "disc": _ragged_batch(cards["discard_flat"], cards["discard_off"], r, 30, device),
                "opp": _ragged_batch(cards["opp_flat"], cards["opp_off"], r, 30, device),
            }
        out[i:i + len(r)] = torch.sigmoid(model(Xt[r].to(device), **kw)).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--subsample", type=int, default=0, help="train行の上限(0=全件)")
    ap.add_argument("--out-prefix", default=str(_HERE / "hand_value"))
    args = ap.parse_args()

    device = "cpu"
    d = np.load(_HERE / "features.npz", allow_pickle=True)
    X, y, W, split, turn = d["X"], d["y"], d["weight"], d["split"], d["turn"]
    c = np.load(_HERE / "card_features.npz")
    cards = {k: c[k] for k in ("hand_flat", "hand_off", "discard_flat",
                               "discard_off", "opp_flat", "opp_off")}

    # 標準化(train統計のみ。test を覗かない)
    tr = np.where(split == 0)[0]
    if args.subsample and len(tr) > args.subsample:
        rs = np.random.RandomState(0)
        keep = rs.choice(tr, args.subsample, replace=False)
        drop = np.setdiff1d(tr, keep)
        split = split.copy()
        split[drop] = 3               # 学習から除外
        tr = keep
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std == 0] = 1.0
    Xs = ((X - mean) / std).astype(np.float32)

    te = np.where(split == 2)[0]
    Xt = torch.from_numpy(Xs)
    report = {"split_counts": {"train": int(len(tr)), "val": int((split == 1).sum()),
                               "test": int(len(te))},
              "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "seeds": args.seeds, "arms": {}}

    for use_cards, name in ((False, "V0p_state_only"), (True, "V1_hand_aware")):
        per_seed = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            t0 = time.time()
            model, bestval = train_model(use_cards, Xs, y, W, split, cards, args, device, seed)
            pt = predict(model, use_cards, Xt, cards, te, device, args.batch)
            m = _metrics(pt, y[te])
            m["val_best_auc"] = bestval["auc"]
            m["train_sec"] = round(time.time() - t0, 1)
            # ターン帯別
            m["by_turn"] = {}
            for lo, hi, nm in ((1, 5, "early"), (6, 10, "mid"), (11, 10**9, "late")):
                sel = (turn[te] >= lo) & (turn[te] <= hi)
                if sel.sum() > 50:
                    m["by_turn"][nm] = _metrics(pt[sel], y[te][sel])["auc"]
            per_seed.append(m)
            if use_cards and seed == int(args.seeds.split(",")[0]):
                torch.save({"state_dict": model.state_dict(),
                            "mean": mean, "std": std, "use_cards": True},
                           f"{args.out_prefix}_v1.pt")
            if (not use_cards) and seed == int(args.seeds.split(",")[0]):
                torch.save({"state_dict": model.state_dict(),
                            "mean": mean, "std": std, "use_cards": False},
                           f"{args.out_prefix}_v0p.pt")
        report["arms"][name] = {
            "per_seed": per_seed,
            "mean_auc": round(float(np.mean([m["auc"] for m in per_seed])), 4),
            "mean_logloss": round(float(np.mean([m["logloss"] for m in per_seed])), 4),
            "mean_brier": round(float(np.mean([m["brier"] for m in per_seed])), 4),
            "mean_cal_err": round(float(np.mean([m["calibration_error"] for m in per_seed])), 4),
        }
        print(f"  [{name}] test AUC={report['arms'][name]['mean_auc']} "
              f"LL={report['arms'][name]['mean_logloss']}", file=sys.stderr, flush=True)

    a, b = report["arms"]["V0p_state_only"], report["arms"]["V1_hand_aware"]
    report["delta_V1_minus_V0p"] = {
        "auc": round(b["mean_auc"] - a["mean_auc"], 4),
        "logloss": round(b["mean_logloss"] - a["mean_logloss"], 4),
        "brier": round(b["mean_brier"] - a["mean_brier"], 4),
        "cal_err": round(b["mean_cal_err"] - a["mean_cal_err"], 4),
    }
    report["direction_reproduced_all_seeds"] = all(
        v1["auc"] > v0["auc"] for v0, v1 in
        zip(report["arms"]["V0p_state_only"]["per_seed"],
            report["arms"]["V1_hand_aware"]["per_seed"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(f"{args.out_prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
