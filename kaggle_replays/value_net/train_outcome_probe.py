"""Phase14 §20(長期側): 「その情報は最終勝敗の予測にも効くか」を state 単位で見る。

短期教師(§19 Target A)とは独立な参照として、**その試合の最終結果**を使う。
Phase13 の L1/L2 データには raw entity が入っていないため Target B の学習には使えない。
そこで 19,969 states に後埋めした game_outcome を長期参照として代用する。

注意: 1 試合の全 state が同じラベルを共有するので、**必ず試合単位で split** する。
candidate 単位の delta は state 単位では定義できないためこの probe には含めない。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import entity_probe as EP  # noqa: E402
import train_probe as TP  # noqa: E402

ARMS = {"O0_base": (), "O1_hand": ("hand",), "O2_board": ("board",),
        "O3_opponent": ("opp",), "O4_deck": ("deck",), "O5_history": ("history",)}


class OutcomeNet(nn.Module):
    def __init__(self, state_dim, families, hidden=64):
        super().__init__()
        self.families = tuple(families)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        ctx = hidden
        self.enc = nn.ModuleDict()
        if "hand" in families:
            self.enc["hand"] = EP.BagEncoder(); ctx += 32
        if "board" in families:
            self.enc["board"] = EP.TokenEncoder(2, 10); ctx += 32
        if "opp" in families:
            self.enc["opp"] = EP.TokenEncoder(2, 10); ctx += 32
            self.enc["opp_disc"] = EP.BagEncoder(); ctx += 32
        if "deck" in families:
            self.enc["disc"] = EP.BagEncoder(); ctx += 32
            self.enc["unseen"] = EP.BagEncoder(); ctx += 32
        if "history" in families:
            self.enc["history"] = EP.TokenEncoder(1, 3); ctx += 32
        self.head = nn.Sequential(nn.Linear(ctx, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        p = [self.state_mlp(b["state"])]
        if "hand" in self.families:
            p.append(self.enc["hand"](b["hand"]))
        if "board" in self.families:
            p.append(self.enc["board"](b["board_ids"], b["board_num"], b["board_mask"]))
        if "opp" in self.families:
            p.append(self.enc["opp"](b["opp_ids"], b["opp_num"], b["opp_mask"]))
            p.append(self.enc["opp_disc"](b["opp_disc"]))
        if "deck" in self.families:
            p.append(self.enc["disc"](b["disc"]))
            p.append(self.enc["unseen"](b["unseen"]))
        if "history" in self.families:
            p.append(self.enc["history"](b["hist_ids"], b["hist_num"], b["hist_mask"]))
        return self.head(torch.cat(p, dim=-1)).squeeze(-1)


def auc(y, s):
    pairs = sorted(zip(s, y))
    r = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2.0 + 1
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    sr = sum(r[k] for k, (_, yy) in enumerate(pairs) if yy == 1)
    return (sr - pos * (pos + 1) / 2) / (pos * neg)


def batch(rows, fam, smean, sstd, deck):
    out = {"state": torch.from_numpy(
        ((np.asarray([r["state_feat"] for r in rows], np.float32) - smean) / sstd))}
    if "hand" in fam:
        out["hand"] = TP._pad_ids([r["_f"]["hand"] for r in rows], 20)
    if "board" in fam:
        out["board_ids"], out["board_num"], out["board_mask"] = TP._pad_tokens(
            [r["_f"]["board"] for r in rows], 2, 10, 6)
    if "opp" in fam:
        out["opp_ids"], out["opp_num"], out["opp_mask"] = TP._pad_tokens(
            [r["_f"]["opp"] for r in rows], 2, 10, 6)
        out["opp_disc"] = TP._pad_ids([r["_f"]["opp_disc"] for r in rows], 60)
    if "deck" in fam:
        out["disc"] = TP._pad_ids([r["_f"]["disc"] for r in rows], 60)
        out["unseen"] = TP._pad_ids([r["_f"]["unseen"] for r in rows], 60)
    if "history" in fam:
        out["hist_ids"], out["hist_num"], out["hist_mask"] = TP._pad_tokens(
            [r["_f"]["history"] for r in rows], 1, 3, EP.HIST_N)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(_HERE.parent / "_ent_states_w*.jsonl.gz"))
    ap.add_argument("--deck", default=str(_HERE.parent.parent / "sample_submission" / "deck.csv"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(_HERE / "phase14_outcome.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    rows = []
    for f in sorted(glob.glob(args.states)):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(l)
            if r.get("game_outcome") is not None:
                rows.append(r)
    deck = TP.deck_ids(args.deck)
    for r in rows:
        e = r["entity"]
        r["_f"] = {"hand": EP.hand_bag(e), "board": EP.board_tokens(e, "self"),
                   "opp": EP.board_tokens(e, "opp"),
                   "opp_disc": EP.discard_bag(e, "opp"),
                   "disc": EP.discard_bag(e, "self"),
                   "unseen": EP.unseen_deck_bag(e, deck),
                   "history": EP.history_tokens(r["history"])}
    games = sorted({r["game"] for r in rows})
    rs = np.random.RandomState(0)
    rs.shuffle(games)
    n = len(games)
    split = {g: (0 if i < 0.7 * n else 1 if i < 0.85 * n else 2) for i, g in enumerate(games)}
    tr = [r for r in rows if split[r["game"]] == 0]
    va = [r for r in rows if split[r["game"]] == 1]
    te = [r for r in rows if split[r["game"]] == 2]
    print(f"states train={len(tr)} val={len(va)} test={len(te)} games={n}", file=sys.stderr)

    X = np.asarray([r["state_feat"] for r in tr], np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0
    rep = {"n_states": len(rows), "n_games": n,
           "split": {"train": len(tr), "val": len(va), "test": len(te)},
           "base_rate": round(statistics.mean([r["game_outcome"] for r in rows]), 4),
           "arms": {}}
    bce = nn.BCEWithLogitsLoss()

    for name, fam in ARMS.items():
        aucs = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = OutcomeNet(len(smean), fam)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            best, best_state = -1, None
            for _ep in range(args.epochs):
                m.train()
                perm = np.random.permutation(len(tr))
                for i in range(0, len(perm), args.batch):
                    gs = [tr[j] for j in perm[i:i + args.batch]]
                    y = torch.tensor([g["game_outcome"] for g in gs], dtype=torch.float32)
                    loss = bce(m(batch(gs, fam, smean, sstd, deck)), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    sv = m(batch(va, fam, smean, sstd, deck)).tolist()
                a = auc([r["game_outcome"] for r in va], sv) or 0
                if a > best:
                    best = a
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            m.load_state_dict(best_state)
            m.eval()
            with torch.no_grad():
                st = m(batch(te, fam, smean, sstd, deck)).tolist()
            aucs.append(auc([r["game_outcome"] for r in te], st))
        rep["arms"][name] = {"test_auc": round(statistics.mean(aucs), 4),
                             "per_seed": [round(a, 4) for a in aucs],
                             "sd": round(statistics.stdev(aucs), 4) if len(aucs) > 1 else None}
        print(f"  [{name}] test_auc={rep['arms'][name]['test_auc']} "
              f"seeds={rep['arms'][name]['per_seed']}", file=sys.stderr, flush=True)
    base = rep["arms"]["O0_base"]["test_auc"]
    for k, v in rep["arms"].items():
        v["delta_vs_base"] = round(v["test_auc"] - base, 4)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep["arms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
