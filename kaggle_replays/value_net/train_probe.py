"""Phase14 §10-§22: feature-family probe の学習・評価・paired bootstrap。

target は Phase12 S3(block-support soft + confidence weight)で固定する。今回 target は
いじらない(§19)。評価も Phase12 と同じ soft_eval を使うので数値の意味が揃う。

全 arm で split / optimizer / updates / loss / seed / 評価 を同一にし、
**変えるのは入力 family だけ**(§11)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import entity_probe as EP  # noqa: E402
import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
import train_soft as TS  # noqa: E402

ARMS = {
    "P0_base": (),
    "P1_hand": ("hand",),
    "P2_board": ("board",),
    "P3_opponent": ("opp",),
    "P4_deck": ("deck",),
    "P5_history": ("history",),
    "P6_delta": ("delta",),
    "P6c_delta_shuffled": ("delta_shuf",),          # 容量対照: delta を候補間でずらす
    "P7_delta_deck": ("delta", "deck"),
}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def deck_ids(path):
    """deck.csv は**ヘッダ無し**の 1 行 1 カード ID(全 60 行)。DictReader は使わない。"""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tok = line.strip().split(",")[0].strip()
        if tok.isdigit():
            out.append(int(tok))
    return out


def _pad_ids(seqs, cap):
    L = max(1, min(cap, max((len(s) for s in seqs), default=1)))
    a = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:L]):
            a[i, j] = c
    return torch.from_numpy(a)


def _pad_tokens(seqs, n_ids, n_num, cap):
    L = max(1, min(cap, max((len(s) for s in seqs), default=1)))
    ids = np.zeros((len(seqs), L, n_ids), dtype=np.int64)
    num = np.zeros((len(seqs), L, n_num), dtype=np.float32)
    msk = np.zeros((len(seqs), L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, (a, b) in enumerate(s[:L]):
            ids[i, j] = a
            num[i, j] = b
            msk[i, j] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(num), torch.from_numpy(msk)


def featurize(groups, deck):
    """entity -> token/bag 化は group ごとに 1 度だけ(epoch ごとに作り直さない)。"""
    for g in groups:
        if "_feat" in g:
            continue
        e = g["entity"]
        g["_feat"] = {
            "hand": EP.hand_bag(e),
            "board": EP.board_tokens(e, "self"),
            "opp": EP.board_tokens(e, "opp"),
            "opp_disc": EP.discard_bag(e, "opp"),
            "disc": EP.discard_bag(e, "self"),
            "unseen": EP.unseen_deck_bag(e, deck),
            "history": EP.history_tokens(g["history"]),
            "delta": [EP.delta_vec(g["before"], c.get("after")) for c in g["candidates"]],
        }


def make_batch(groups, smean, sstd, arm_families, deck):
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    state = np.zeros((B, len(smean)), dtype=np.float32)
    opt = np.zeros((B, C, D), dtype=np.float32)
    card = np.zeros((B, C), dtype=np.int64)
    delta = np.zeros((B, C, EP.N_SUMMARY + 1), dtype=np.float32)
    mask = np.zeros((B, C), dtype=bool)
    tgt = np.zeros((B, C, C), dtype=np.float32)
    wgt = np.zeros((B, C, C), dtype=np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], dtype=np.float32) - smean) / sstd
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            cid = cd["action_card_id"]
            card[b, c] = cid if 0 < cid < EP.VOCAB else 0
            mask[b, c] = True
            if "delta" in arm_families:
                delta[b, c] = g["_feat"]["delta"][c]
            elif "delta_shuf" in arm_families:      # 容量は同じ・情報だけ壊す対照
                nC = len(g["candidates"])
                delta[b, c] = g["_feat"]["delta"][(c + 1) % nC]
        for p in g["_pairs"]:
            t, w = ST.arm_target(p, "S3")
            if w > 0:
                tgt[b, p["i"], p["j"]] = t
                wgt[b, p["i"], p["j"]] = w
    out = {"state": torch.from_numpy(state), "opt": torch.from_numpy(opt),
           "card": torch.from_numpy(card), "mask": torch.from_numpy(mask),
           "delta": torch.from_numpy(delta), "tgt": torch.from_numpy(tgt),
           "wgt": torch.from_numpy(wgt)}
    if "hand" in arm_families:
        out["hand"] = _pad_ids([g["_feat"]["hand"] for g in groups], 20)
    if "board" in arm_families:
        i, n, m = _pad_tokens([g["_feat"]["board"] for g in groups], 2, 10, 6)
        out["board_ids"], out["board_num"], out["board_mask"] = i, n, m
    if "opp" in arm_families:
        i, n, m = _pad_tokens([g["_feat"]["opp"] for g in groups], 2, 10, 6)
        out["opp_ids"], out["opp_num"], out["opp_mask"] = i, n, m
        out["opp_disc"] = _pad_ids([g["_feat"]["opp_disc"] for g in groups], 60)
    if "deck" in arm_families:
        out["disc"] = _pad_ids([g["_feat"]["disc"] for g in groups], 60)
        out["unseen"] = _pad_ids([g["_feat"]["unseen"] for g in groups], 60)
    if "history" in arm_families:
        i, n, m = _pad_tokens([g["_feat"]["history"] for g in groups], 1, 3, EP.HIST_N)
        out["hist_ids"], out["hist_num"], out["hist_mask"] = i, n, m
    return out


@torch.no_grad()
def score_group(model, g, smean, sstd, fam, deck):
    model.eval()
    b = make_batch([g], smean, sstd, fam, deck)
    q = model(b)
    return q[0][b["mask"][0]].tolist()


def train_arm(tr, va, fam, smean, sstd, args, seed, deck):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(tr[0]["candidates"][0]["option_feat"])
    model = EP.ProbeNet(len(smean), D, families=fam, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot = nb = 0
        for i in range(0, len(perm), args.batch):
            gs = [tr[j] for j in perm[i:i + args.batch]]
            b = make_batch(gs, smean, sstd, fam, deck)
            q = model(b).masked_fill(~b["mask"], 0.0)
            loss = TS.pair_loss(q, b["tgt"], b["wgt"], b["mask"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        m = SE.evaluate(va, lambda g: score_group(model, g, smean, sstd, fam, deck))
        sel = m["stable"] or 0.0
        print(f"    ep{ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
              f"val_stable={m['stable']}", file=sys.stderr, flush=True)
        if sel > best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_ent_probes_w*.jsonl.gz"))
    ap.add_argument("--deck", default=str(_HERE.parent.parent / "sample_submission" / "deck.csv"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--ref", default="P0_base")
    ap.add_argument("--out", default=str(_HERE / "phase14_probe.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    groups = load(args.data)
    TS.stratified_split(groups)
    TS.attach_pairs(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    deck = deck_ids(args.deck)
    featurize(groups, deck)
    print(f"groups train={len(tr)} val={len(va)} test={len(te)} deck={len(deck)}",
          file=sys.stderr)

    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    allp = [p for g in groups for p in g["_pairs"]]
    rep = {"data": args.data, "target": "Phase12 S3 (soft block-support + confidence)",
           "groups": {"total": len(groups), "train": len(tr), "val": len(va), "test": len(te)},
           "split_audit": TS.split_audit(groups),
           "pairs": {"total": len(allp),
                     **{k: round(sum(1 for p in allp if p["cls"] == k) / max(1, len(allp)), 4)
                        for k in ("stable", "mostly", "unstable", "always_tie")}},
           "arms": {}, "bootstrap": {}, "per_seed": {}}

    PG = {}
    for name in args.arms.split(","):
        fam = ARMS[name]
        t0 = time.time()
        per, pgs = [], []
        for seed in [int(s) for s in args.seeds.split(",")]:
            model = train_arm(tr, va, fam, smean, sstd, args, seed, deck)
            m = SE.evaluate(te, lambda g: score_group(model, g, smean, sstd, fam, deck))
            pgs.append(m.pop("_per_group"))
            per.append(m)
        keys = [k for k in per[0] if isinstance(per[0][k], (int, float))
                and not k.endswith("_n")]
        agg = {k: round(statistics.mean([p[k] for p in per if p[k] is not None]), 4)
               for k in keys if any(p[k] is not None for p in per)}
        for k in ("stable", "support_weighted", "margin_weighted", "regret"):
            agg["per_seed_" + k] = [p[k] for p in per]
            vals = [p[k] for p in per if p[k] is not None]
            agg["sd_" + k] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
        agg["train_sec"] = round(time.time() - t0, 1)
        agg["families"] = list(fam)
        agg["n_params"] = sum(p.numel() for p in
                              EP.ProbeNet(len(smean), len(tr[0]["candidates"][0]["option_feat"]),
                                          families=fam, hidden=args.hidden).parameters())
        rep["arms"][name] = agg
        n = min(len(x) for x in pgs)
        PG[name] = [{**{k: (statistics.mean([x[i][k][0] for x in pgs]),
                          statistics.mean([x[i][k][1] for x in pgs])) for k in SE._METRICS},
                     "_regret": statistics.mean([x[i]["_regret"] for x in pgs])}
                    for i in range(n)]
        print(f"  [{name}] stable={agg.get('stable')} supp={agg.get('support_weighted')} "
              f"marg={agg.get('margin_weighted')} regret={agg.get('regret')} "
              f"params={agg['n_params']}", file=sys.stderr, flush=True)

    ref = args.ref if args.ref in PG else "P0_base"
    for name in PG:
        if name == ref:
            continue
        for met in ("stable", "support_weighted", "margin_weighted", "large_margin",
                    "all_pairwise", "regret"):
            rep["bootstrap"][f"{name}-{ref}_{met}"] = TS.paired_bootstrap(
                PG[name], PG[ref], met)

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": rep["groups"], "pairs": rep["pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
