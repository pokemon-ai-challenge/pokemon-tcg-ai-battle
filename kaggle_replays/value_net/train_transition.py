"""Phase17 §22-§39: T0/T1/T2/T4 と必須対照(after-shuffle / no-relation)の比較。

target は Phase12/15 と同じ block-support soft confidence(§27)。変更しない。
全 arm で split / optimizer / LR / batch / epoch / model selection / seed を固定(§37)。

  T0            : state166 + option65 + action card id
  T1            : T0 + handcrafted delta23(= Phase15 D1)
  T2            : raw entity Transformer(current only)
  T3            : before/after(latent delta なし)
  T4            : before/after + latent delta  ← 本命
  T4_shuffle    : T4 と**パラメータ完全一致**、after だけ group 内でずらす(§23)
  T4_norel      : T4 から relation bias のみ除去(§24)
  EPool         : Transformer を使わず entity MLP + masked mean の transition(§25)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import delta_features as DF  # noqa: E402
import entity_tokens as ET  # noqa: E402
import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
import train_delta_q as TD  # noqa: E402
import train_soft as TS  # noqa: E402
import transition_net as TN  # noqa: E402

ARMS = {
    "T0":         {"kind": "mlp", "delta": False},
    "T1":         {"kind": "mlp", "delta": True},
    "T2":         {"kind": "tf", "mode": "current", "rel": True, "shuffle": False},
    "T3":         {"kind": "tf", "mode": "before_after", "rel": True, "shuffle": False},
    "T4":         {"kind": "tf", "mode": "transition", "rel": True, "shuffle": False},
    "T4_shuffle": {"kind": "tf", "mode": "transition", "rel": True, "shuffle": True},
    "T4_norel":   {"kind": "tf", "mode": "transition", "rel": False, "shuffle": False},
    "EPool":      {"kind": "pool", "mode": "transition"},
    "EP_C":       {"kind": "pool", "mode": "current"},        # §14 Current-only(after 不使用)
    # Phase17 R1 との整合を取るための追加対照: pooling 版 after-shuffle(容量完全一致)
    "EPool_shuffle": {"kind": "pool", "mode": "transition", "shuffle": True},
    # §38 容量対照: state166+option65 のまま entity 系と同規模まで MLP を太らせる
    "T0_cap":     {"kind": "mlp", "delta": False, "hidden": 256},
    "T1_cap":     {"kind": "mlp", "delta": True, "hidden": 256},
}
MAX_TOK = 40


# ---------------- tokenization cache ----------------

def prepare(groups):
    for g in groups:
        if "_tok" in g:
            continue
        g["_tok"] = ET.tokenize(g["entity"])
        g["_rel"] = ET.relation_matrix(g["_tok"])
        atoks, arels = [], []
        for c in g["candidates"]:
            ae = c.get("after_entity")
            t = ET.tokenize(ae) if ae else g["_tok"]
            atoks.append(t)
            arels.append(ET.relation_matrix(t) if ae else g["_rel"])
        g["_atok"], g["_arel"] = atoks, arels
        g["_delta"] = [DF.single(g["before"], (c.get("afters") or [None])[0])
                       for c in g["candidates"]]
        # §30 delta-collision: 同一 group 内で delta23 が完全一致する候補ペア
        n = len(g["candidates"])
        g["_collision"] = [[DF.pair_distance(g["_delta"][i], g["_delta"][j]) < 1e-9
                            for j in range(n)] for i in range(n)]


def _pack(toks, rels, cap=MAX_TOK):
    B = len(toks)
    T = max(1, min(cap, max(len(t["card_id"]) for t in toks)))
    cid = np.zeros((B, T), np.int64); typ = np.zeros((B, T), np.int64)
    own = np.zeros((B, T), np.int64); zon = np.zeros((B, T), np.int64)
    slt = np.zeros((B, T), np.int64); num = np.zeros((B, T, ET.N_NUM), np.float32)
    msk = np.zeros((B, T), bool); rel = np.zeros((B, T, T), np.int64)
    for b, (t, r) in enumerate(zip(toks, rels)):
        n = min(T, len(t["card_id"]))
        cid[b, :n] = t["card_id"][:n]; typ[b, :n] = t["type"][:n]
        own[b, :n] = t["owner"][:n]; zon[b, :n] = t["zone"][:n]
        slt[b, :n] = t["slot"][:n]; num[b, :n] = np.asarray(t["num"][:n], np.float32)
        msk[b, :n] = True
        for i in range(n):
            rel[b, i, :n] = r[i][:n]
    return {"card_id": torch.from_numpy(cid), "type": torch.from_numpy(typ),
            "owner": torch.from_numpy(own), "zone": torch.from_numpy(zon),
            "slot": torch.from_numpy(slt), "num": torch.from_numpy(num),
            "mask": torch.from_numpy(msk), "rel": torch.from_numpy(rel)}


def make_batch(groups, smean, sstd, arm):
    cfg = ARMS[arm]
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    opt = np.zeros((B, C, D), np.float32)
    card = np.zeros((B, C), np.int64)
    mask = np.zeros((B, C), bool)
    tgt = np.zeros((B, C, C), np.float32)
    wgt = np.zeros((B, C, C), np.float32)
    state = np.zeros((B, len(smean)), np.float32)
    dlt = np.zeros((B, C, DF.N_DELTA), np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], np.float32) - smean) / sstd
        n = len(g["candidates"])
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            ci = cd["action_card_id"]
            card[b, c] = ci if 0 < ci < ET.VOCAB else 0
            mask[b, c] = True
            dlt[b, c] = g["_delta"][c]
        for p in g["_pairs"]:
            t, w = ST.arm_target(p, "S3")
            if w > 0:
                tgt[b, p["i"], p["j"]] = t
                wgt[b, p["i"], p["j"]] = w
    out = {"opt": torch.from_numpy(opt), "card": torch.from_numpy(card),
           "mask": torch.from_numpy(mask), "tgt": torch.from_numpy(tgt),
           "wgt": torch.from_numpy(wgt), "state": torch.from_numpy(state),
           "delta": torch.from_numpy(dlt)}
    if cfg["kind"] in ("tf", "pool"):
        out["before"] = _pack([g["_tok"] for g in groups], [g["_rel"] for g in groups])
        if cfg.get("mode") != "current":
            at, ar = [], []
            for g in groups:
                n = len(g["candidates"])
                for c in range(C):
                    if c < n:
                        # §23 after-shuffle: 対応だけを 1 つずらす
                        idx = (c + 1) % n if cfg.get("shuffle") else c
                        at.append(g["_atok"][idx]); ar.append(g["_arel"][idx])
                    else:
                        at.append(g["_tok"]); ar.append(g["_rel"])
            packed = _pack(at, ar)
            out["after"] = {k: v.view(B, C, *v.shape[1:]) for k, v in packed.items()}
    return out


class MLPArm(torch.nn.Module):
    """T0 / T1(Phase15 DeltaQNet と同形)。hidden で容量対照も作れる。"""

    def __init__(self, state_dim, option_dim, use_delta, hidden=64):
        super().__init__()
        self.net = TD.DeltaQNet(state_dim, option_dim,
                                delta_dim=DF.N_DELTA if use_delta else 0, hidden=hidden)

    def forward(self, b):
        return self.net(b)


class PoolArm(torch.nn.Module):
    """§25 Transformer なしの entity pooling。mode="current" なら after を使わない。"""

    def __init__(self, option_dim, hidden=64, mode="transition"):
        super().__init__()
        self.mode = mode
        d = TN.D_MODEL
        self.card = torch.nn.Embedding(ET.VOCAB, d, padding_idx=0)
        self.typ = torch.nn.Embedding(ET.N_TYPE, d)
        self.own = torch.nn.Embedding(ET.N_OWNER, d)
        self.zon = torch.nn.Embedding(ET.N_ZONE, d)
        self.slt = torch.nn.Embedding(ET.N_SLOT, d)
        self.num = torch.nn.Linear(ET.N_NUM, d)
        self.tok = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU())
        self.act = torch.nn.Sequential(torch.nn.Linear(option_dim + d, hidden),
                                       torch.nn.ReLU())
        self.act_card = torch.nn.Embedding(ET.VOCAB, d, padding_idx=0)
        n_state = 1 if mode == "current" else 3
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d * n_state + hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def _pool(self, t):
        x = (self.card(t["card_id"]) + self.typ(t["type"]) + self.own(t["owner"])
             + self.zon(t["zone"]) + self.slt(t["slot"]) + self.num(t["num"]))
        h = self.tok(x) * t["mask"].unsqueeze(-1)
        return h.sum(1) / t["mask"].sum(1, keepdim=True).clamp(min=1)

    def forward(self, b):
        B, C = b["opt"].shape[0], b["opt"].shape[1]
        zb = self._pool(b["before"]).unsqueeze(1).expand(B, C, -1)
        a = self.act(torch.cat([b["opt"], self.act_card(b["card"])], dim=-1))
        if self.mode == "current":
            parts = [zb, a]                     # after を一切見ない
        else:
            flat = {k: v.reshape(B * C, *v.shape[2:]) for k, v in b["after"].items()}
            za = self._pool(flat).view(B, C, -1)
            parts = [zb, za, za - zb, a]
        q = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return q.masked_fill(~b["mask"], -1e9)


def build_model(arm, state_dim, option_dim):
    cfg = ARMS[arm]
    if cfg["kind"] == "mlp":
        return MLPArm(state_dim, option_dim, cfg["delta"], cfg.get("hidden", 64))
    if cfg["kind"] == "pool":
        return PoolArm(option_dim, mode=cfg.get("mode", "transition"))
    return TN.TransitionQNet(option_dim, mode=cfg["mode"], use_relation=cfg["rel"])


@torch.no_grad()
def score_group(model, g, smean, sstd, arm):
    model.eval()
    b = make_batch([g], smean, sstd, arm)
    return model(b)[0][b["mask"][0]].tolist()


def train_arm(tr, va, arm, smean, sstd, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(tr[0]["candidates"][0]["option_feat"])
    model = build_model(arm, len(smean), D)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot = nb = 0
        for i in range(0, len(perm), args.batch):
            gs = [tr[j] for j in perm[i:i + args.batch]]
            b = make_batch(gs, smean, sstd, arm)
            q = model(b).masked_fill(~b["mask"], 0.0)
            loss = TS.pair_loss(q, b["tgt"], b["wgt"], b["mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        m = SE.evaluate(va, lambda g: score_group(model, g, smean, sstd, arm))
        sel = m["stable"] or 0.0
        print(f"    [{arm} s{seed}] ep{ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
              f"val_stable={m['stable']}", file=sys.stderr, flush=True)
        if sel > best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------- §30 collision subset 評価 ----------------

def subset_pairwise(model, groups, smean, sstd, arm, want_collision):
    """delta23 が完全一致するペアだけ(または一致しないペアだけ)の pairwise。"""
    num = den = 0.0
    per = []
    for g in groups:
        pred = score_group(model, g, smean, sstd, arm)
        gn = gd = 0.0
        for p in g["_pairs"]:
            if g["_collision"][p["i"]][p["j"]] != want_collision:
                continue
            d = 1 if p["support"] > 0.5 else (-1 if p["support"] < 0.5 else 0)
            if d == 0:
                continue
            diff = pred[p["i"]] - pred[p["j"]]
            gn += 0.5 if diff == 0 else (1.0 if diff * d > 0 else 0.0)
            gd += 1.0
        num += gn
        den += gd
        per.append((gn, gd))
    return (round(num / den, 4) if den else None), int(den), per


def boot_pairs(pa, pb, B=10000, seed=0):
    n = min(len(pa), len(pb))
    if n == 0:
        return None
    an = np.array([x[0] for x in pa[:n]]); ad = np.array([x[1] for x in pa[:n]])
    bn = np.array([x[0] for x in pb[:n]]); bd = np.array([x[1] for x in pb[:n]])
    if ad.sum() == 0 or bd.sum() == 0:
        return None
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, n, size=(B, n))
    m = (an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-9)
         - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-9))
    return {"mean_diff": round(float(an.sum() / ad.sum() - bn.sum() / bd.sum()), 4),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)], "n_groups": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_p17_probes_w*.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="T0,T1,T2,T4,T4_shuffle,T4_norel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(_HERE / "phase17_report.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    groups = []
    for f in sorted(glob.glob(args.data)):
        groups += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    if args.limit:
        groups = groups[:args.limit]
    TS.stratified_split(groups)
    TS.attach_pairs(groups)
    prepare(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    print(f"groups train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    X = np.asarray([g["state_feat"] for g in tr], np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    allp = [p for g in groups for p in g["_pairs"]]
    ncol = sum(1 for g in groups for p in g["_pairs"]
               if g["_collision"][p["i"]][p["j"]])
    rep = {"data": args.data, "target": "Phase12 S3 soft block-support + confidence",
           "groups": {"total": len(groups), "train": len(tr), "val": len(va),
                      "test": len(te), "games": len({g["game"] for g in groups}),
                      "candidates": sum(len(g["candidates"]) for g in groups)},
           "split_audit": TS.split_audit(groups),
           "pairs": {"total": len(allp), "delta_collision": ncol,
                     "collision_rate": round(ncol / max(1, len(allp)), 4),
                     **{k: round(sum(1 for p in allp if p["cls"] == k) / max(1, len(allp)), 4)
                        for k in ("stable", "mostly", "unstable", "always_tie")}},
           "arch": {"layers": TN.N_LAYER, "d_model": TN.D_MODEL, "heads": TN.N_HEAD,
                    "ffn": TN.D_FF, "max_tokens": MAX_TOK},
           "arms": {}, "bootstrap": {}, "subsets": {}}

    PG, COL, NCOL = {}, {}, {}
    for arm in args.arms.split(","):
        t0 = time.time()
        per, pgs, models = [], [], []
        for seed in [int(s) for s in args.seeds.split(",")]:
            model = train_arm(tr, va, arm, smean, sstd, args, seed)
            models.append(model)
            m = SE.evaluate(te, lambda g: score_group(model, g, smean, sstd, arm))
            pgs.append(m.pop("_per_group"))
            per.append(m)
        keys = [k for k in per[0] if isinstance(per[0][k], (int, float))
                and not k.endswith("_n")]
        agg = {k: round(statistics.mean([p[k] for p in per if p[k] is not None]), 4)
               for k in keys if any(p[k] is not None for p in per)}
        for k in ("stable", "support_weighted", "margin_weighted", "regret"):
            agg["per_seed_" + k] = [p[k] for p in per]
            v = [p[k] for p in per if p[k] is not None]
            agg["sd_" + k] = round(statistics.stdev(v), 4) if len(v) > 1 else None
        agg["n_params"] = sum(p.numel() for p in models[0].parameters())
        agg["train_sec"] = round(time.time() - t0, 1)
        c_acc, c_n, c_per = subset_pairwise(models[0], te, smean, sstd, arm, True)
        n_acc, n_n, n_per = subset_pairwise(models[0], te, smean, sstd, arm, False)
        agg["collision_pairwise"] = c_acc
        agg["collision_pairs"] = c_n
        agg["noncollision_pairwise"] = n_acc
        agg["noncollision_pairs"] = n_n
        if ARMS[arm]["kind"] == "tf":
            b = make_batch(te[:8], smean, sstd, arm)
            with torch.no_grad():
                models[0](b)
            agg["attention_entropy"] = round(models[0].enc.attention_entropy() or 0, 4)
        COL[arm], NCOL[arm] = c_per, n_per
        rep["arms"][arm] = agg
        n = min(len(x) for x in pgs)
        PG[arm] = [{**{k: (statistics.mean([x[i][k][0] for x in pgs]),
                          statistics.mean([x[i][k][1] for x in pgs])) for k in SE._METRICS},
                    "_regret": statistics.mean([x[i]["_regret"] for x in pgs])}
                   for i in range(n)]
        print(f"  [{arm}] stable={agg.get('stable')} supp={agg.get('support_weighted')} "
              f"collision={c_acc}({c_n}) noncol={n_acc} params={agg['n_params']}",
              file=sys.stderr, flush=True)

    for a, b in (("T4", "T1"), ("T4", "T2"), ("T4", "T4_shuffle"), ("T4", "T4_norel"),
                 ("T4", "T0"), ("T1", "T0"), ("T2", "T0"), ("T3", "T4"), ("EPool", "T4"),
                 ("T4", "T0_cap"), ("T2", "T0_cap"), ("EPool", "T0_cap"),
                 ("T0_cap", "T0"), ("T1_cap", "T0_cap"), ("EP_C", "EPool"), ("EP_C", "T0"),
                 ("EPool", "EPool_shuffle")):
        if a in PG and b in PG:
            for met in ("stable", "support_weighted", "margin_weighted",
                        "large_margin", "regret"):
                rep["bootstrap"][f"{a}-{b}_{met}"] = TS.paired_bootstrap(PG[a], PG[b], met)
            rep["bootstrap"][f"{a}-{b}_collision"] = boot_pairs(COL[a], COL[b])
            rep["bootstrap"][f"{a}-{b}_noncollision"] = boot_pairs(NCOL[a], NCOL[b])

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": rep["groups"], "pairs": rep["pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
