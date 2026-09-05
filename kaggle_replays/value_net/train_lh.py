"""Phase19 §20-§28: Long-Horizon Action-Q(target だけを terminal 勝率へ変更)。

  LH0 : state166 + option65 + action card id            (MLP)
  LH1 : raw current entity pooling + action             (after 不使用)
  LH2 : EPool 構造(before/after pooling)               (Phase17 EntityQ と同構造)

target: y_LH(s,a) = M 回の terminal 継続の平均勝敗(§5)。短期 teacher は使わない。
Loss  : Huber(value) + lambda_rank * pairwise(§22)。lambda は固定、探索しない。

**§32**: test では学習に使っていない continuation block(audit の block B)でも評価する。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
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

import delta_features as DF  # noqa: E402
import entity_tokens as ET  # noqa: E402
import train_delta_q as TD  # noqa: E402
import train_transition as TT  # noqa: E402

LAMBDA_RANK = 1.0        # 既存 Action-Q と同じ重み。探索しない(§22)
ARMS = {"TD0": "mlp", "TD1": "pool_current", "TD2": "pool_transition"}
HSTAR = ["H1"]          # §28 で選択した horizon


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def split_games(groups, fracs=(0.6, 0.2, 0.2), seed=0):
    """§12 game 単位 split。archetype/turn/先後を均衡化(教師値は使わない)。"""
    from collections import defaultdict
    by = defaultdict(list)
    for g in groups:
        by[g["game"]].append(g)
    cells = {}
    for game, gs in by.items():
        c = Counter()
        for x in gs:
            c["t:" + x["turn_band"]] += 1
            c["a:" + x["arch"]] += 1
            c["f:" + str(x["me_first"])] += 1
            c["N"] += 1
        cells[game] = c
    total = Counter()
    for c in cells.values():
        total.update(c)
    rs = np.random.RandomState(seed)
    order = sorted(cells, key=lambda g: (-cells[g]["N"], g))
    have = [Counter() for _ in range(3)]

    def cost(s, gc):
        return sum(((have[s][k] + gc[k]) - fracs[s] * n) ** 2 / n
                   for k, n in total.items() if n > 0)
    assign = {}
    for game in order:
        gc = cells[game]
        base = [cost(s, Counter()) for s in range(3)]
        s = min(range(3), key=lambda i: cost(i, gc) - base[i])
        assign[game] = s
        have[s].update(gc)
    for g in groups:
        g["_split"] = assign[g["game"]]
    return assign


def prepare(groups):
    for g in groups:
        if "_tok" in g:
            continue
        g["_tok"] = ET.tokenize(g["entity"])
        g["_rel"] = ET.relation_matrix(g["_tok"])
        at, ar = [], []
        for c in g["candidates"]:
            ae = c.get("after_entity")
            t = ET.tokenize(ae) if ae else g["_tok"]
            at.append(t)
            ar.append(ET.relation_matrix(t) if ae else g["_rel"])
        g["_atok"], g["_arel"] = at, ar
        h = HSTAR[0]
        def _v(c, blk):
            b = c.get(blk)
            if b is None:
                return None
            if h == "HT":
                return b.get("mean")
            z = b.get("hz", {}).get(h)
            return z["mean"] if z else None
        ya = [_v(c, "lh_a") for c in g["candidates"]]
        yb = [_v(c, "lh_b") for c in g["candidates"]]
        g["_yA"] = ya if all(x is not None for x in ya) else None
        g["_yB"] = yb if all(x is not None for x in yb) else None


def make_batch(groups, smean, sstd, arm, target="A"):
    kind = ARMS[arm]
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    opt = np.zeros((B, C, D), np.float32)
    card = np.zeros((B, C), np.int64)
    mask = np.zeros((B, C), bool)
    state = np.zeros((B, len(smean)), np.float32)
    y = np.zeros((B, C), np.float32)
    dlt = np.zeros((B, C, DF.N_DELTA), np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], np.float32) - smean) / sstd
        ys = g["_yA"] if target == "A" else (g["_yB"] or g["_yA"])
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            ci = cd["action_card_id"]
            card[b, c] = ci if 0 < ci < ET.VOCAB else 0
            mask[b, c] = True
            y[b, c] = ys[c]
    out = {"opt": torch.from_numpy(opt), "card": torch.from_numpy(card),
           "mask": torch.from_numpy(mask), "state": torch.from_numpy(state),
           "y": torch.from_numpy(y), "delta": torch.from_numpy(dlt)}
    if kind.startswith("pool"):
        out["before"] = TT._pack([g["_tok"] for g in groups], [g["_rel"] for g in groups])
        if kind == "pool_transition":
            at, ar = [], []
            for g in groups:
                n = len(g["candidates"])
                for c in range(C):
                    at.append(g["_atok"][c] if c < n else g["_tok"])
                    ar.append(g["_arel"][c] if c < n else g["_rel"])
            p = TT._pack(at, ar)
            out["after"] = {k: v.view(B, C, *v.shape[1:]) for k, v in p.items()}
    return out


def build_model(arm, state_dim, option_dim):
    kind = ARMS[arm]
    if kind == "mlp":
        return TD.DeltaQNet(state_dim, option_dim, delta_dim=0, hidden=64)
    return TT.PoolArm(option_dim,
                      mode="current" if kind == "pool_current" else "transition")


def lh_loss(q, y, mask):
    """Huber(value) + pairwise ranking(§22)。"""
    m = mask.float()
    p = torch.sigmoid(q)
    l_val = (nn.functional.smooth_l1_loss(p, y, reduction="none", beta=0.1) * m).sum() \
        / m.sum().clamp(min=1)
    qi, qj = q.unsqueeze(2), q.unsqueeze(1)
    yi, yj = y.unsqueeze(2), y.unsqueeze(1)
    valid = (mask.unsqueeze(2) & mask.unsqueeze(1)) & (yi > yj)
    l_rank = torch.tensor(0.0)
    if valid.any():
        diff = (qi - qj)[valid].clamp(-30, 30)
        w = (yi - yj)[valid]                      # 差が大きいペアを重く
        l_rank = (-nn.functional.logsigmoid(diff) * w).sum() / w.sum().clamp(min=1e-6)
    return l_val + LAMBDA_RANK * l_rank


# ---------------- 評価(§24) ----------------

def _pairwise(pred, y):
    tot = ok = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if y[i] == y[j]:
                continue
            tot += 1
            d = pred[i] - pred[j]
            ok += 0.5 if d == 0 else (1.0 if d * (y[i] - y[j]) > 0 else 0.0)
    return (ok, tot)


def _spearman(a, b):
    if len(a) < 3:
        return None

    def rk(v):
        n = len(v)
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


def evaluate(groups, score_fn, target="A"):
    per = []
    for g in groups:
        y = g["_yA"] if target == "A" else g["_yB"]
        if y is None:
            continue
        pred = score_fn(g)
        if pred is None or len(pred) != len(y):
            continue
        ok, tot = _pairwise(pred, y)
        bi = max(range(len(pred)), key=lambda i: pred[i])
        per.append({"pw": (ok, tot), "top1": 1.0 if y[bi] == max(y) else 0.0,
                    "regret": max(y) - y[bi], "sp": _spearman(pred, y)})
    if not per:
        return None
    num = sum(p["pw"][0] for p in per)
    den = sum(p["pw"][1] for p in per)
    sp = [p["sp"] for p in per if p["sp"] is not None]
    return {"n_groups": len(per),
            "long_pairwise": round(num / den, 4) if den else None,
            "long_top1": round(st.mean([p["top1"] for p in per]), 4),
            "long_regret": round(st.mean([p["regret"] for p in per]), 4),
            "long_spearman": round(st.mean(sp), 4) if sp else None,
            "_per": per}


def short_stable_pairwise(groups, score_fn):
    """§25 短期 teacher の stable pair 一致率(参考指標)。"""
    ok = tot = 0.0
    for g in groups:
        bl = [c.get("short_blocks") for c in g["candidates"]]
        if any(b is None for b in bl):
            continue
        pred = score_fn(g)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                d = [bl[i][b] - bl[j][b] for b in range(len(bl[i]))]
                sg = [0 if abs(x) < 0.005 else (1 if x > 0 else -1) for x in d]
                nz = [x for x in sg if x != 0]
                if not nz or max(nz.count(1), nz.count(-1)) != len(d):
                    continue
                tot += 1
                diff = pred[i] - pred[j]
                ok += 0.5 if diff == 0 else (1.0 if diff * st.mean(d) > 0 else 0.0)
    return round(ok / tot, 4) if tot else None


def boot(per_a, per_b, key, B=10000, seed=0):
    n = min(len(per_a), len(per_b))
    if n == 0:
        return None
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, n, size=(B, n))
    if key == "pairwise":
        an = np.array([p["pw"][0] for p in per_a[:n]])
        ad = np.array([p["pw"][1] for p in per_a[:n]])
        bn = np.array([p["pw"][0] for p in per_b[:n]])
        bd = np.array([p["pw"][1] for p in per_b[:n]])
        obs = an.sum() / ad.sum() - bn.sum() / bd.sum()
        m = (an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-9)
             - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-9))
    else:
        a = np.array([p[key] for p in per_a[:n]])
        b = np.array([p[key] for p in per_b[:n]])
        d = a - b
        obs = d.mean()
        m = d[idx].mean(1)
    return {"mean_diff": round(float(obs), 4),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)], "n_groups": n}


def train_arm(tr, va, arm, smean, sstd, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(tr[0]["candidates"][0]["option_feat"])
    model = build_model(arm, len(smean), D)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot = nb = 0
        for i in range(0, len(perm), args.batch):
            gs = [tr[j] for j in perm[i:i + args.batch]]
            b = make_batch(gs, smean, sstd, arm)
            q = model(b).masked_fill(~b["mask"], 0.0)
            loss = lh_loss(q, b["y"], b["mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        m = evaluate(va, lambda g: score_group(model, g, smean, sstd, arm))
        sel = m["long_regret"] if m else 1e9      # §31 選択は regret 優先
        print(f"    [{arm} s{seed}] ep{ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
              f"val_regret={sel} val_pw={m['long_pairwise'] if m else None}",
              file=sys.stderr, flush=True)
        if sel < best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_group(model, g, smean, sstd, arm):
    model.eval()
    b = make_batch([g], smean, sstd, arm)
    return model(b)[0][b["mask"][0]].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_lh_w*.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="TD0,TD1")
    ap.add_argument("--hstar", default="H1")
    ap.add_argument("--out", default=str(_HERE / "phase19_report.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    HSTAR[0] = args.hstar

    G = [g for g in load(args.data)]
    split_games(G)
    prepare(G)
    G = [g for g in G if g["_yA"] is not None]
    tr = [g for g in G if g["_split"] == 0]
    va = [g for g in G if g["_split"] == 1]
    te = [g for g in G if g["_split"] == 2]
    te_b = [g for g in te if g["_yB"] is not None]
    print(f"groups train={len(tr)} val={len(va)} test={len(te)} test_blockB={len(te_b)}",
          file=sys.stderr)

    X = np.asarray([g["state_feat"] for g in tr], np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    rep = {"data": args.data, "target": "%s bootstrapped (M=%d)" % (HSTAR[0], G[0]["m"]),
           "groups": {"total": len(G), "train": len(tr), "val": len(va), "test": len(te),
                      "test_blockB": len(te_b), "games": len({g["game"] for g in G})},
           "arms": {}, "bootstrap": {}, "frozen_ref": {}}

    PER = {}
    # 凍結モデル(短期 target で学習済み)を long-horizon 指標で評価
    for name, key in (("Q0_frozen", "q0_score"), ("EntityQ_frozen", "entityq_score")):
        m = evaluate(te, lambda g, k=key: [c[k] for c in g["candidates"]])
        PER[name] = m.pop("_per")
        m["short_stable_pairwise"] = short_stable_pairwise(
            te, lambda g, k=key: [c[k] for c in g["candidates"]])
        mb = evaluate(te_b, lambda g, k=key: [c[k] for c in g["candidates"]], target="B")
        m["blockB"] = {k: mb[k] for k in mb if k != "_per"} if mb else None
        rep["arms"][name] = m
        print(f"  [{name}] pw={m['long_pairwise']} top1={m['long_top1']} "
              f"regret={m['long_regret']}", file=sys.stderr, flush=True)

    for arm in args.arms.split(","):
        t0 = time.time()
        per_seed, pers, models = [], [], []
        for seed in [int(s) for s in args.seeds.split(",")]:
            model = train_arm(tr, va, arm, smean, sstd, args, seed)
            models.append(model)
            m = evaluate(te, lambda g: score_group(model, g, smean, sstd, arm))
            pers.append(m.pop("_per"))
            per_seed.append(m)
        agg = {k: round(st.mean([p[k] for p in per_seed if p[k] is not None]), 4)
               for k in ("long_pairwise", "long_top1", "long_regret", "long_spearman")}
        for k in ("long_pairwise", "long_top1", "long_regret"):
            agg["per_seed_" + k] = [p[k] for p in per_seed]
            v = [p[k] for p in per_seed if p[k] is not None]
            agg["sd_" + k] = round(st.stdev(v), 4) if len(v) > 1 else None
        agg["n_params"] = sum(p.numel() for p in models[0].parameters())
        agg["short_stable_pairwise"] = short_stable_pairwise(
            te, lambda g: score_group(models[0], g, smean, sstd, arm))
        mb = evaluate(te_b, lambda g: score_group(models[0], g, smean, sstd, arm),
                      target="B")
        agg["blockB"] = {k: mb[k] for k in mb if k != "_per"} if mb else None
        agg["train_sec"] = round(time.time() - t0, 1)
        rep["arms"][arm] = agg
        n = min(len(x) for x in pers)
        PER[arm] = [{"pw": (st.mean([x[i]["pw"][0] for x in pers]),
                            st.mean([x[i]["pw"][1] for x in pers])),
                     "top1": st.mean([x[i]["top1"] for x in pers]),
                     "regret": st.mean([x[i]["regret"] for x in pers])}
                    for i in range(n)]
        print(f"  [{arm}] pw={agg['long_pairwise']} top1={agg['long_top1']} "
              f"regret={agg['long_regret']} params={agg['n_params']}",
              file=sys.stderr, flush=True)

    for a, b in (("TD1", "EntityQ_frozen"), ("TD0", "EntityQ_frozen"),
                 ("TD1", "TD0"), ("TD0", "Q0_frozen"),
                 ("TD1", "Q0_frozen"), ("EntityQ_frozen", "Q0_frozen")):
        if a in PER and b in PER:
            for key in ("pairwise", "top1", "regret"):
                rep["bootstrap"][f"{a}-{b}_{key}"] = boot(PER[a], PER[b], key)

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": rep["groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
