"""Phase15 §7/§16/§20: Action-Delta Encoder の正式比較。

構造は Q0-expanded と同形の base に **Delta Encoder だけ**を足す(§4)。
アーム間で違うのは delta の作り方だけで、他は完全に同一条件(§15)。

  D0          : delta なし(正式 base)
  D0_cap      : delta なしのまま Delta Encoder 相当の容量を足す(容量分離, §16)
  D1          : single delta(determinization 1本の実現値)
  D1_shuffle  : D1 と**パラメータ数完全一致**で、group 内の delta 対応だけ破壊(§7 必須)
  D2          : expected delta(M=4 平均)
  D3          : expected delta + std(D2 が D1 を上回った場合のみ, §24)
  D4          : next-state のみ(差分でなく s' 自体, §6 診断)
  Ddet        : 事前に count と分類した次元だけの expected delta(§9)

checkpoint 選択は validation の stable pairwise のみ。test は見ない(§15)。
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
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import delta_features as DF  # noqa: E402
import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
import train_soft as TS  # noqa: E402
from entity_probe import CARD_D, VOCAB  # noqa: E402

# arm -> (delta 種別, 追加容量を持つか)
ARMS = {
    "D0":         ("none", False),
    "D0_cap":     ("none", True),
    "D1":         ("single", True),
    "D1_shuffle": ("single_shuffle", True),
    "D2":         ("expected", True),
    "D3":         ("expected_std", True),
    "D4":         ("next_state", True),
    "Ddet":       ("deterministic", True),
}
DELTA_DIM = {"none": 0, "single": DF.N_DELTA, "single_shuffle": DF.N_DELTA,
             "expected": DF.N_DELTA, "expected_std": DF.N_DELTA + DF.N_BASE,
             "next_state": DF.N_DELTA, "deterministic": DF.N_DELTA}


class DeltaQNet(nn.Module):
    def __init__(self, state_dim, option_dim, delta_dim=0, cap_dim=0, hidden=64):
        super().__init__()
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.card_emb = nn.Embedding(VOCAB, CARD_D, padding_idx=0)
        self.act_mlp = nn.Sequential(nn.Linear(option_dim + CARD_D, hidden), nn.ReLU())
        self.delta_dim = delta_dim
        # D0_cap 用: delta を持たない場合でも同等の容量を持つダミー枝を用意する
        self.delta_mlp = (nn.Sequential(nn.Linear(max(1, delta_dim or cap_dim), 32),
                                        nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
                          if (delta_dim or cap_dim) else None)
        extra = 32 if self.delta_mlp is not None else 0
        self.fuse = nn.Sequential(nn.Linear(hidden * 2 + extra, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.q_head = nn.Linear(hidden, 1)

    def forward(self, b):
        B, C, _ = b["opt"].shape
        ctx = self.state_mlp(b["state"]).unsqueeze(1).expand(B, C, -1)
        act = self.act_mlp(torch.cat([b["opt"], self.card_emb(b["card"])], dim=-1))
        parts = [ctx, act]
        if self.delta_mlp is not None:
            parts.append(self.delta_mlp(b["delta"]))
        q = self.q_head(self.fuse(torch.cat(parts, dim=-1))).squeeze(-1)
        return q.masked_fill(~b["mask"], -1e9)


def build_delta(g, kind, n_cand):
    """group -> [C, delta_dim] の delta 行列。"""
    before = g["before"]
    rows = []
    for c in g["candidates"][:n_cand]:
        af = c.get("afters") or []
        if kind == "single":
            rows.append(DF.single(before, af[0] if af else None))
        elif kind == "expected":
            rows.append(DF.expected(before, af))
        elif kind == "expected_std":
            rows.append(DF.expected(before, af) + DF.std(before, af))
        elif kind == "next_state":
            rows.append(DF.expected_next_state(af))
        elif kind == "deterministic":
            rows.append(DF.deterministic_only(before, af))
        else:
            rows.append([])
    if kind == "single_shuffle" or kind == "none":
        pass
    return rows


def featurize(groups):
    for g in groups:
        if "_dcache" in g:
            continue
        before = g["before"]
        cache = {}
        singles = [DF.single(before, (c.get("afters") or [None])[0]) for c in g["candidates"]]
        cache["single"] = singles
        n = len(singles)
        # 容量一致対照: delta の**対応だけ**を 1 つずらす(値の分布は不変)
        cache["single_shuffle"] = [singles[(i + 1) % n] for i in range(n)]
        cache["expected"] = [DF.expected(before, c.get("afters") or []) for c in g["candidates"]]
        cache["expected_std"] = [
            DF.expected(before, c.get("afters") or []) + DF.std(before, c.get("afters") or [])
            for c in g["candidates"]]
        cache["next_state"] = [DF.expected_next_state(c.get("afters") or [])
                               for c in g["candidates"]]
        cache["deterministic"] = [DF.deterministic_only(before, c.get("afters") or [])
                                  for c in g["candidates"]]
        cache["var"] = [DF.variance_scalar(before, c.get("afters") or [])
                        for c in g["candidates"]]
        g["_dcache"] = cache


def delta_width(arm):
    """モデルが受け取る delta 入力幅。D0_cap は容量だけ合わせるのでゼロ埋め幅を持つ。"""
    kind, cap = ARMS[arm]
    d = DELTA_DIM[kind]
    return d if d else (DF.N_DELTA if cap else 0)


def make_batch(groups, smean, sstd, arm):
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    kind = ARMS[arm][0]
    dd = delta_width(arm)
    fill = DELTA_DIM[kind]          # 実際に値を入れる幅(D0_cap は 0 = ゼロのまま)
    state = np.zeros((B, len(smean)), dtype=np.float32)
    opt = np.zeros((B, C, D), dtype=np.float32)
    card = np.zeros((B, C), dtype=np.int64)
    delta = np.zeros((B, C, max(1, dd)), dtype=np.float32)
    mask = np.zeros((B, C), dtype=bool)
    tgt = np.zeros((B, C, C), dtype=np.float32)
    wgt = np.zeros((B, C, C), dtype=np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], dtype=np.float32) - smean) / sstd
        rows = g["_dcache"].get(kind) if fill else None
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            cid = cd["action_card_id"]
            card[b, c] = cid if 0 < cid < VOCAB else 0
            mask[b, c] = True
            if rows is not None:
                delta[b, c, :fill] = rows[c]
        for p in g["_pairs"]:
            t, w = ST.arm_target(p, "S3")
            if w > 0:
                tgt[b, p["i"], p["j"]] = t
                wgt[b, p["i"], p["j"]] = w
    return {"state": torch.from_numpy(state), "opt": torch.from_numpy(opt),
            "card": torch.from_numpy(card), "mask": torch.from_numpy(mask),
            "delta": torch.from_numpy(delta), "tgt": torch.from_numpy(tgt),
            "wgt": torch.from_numpy(wgt)}


def build_model(arm, state_dim, option_dim, hidden):
    kind, cap = ARMS[arm]
    dd = DELTA_DIM[kind]
    return DeltaQNet(state_dim, option_dim, delta_dim=dd,
                     cap_dim=(DF.N_DELTA if (cap and dd == 0) else 0), hidden=hidden)


@torch.no_grad()
def score_group(model, g, smean, sstd, arm):
    model.eval()
    b = make_batch([g], smean, sstd, arm)
    q = model(b)
    return q[0][b["mask"][0]].tolist()


def train_arm(tr, va, arm, smean, sstd, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(tr[0]["candidates"][0]["option_feat"])
    model = build_model(arm, len(smean), D, args.hidden)
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


def subset_eval(model, groups, smean, sstd, arm, keyfn):
    out = {}
    for k in sorted({keyfn(g) for g in groups}):
        sub = [g for g in groups if keyfn(g) == k]
        if len(sub) < 5:
            continue
        m = SE.evaluate(sub, lambda g: score_group(model, g, smean, sstd, arm))
        m.pop("_per_group")
        out[str(k)] = {"groups": len(sub), "stable": m["stable"],
                       "support_weighted": m["support_weighted"], "regret": m["regret"]}
    return out


def var_band(g):
    v = max(g["_dcache"]["var"])
    return "low" if v < 1e-9 else "medium" if v < 0.05 else "high"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_dlt_probes_w*.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="D0,D0_cap,D1,D1_shuffle")
    ap.add_argument("--out", default=str(_HERE / "phase15_delta.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    groups = []
    for f in sorted(glob.glob(args.data)):
        groups += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    TS.stratified_split(groups)
    TS.attach_pairs(groups)
    featurize(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    print(f"groups train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    allp = [p for g in groups for p in g["_pairs"]]
    # §18.1 候補間 delta 距離 / §18.2 variance
    dists = []
    for g in groups:
        e = g["_dcache"]["expected"]
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                dists.append(DF.pair_distance(e[i], e[j]))
    allvar = [v for g in groups for v in g["_dcache"]["var"]]
    rep = {
        "data": args.data, "delta_schema": DF.schema(),
        "target": "Phase12 S3 (block-support soft + confidence)",
        "groups": {"total": len(groups), "train": len(tr), "val": len(va), "test": len(te),
                   "games": len({g["game"] for g in groups}),
                   "candidates": sum(len(g["candidates"]) for g in groups),
                   "mean_candidates": round(statistics.mean(
                       [len(g["candidates"]) for g in groups]), 2)},
        "split_audit": TS.split_audit(groups),
        "pairs": {"total": len(allp),
                  **{k: round(sum(1 for p in allp if p["cls"] == k) / max(1, len(allp)), 4)
                     for k in ("stable", "mostly", "unstable", "always_tie")}},
        "delta_diagnostics": {
            "candidate_pair_distance_mean": round(statistics.mean(dists), 4),
            "candidate_pair_distance_zero_rate": round(
                sum(1 for d in dists if d < 1e-9) / len(dists), 4),
            "determinization_variance_mean": round(statistics.mean(allvar), 6),
            "determinization_variance_zero_rate": round(
                sum(1 for v in allvar if v < 1e-9) / len(allvar), 4),
            "var_band_groups": {b: sum(1 for g in groups if var_band(g) == b)
                                for b in ("low", "medium", "high")},
        },
        "arms": {}, "bootstrap": {}, "subsets": {}}

    PG = {}
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
        agg["delta_kind"] = ARMS[arm][0]
        agg["delta_dim"] = delta_width(arm)
        agg["train_sec"] = round(time.time() - t0, 1)
        rep["arms"][arm] = agg
        rep["subsets"][arm] = {
            "variance": subset_eval(models[0], te, smean, sstd, arm, var_band),
            "action_type": subset_eval(models[0], te, smean, sstd, arm,
                                       lambda g: g["candidates"][0]["option_type"]),
            "turn": subset_eval(models[0], te, smean, sstd, arm, lambda g: g["turn_band"]),
        }
        n = min(len(x) for x in pgs)
        PG[arm] = [{**{k: (statistics.mean([x[i][k][0] for x in pgs]),
                          statistics.mean([x[i][k][1] for x in pgs])) for k in SE._METRICS},
                    "_regret": statistics.mean([x[i]["_regret"] for x in pgs])}
                   for i in range(n)]
        print(f"  [{arm}] stable={agg.get('stable')} supp={agg.get('support_weighted')} "
              f"marg={agg.get('margin_weighted')} regret={agg.get('regret')} "
              f"params={agg['n_params']}", file=sys.stderr, flush=True)

    for a, b in (("D1", "D0"), ("D1", "D1_shuffle"), ("D1", "D0_cap"),
                 ("D2", "D1"), ("D2", "D0"), ("D3", "D2"), ("D4", "D1"), ("Ddet", "D2")):
        if a in PG and b in PG:
            for met in ("stable", "support_weighted", "margin_weighted",
                        "large_margin", "all_pairwise", "regret"):
                rep["bootstrap"][f"{a}-{b}_{met}"] = TS.paired_bootstrap(PG[a], PG[b], met)

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": rep["groups"], "pairs": rep["pairs"],
                      "delta_diagnostics": rep["delta_diagnostics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
