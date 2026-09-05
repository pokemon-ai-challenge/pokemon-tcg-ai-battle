"""Phase12: Search Distillation v0 — S0/S1/S2/S3 の同条件学習と評価。

モデル構造は Q0-expanded と同一(ActionQNet, use_cards=False, use_action_card=True)で固定。
アーム間で違うのは **target と pair weight だけ**(§1)。Loss 実装も 1 本に統一する:

    L_pair = Σ w_ij * BCEWithLogits(Q_i - Q_j, t_ij) / Σ w_ij

  S0: t = hard(block A), w = 1        (従来方式のアンカー)
  S1: t = hard(4block 多数決), w = 1  (2/2 は除外)
  S2: t = block-support p_ij, w = 1
  S3: t = block-support p_ij, w = confidence_ij   (本命)

**制約(正直に記録)**: 教師 block データセットには `selected_option` / `outcome` /
zone card ids が保存されていないため、Phase7〜11 の `L_return` / `L_policy_aux` は
本フェーズでは使用できない。全アームで同一に落としているのでアーム間比較は汚染されないが、
絶対値を Phase 9C の Q0-expanded と直接比較することはできない(§24-A に明記)。

checkpoint 選択は **validation の stable pairwise**(主指標)で行う。test は一切見ない(§22)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
from action_q import ActionQNet  # noqa: E402

ARMS = ("S0", "S1", "S2", "S3")
OPT_ARMS = ("S4",)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]


# ---------------- §12 層化 split(試合単位・教師スコア不使用) ----------------

def stratified_split(groups, fracs=(0.62, 0.17, 0.21), seed=0):
    """試合単位 split。turn 帯 / candidate 数帯 / archetype の**周辺分布**を揃える。

    試合は分割できない(§12)ので、各試合をひとかたまりとして貪欲に配置し、
    「割り当て後の各 split の周辺分布が目標比率からどれだけずれるか」を最小化する。
    大きい試合から先に置く(bin packing と同じ理由)。

    教師 score・モデル性能は使用しない(§12 禁止事項)。
    """
    by_game = defaultdict(list)
    for g in groups:
        by_game[g["game"]].append(g)

    def cells(gs):
        c = Counter()
        for x in gs:
            c["t:" + x["turn_band"]] += 1
            c["c:" + x["cand_band"]] += 1
            c["a:" + x["arch"]] += 1
            c["N"] += 1
        return c

    game_cells = {game: cells(gs) for game, gs in by_game.items()}
    total = Counter()
    for c in game_cells.values():
        total.update(c)

    rs = np.random.RandomState(seed)
    order = sorted(game_cells, key=lambda g: (-game_cells[g]["N"], g))
    # 同サイズ内の順序だけランダム化(決定性は seed で担保)
    buckets = defaultdict(list)
    for g in order:
        buckets[game_cells[g]["N"]].append(g)
    order = []
    for n in sorted(buckets, reverse=True):
        b = buckets[n]
        rs.shuffle(b)
        order.extend(b)

    have = [Counter() for _ in range(3)]

    def cost(s, gc):
        """split s に gc を足した後の、s の全 cell の相対二乗ずれ。"""
        e = 0.0
        for k, n_tot in total.items():
            if n_tot <= 0:
                continue
            want = fracs[s] * n_tot
            e += ((have[s][k] + gc[k]) - want) ** 2 / n_tot
        return e

    assign = {}
    for game in order:
        gc = game_cells[game]
        base = [cost(s, Counter()) for s in range(3)]
        s = min(range(3), key=lambda i: cost(i, gc) - base[i])
        assign[game] = s
        have[s].update(gc)
    for g in groups:
        g["_split"] = assign[g["game"]]
    return assign


def split_audit(groups):
    out = {}
    for s, name in enumerate(("train", "validation", "test")):
        gs = [g for g in groups if g["_split"] == s]
        n = max(1, len(gs))
        out[name] = {
            "groups": len(gs), "games": len({g["game"] for g in gs}),
            "candidates": sum(len(g["candidates"]) for g in gs),
            "turn_band": {k: round(v / n, 4) for k, v in
                          sorted(Counter(g["turn_band"] for g in gs).items())},
            "cand_band": {k: round(v / n, 4) for k, v in
                          sorted(Counter(g["cand_band"] for g in gs).items())},
            "arch": {k: round(v / n, 4) for k, v in
                     sorted(Counter(g["arch"] for g in gs).items())},
            "mean_turn": round(statistics.mean([g["turn"] for g in gs]), 2) if gs else None,
        }
    a, b = out["validation"], out["test"]
    out["val_test_max_abs_diff"] = {
        f: round(max(abs(a[f].get(k, 0) - b[f].get(k, 0))
                     for k in set(a[f]) | set(b[f])), 4)
        for f in ("turn_band", "cand_band", "arch")}
    return out


# ---------------- pair 前処理 ----------------

def attach_pairs(groups, n_blocks=ST.NB):
    for g in groups:
        g["_pairs"] = ST.group_pairs(g, n_blocks=n_blocks)
        g["_tmean"] = [statistics.mean(c["blocks"][:n_blocks]) for c in g["candidates"]]


def coverage(groups, arm):
    used = tot = 0
    for g in groups:
        for p in g["_pairs"]:
            tot += 1
            if ST.arm_target(p, arm)[1] > 0:
                used += 1
    return round(used / tot, 4) if tot else None


# ---------------- batching ----------------

def make_batch(groups, smean, sstd, arm):
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    state = np.zeros((B, len(smean)), dtype=np.float32)
    opt = np.zeros((B, C, D), dtype=np.float32)
    cid = np.zeros((B, C), dtype=np.int64)
    otp = np.zeros((B, C), dtype=np.int64)
    mask = np.zeros((B, C), dtype=bool)
    tgt = np.zeros((B, C, C), dtype=np.float32)
    wgt = np.zeros((B, C, C), dtype=np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], dtype=np.float32) - smean) / sstd
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            cid[b, c] = max(0, cd["action_card_id"] + 1) if cd["action_card_id"] >= 0 else 0
            otp[b, c] = min(cd["option_type"], 63)
            mask[b, c] = True
        for p in g["_pairs"]:
            t, w = ST.arm_target(p, arm)
            if w > 0:
                tgt[b, p["i"], p["j"]] = t
                wgt[b, p["i"], p["j"]] = w
    zeros = torch.zeros((B, 1), dtype=torch.int64)
    return (torch.from_numpy(state), [zeros, zeros, zeros], torch.from_numpy(opt),
            torch.from_numpy(cid), torch.from_numpy(otp), torch.from_numpy(mask),
            torch.from_numpy(tgt), torch.from_numpy(wgt))


def pair_loss(q, tgt, wgt, mask):
    """Σ w*BCEWithLogits(Q_i-Q_j, t) / Σ w。padding 候補は mask で必ず除外。"""
    valid = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
    w = wgt * valid
    tot = w.sum()
    if float(tot) <= 0:
        return q.sum() * 0.0
    diff = q.unsqueeze(2) - q.unsqueeze(1)          # [B, C, C] = Q_i - Q_j
    diff = diff.clamp(-30.0, 30.0)
    l = nn.functional.binary_cross_entropy_with_logits(diff, tgt, reduction="none")
    return (l * w).sum() / tot


@torch.no_grad()
def score_group(model, g, smean, sstd):
    model.eval()
    st, zones, of, ci, ot, mk, *_ = make_batch([g], smean, sstd, "S2")
    q, _ = model(st, zones, of, ci, ot, mk)
    return q[0][mk[0]].tolist()


def train_arm(tr, va, arm, smean, sstd, args, seed, log=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(tr[0]["candidates"][0]["option_feat"])
    model = ActionQNet(len(smean), D, use_cards=False, use_action_card=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, hist = -1.0, None, []
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            gs = [tr[j] for j in perm[i:i + args.batch]]
            st, zones, of, ci, ot, mk, tg, wg = make_batch(gs, smean, sstd, arm)
            q, _ = model(st, zones, of, ci, ot, mk)
            q = q.masked_fill(~mk, 0.0)             # -1e9 を pair 差に混ぜない
            loss = pair_loss(q, tg, wg, mk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        m = SE.evaluate(va, lambda g: score_group(model, g, smean, sstd))
        sel = m["stable"] or 0.0                    # checkpoint 選択は val 主指標のみ
        hist.append({"ep": ep + 1, "loss": round(tot / max(1, nb), 4),
                     "val_stable": m["stable"], "val_all": m["all_pairwise"]})
        if log:
            print(f"    [{arm} s{seed}] ep{ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
                  f"val_stable={m['stable']} val_all={m['all_pairwise']}",
                  file=sys.stderr, flush=True)
        if sel > best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


# ---------------- bootstrap (§17) ----------------

def paired_bootstrap(pg_a, pg_b, metric, B=10000, seed=0):
    """group 単位 paired bootstrap。pooled num/den 比の差を再集計する。"""
    n = min(len(pg_a), len(pg_b))
    if n == 0:
        return None
    rs = np.random.RandomState(seed)
    if metric == "regret":
        a = np.asarray([g["_regret"] for g in pg_a[:n]])
        b = np.asarray([g["_regret"] for g in pg_b[:n]])
        idx = rs.randint(0, n, size=(B, n))
        d = (a - b)
        means = d[idx].mean(axis=1)
        obs = float(d.mean())
    else:
        an = np.asarray([g[metric][0] for g in pg_a[:n]])
        ad = np.asarray([g[metric][1] for g in pg_a[:n]])
        bn = np.asarray([g[metric][0] for g in pg_b[:n]])
        bd = np.asarray([g[metric][1] for g in pg_b[:n]])
        if ad.sum() == 0 or bd.sum() == 0:
            return None
        obs = float(an.sum() / ad.sum() - bn.sum() / bd.sum())
        idx = rs.randint(0, n, size=(B, n))
        means = (an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-9)
                 - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-9))
    return {"mean_diff": round(obs, 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4),
                     round(float(np.percentile(means, 97.5)), 4)],
            "n_groups": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_teacher_blocks_p12.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--frozen", default=str(_HERE / "frozen" / "Q0-expanded.pt"))
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--out", default=str(_HERE / "phase12_report.json"))
    args = ap.parse_args()

    groups = load(Path(args.data))
    if args.limit_groups:
        groups = groups[:args.limit_groups]
    stratified_split(groups)
    attach_pairs(groups)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    print(f"groups train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    allp = [p for g in groups for p in g["_pairs"]]
    cls = Counter(p["cls"] for p in allp)
    report = {
        "data": args.data, "n_blocks": ST.NB, "tie_eps": ST.TIE,
        "model": "ActionQNet(use_cards=False, use_action_card=True)  # Q0-expanded と同一構造",
        "loss": "sum_w * BCEWithLogits(Q_i-Q_j, t) / sum_w  (L_return/L_policy_aux は"
                " 教師blockデータに selected_option/outcome が無いため全アーム共通で不使用)",
        "split_audit": split_audit(groups),
        "pairs": {"total": len(allp),
                  **{k: round(cls[k] / max(1, len(allp)), 4)
                     for k in ("stable", "mostly", "unstable", "always_tie")}},
        "coverage": {a: {"train": coverage(tr, a), "test": coverage(te, a)}
                     for a in args.arms.split(",")},
        "arms": {}, "bootstrap": {}, "per_seed": {},
    }

    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    PG = {}
    if args.frozen:
        try:
            import torch as _t
            from action_q import ActionQNet as _AQ
            ck = _t.load(args.frozen, map_location="cpu", weights_only=False)
            fm = _AQ(ck["state_dim"], ck["option_dim"], use_cards=False)
            fm.load_state_dict(ck["state_dict"])
            fm.eval()
            fmean, fstd = np.asarray(ck["mean"]), np.asarray(ck["std"])
            frz = SE.evaluate(te, lambda g: score_group(fm, g, fmean, fstd))
            PG["Q0_expanded_frozen"] = frz.pop("_per_group")
            report["arms"]["Q0_expanded_frozen"] = frz
        except Exception as exc:                       # noqa: BLE001
            report["arms"]["Q0_expanded_frozen"] = {"error": str(exc)}
    pol = SE.evaluate(te, lambda g: [c["policy_score"] for c in g["candidates"]])
    PG["policy"] = pol.pop("_per_group")
    report["arms"]["policy"] = pol

    for arm in args.arms.split(","):
        t0 = time.time()
        per_seed, pgs = [], []
        for seed in [int(s) for s in args.seeds.split(",")]:
            model, hist = train_arm(tr, va, arm, smean, sstd, args, seed)
            m = SE.evaluate(te, lambda g: score_group(model, g, smean, sstd))
            pgs.append(m.pop("_per_group"))
            m["val_last"] = hist[-1]
            per_seed.append(m)
        keys = [k for k in per_seed[0]
                if isinstance(per_seed[0][k], (int, float)) and not k.endswith("_n")]
        agg = {k: round(statistics.mean([p[k] for p in per_seed if p[k] is not None]), 4)
               for k in keys if any(p[k] is not None for p in per_seed)}
        agg["per_seed_stable"] = [p["stable"] for p in per_seed]
        agg["per_seed_support_weighted"] = [p["support_weighted"] for p in per_seed]
        agg["per_seed_all_pairwise"] = [p["all_pairwise"] for p in per_seed]
        agg["per_seed_regret"] = [p["regret"] for p in per_seed]
        agg["abs_q_diff_by_margin"] = per_seed[0]["abs_q_diff_by_margin"]
        agg["calibration"] = per_seed[0]["calibration"]
        agg["train_sec"] = round(time.time() - t0, 1)
        report["arms"][arm] = agg
        report["per_seed"][arm] = [
            {k: p[k] for k in ("stable", "mostly", "support_weighted", "margin_weighted",
                               "large_margin", "all_pairwise", "regret", "spearman")}
            for p in per_seed]
        # seed 平均の per-group(paired bootstrap 用)
        n = min(len(x) for x in pgs)
        merged = []
        for i in range(n):
            e = {}
            for k in SE._METRICS:
                e[k] = (statistics.mean([x[i][k][0] for x in pgs]),
                        statistics.mean([x[i][k][1] for x in pgs]))
            e["_regret"] = statistics.mean([x[i]["_regret"] for x in pgs])
            merged.append(e)
        PG[arm] = merged
        print(f"  [{arm}] stable={agg.get('stable')} support_w={agg.get('support_weighted')} "
              f"all={agg.get('all_pairwise')} regret={agg.get('regret')}",
              file=sys.stderr, flush=True)

    for a, b in (("S3", "S0"), ("S3", "S2"), ("S3", "S1"), ("S2", "S0"), ("S1", "S0"),
                 ("S4", "S3"), ("S3", "policy")):
        if a in PG and b in PG:
            for met in ("stable", "support_weighted", "margin_weighted",
                        "large_margin", "all_pairwise", "regret"):
                report["bootstrap"][f"{a}-{b}_{met}"] = paired_bootstrap(PG[a], PG[b], met)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pairs", "coverage")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
