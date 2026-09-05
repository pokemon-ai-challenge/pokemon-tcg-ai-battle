"""Phase7 Track A: MAIN Action-Q(Q0/Q1)の学習と候補順位評価。

split は **decision group 単位 かつ 試合単位**(同一 game の group は同じ split へ)。
Loss は役割ごとに分けて記録する:

    L = 1.0*L_return + 1.0*L_rank + 0.2*L_policy_aux     (§7.5)

  L_return : 実選択行動の Q を最終勝敗へ回帰(BCE)
  L_rank   : 同一 group 内の教師順位への pairwise(Bradley-Terry)。teacher confidence で重み付け
  L_policy_aux : 実選択行動の模倣(**Q教師ではない**。分離して報告)

比較アンカー: V0-post / V1-post(行動後状態を Value で評価する従来方式)。
評価は CARD/MAIN を混ぜない(本スクリプトは生成時の select 種別をそのまま使う)。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from action_q import ActionQNet, CandidateSetQNet, CrossActionQNet  # noqa: E402

MAX_HAND, MAX_DISC, MAX_OPP = 20, 30, 30


def load_groups(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_of(g: dict) -> int:
    """**試合単位**で split(同一 game の全 group は同じ split)。"""
    h = int(hashlib.md5(f"game{g['game']}".encode()).hexdigest(), 16) % 100
    return 0 if h < 70 else 1 if h < 85 else 2


# ---------- 指標 ----------

def _rank(v):
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


def _kendall(a, b):
    n = len(a)
    if n < 2:
        return None
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    return (con - dis) / (con + dis) if (con + dis) else None


def _pairwise(pred, teach):
    tot = ok = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if teach[i] == teach[j]:
                continue
            tot += 1
            d = pred[i] - pred[j]
            ok += 0.5 if d == 0 else (1.0 if d * (teach[i] - teach[j]) > 0 else 0.0)
    return (ok / tot) if tot else None


def _ndcg(pred, teach):
    n = len(pred)
    order = sorted(range(n), key=lambda i: pred[i], reverse=True)
    ideal = sorted(teach, reverse=True)
    dcg = sum(teach[order[i]] / math.log2(i + 2) for i in range(n))
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(n))
    return (dcg / idcg) if idcg > 0 else None


def eval_ranker(groups, score_fn):
    """score_fn(group) -> 候補ごとのスコア列。教師順位に対する各種指標を返す。"""
    pw, sp, kd, t1, t2, nd, rg = [], [], [], [], [], [], []
    for g in groups:
        teach = [c["teacher_mean"] for c in g["candidates"]]
        if len(teach) < 2 or max(teach) == min(teach):
            continue
        pred = score_fn(g)
        if pred is None or len(pred) != len(teach):
            continue
        v = _pairwise(pred, teach)
        if v is not None:
            pw.append(v)
        v = _spearman(pred, teach)
        if v is not None:
            sp.append(v)
        v = _kendall(pred, teach)
        if v is not None:
            kd.append(v)
        bi = max(range(len(pred)), key=lambda i: pred[i])
        ti = max(range(len(teach)), key=lambda i: teach[i])
        t1.append(1.0 if bi == ti else 0.0)
        top2 = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:2]
        t2.append(1.0 if ti in top2 else 0.0)
        v = _ndcg(pred, teach)
        if v is not None:
            nd.append(v)
        rg.append(teach[ti] - teach[bi])
    m = lambda x: round(statistics.mean(x), 4) if x else None  # noqa: E731
    return {"n_groups": len(pw), "pairwise": m(pw), "spearman": m(sp), "kendall": m(kd),
            "top1": m(t1), "top2_recall": m(t2), "ndcg": m(nd), "regret": m(rg),
            "_per_group": {"pairwise": pw, "regret": rg, "spearman": sp}}


# ---------- バッチ化 ----------

def _pad_ids(lists, cap):
    L = max(1, min(cap, max((len(v) for v in lists), default=1)))
    a = np.zeros((len(lists), L), dtype=np.int64)
    for i, v in enumerate(lists):
        for j, c in enumerate(v[:L]):
            a[i, j] = c + 1
    return torch.from_numpy(a)


def make_batch(groups, smean, sstd):
    B = len(groups)
    C = max(len(g["candidates"]) for g in groups)
    D = len(groups[0]["candidates"][0]["option_feat"])
    state = np.zeros((B, len(smean)), dtype=np.float32)
    opt = np.zeros((B, C, D), dtype=np.float32)
    cid = np.zeros((B, C), dtype=np.int64)
    otp = np.zeros((B, C), dtype=np.int64)
    mask = np.zeros((B, C), dtype=bool)
    teach = np.zeros((B, C), dtype=np.float32)
    conf = np.zeros((B, C), dtype=np.float32)
    sel = np.full(B, -1, dtype=np.int64)
    outc = np.zeros(B, dtype=np.float32)
    for b, g in enumerate(groups):
        state[b] = (np.asarray(g["state_feat"], dtype=np.float32) - smean) / sstd
        for c, cd in enumerate(g["candidates"]):
            opt[b, c] = cd["option_feat"]
            cid[b, c] = max(0, cd["action_card_id"] + 1) if cd["action_card_id"] >= 0 else 0
            otp[b, c] = min(cd["option_type"], 63)
            teach[b, c] = cd["teacher_mean"]
            conf[b, c] = 1.0 / (1.0 + cd["teacher_std"])
            mask[b, c] = True
            if g.get("selected_option") == cd["option_index"]:
                sel[b] = c
        outc[b] = g["outcome"]
    zones = [_pad_ids([g["hand_ids"] for g in groups], MAX_HAND),
             _pad_ids([g["discard_ids"] for g in groups], MAX_DISC),
             _pad_ids([g["opp_visible_ids"] for g in groups], MAX_OPP)]
    return (torch.from_numpy(state), zones, torch.from_numpy(opt),
            torch.from_numpy(cid), torch.from_numpy(otp), torch.from_numpy(mask),
            torch.from_numpy(teach), torch.from_numpy(conf),
            torch.from_numpy(sel), torch.from_numpy(outc))


def build_model(kind, state_dim, D, use_cards=True, use_action_card=True):
    if kind == "pool":
        return ActionQNet(state_dim, D, use_cards=use_cards,
                          use_action_card=use_action_card)
    if kind == "cross":
        return CrossActionQNet(state_dim, D, use_action_card=use_action_card)
    if kind == "capacity":
        return CrossActionQNet(state_dim, D, use_action_card=use_action_card,
                               capacity_only=True)
    if kind == "mean_attn":
        return CrossActionQNet(state_dim, D, use_action_card=use_action_card,
                               mean_attention=True)
    if kind.startswith("cset:"):          # Phase10 候補間モデル
        return CandidateSetQNet(state_dim, D, mode=kind.split(":", 1)[1],
                                use_action_card=use_action_card)
    raise ValueError(kind)


def _fwd(model, st, zones, of, ci, ot, mk):
    """pool系は zones(3ゾーン)、cross系は手札 ids のみを取る。"""
    if isinstance(model, CrossActionQNet):
        return model(st, zones[0], of, ci, ot, mk)
    return model(st, zones, of, ci, ot, mk)


def train(groups_tr, groups_va, kind, use_cards, use_action_card, use_rank,
          smean, sstd, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = len(groups_tr[0]["candidates"][0]["option_feat"])
    model = build_model(kind, len(smean), D, use_cards=use_cards,
                        use_action_card=use_action_card)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(groups_tr))
        tot = {"return": 0.0, "rank": 0.0, "aux": 0.0}
        nb = 0
        for i in range(0, len(perm), args.batch):
            gs = [groups_tr[j] for j in perm[i:i + args.batch]]
            st, zones, of, ci, ot, mk, te, cf, sel, oc = make_batch(gs, smean, sstd)
            q, p = _fwd(model, st, zones, of, ci, ot, mk)
            # L_return: 実選択行動の Q を勝敗へ
            has = sel >= 0
            l_ret = torch.tensor(0.0)
            if has.any():
                qs = q[torch.arange(len(gs))[has], sel[has]]
                l_ret = bce(qs, oc[has]).mean()
            # L_rank: 同一 group 内 pairwise(Bradley-Terry), teacher confidence 重み
            l_rank = torch.tensor(0.0)
            if use_rank:
                qi = q.unsqueeze(2)
                qj = q.unsqueeze(1)
                ti = te.unsqueeze(2)
                tj = te.unsqueeze(1)
                valid = (mk.unsqueeze(2) & mk.unsqueeze(1)) & (ti > tj)
                if valid.any():
                    w = (cf.unsqueeze(2) * cf.unsqueeze(1))[valid]
                    diff = (qi - qj)[valid]
                    l_rank = (-torch.nn.functional.logsigmoid(diff) * w).sum() / w.sum()
            # L_policy_aux: 実選択の模倣(Q教師ではない)
            l_aux = torch.tensor(0.0)
            if has.any():
                l_aux = nn.functional.cross_entropy(p[has], sel[has])
            loss = 1.0 * l_ret + (1.0 * l_rank if use_rank else 0.0) + 0.2 * l_aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["return"] += float(l_ret)
            tot["rank"] += float(l_rank)
            tot["aux"] += float(l_aux)
            nb += 1
        m = eval_ranker(groups_va, lambda g: score_groups(model, [g], smean, sstd)[0])
        pw = m["pairwise"] or 0
        print(f"    ep{ep+1}/{args.epochs} ret={tot['return']/nb:.4f} "
              f"rank={tot['rank']/nb:.4f} aux={tot['aux']/nb:.4f} val_pairwise={pw}",
              file=sys.stderr, flush=True)
        if pw > best:
            best = pw
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_groups(model, groups, smean, sstd):
    model.eval()
    st, zones, of, ci, ot, mk, *_ = make_batch(groups, smean, sstd)
    q, _ = _fwd(model, st, zones, of, ci, ot, mk)
    return [q[b][mk[b]].tolist() for b in range(len(groups))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parent / "_actionq_dataset_main.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=str(_HERE / "action_q_report.json"))
    args = ap.parse_args()

    groups = load_groups(Path(args.data))
    for g in groups:
        g["_split"] = split_of(g)
    tr = [g for g in groups if g["_split"] == 0]
    va = [g for g in groups if g["_split"] == 1]
    te = [g for g in groups if g["_split"] == 2]
    print(f"groups: total={len(groups)} train={len(tr)} val={len(va)} test={len(te)}",
          file=sys.stderr)
    if len(tr) < 20 or len(te) < 10:
        print(json.dumps({"error": "insufficient groups", "train": len(tr),
                          "test": len(te)}, ensure_ascii=False))
        return

    X = np.asarray([g["state_feat"] for g in tr], dtype=np.float32)
    smean, sstd = X.mean(0), X.std(0)
    sstd[sstd == 0] = 1.0

    # 教師信頼度による高信頼 subset(モデル結果を見て後付け変更しない)
    def self_pair(g):
        a = [c["teacher_splitA"] for c in g["candidates"]]
        b = [c["teacher_splitB"] for c in g["candidates"]]
        return _pairwise(a, b)

    hi_te = [g for g in te if (self_pair(g) or 0) >= 0.75]

    report = {"data": args.data, "budget_regime": "research_accuracy",
              "groups": {"total": len(groups), "train": len(tr), "val": len(va),
                         "test": len(te), "high_conf_test": len(hi_te)},
              "teacher_self": {
                  "all_test_pairwise": round(statistics.mean(
                      [self_pair(g) for g in te if self_pair(g) is not None]), 4) if te else None,
                  "high_conf_test_pairwise": round(statistics.mean(
                      [self_pair(g) for g in hi_te if self_pair(g) is not None]), 4)
                  if hi_te else None},
              "arms": {}}

    # アンカー: Policy スコア順(= 現行の候補生成順)
    PERGROUP = {}
    report["arms"]["policy_score"] = {
        "all": eval_ranker(te, lambda g: [c["policy_score"] for c in g["candidates"]]),
        "high_conf": eval_ranker(hi_te, lambda g: [c["policy_score"] for c in g["candidates"]])}
    _pol = eval_ranker(te, lambda g: [c["policy_score"] for c in g["candidates"]])
    PERGROUP["policy_score"] = {k: _pol["_per_group"][k] for k in ("pairwise", "regret")}

    arms = [
        ("Q0_state_action", dict(kind="pool", use_cards=False, use_action_card=True, use_rank=True)),
        ("Q1_pool", dict(kind="pool", use_cards=True, use_action_card=True, use_rank=True)),
        ("QC_capacity", dict(kind="capacity", use_cards=True, use_action_card=True, use_rank=True)),
        ("QX_cross", dict(kind="cross", use_cards=True, use_action_card=True, use_rank=True)),
        ("QX_mean_attention", dict(kind="mean_attn", use_cards=True, use_action_card=True, use_rank=True)),
        ("QX_no_action_card", dict(kind="cross", use_cards=True, use_action_card=False, use_rank=True)),
        ("QX_no_ranking", dict(kind="cross", use_cards=True, use_action_card=True, use_rank=False)),
    ]
    for name, kw in arms:
        per = []
        t0 = time.time()
        for seed in [int(s) for s in args.seeds.split(",")]:
            model = train(tr, va, kw["kind"], kw["use_cards"], kw["use_action_card"],
                          kw["use_rank"], smean, sstd, args, seed)
            per.append({
                "all": eval_ranker(te, lambda g: score_groups(model, [g], smean, sstd)[0]),
                "high_conf": eval_ranker(hi_te, lambda g: score_groups(model, [g], smean, sstd)[0]),
            })
        agg = {}
        for scope in ("all", "high_conf"):
            for k in ("pairwise", "spearman", "kendall", "top1", "top2_recall", "ndcg", "regret"):
                vals = [p[scope][k] for p in per if p[scope][k] is not None]
                agg.setdefault(scope, {})[k] = round(statistics.mean(vals), 4) if vals else None
        agg["per_seed_pairwise"] = [p["all"]["pairwise"] for p in per]
        agg["per_seed_spearman"] = [p["all"]["spearman"] for p in per]
        agg["per_seed_regret"] = [p["all"]["regret"] for p in per]
        agg["train_sec"] = round(time.time() - t0, 1)
        # seed 平均の per-group 値(paired bootstrap 用)
        pg = {}
        for key in ("pairwise", "regret"):
            mats = [p["all"]["_per_group"][key] for p in per]
            n = min(len(m) for m in mats)
            pg[key] = [statistics.mean([m[i] for m in mats]) for i in range(n)]
        PERGROUP[name] = pg
        report["arms"][name] = agg
        print(f"  [{name}] test pairwise={agg['all']['pairwise']} "
              f"spearman={agg['all']['spearman']}", file=sys.stderr, flush=True)

    # ---- paired group bootstrap(同じ test group を使うので対応ありでよい) ----
    rs = np.random.RandomState(0)

    def boot(a_name, b_name, key="pairwise", B=4000):
        if a_name not in PERGROUP or b_name not in PERGROUP:
            return None
        a, b = PERGROUP[a_name][key], PERGROUP[b_name][key]
        n = min(len(a), len(b))
        d = np.asarray(a[:n]) - np.asarray(b[:n])
        if n == 0:
            return None
        idx = rs.randint(0, n, size=(B, n))
        means = d[idx].mean(axis=1)
        return {"mean_diff": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(means, 2.5)), 4),
                         round(float(np.percentile(means, 97.5)), 4)],
                "n_groups": n}

    report["bootstrap_paired"] = {
        "QX-Q0_pairwise": boot("QX_cross", "Q0_state_action"),
        "QX-Q1_pairwise": boot("QX_cross", "Q1_pool"),
        "QX-QC_pairwise": boot("QX_cross", "QC_capacity"),
        "QX-policy_pairwise": boot("QX_cross", "policy_score"),
        "QX-Q0_regret": boot("QX_cross", "Q0_state_action", key="regret"),
    }
    # 内部用の per-group は出力から落とす
    for a in report["arms"].values():
        for scope in ("all", "high_conf"):
            if isinstance(a.get(scope), dict):
                a[scope].pop("_per_group", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
