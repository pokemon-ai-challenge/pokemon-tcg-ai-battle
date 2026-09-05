"""Phase19.17 LOOP1: entity-link 単離テスト(Codex 反論 #8 / #3 を反映)。

Phase19.16 では bind ブロック丸ごと / traj ブロック丸ごとを shuffle していたので、
「entity 対応(link)が効いている」のか「非 relation 情報が増えただけ」なのかを分離できていなかった。

ここでは特徴を4ブロックに分ける:
  BASE    : binding も trajectory も使わない盤面/policy 情報
  ENT     : source/target **entity の属性**(HP・エネ・zone・owner …)。link ではない
  TRAJ_NL : trajectory の **非 link** 情報(本数・種類・長さ …)
  LINK    : ★仮説の本体★ root action の source/target と、その後の trajectory が
            **同じ entity に触るか**という対応フラグだけ

arms:
  FULL       = BASE+ENT+TRAJ_NL+LINK
  LINK_SHUF  = FULL だが LINK だけ root 内で候補間置換(値分布は保存、対応だけ破壊)
  LINK_ERASE = FULL だが LINK だけ 0 埋め
  LINK_NOISE = FULL だが LINK を **同スケールの無関係乱数**へ置換(Codex #3: 0埋めは
               追加次元の第一層重みに勾配が流れず容量が一致しないため)
primary = FULL − LINK_SHUF と FULL − LINK_NOISE。両方が正でなければ仮説は反証。

CI は **game クラスタ bootstrap**(同一 game の root は相関するため)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
OPT_TYPES = list(range(20))


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern))
            for l in gzip.open(f, "rt", encoding="utf-8")]


def _ent_index(row):
    return {e["serial"]: e for e in row["before_entities"] if e["serial"] is not None}


def f_base(row, c):
    g = row["before_global"]
    ot = [0.0] * len(OPT_TYPES)
    t = c["option_raw"]["type"]
    if isinstance(t, int) and 0 <= t < len(OPT_TYPES):
        ot[t] = 1.0
    d = c["delta_by_serial"]
    me = row["me"]
    steps = [x for x in c["trajectory"][1:] if x.get("actor") == me]
    return ot + [
        c["policy_score"], c["policy_prob"], math.log1p(row["n_legal_options"]),
        g["prize_self"], g["prize_opp"], g["prize_opp"] - g["prize_self"],
        g["hand_count_self"] / 10.0, g["hand_count_opp"] / 10.0,
        g["deck_self"] / 60.0, g["deck_opp"] / 60.0,
        len(g["discard_self_ids"]) / 60.0, len(g["discard_opp_ids"]) / 60.0,
        g["turn"] / 20.0, float(g["supporter_played"]), float(g["energy_attached"]),
        float(g["retreated"]), 1.0 if g["stadium_id"] else 0.0,
        len(row["before_hand"]) / 10.0,
        # 盤面全体の変化量(source/target を知らなくても取れる)
        sum(1 for v in d.values() if v["hp_delta"] < 0) / 6.0,
        sum(1 for v in d.values() if v["disappeared"]) / 6.0,
        sum(1 for v in d.values() if v["appeared"]) / 6.0,
        sum(v["hp_delta"] for v in d.values()) / 200.0,
        sum(v["energy_delta"] for v in d.values()) / 4.0,
        len(steps) / 10.0,
    ]


def f_ent(row, c):
    """source/target entity の属性。link ではない(どの entity かは使うが対応関係は問わない)。"""
    idx = _ent_index(row)
    out = []
    for key in ("source", "target"):
        b = c[key]
        e = idx.get(b["serial"]) if b["serial"] is not None else None
        out += [
            1.0 if b["serial"] is not None else 0.0,
            1.0 if b["card_id"] is not None else 0.0,
            (e["hp"] / 200.0) if e and e["hp"] is not None else 0.0,
            (e["hp"] / e["max_hp"]) if e and e.get("max_hp") else 0.0,
            (e["energies"] / 4.0) if e else 0.0,
            1.0 if (e and e["zone"] == "active") else 0.0,
            1.0 if (e and e["owner"] == 1) else 0.0,
            1.0 if (e and e["appear_this_turn"]) else 0.0,
            len(e["tool_card_ids"]) if e else 0.0,
            (b["area"] or 0) / 12.0,
        ]
        d = c["delta_by_serial"].get(str(b["serial"])) if b["serial"] is not None else None
        out += [(d["hp_delta"] / 200.0) if d else 0.0,
                (d["energy_delta"] / 4.0) if d else 0.0,
                1.0 if (d and d["zone_change"]) else 0.0,
                1.0 if (d and d["disappeared"]) else 0.0]
    return out


def f_traj_nl(row, c):
    """trajectory の非 link 情報。どの entity かは問わない。"""
    me = row["me"]
    steps = [x for x in c["trajectory"][1:] if x.get("actor") == me]
    tc = [0.0] * len(OPT_TYPES)
    for x in steps:
        t = x["option_raw"]["type"]
        if isinstance(t, int) and 0 <= t < len(OPT_TYPES):
            tc[t] += 1.0
    src = [x["source"]["serial"] for x in steps if x["source"]["serial"] is not None]
    tgt = [x["target"]["serial"] for x in steps if x["target"]["serial"] is not None]
    return tc + [
        len(set(src)) / 5.0, len(set(tgt)) / 5.0,
        sum(1 for x in steps if x["source"]["area"] == 2) / 5.0,
        1.0 if any(x.get("select_type") == 6 for x in steps) else 0.0,
        len([x for x in c["trajectory"] if x.get("actor") != me]) / 10.0,
        c["traj_len"] / 20.0,
    ]


def f_link(row, c):
    """★仮説の本体★ root action の entity と trajectory の entity の **対応** だけ。"""
    me = row["me"]
    steps = [x for x in c["trajectory"][1:] if x.get("actor") == me]
    src = [x["source"]["serial"] for x in steps if x["source"]["serial"] is not None]
    tgt = [x["target"]["serial"] for x in steps if x["target"]["serial"] is not None]
    rs, rt = c["source"]["serial"], c["target"]["serial"]
    consec = 0.0
    prev = None
    for x in steps:
        s = x["source"]["serial"]
        if s is not None and s == prev:
            consec += 1.0
        prev = s
    idx = _ent_index(row)
    se, te = idx.get(rs), idx.get(rt)
    return [
        1.0 if (rs is not None and rs in src) else 0.0,   # 後で自分がまた同じ札/個体を使う
        1.0 if (rt is not None and rt in tgt) else 0.0,   # 後で同じ相手を狙い続ける
        1.0 if (rs is not None and rs in tgt) else 0.0,   # 動かした個体が後で狙われる側になる
        1.0 if (rt is not None and rt in src) else 0.0,
        consec / 5.0,                                     # 連続して同じ entity に触る回数
        1.0 if (rs is not None and rs == rt) else 0.0,
        1.0 if (se and te and se["owner"] == te["owner"]) else 0.0,
        len(set(src) & set(tgt)) / 5.0,
    ]


BLOCKS = ("base", "ent", "traj_nl", "link")
FN = {"base": f_base, "ent": f_ent, "traj_nl": f_traj_nl, "link": f_link}


class MLP(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build(rows, target):
    out = []
    for r in rows:
        cs = r["candidates"]
        if len(cs) < 2:
            continue
        y, blocks = [], {b: [] for b in BLOCKS}
        ok = True
        for c in cs:
            v = c["teacher"].get(target)
            if not v:
                ok = False
                break
            y.append(st.mean(v))
            for b in BLOCKS:
                blocks[b].append(FN[b](r, c))
        if not ok:
            continue
        g = {"game": r["game"], "y": np.array(y, np.float32),
             "turn_band": r["turn_band"], "arch": r["arch"]}
        for b in BLOCKS:
            g[b] = np.array(blocks[b], np.float32)
        out.append(g)
    return out


def assemble(G, arm, rng):
    X = []
    for g in G:
        link = g["link"]
        if arm == "full":
            L = link
        elif arm == "link_shuf":
            L = link[rng.permutation(len(link))]
        elif arm == "link_erase":
            L = np.zeros_like(link)
        elif arm == "link_noise":
            L = rng.normal(link.mean(0), link.std(0) + 1e-6,
                           size=link.shape).astype(np.float32)
        elif arm == "no_link":
            L = None
        else:
            raise ValueError(arm)
        parts = [g["base"], g["ent"], g["traj_nl"]] + ([] if L is None else [L])
        X.append(np.concatenate(parts, 1))
    return X


def pw(pred, y):
    ok = tot = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                continue
            tot += 1
            d = pred[i] - pred[j]
            ok += 0.5 if d == 0 else (1.0 if d * (y[i] - y[j]) > 0 else 0.0)
    return ok, tot


def per_group(model, X, G):
    model.eval()
    out = []
    with torch.no_grad():
        for x, g in zip(X, G):
            p = model(torch.from_numpy(x)).numpy()
            o, t = pw(p, g["y"])
            bi = int(np.argmax(p))
            out.append({"game": g["game"], "ok": o, "tot": t,
                        "regret": float(g["y"].max() - g["y"][bi]),
                        "top1": 1.0 if g["y"][bi] == g["y"].max() else 0.0})
    return out


def summarize(rows):
    tot = sum(r["tot"] for r in rows)
    return {"pairwise": round(sum(r["ok"] for r in rows) / tot, 4) if tot else None,
            "regret": round(st.mean([r["regret"] for r in rows]), 4),
            "top1": round(st.mean([r["top1"] for r in rows]), 4), "n": len(rows)}


def train_arm(tr, va, te, arm, seed, epochs=60):
    rng = np.random.RandomState(1000 + seed)
    Xtr, Xva, Xte = (assemble(s, arm, rng) for s in (tr, va, te))
    torch.manual_seed(seed)
    model = MLP(Xtr[0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = list(range(len(tr)))
    best, best_v = None, -1.0
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        for k in order:
            x = torch.from_numpy(Xtr[k])
            y = torch.from_numpy(tr[k]["y"])
            p = model(x)
            loss = ((p - y) ** 2).mean()
            if len(y) > 1:
                di, dy = p[:, None] - p[None, :], y[:, None] - y[None, :]
                m = dy.abs() > 1e-6
                if m.any():
                    loss = loss + torch.nn.functional.softplus(
                        -di[m] * torch.sign(dy[m])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            v = summarize(per_group(model, Xva, va))["pairwise"] or 0.0
            if v > best_v:
                best_v, best = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return best_v, per_group(model, Xte, te)


def cluster_boot(a_rows, b_rows, B=10000, seed=0):
    """game クラスタ bootstrap で pairwise 差の CI。a と b は同じ group 順。"""
    by = defaultdict(list)
    for i, r in enumerate(a_rows):
        by[r["game"]].append(i)
    games = list(by)
    rs = np.random.RandomState(seed)
    diffs = []
    for _ in range(B):
        pick = rs.randint(0, len(games), len(games))
        ao = at = bo = bt = 0.0
        for gi in pick:
            for i in by[games[gi]]:
                ao += a_rows[i]["ok"]
                at += a_rows[i]["tot"]
                bo += b_rows[i]["ok"]
                bt += b_rows[i]["tot"]
        if at and bt:
            diffs.append(ao / at - bo / bt)
    obs = (sum(r["ok"] for r in a_rows) / sum(r["tot"] for r in a_rows)
           - sum(r["ok"] for r in b_rows) / sum(r["tot"] for r in b_rows))
    return {"delta": round(obs, 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "P_gt_0": round(float(np.mean(np.array(diffs) > 0)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="C1b")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1917_link.json")
    args = ap.parse_args()
    rows = load(args.data)
    G = build(rows, args.target)
    games = sorted({g["game"] for g in G})
    rs = np.random.RandomState(20260810)
    rs.shuffle(games)
    n = len(games)
    tr_g, va_g = set(games[: int(.6 * n)]), set(games[int(.6 * n): int(.8 * n)])
    te_g = set(games[int(.8 * n):])
    tr = [g for g in G if g["game"] in tr_g]
    va = [g for g in G if g["game"] in va_g]
    te = [g for g in G if g["game"] in te_g]
    assert not (tr_g & va_g | tr_g & te_g | va_g & te_g)

    ok = tot = 0.0
    for r in rows:
        cs = [c for c in r["candidates"] if len(c["teacher"].get(args.target) or []) >= 4]
        if len(cs) < 2:
            continue
        h = min(len(c["teacher"][args.target]) for c in cs) // 2
        o, t = pw([st.mean(c["teacher"][args.target][:h]) for c in cs],
                  [st.mean(c["teacher"][args.target][h:2 * h]) for c in cs])
        ok += o
        tot += t

    rep = {"target": args.target, "roots": len(G), "games": n,
           "split": {"train": len(tr), "val": len(va), "test": len(te)},
           "teacher_self_pairwise": round(ok / tot, 4) if tot else None,
           "turn_band": dict(Counter(g["turn_band"] for g in G)), "arms": {}}
    per = {}
    for arm in ("full", "link_shuf", "link_noise", "link_erase", "no_link"):
        vs, rows_seed = [], []
        for s in range(args.seeds):
            v, tr_rows = train_arm(tr, va, te, arm, s)
            vs.append(v)
            rows_seed.append(tr_rows)
        avg = [{"game": rows_seed[0][i]["game"],
                "ok": st.mean([rs_[i]["ok"] for rs_ in rows_seed]),
                "tot": rows_seed[0][i]["tot"],
                "regret": st.mean([rs_[i]["regret"] for rs_ in rows_seed]),
                "top1": st.mean([rs_[i]["top1"] for rs_ in rows_seed])}
               for i in range(len(te))]
        per[arm] = avg
        rep["arms"][arm] = {"val_pairwise": round(st.mean(vs), 4),
                            "val_sd": round(st.pstdev(vs), 4), **summarize(avg)}
        print(arm, json.dumps(rep["arms"][arm]), flush=True)
    for nm, b in (("full_minus_link_shuf", "link_shuf"),
                  ("full_minus_link_noise", "link_noise"),
                  ("full_minus_link_erase", "link_erase"),
                  ("full_minus_no_link", "no_link")):
        rep[nm] = cluster_boot(per["full"], per[b])
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in rep if k != "arms"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
