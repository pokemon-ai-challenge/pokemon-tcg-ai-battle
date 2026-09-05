"""Phase19.16-T0.6: binding 仮説の最小反証実験(Transformer を作る前に回す)。

問い: 「action の source/target を **明示的に entity へ binding した特徴**」は、
候補の teacher(C1b / C4 Frozen Value)の順位付けに実際に効くのか。

効かないなら Transformer を作っても意味がないので branch を止める(§21/§47/§50)。

対照(§20):
  BASE          : binding なし(policy score / option type / global 盤面のみ)
  BASE+BIND     : + source/target entity の属性と delta
  BASE+BIND_SHUF: binding 特徴を **同一 root 内の候補間で置換**(値の分布は同じ、対応だけ壊す)
  BASE+BIND_ERAS: binding 特徴を 0 埋め(BASE と同義。実装健全性の確認用)
  BIND_ONLY     : binding 特徴だけ

split は **game 単位**(§51)。checkpoint 選択は validation のみ、test は最後に1回。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import statistics as st
from collections import Counter
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


def traj_feats(row, c) -> list[float]:
    """§57 のもう半分: 「この Action の後、同じターン内でどんな短い trajectory を作るか」。

    root の me だけを見る(相手手番の step は除く)。root action の source/target と
    **後続 action が同じ entity に触るか**という binding x trajectory の交互作用も入れる。
    """
    me = row["me"]
    steps = [t for t in c["trajectory"][1:] if t.get("actor") == me]
    n = len(steps)
    types = [t["option_raw"]["type"] for t in steps]
    src_serials = [t["source"]["serial"] for t in steps if t["source"]["serial"] is not None]
    tgt_serials = [t["target"]["serial"] for t in steps if t["target"]["serial"] is not None]
    root_src, root_tgt = c["source"]["serial"], c["target"]["serial"]
    tcount = [0.0] * len(OPT_TYPES)
    for t in types:
        if isinstance(t, int) and 0 <= t < len(OPT_TYPES):
            tcount[t] += 1.0
    return tcount + [
        n / 10.0,
        len(set(src_serials)) / 5.0,
        len(set(tgt_serials)) / 5.0,
        1.0 if (root_src is not None and root_src in src_serials) else 0.0,
        1.0 if (root_tgt is not None and root_tgt in tgt_serials) else 0.0,
        1.0 if (root_src is not None and root_src in tgt_serials) else 0.0,
        sum(1 for t in steps if t["source"]["area"] == 2) / 5.0,
        1.0 if any(t.get("select_type") == 6 for t in steps) else 0.0,
        len([t for t in c["trajectory"] if t.get("actor") != me]) / 10.0,
    ]


def unbound_change_feats(row, c) -> list[float]:
    """**source/target を知らなくても取れる**「何かが変わった」量。BASE 側に置く。

    これを BIND 側に入れると、binding の効果ではなく単なる after-state 情報の効果を
    binding の手柄として数えてしまい、shuffle 対照でも同時に壊れるので二重に誤る。
    """
    d = c["delta_by_serial"]
    return [
        sum(1 for v in d.values() if v["hp_delta"] < 0) / 6.0,
        sum(1 for v in d.values() if v["disappeared"]) / 6.0,
        sum(1 for v in d.values() if v["appeared"]) / 6.0,
        sum(v["hp_delta"] for v in d.values()) / 200.0,
        sum(v["energy_delta"] for v in d.values()) / 4.0,
        c["traj_len"] / 20.0,
    ]


def base_feats(row, c) -> list[float]:
    g = row["before_global"]
    ot = [0.0] * len(OPT_TYPES)
    t = c["option_raw"]["type"]
    if isinstance(t, int) and 0 <= t < len(OPT_TYPES):
        ot[t] = 1.0
    return ot + [
        c["policy_score"], c["policy_prob"], math.log1p(row["n_legal_options"]),
        g["prize_self"], g["prize_opp"], g["prize_opp"] - g["prize_self"],
        g["hand_count_self"] / 10.0, g["hand_count_opp"] / 10.0,
        g["deck_self"] / 60.0, g["deck_opp"] / 60.0,
        len(g["discard_self_ids"]) / 60.0, len(g["discard_opp_ids"]) / 60.0,
        g["turn"] / 20.0, float(g["supporter_played"]), float(g["energy_attached"]),
        float(g["retreated"]), 1.0 if g["stadium_id"] else 0.0,
        len(row["before_hand"]) / 10.0,
    ] + unbound_change_feats(row, c)


def bind_feats(row, c) -> list[float]:
    """source/target を実際の entity へ binding した特徴。ここが仮説の本体。"""
    idx = _ent_index(row)
    out = []
    for key in ("source", "target"):
        b = c[key]
        e = idx.get(b["serial"]) if b["serial"] is not None else None
        out += [
            1.0 if b["serial"] is not None else 0.0,
            1.0 if b["card_id"] is not None else 0.0,
            1.0 if e is not None else 0.0,
            (e["hp"] / 200.0) if e and e["hp"] is not None else 0.0,
            (e["hp"] / e["max_hp"]) if e and e.get("max_hp") else 0.0,
            (e["energies"] / 4.0) if e else 0.0,
            1.0 if (e and e["zone"] == "active") else 0.0,
            1.0 if (e and e["owner"] == 1) else 0.0,
            1.0 if (e and e["appear_this_turn"]) else 0.0,
            len(e["tool_card_ids"]) if e else 0.0,
            (b["area"] or 0) / 12.0,
        ]
    src, tgt = c["source"], c["target"]
    se, te = idx.get(src["serial"]), idx.get(tgt["serial"])
    d = c["delta_by_serial"]
    ds = d.get(str(src["serial"])) if src["serial"] is not None else None
    dt = d.get(str(tgt["serial"])) if tgt["serial"] is not None else None
    out += [
        1.0 if (src["serial"] is not None and src["serial"] == tgt["serial"]) else 0.0,
        1.0 if (se and te and se["owner"] == te["owner"]) else 0.0,
        1.0 if (te and te["owner"] == 1) else 0.0,
        (ds["hp_delta"] / 200.0) if ds else 0.0,
        (ds["energy_delta"] / 4.0) if ds else 0.0,
        1.0 if (ds and ds["zone_change"]) else 0.0,
        1.0 if (ds and ds["disappeared"]) else 0.0,
        (dt["hp_delta"] / 200.0) if dt else 0.0,
        (dt["energy_delta"] / 4.0) if dt else 0.0,
        1.0 if (dt and dt["zone_change"]) else 0.0,
        1.0 if (dt and dt["disappeared"]) else 0.0,
    ]
    return out


N_BASE = len(OPT_TYPES) + 18 + 6
N_BIND = 22 + 11


class MLP(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build(rows, target):
    """root ごとに (特徴行列, teacher) を作る。"""
    out = []
    for r in rows:
        cs = r["candidates"]
        if len(cs) < 2:
            continue
        y = []
        B, Bd, Tj = [], [], []
        for c in cs:
            v = c["teacher"][target]
            if not v:
                break
            y.append(st.mean(v))
            B.append(base_feats(r, c))
            Bd.append(bind_feats(r, c))
            Tj.append(traj_feats(r, c))
        if len(y) != len(cs):
            continue
        out.append({"game": r["game"], "y": np.array(y, np.float32),
                    "base": np.array(B, np.float32), "bind": np.array(Bd, np.float32),
                    "traj": np.array(Tj, np.float32),
                    "turn_band": r["turn_band"], "arch": r["arch"]})
    return out


def assemble(groups, arm, rng):
    X = []
    for g in groups:
        b = g["bind"]
        if arm == "base":
            x = g["base"]
        elif arm == "bind":
            x = np.concatenate([g["base"], b], 1)
        elif arm == "bind_shuf":
            p = rng.permutation(len(b))
            x = np.concatenate([g["base"], b[p]], 1)
        elif arm == "bind_erase":
            x = np.concatenate([g["base"], np.zeros_like(b)], 1)
        elif arm == "bind_only":
            x = b
        elif arm == "traj":
            x = np.concatenate([g["base"], g["traj"]], 1)
        elif arm == "traj_erase":
            x = np.concatenate([g["base"], np.zeros_like(g["traj"])], 1)
        elif arm == "traj_shuf":
            p = rng.permutation(len(g["traj"]))
            x = np.concatenate([g["base"], g["traj"][p]], 1)
        elif arm == "bind_traj":
            x = np.concatenate([g["base"], b, g["traj"]], 1)
        elif arm == "bind_traj_erase":
            x = np.concatenate([g["base"], np.zeros_like(b), np.zeros_like(g["traj"])], 1)
        else:
            raise ValueError(arm)
        X.append(x)
    return X


def pairwise(pred, y):
    ok = tot = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                continue
            tot += 1
            d = pred[i] - pred[j]
            ok += 0.5 if d == 0 else (1.0 if d * (y[i] - y[j]) > 0 else 0.0)
    return ok, tot


def evaluate(model, X, G):
    model.eval()
    ok = tot = 0.0
    reg, t1 = [], []
    with torch.no_grad():
        for x, g in zip(X, G):
            p = model(torch.from_numpy(x)).numpy()
            o, t = pairwise(p, g["y"])
            ok += o
            tot += t
            bi = int(np.argmax(p))
            reg.append(float(g["y"].max() - g["y"][bi]))
            t1.append(1.0 if g["y"][bi] == g["y"].max() else 0.0)
    return {"pairwise": round(ok / tot, 4) if tot else None, "pairs": int(tot),
            "regret": round(st.mean(reg), 4) if reg else None,
            "top1": round(st.mean(t1), 4) if t1 else None, "n": len(G)}


def train_arm(tr, va, te, arm, seed, epochs=60, lam_rank=1.0):
    rng = np.random.RandomState(seed)
    Xtr, Xva, Xte = (assemble(s, arm, rng) for s in (tr, va, te))
    d = Xtr[0].shape[1]
    torch.manual_seed(seed)
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = list(range(len(tr)))
    best = (None, -1)
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        for k in order:
            x = torch.from_numpy(Xtr[k])
            y = torch.from_numpy(tr[k]["y"])
            p = model(x)
            loss = ((p - y) ** 2).mean()
            n = len(y)
            if n > 1:
                di = p[:, None] - p[None, :]
                dy = y[:, None] - y[None, :]
                m = (dy.abs() > 1e-6)
                if m.any():
                    loss = loss + lam_rank * torch.nn.functional.softplus(
                        -di[m] * torch.sign(dy[m])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            v = evaluate(model, Xva, va)["pairwise"] or 0.0
            if v > best[1]:
                best = ({k: t.clone() for k, t in model.state_dict().items()}, v)
    if best[0] is not None:
        model.load_state_dict(best[0])
    return {"arm": arm, "val_pairwise": round(best[1], 4),
            "test": evaluate(model, Xte, te)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_p1916b_b*.jsonl.gz")
    ap.add_argument("--target", default="C1b")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="base,bind,bind_shuf,bind_erase,bind_only")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase1916_bind_probe.json")
    args = ap.parse_args()
    rows = load(args.data)
    G = build(rows, args.target)
    games = sorted({g["game"] for g in G})
    rs = np.random.RandomState(0)
    rs.shuffle(games)
    n = len(games)
    tr_g = set(games[: int(0.6 * n)])
    va_g = set(games[int(0.6 * n): int(0.8 * n)])
    te_g = set(games[int(0.8 * n):])
    tr = [g for g in G if g["game"] in tr_g]
    va = [g for g in G if g["game"] in va_g]
    te = [g for g in G if g["game"] in te_g]
    assert not (tr_g & va_g) and not (tr_g & te_g) and not (va_g & te_g)

    # teacher 自身の信頼性(M/2 vs M/2)。probe の上限の目安。
    ok = tot = 0.0
    for r in rows:
        cs = [c for c in r["candidates"] if len(c["teacher"][args.target]) >= 4]
        if len(cs) < 2:
            continue
        h = min(len(c["teacher"][args.target]) for c in cs) // 2
        A = [st.mean(c["teacher"][args.target][:h]) for c in cs]
        B = [st.mean(c["teacher"][args.target][h:2 * h]) for c in cs]
        o, t = pairwise(A, B)
        ok += o
        tot += t

    rep = {"target": args.target, "roots": len(G), "games": n,
           "split": {"train": len(tr), "val": len(va), "test": len(te)},
           "teacher_self_pairwise": round(ok / tot, 4) if tot else None,
           "turn_band": dict(Counter(g["turn_band"] for g in G)),
           "arms": {}}
    if min(len(tr), len(va), len(te)) < 5:
        raise SystemExit("root 数が足りない: train/val/test = {}/{}/{}".format(
            len(tr), len(va), len(te)))
    for arm in (args.arms.split(",")):
        runs = [train_arm(tr, va, te, arm, s) for s in range(args.seeds)]
        rep["arms"][arm] = {
            "seeds": args.seeds,
            "val_pairwise": round(st.mean([r["val_pairwise"] for r in runs]), 4),
            "test_pairwise": round(st.mean([r["test"]["pairwise"] for r in runs]), 4),
            "test_pairwise_sd": round(st.pstdev([r["test"]["pairwise"] for r in runs]), 4),
            "test_regret": round(st.mean([r["test"]["regret"] for r in runs]), 4),
            "test_top1": round(st.mean([r["test"]["top1"] for r in runs]), 4),
            "test_pairs": runs[0]["test"]["pairs"]}
        print(arm, json.dumps(rep["arms"][arm]), flush=True)
    A = rep["arms"]
    for nm, x, y in (("traj_minus_erased", "traj", "traj_erase"),
                     ("traj_minus_shuffled", "traj", "traj_shuf"),
                     ("bind_traj_minus_erased", "bind_traj", "bind_traj_erase")):
        if x in A and y in A:
            rep[nm] = round(A[x]["test_pairwise"] - A[y]["test_pairwise"], 4)
    if "bind" in A:
        b = A["bind"]["test_pairwise"]
        for name, key in (("bind_minus_base", "base"),
                          ("bind_minus_shuffled", "bind_shuf"),
                          ("bind_minus_erased", "bind_erase")):
            if key in A:
                rep[name] = round(b - A[key]["test_pairwise"], 4)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
