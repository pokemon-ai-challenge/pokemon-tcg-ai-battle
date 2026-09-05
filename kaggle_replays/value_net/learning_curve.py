"""Phase19.6 §7-§48: H1 target を固定し、**学習データ量だけ**を変えた learning curve。

nested subset(§8): 大きい pool の prefix を使うので Train632 ⊂ Train1000 ⊂ ... となる。
val / testA / testB は全 N で完全固定(§10-§12)。checkpoint 選択は validation の
H1 regret のみ(§21)。test は一切見ない。

出力は Curve A(H1 held-out regret)/ Curve B(terminal-ref regret)/
Curve C(H1 oracle gap)と、train/val gap による underfit/overfit 診断(§47/§48)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import train_lh as L  # noqa: E402

NS = (632, 1000, 1500, 2500)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def ghash(gs):
    return hashlib.sha256(
        json.dumps(sorted(g["group_id"] for g in gs)).encode()).hexdigest()[:16]


def nested_prefix(pool_games, order, n_target, by_game):
    """game 単位で n_target 以下の最大 prefix を取る(§8/§9)。"""
    out, seen = [], []
    for game in order:
        if len(out) + len(by_game[game]) > n_target and out:
            break
        out += by_game[game]
        seen.append(game)
    return out, seen


def dist(gs):
    n = max(1, len(gs))
    return {"early_mid_late": {k: round(v / n, 3) for k, v in
                               sorted(Counter(g["turn_band"] for g in gs).items())},
            "archetype": {k: round(v / n, 3) for k, v in
                          sorted(Counter(g["arch"] for g in gs).items())},
            "first_rate": round(sum(1 for g in gs if g["me_first"]) / n, 3),
            "cand_mean": round(st.mean([len(g["candidates"]) for g in gs]), 2),
            "h1_spread": round(st.mean([max(g["_yA"]) - min(g["_yA"]) for g in gs]), 4),
            "h1_mean": round(st.mean([x for g in gs for x in g["_yA"]]), 4)}


def terminal_eval(model_fn, R):
    reg, ok, tot, t1, sp = [], 0.0, 0.0, [], []
    for g in R:
        t = [c["lh_a"]["mean"] for c in g["candidates"]]
        if any(x is None for x in t):
            continue
        p = model_fn(g)
        if p is None or len(p) != len(t):
            continue
        o, tt = L._pairwise(p, t)
        ok += o
        tot += tt
        bi = max(range(len(p)), key=lambda i: p[i])
        reg.append(max(t) - t[bi])
        t1.append(1.0 if t[bi] == max(t) else 0.0)
        v = L._spearman(p, t)
        if v is not None:
            sp.append(v)
    return {"terminal_regret": round(st.mean(reg), 4) if reg else None,
            "terminal_pairwise": round(ok / tot, 4) if tot else None,
            "terminal_top1": round(st.mean(t1), 4) if t1 else None,
            "terminal_spearman": round(st.mean(sp), 4) if sp else None,
            "n": len(reg)}


def subset_regret(groups, fn, keyfn):
    out = {}
    for k in sorted({keyfn(g) for g in groups}):
        sub = [g for g in groups if keyfn(g) == k]
        if len(sub) < 8:
            continue
        r = []
        for g in sub:
            p = fn(g)
            y = g["_yA"]
            bi = max(range(len(p)), key=lambda i: p[i])
            r.append(max(y) - y[bi])
        out[str(k)] = {"groups": len(sub), "h1_regret": round(st.mean(r), 4)}
    return out


def margin_pairwise(groups, fn):
    bands = {"<0.05": [0.0, 0.0], "0.05-0.10": [0.0, 0.0],
             "0.10-0.20": [0.0, 0.0], ">=0.20": [0.0, 0.0]}
    for g in groups:
        p, y = fn(g), g["_yA"]
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                m = abs(y[i] - y[j])
                if m == 0:
                    continue
                k = ("<0.05" if m < 0.05 else "0.05-0.10" if m < 0.10
                     else "0.10-0.20" if m < 0.20 else ">=0.20")
                d = p[i] - p[j]
                bands[k][0] += 0.5 if d == 0 else (1.0 if d * (y[i] - y[j]) > 0 else 0.0)
                bands[k][1] += 1
    return {k: {"pairs": int(v[1]), "pairwise": round(v[0] / v[1], 4)}
            for k, v in bands.items() if v[1] > 0}


def calib(groups, fn):
    e, se = [], []
    for g in groups:
        p = torch.sigmoid(torch.tensor(fn(g))).tolist()
        for x, y in zip(p, g["_yA"]):
            e.append(abs(x - y))
            se.append((x - y) ** 2)
    return {"mae": round(st.mean(e), 4), "rmse": round(st.mean(se) ** 0.5, 4)}


def scaling_fit(ns, vals):
    """§36 a/sqrt(N)+c と a*N^-b+c の簡易 fit(診断用。外挿しない)。"""
    x = np.asarray(ns, float)
    y = np.asarray(vals, float)
    A = np.vstack([1 / np.sqrt(x), np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = 1 - ((y - pred) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
    out = {"sqrt_fit": {"a": round(float(coef[0]), 4), "floor_c": round(float(coef[1]), 4),
                        "r2": round(float(ss), 4)}}
    try:
        best = None
        for b in np.linspace(0.05, 1.5, 60):
            A2 = np.vstack([x ** (-b), np.ones_like(x)]).T
            c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
            p2 = A2 @ c2
            r2 = 1 - ((y - p2) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
            if best is None or r2 > best[0]:
                best = (r2, b, c2)
        out["power_fit"] = {"b": round(float(best[1]), 3),
                            "a": round(float(best[2][0]), 4),
                            "floor_c": round(float(best[2][1]), 4),
                            "r2": round(float(best[0]), 4)}
    except Exception:                                   # noqa: BLE001
        pass
    out["relative_reduction_first_to_last"] = round(
        (vals[0] - vals[-1]) / vals[0], 4) if vals[0] else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(_HERE.parent / "_mh_t*.jsonl.gz"),
                    help="Phase19.5 データ。ここから val / testA を確定させる(§10)")
    ap.add_argument("--extra", default=str(_HERE.parent / "_mh_p*.jsonl.gz"),
                    help="追加分。すべて training pool へ入れる")
    ap.add_argument("--highm", default=str(_HERE.parent / "_mh_r*.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="TD1,TD0")
    ap.add_argument("--ns", default=",".join(str(n) for n in NS))
    ap.add_argument("--out", default=str(_HERE / "phase196_curve.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    L.HSTAR[0] = "H1"

    # §10: val / testA は **Phase19.5 のデータだけ**から確定させ、以降固定。
    base = load(args.base)
    L.prepare(base)
    base = [g for g in base if g["_yA"] is not None]
    L.split_games(base)
    val = [g for g in base if g["_split"] == 1]
    testA = [g for g in base if g["_split"] == 2]
    pool = [g for g in base if g["_split"] == 0]
    val_games = {g["game"] for g in val}
    ta_games = {g["game"] for g in testA}
    # 追加分はすべて training pool(val/testA の game と重ならないことを検査)
    extra = load(args.extra)
    if extra:
        L.prepare(extra)
        extra = [g for g in extra if g["_yA"] is not None]
        eg = {g["game"] for g in extra}
        assert not (eg & val_games) and not (eg & ta_games), "extra が val/testA と重複"
        pool = pool + extra
    G = base + extra

    by_game = defaultdict(list)
    for g in pool:
        by_game[g["game"]].append(g)
    rs = np.random.RandomState(0)
    order = sorted(by_game)
    rs.shuffle(order)

    ns = [int(x) for x in args.ns.split(",")]
    R = load(args.highm)
    if R:
        L.prepare(R)

    rep = {"hstar": "H1", "pool_groups": len(G), "pool_train_available": len(pool),
           "val": {"groups": len(val), "hash": ghash(val)},
           "testA": {"groups": len(testA), "hash": ghash(testA)},
           "highm": {"groups": len(R), "m_ref": R[0]["m"] if R else None},
           "config": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                      "seeds": args.seeds, "lambda_rank": L.LAMBDA_RANK},
           "points": {}, "curves": {}}

    # 凍結参照
    if R:
        rep["highm"]["h1_oracle_terminal"] = terminal_eval(
            lambda g: [c["lh_a"]["hz"]["H1"]["mean"] for c in g["candidates"]], R)
        rep["highm"]["entityq_frozen"] = terminal_eval(
            lambda g: [c["entityq_score"] for c in g["candidates"]], R)
        rep["highm"]["q0_frozen"] = terminal_eval(
            lambda g: [c["q0_score"] for c in g["candidates"]], R)

    curves = defaultdict(lambda: defaultdict(list))
    for n_target in ns:
        tr, games = nested_prefix(pool, order, n_target, by_game)
        assert not ({g["game"] for g in tr} & val_games), "train/val leak"
        assert not ({g["game"] for g in tr} & ta_games), "train/testA leak"
        X = np.asarray([g["state_feat"] for g in tr], np.float32)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        pt = {"target_n": n_target, "actual_groups": len(tr), "games": len(games),
              "candidates": sum(len(g["candidates"]) for g in tr),
              "train_hash": ghash(tr), "distribution": dist(tr),
              "updates_per_epoch": (len(tr) + args.batch - 1) // args.batch,
              "arms": {}}
        for arm in args.arms.split(","):
            per, models = [], []
            t0 = time.time()
            for seed in [int(s) for s in args.seeds.split(",")]:
                m = L.train_arm(tr, val, arm, mu, sd, args, seed)
                models.append(m)
                fn = (lambda g, m=m, a=arm: L.score_group(m, g, mu, sd, a))
                ea = L.evaluate(testA, fn)
                eb = L.evaluate([g for g in testA if g["_yB"] is not None], fn,
                                target="B")
                etr = L.evaluate(tr, fn)
                eva = L.evaluate(val, fn)
                per.append({"testA": {k: ea[k] for k in ea if k != "_per"},
                            "testA_blockB": ({k: eb[k] for k in eb if k != "_per"}
                                             if eb else None),
                            "train": {"long_regret": etr["long_regret"]},
                            "val": {"long_regret": eva["long_regret"]}})
            agg = {}
            for k in ("long_regret", "long_top1", "long_pairwise", "long_spearman"):
                v = [p["testA"][k] for p in per if p["testA"][k] is not None]
                agg[k] = round(st.mean(v), 4) if v else None
                agg["sd_" + k] = round(st.stdev(v), 4) if len(v) > 1 else None
            agg["blockB_regret"] = round(st.mean(
                [p["testA_blockB"]["long_regret"] for p in per
                 if p["testA_blockB"]]), 4) if per[0]["testA_blockB"] else None
            agg["train_regret"] = round(st.mean([p["train"]["long_regret"] for p in per]), 4)
            agg["val_regret"] = round(st.mean([p["val"]["long_regret"] for p in per]), 4)
            agg["train_val_gap"] = round(agg["val_regret"] - agg["train_regret"], 4)
            agg["train_sec"] = round(time.time() - t0, 1)
            agg["n_params"] = sum(p.numel() for p in models[0].parameters())
            fn0 = (lambda g, m=models[0], a=arm: L.score_group(m, g, mu, sd, a))
            agg["by_action_type"] = subset_regret(testA, fn0,
                                                  lambda g: g["candidates"][0]["option_type"])
            agg["by_turn"] = subset_regret(testA, fn0, lambda g: g["turn_band"])
            agg["margin_pairwise"] = margin_pairwise(testA, fn0)
            agg["calibration"] = calib(testA, fn0)
            if R:
                agg["terminal"] = terminal_eval(fn0, R)
                orc = rep["highm"]["h1_oracle_terminal"]["terminal_regret"]
                if orc is not None and agg["terminal"]["terminal_regret"] is not None:
                    agg["oracle_gap"] = round(
                        agg["terminal"]["terminal_regret"] - orc, 4)
            pt["arms"][arm] = agg
            curves[arm]["n"].append(len(tr))
            curves[arm]["h1_regret"].append(agg["long_regret"])
            curves[arm]["terminal_regret"].append(
                (agg.get("terminal") or {}).get("terminal_regret"))
            curves[arm]["oracle_gap"].append(agg.get("oracle_gap"))
            print(f"  [N={len(tr)} {arm}] H1 regret={agg['long_regret']} "
                  f"top1={agg['long_top1']} train_gap={agg['train_val_gap']} "
                  f"terminal={(agg.get('terminal') or {}).get('terminal_regret')}",
                  file=sys.stderr, flush=True)
        rep["points"][str(n_target)] = pt

    for arm, c in curves.items():
        rep["curves"][arm] = dict(c)
        vals = [v for v in c["h1_regret"] if v is not None]
        if len(vals) >= 3:
            rep["curves"][arm]["scaling_fit_h1"] = scaling_fit(c["n"][:len(vals)], vals)
        tv = [v for v in c["terminal_regret"] if v is not None]
        if len(tv) >= 3:
            rep["curves"][arm]["scaling_fit_terminal"] = scaling_fit(c["n"][:len(tv)], tv)

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pool": rep["pool_groups"], "val": rep["val"],
                      "testA": rep["testA"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
