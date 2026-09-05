"""Phase19.9 §7-§29: TES の M 選択 / scorer 比較 / plan-length 解析。

主比較(§12/§21): IV(行動直後の Value)vs TES(ターン終端の Value)。
違いは「どこで止めるか」だけなので、intra-turn を読むこと自体の効果を分離できる。
評価 target は独立 seed family の H1(§16)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np


def load(p):
    rows = []
    for f in sorted(glob.glob(p)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


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


def spearman(a, b):
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


def evaluate(G, fn, ykey="h1_a"):
    per = []
    for g in G:
        y = [st.mean(c[ykey]) for c in g["candidates"] if c[ykey]]
        if len(y) != len(g["candidates"]) or max(y) == min(y):
            continue
        p = fn(g)
        if p is None or len(p) != len(y):
            continue
        ok, tot = pw(p, y)
        bi = max(range(len(p)), key=lambda i: p[i])
        per.append({"pw": (ok, tot), "regret": max(y) - y[bi],
                    "top1": 1.0 if y[bi] == max(y) else 0.0,
                    "sp": spearman(p, y), "gid": g["group_id"],
                    # 試合クラスタブートストラップ(--cluster-bootstrap)用。既存消費者は
                    # "pw"/"regret"/"top1"/"sp"/"gid" のみ参照するため追加キーでも非破壊。
                    "game": g.get("game")})
    if not per:
        return None
    num = sum(x["pw"][0] for x in per)
    den = sum(x["pw"][1] for x in per)
    sp = [x["sp"] for x in per if x["sp"] is not None]
    return {"n": len(per), "regret": round(st.mean([x["regret"] for x in per]), 4),
            "pairwise": round(num / den, 4) if den else None,
            "top1": round(st.mean([x["top1"] for x in per]), 4),
            "spearman": round(st.mean(sp), 4) if sp else None, "_per": per}


def boot(a, b, key, B=10000, seed=0, cluster_bootstrap=False):
    """block A/B(または scorer 間)の差のブートストラップ95%CI。

    既定(cluster_bootstrap=False)は従来どおり root(候補group)単位の復元抽出
    (出力キーも従来と完全に同一)。cluster_bootstrap=True だと試合(game)単位で
    クラスタ復元抽出する(同一試合内の root は独立でないため、root単位resampleは
    CIを過小評価しうる。§12/§21の TES vs IV/TD1 等の差を試合クラスタで検定する用)。
    a/b は同一 G から `evaluate()` が作るため要素が1:1対応しており、game割当は
    a 側の "game" キーで代表させてよい。
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rs = np.random.RandomState(seed)
    if key == "pairwise":
        an = np.array([x["pw"][0] for x in a]); ad = np.array([x["pw"][1] for x in a])
        bn = np.array([x["pw"][0] for x in b]); bd = np.array([x["pw"][1] for x in b])
        obs = an.sum() / ad.sum() - bn.sum() / bd.sum()
    else:
        x = np.array([p[key] for p in a]) - np.array([p[key] for p in b])
        obs = x.mean()

    if not cluster_bootstrap:
        idx = rs.randint(0, n, size=(B, n))
        if key == "pairwise":
            m = (an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-9)
                 - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-9))
        else:
            m = x[idx].mean(1)
        return {"mean_diff": round(float(obs), 4),
                "ci95": [round(float(np.percentile(m, 2.5)), 4),
                         round(float(np.percentile(m, 97.5)), 4)], "n_groups": n}

    by_game: dict = {}
    for i, row in enumerate(a):
        by_game.setdefault(row.get("game"), []).append(i)
    games = list(by_game)
    n_games = len(games)
    diffs = []
    if n_games:
        game_idx = rs.randint(0, n_games, size=(B, n_games))
        for b_i in range(B):
            picked = [pos for gi in game_idx[b_i] for pos in by_game[games[gi]]]
            if not picked:
                continue
            picked = np.asarray(picked)
            if key == "pairwise":
                ad_s = ad[picked].sum(); bd_s = bd[picked].sum()
                diffs.append(
                    (an[picked].sum() / ad_s if ad_s else 0.0)
                    - (bn[picked].sum() / bd_s if bd_s else 0.0))
            else:
                diffs.append(float(x[picked].mean()))
    m = np.asarray(diffs) if diffs else np.asarray([float(obs)])
    return {"mean_diff": round(float(obs), 4),
            "ci95": [round(float(np.percentile(m, 2.5)), 4),
                     round(float(np.percentile(m, 97.5)), 4)], "n_groups": n,
            "n_games": n_games, "method": "cluster_bootstrap_game"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_tes_w*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase199_tes.json")
    ap.add_argument("--cluster-bootstrap", action="store_true",
                    help="bootstrap セクションを root単位ではなく試合(game)単位のクラスタ復元抽出にする"
                         "(既定OFF=従来のroot単位resample。TES vs IV/TD1等の差を試合クラスタでCI検定)。")
    args = ap.parse_args()
    G = load(args.data)
    rep = {"groups": len(G), "games": len({g["game"] for g in G}),
           "candidates": sum(len(g["candidates"]) for g in G),
           "turn_band": dict(Counter(g["turn_band"] for g in G)),
           "arch": dict(Counter(g["arch"] for g in G)),
           "first_rate": round(sum(1 for g in G if g["me_first"]) / max(1, len(G)), 3)}

    # ---- §7/§8 M reliability audit(block A vs B の prefix)----
    mrep = {}
    for M in (1, 2, 4, 8):
        ok = tot = 0.0
        t1, mad = [], []
        lg_ok = lg_tot = 0.0
        for g in G:
            a = [st.mean(c["tes_A"][:M]) for c in g["candidates"]]
            b = [st.mean(c["tes_B"][:M]) for c in g["candidates"]]
            o, t = pw(a, b)
            ok += o
            tot += t
            t1.append(1.0 if a.index(max(a)) == b.index(max(b)) else 0.0)
            mad += [abs(x - y) for x, y in zip(a, b)]
            for i, j in itertools.combinations(range(len(a)), 2):
                if abs(a[i] - a[j]) < 0.10 or b[i] == b[j]:
                    continue
                lg_tot += 1
                lg_ok += 1.0 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else 0.0
        mrep[str(M)] = {"pairwise": round(ok / tot, 4) if tot else None,
                        "top1": round(st.mean(t1), 4),
                        "large_margin": round(lg_ok / lg_tot, 4) if lg_tot else None,
                        "mean_abs_diff": round(st.mean(mad), 4)}
    rep["m_audit"] = mrep
    sel = None
    for M in (1, 2, 4, 8):
        r = mrep[str(M)]
        if r["pairwise"] and r["pairwise"] >= 0.80 and r["large_margin"] \
                and r["large_margin"] >= 0.90:
            sel = M
            break
    rep["selected_M"] = sel
    rep["R0"] = "PASS" if sel else "FAIL"
    M = sel or 8

    # ---- H1 reference の信頼性 ----
    ok = tot = 0.0
    nf = []
    for g in G:
        a = [st.mean(c["h1_a"]) for c in g["candidates"]]
        b = [st.mean(c["h1_b"]) for c in g["candidates"] if c["h1_b"]]
        if len(b) != len(a):
            continue
        o, t = pw(a, b)
        ok += o
        tot += t
        bi = max(range(len(a)), key=lambda i: a[i])
        nf.append(max(b) - b[bi])
    rep["h1_reference"] = {"pairwise_self_reliability": round(ok / tot, 4) if tot else None,
                           "noise_floor_regret": round(st.mean(nf), 4) if nf else None,
                           "n": len(nf)}

    # ---- §18 scorer 比較 ----
    scorers = {
        "Q0": lambda g: [c["q0"] for c in g["candidates"]],
        "EntityQ": lambda g: [c["entityq"] for c in g["candidates"]],
        "TD1": lambda g: [c["td1"] for c in g["candidates"]],
        "IV": lambda g: [st.mean(c["iv"][:M]) if c["iv"] else 0.0 for c in g["candidates"]],
        "TES": lambda g: [st.mean(c["tes_A"][:M]) for c in g["candidates"]],
    }
    PER = {}
    rep["scorers"] = {}
    for name, fn in scorers.items():
        r = evaluate(G, fn)
        PER[name] = r.pop("_per")
        rep["scorers"][name] = r

    rep["bootstrap"] = {}
    for a, b in (("TES", "IV"), ("TES", "EntityQ"), ("TES", "Q0"), ("TES", "TD1"),
                 ("IV", "EntityQ")):
        for k in ("regret", "pairwise", "top1"):
            rep["bootstrap"][f"{a}-{b}_{k}"] = boot(
                PER[a], PER[b], k, cluster_bootstrap=args.cluster_bootstrap)

    # ---- §23-§25 plan length ----
    def planlen(g):
        return int(round(st.mean([st.mean(c["plan_len"]) for c in g["candidates"]])))
    bands = {"0": lambda n: n == 0, "1": lambda n: n == 1,
             "2": lambda n: n == 2, "3+": lambda n: n >= 3}
    pl = {}
    for name, f in bands.items():
        sub = [g for g in G if f(planlen(g))]
        if len(sub) < 8:
            continue
        rt = evaluate(sub, scorers["TES"])
        ri = evaluate(sub, scorers["IV"])
        re_ = evaluate(sub, scorers["EntityQ"])
        pl[name] = {"groups": len(sub), "TES_regret": rt["regret"],
                    "IV_regret": ri["regret"], "EntityQ_regret": re_["regret"],
                    "TES_minus_IV": round(rt["regret"] - ri["regret"], 4)}
    rep["plan_length"] = pl

    # ---- §26/§27 action type / turn ----
    def subset(keyfn):
        out = {}
        for k in sorted({keyfn(g) for g in G}):
            sub = [g for g in G if keyfn(g) == k]
            if len(sub) < 10:
                continue
            out[str(k)] = {"groups": len(sub),
                           "TES": evaluate(sub, scorers["TES"])["regret"],
                           "IV": evaluate(sub, scorers["IV"])["regret"],
                           "EntityQ": evaluate(sub, scorers["EntityQ"])["regret"]}
        return out
    ATY = {7: "play", 8: "attach", 9: "evolve", 10: "ability", 13: "attack"}
    rep["by_action_type"] = subset(
        lambda g: ATY.get(g["candidates"][0]["option_type"],
                          "other%d" % g["candidates"][0]["option_type"]))
    rep["by_turn"] = subset(lambda g: g["turn_band"])

    # ---- §36 disagreement ----
    def top(g, fn):
        p = fn(g)
        return max(range(len(p)), key=lambda i: p[i])
    rep["disagreement"] = {
        "TES_vs_Q0": round(st.mean([0.0 if top(g, scorers["TES"]) == top(g, scorers["Q0"])
                                    else 1.0 for g in G]), 4),
        "TES_vs_EntityQ": round(st.mean([0.0 if top(g, scorers["TES"])
                                         == top(g, scorers["EntityQ"]) else 1.0
                                         for g in G]), 4),
        "TES_vs_IV": round(st.mean([0.0 if top(g, scorers["TES"]) == top(g, scorers["IV"])
                                    else 1.0 for g in G]), 4)}

    # ---- §28 runtime ----
    ms = sorted(g["tes_ms"] for g in G)
    per_cand = [g["tes_ms"] / (2 * 8 * len(g["candidates"])) for g in G]  # A+B, M=8
    rep["runtime"] = {
        "measured_AB_M8_ms_mean": round(st.mean(ms), 1),
        "ms_per_candidate_per_rollout": round(st.mean(per_cand), 2),
        "estimated_decision_ms_at_selected_M": round(
            st.mean(per_cand) * M * st.mean([len(g["candidates"]) for g in G]), 1),
        "p50_AB": round(ms[len(ms) // 2], 1),
        "p90_AB": round(ms[int(0.9 * (len(ms) - 1))], 1),
        "production_budget_ms": 1446.7}
    rep["runtime"]["estimated_budget_share"] = round(
        100 * rep["runtime"]["estimated_decision_ms_at_selected_M"] / 1446.7, 1)

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("groups", "m_audit", "selected_M", "R0",
                                          "h1_reference", "scorers", "bootstrap",
                                          "plan_length", "disagreement", "runtime")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
