"""Phase19.8 §19-§41: trajectory divergence の解析。

主眼は「H1 の価値差がどの checkpoint から Frozen Value で読めるようになるか」(§20 ladder)。
向きは H1 winner - loser に統一する(§19)。C4 でも 100% にならないので、C4 を実測 ceiling
として扱う(§21)。structural divergence と value predictability は別指標(§27)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

CK = ("C1", "C1b", "C2", "C3", "C4")


def load(p):
    rows = []
    for f in sorted(glob.glob(p)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def vals_at(traj_list, ck):
    return [t["ckpt"][ck]["value"] for t in traj_list
            if t.get("ckpt") and t["ckpt"].get(ck)]


def ent_dist(a, b):
    """observable entity の identity 距離(§24)。"""
    def bag(e):
        c = Counter()
        s = e["self"]
        for x in s.get("hand", []):
            c[("h", x)] += 1
        for p in [s["active"]] + list(s["bench"]):
            if p:
                c[("b", p["id"])] += 1
                for t in p["tools"]:
                    c[("t", t)] += 1
        for x in s["discard"]:
            c[("d", x)] += 1
        for p in [e["opp"]["active"]] + list(e["opp"]["bench"]):
            if p:
                c[("ob", p["id"])] += 1
        return c
    ca, cb = bag(a), bag(b)
    return sum(abs(ca[k] - cb[k]) for k in set(ca) | set(cb))


def hand_dist(a, b):
    ca = Counter(a["self"].get("hand", []))
    cb = Counter(b["self"].get("hand", []))
    return sum(abs(ca[k] - cb[k]) for k in set(ca) | set(cb))


def s166_dist(a, b):
    x = np.asarray(a, np.float32)
    y = np.asarray(b, np.float32)
    return float(np.linalg.norm(x - y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kaggle_replays/_tj_w*.jsonl.gz")
    ap.add_argument("--out", default="kaggle_replays/value_net/phase198_traj.json")
    args = ap.parse_args()
    G = load(args.data)
    rep = {"root_groups": len(G),
           "games": len({g["game"] for g in G}),
           "arch": dict(Counter(g["arch"] for g in G)),
           "turn_band": dict(Counter(g["turn_band"] for g in G))}

    items = []
    for g in G:
        for pr in g["pairs"]:
            p = pr["pair"]
            # H1 winner を A/B のどちらかへ揃える
            win_is_A = (p["winner_idx"] == p["a"])
            W = pr["blockC"]["A" if win_is_A else "B"]
            Lo = pr["blockC"]["B" if win_is_A else "A"]
            items.append({"kind": pr["kind"], "margin": p["margin"],
                          "gid": g["group_id"], "turn_band": g["turn_band"],
                          "arch": g["arch"], "W": W, "L": Lo})
    rep["pairs"] = {"total": len(items),
                    **dict(Counter(i["kind"] for i in items)),
                    "mean_margin": round(st.mean([i["margin"] for i in items]), 4)}

    # ---- §20 Value predictability ladder ----
    ladder = {}
    for ck in CK:
        row = {}
        for scope in ("core_failure", "matched_correct", "all"):
            sub = [i for i in items if scope == "all" or i["kind"] == scope]
            ok = n = 0
            dv = []
            for i in sub:
                w, l = vals_at(i["W"], ck), vals_at(i["L"], ck)
                if not w or not l:
                    continue
                d = st.mean(w) - st.mean(l)
                dv.append(d)
                n += 1
                ok += 1 if d > 0 else (0.5 if d == 0 else 0)
            row[scope] = {"n": n, "agreement": round(ok / n, 4) if n else None,
                          "ci95": wilson(ok, n),
                          "mean_dV": round(st.mean(dv), 4) if dv else None,
                          "median_dV": round(st.median(dv), 4) if dv else None}
        ladder[ck] = row
    rep["value_ladder"] = ladder

    # §22 normalized emergence(C4 を ceiling として)
    c4 = ladder["C4"]["all"]["agreement"]
    if c4 and c4 > 0.55:
        rep["emergence_score"] = {
            ck: round((ladder[ck]["all"]["agreement"] - 0.5) / (c4 - 0.5), 4)
            for ck in CK}
    else:
        rep["emergence_score"] = "C4 agreement が 0.55 未満のため算出しない(§22)"

    # ---- §18/§25 same-action null ----
    nulls = [g["null"] for g in G if g.get("null")]
    nl = {}
    for ck in CK:
        d, ed = [], []
        for nu in nulls:
            x, y = vals_at(nu["X"], ck), vals_at(nu["Y"], ck)
            if x and y:
                d.append(abs(st.mean(x) - st.mean(y)))
            ex = [t["ckpt"][ck]["entity"] for t in nu["X"] if t.get("ckpt") and t["ckpt"].get(ck)]
            ey = [t["ckpt"][ck]["entity"] for t in nu["Y"] if t.get("ckpt") and t["ckpt"].get(ck)]
            if ex and ey:
                ed.append(ent_dist(ex[0], ey[0]))
        nl[ck] = {"n": len(d), "mean_abs_dV": round(st.mean(d), 4) if d else None,
                  "mean_entity_dist": round(st.mean(ed), 2) if ed else None}
    rep["same_action_null"] = {"cases": len(nulls), "by_checkpoint": nl}

    # ---- §24/§25 structural divergence(null 正規化)----
    sd = {}
    for ck in CK:
        ed, hd, s1 = [], [], []
        for i in items:
            ew = [t["ckpt"][ck] for t in i["W"] if t.get("ckpt") and t["ckpt"].get(ck)]
            el = [t["ckpt"][ck] for t in i["L"] if t.get("ckpt") and t["ckpt"].get(ck)]
            if not ew or not el:
                continue
            ed.append(ent_dist(ew[0]["entity"], el[0]["entity"]))
            hd.append(hand_dist(ew[0]["entity"], el[0]["entity"]))
            s1.append(s166_dist(ew[0]["state166"], el[0]["state166"]))
        base = nl[ck]["mean_entity_dist"]
        sd[ck] = {"entity_dist": round(st.mean(ed), 2) if ed else None,
                  "hand_dist": round(st.mean(hd), 2) if hd else None,
                  "state166_l2": round(st.mean(s1), 3) if s1 else None,
                  "null_normalized_entity": (round(st.mean(ed) - base, 2)
                                             if ed and base is not None else None)}
    rep["structural_divergence"] = sd

    # ---- §16/§36 continuation divergence ----
    def first_div(i):
        """W/L branch の action log が最初に食い違う位置を分類。"""
        aw = [t["actions"] for t in i["W"] if t.get("actions")]
        al = [t["actions"] for t in i["L"] if t.get("actions")]
        if not aw or not al:
            return "unknown"
        a, b = aw[0], al[0]
        for k in range(min(len(a), len(b))):
            if (a[k]["actor"], a[k]["type"], a[k]["card"]) != \
               (b[k]["actor"], b[k]["type"], b[k]["card"]):
                return "opponent" if a[k]["actor"] != i["W"][0]["ckpt"]["C1"]["actor"] \
                    else "own"
        return "same_prefix"
    dvc = {}
    for scope in ("core_failure", "matched_correct"):
        sub = [i for i in items if i["kind"] == scope]
        dvc[scope] = dict(Counter(first_div(i) for i in sub))
    rep["continuation_divergence"] = dvc

    # ---- §39/§40 earliest value-relevant divergence(8 trajectory 多数決)----
    def earliest(i):
        for ck in CK:
            w, l = vals_at(i["W"], ck), vals_at(i["L"], ck)
            n = min(len(w), len(l))
            if n < 4:
                continue
            votes = sum(1 for k in range(n) if w[k] > l[k])
            if votes >= math.ceil(0.75 * n):
                return {"C1": "D0_immediate", "C1b": "D0b_own_turn_end",
                        "C2": "D1_opponent", "C3": "D2_own_start",
                        "C4": "D3_endpoint"}[ck]
        return "DN_unstable"
    ed = {}
    for scope in ("core_failure", "matched_correct"):
        sub = [i for i in items if i["kind"] == scope]
        c = Counter(earliest(i) for i in sub)
        ed[scope] = {"n": len(sub), **{k: c[k] for k in
                                       ("D0_immediate", "D0b_own_turn_end", "D1_opponent",
                                        "D2_own_start", "D3_endpoint", "DN_unstable")}}
    rep["earliest_divergence"] = ed

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("root_groups", "pairs", "value_ladder", "emergence_score",
                       "same_action_null", "earliest_divergence")}, ensure_ascii=False,
                     indent=2))


if __name__ == "__main__":
    main()
