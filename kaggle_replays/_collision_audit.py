"""Phase14 §6-§9: State Collision Audit と Candidate Recall Audit。

Collision: state166 空間で近いのに raw entity が大きく違うペアを探し、
          「同じ特徴に潰れているが実際は別局面」がどれだけあるかを測る。
Recall   : policy top-k に教師の最良手が入っているか(ranking ではなく recall の問題か)。

leak 防止: 近傍探索は**別試合どうし**に限る(同一試合の連続局面は自明に似ているため除外)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _l1l2_core as C  # noqa: E402


def load(pattern, limit=0):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            rows.append(json.loads(l))
            if limit and len(rows) >= limit:
                return rows
    return rows


# ---------------- raw entity 距離 ----------------

def _bag(e):
    c = Counter()
    sd = e["self"]
    for x in sd.get("hand", []):
        c[("h", x)] += 1
    for p in [sd["active"]] + list(sd["bench"]):
        if p:
            c[("b", p["id"])] += 1
            for t in p["tools"]:
                c[("t", t)] += 1
    for p in [e["opp"]["active"]] + list(e["opp"]["bench"]):
        if p:
            c[("ob", p["id"])] += 1
    return c


def entity_distance(a, b):
    """カード identity 集合の対称差(= state166 が捨てている情報の量)。"""
    ca, cb = _bag(a), _bag(b)
    keys = set(ca) | set(cb)
    return sum(abs(ca[k] - cb[k]) for k in keys)


def board_identity_diff(a, b):
    def ids(e, side):
        s = e[side]
        return tuple(sorted(p["id"] for p in [s["active"]] + list(s["bench"]) if p))
    return (0 if ids(a, "self") == ids(b, "self") else 1) + \
           (0 if ids(a, "opp") == ids(b, "opp") else 1)


def hand_identity_diff(a, b):
    ca, cb = Counter(a["self"].get("hand", [])), Counter(b["self"].get("hand", []))
    return sum(abs(ca[k] - cb[k]) for k in set(ca) | set(cb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(_HERE / "_ent_states_w*.jsonl.gz"))
    ap.add_argument("--probes", default=str(_HERE / "_ent_probes_w*.jsonl.gz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--out", default=str(_HERE / "value_net" / "phase14_collision.json"))
    args = ap.parse_args()

    S = load(args.states, args.limit)
    print(f"states={len(S)}", file=sys.stderr)
    rep = {"n_states": len(S), "n_games": len({r["game"] for r in S})}

    X = np.asarray([r["state_feat"] for r in S], dtype=np.float32)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    games = np.asarray([r["game"] for r in S])

    # ---- 近傍探索(別試合のみ)----
    rs = np.random.RandomState(0)
    idx = rs.choice(len(S), size=min(3000, len(S)), replace=False)
    pairs = []
    CH = 512
    for a0 in range(0, len(idx), CH):
        q = Z[idx[a0:a0 + CH]]
        d = ((q[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        for r_, i in enumerate(idx[a0:a0 + CH]):
            d[r_][games == games[i]] = np.inf          # 同一試合は除外(§32)
            nn = np.argpartition(d[r_], args.knn)[:args.knn]
            for j in nn:
                if np.isfinite(d[r_][j]):
                    pairs.append((int(i), int(j), float(np.sqrt(d[r_][j]))))
    rep["n_pairs_examined"] = len(pairs)
    if not pairs:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), "utf-8")
        return

    dists = sorted(p[2] for p in pairs)
    thr = dists[int(0.05 * (len(dists) - 1))]          # 最も近い 5%
    rep["feature_distance"] = {
        "p5": round(thr, 4), "median": round(dists[len(dists) // 2], 4),
        "min": round(dists[0], 4)}

    near = [p for p in pairs if p[2] <= thr]
    rows = []
    for i, j, d in near:
        a, b = S[i], S[j]
        ed = entity_distance(a["entity"], b["entity"])
        bd = board_identity_diff(a["entity"], b["entity"])
        hd = hand_identity_diff(a["entity"], b["entity"])
        oa, ob = a.get("game_outcome"), b.get("game_outcome")
        rows.append({"i": i, "j": j, "fd": d, "entity_dist": ed, "board_diff": bd,
                     "hand_diff": hd,
                     "outcome_diff": (None if oa is None or ob is None else abs(oa - ob)),
                     "best_action_card_differs": int(
                         a["cand_card_ids"][int(np.argmax(a["q0_scores"]))]
                         != b["cand_card_ids"][int(np.argmax(b["q0_scores"]))]),
                     "turn_diff": abs(a["turn"] - b["turn"]),
                     "arch_differs": int(a["arch"] != b["arch"])})
    od = [r["outcome_diff"] for r in rows if r["outcome_diff"] is not None]
    rep["near_collision"] = {
        "n_pairs": len(rows),
        "threshold_feature_distance": round(thr, 4),
        "mean_entity_distance": round(st.mean([r["entity_dist"] for r in rows]), 2),
        "median_entity_distance": st.median([r["entity_dist"] for r in rows]),
        "board_identity_differs_rate": round(
            sum(1 for r in rows if r["board_diff"] > 0) / len(rows), 4),
        "hand_identity_differs_rate": round(
            sum(1 for r in rows if r["hand_diff"] > 0) / len(rows), 4),
        "mean_hand_card_diff": round(st.mean([r["hand_diff"] for r in rows]), 2),
        "archetype_differs_rate": round(
            sum(1 for r in rows if r["arch_differs"]) / len(rows), 4),
        "best_action_card_differs_rate": round(
            sum(1 for r in rows if r["best_action_card_differs"]) / len(rows), 4),
        "outcome_differs_rate": round(sum(1 for x in od if x > 0) / len(od), 4) if od else None,
        "n_with_outcome": len(od),
    }
    # 参考: ランダムペアの基準値(近傍が本当に「近い」のか)
    rp = [(int(x), int(y)) for x, y in rs.randint(0, len(S), size=(2000, 2))]
    rp = [(x, y) for x, y in rp if games[x] != games[y]]
    rep["random_baseline"] = {
        "n_pairs": len(rp),
        "mean_entity_distance": round(st.mean(
            [entity_distance(S[x]["entity"], S[y]["entity"]) for x, y in rp]), 2),
        "mean_feature_distance": round(st.mean(
            [float(np.linalg.norm(Z[x] - Z[y])) for x, y in rp]), 4),
        "outcome_differs_rate": round(st.mean(
            [1.0 if (S[x].get("game_outcome") is not None
                     and S[y].get("game_outcome") is not None
                     and S[x]["game_outcome"] != S[y]["game_outcome"]) else 0.0
             for x, y in rp]), 4),
    }
    rep["examples"] = sorted(rows, key=lambda r: (-r["entity_dist"], r["fd"]))[:10]

    # ---- §8/§9 Candidate recall ----
    Pr = load(args.probes)
    if Pr:
        rec = {"groups": len(Pr)}
        hits = Counter()
        wide_better = wide_groups = 0
        for g in Pr:
            tm = [st.mean(c["blocks"]) for c in g["candidates"]]
            best = int(np.argmax(tm))
            for k in (1, 4, 8, 12):
                kk = min(k, len(tm))
                top = sorted(range(len(tm)), key=lambda i: g["candidates"][i]["policy_rank"])[:kk]
                hits[k] += 1 if best in top else 0
            if g.get("wide"):
                wide_groups += 1
                if max(w["block"] for w in g["wide"]) > max(tm):
                    wide_better += 1
        rec["recall"] = {f"top{k}": round(hits[k] / len(Pr), 4) for k in (1, 4, 8, 12)}
        rec["n_legal_mean"] = round(st.mean([g["n_legal"] for g in Pr]), 2)
        rec["n_legal_gt8_rate"] = round(
            sum(1 for g in Pr if g["n_legal"] > 8) / len(Pr), 4)
        rec["wide_checked_groups"] = wide_groups
        rec["rank8to11_beats_top8_rate"] = (round(wide_better / wide_groups, 4)
                                            if wide_groups else None)
        rep["recall_audit"] = rec

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("n_states", "near_collision", "random_baseline", "recall_audit")
                      if k in rep}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
