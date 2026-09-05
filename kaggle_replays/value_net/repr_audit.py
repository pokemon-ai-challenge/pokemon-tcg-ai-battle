"""Phase19.7 §4-§35: action-conditioned representation の診断。

新モデルを作らず、Phase19.6 の TD0/TD1(N=2500)が **候補間の H1 差をどこで取り違えるか**
を分解する。

構造上の要点(§18): TD0/TD1 とも Q head の入力は [z(state), z(action)] で、
z(state) は group 内の全候補で**同一**。したがって候補の識別は z(action) だけから生じる。
z(action) = MLP(option65, action_card_id embedding)。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import train_lh as L  # noqa: E402
import train_transition as TT  # noqa: E402

# option65 のブロック境界(コードから確認済み)
TGT_PK = slice(35, 47)      # target_pokemon_*
TGT_CARD = slice(47, 60)    # target_card_*
ATTACK = slice(60, 65)
OPTTYPE = slice(0, 18)


def load(p):
    rows = []
    for f in sorted(glob.glob(p)):
        rows += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return rows


@torch.no_grad()
def latents(model, g, mu, sd, arm):
    """z(state) と候補ごとの z(action) を取り出す。"""
    model.eval()
    b = L.make_batch([g], mu, sd, arm)
    if arm == "TD1":
        zs = model._pool(b["before"])
        za = model.act(torch.cat([b["opt"], model.act_card(b["card"])], dim=-1))
    else:
        net = model.net
        zs = net.state_mlp(b["state"])
        za = net.act_mlp(torch.cat([b["opt"], net.card_emb(b["card"])], dim=-1))
    return zs[0].numpy(), za[0].numpy()


@torch.no_grad()
def score_with_action_override(model, g, mu, sd, arm, idx, opt_row=None, card=None):
    """候補 idx の action 入力だけ差し替えてスコアを出す(§21-§24 swap test)。"""
    b = L.make_batch([g], mu, sd, arm)
    if opt_row is not None:
        b["opt"][0, idx] = torch.tensor(opt_row, dtype=torch.float32)
    if card is not None:
        b["card"][0, idx] = int(card)
    return model(b)[0][b["mask"][0]].tolist()[idx]


def taxonomy(ca, cb):
    """§9 action 差のカテゴリ(複数該当あり)。"""
    t = []
    if ca["option_type"] != cb["option_type"]:
        t.append("different_action_type")
    else:
        t.append("same_action_type")
        if ca["action_card_id"] != cb["action_card_id"]:
            t.append("different_card")
        else:
            t.append("same_card")
    oa = np.asarray(ca["option_feat"], np.float32)
    ob = np.asarray(cb["option_feat"], np.float32)
    if np.abs(oa[TGT_PK] - ob[TGT_PK]).sum() > 1e-6:
        t.append("different_target_pokemon_features")
    else:
        t.append("same_target_pokemon_features")
    if np.abs(oa[TGT_CARD] - ob[TGT_CARD]).sum() > 1e-6:
        t.append("different_target_card_category")
    if np.abs(oa[ATTACK] - ob[ATTACK]).sum() > 1e-6:
        t.append("different_attack")
    # §9 A10: model から見える action 入力がほぼ同一
    if (np.abs(oa - ob).sum() < 1e-6) and ca["action_card_id"] == cb["action_card_id"]:
        t.append("model_input_identical")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(_HERE.parent / "_mh_t*.jsonl.gz"))
    ap.add_argument("--extra", default=str(_HERE.parent / "_mh_p*.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default=str(_HERE / "phase197_audit.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    L.HSTAR[0] = "H1"

    base = load(args.base)
    L.prepare(base)
    base = [g for g in base if g["_yA"] is not None]
    L.split_games(base)
    val = [g for g in base if g["_split"] == 1]
    testA = [g for g in base if g["_split"] == 2]
    pool = [g for g in base if g["_split"] == 0]
    extra = load(args.extra)
    L.prepare(extra)
    extra = [g for g in extra if g["_yA"] is not None]
    pool = pool + extra
    # Phase19.6 と同じ nested prefix
    from collections import defaultdict
    by = defaultdict(list)
    for g in pool:
        by[g["game"]].append(g)
    rs = np.random.RandomState(0)
    order = sorted(by)
    rs.shuffle(order)
    tr = []
    for game in order:
        if len(tr) + len(by[game]) > args.n and tr:
            break
        tr += by[game]
    X = np.asarray([g["state_feat"] for g in tr], np.float32)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    print(f"train={len(tr)} val={len(val)} testA={len(testA)}", file=sys.stderr)

    models = {}
    for arm in ("TD0", "TD1"):
        models[arm] = L.train_arm(tr, val, arm, mu, sd, args, 0)
        Path(_HERE / "frozen").mkdir(exist_ok=True)
        torch.save({"arm": arm, "n_train": len(tr), "seed": 0,
                    "state_dict": models[arm].state_dict(),
                    "mean": mu.tolist(), "std": sd.tolist(),
                    "option_dim": len(tr[0]["candidates"][0]["option_feat"])},
                   _HERE / "frozen" / f"{arm}_N{len(tr)}.pt")

    rep = {"train_groups": len(tr), "testA_groups": len(testA),
           "option65_blocks": {"target_pokemon": "35-46", "target_card": "47-59",
                               "attack": "60-64"},
           "model_visible_information": {
               "action_card_identity": "YES (explicit: action_card_id embedding)",
               "source_entity_identity": "NO",
               "source_entity_features": "NO",
               "source_slot": "NO",
               "target_entity_identity": "NO (category flags only)",
               "target_slot": "NO (option_position_norm はリスト内位置であり盤面 slot ではない)",
               "target_entity_features": "YES (explicit: option65 35-46)",
               "target_energy_count": "YES (count のみ、型は無し)",
               "energy_type": "NO",
               "source_evolution_line": "NO",
               "target_evolution_line": "NO",
               "selected_hand_card": "YES (action_card_id)",
               "state_entities": "INDIRECT (z(state) に pool 済み。action と binding されない)"}}

    # ---- pair 構築 ----
    pairs = []
    for g in testA:
        sc = {a: L.score_group(models[a], g, mu, sd, a) for a in ("TD0", "TD1")}
        zs, za = latents(models["TD1"], g, mu, sd, "TD1")
        y, yb = g["_yA"], g["_yB"]
        for i, j in itertools.combinations(range(len(g["candidates"])), 2):
            dh = y[i] - y[j]
            if dh == 0:
                continue
            ca, cb = g["candidates"][i], g["candidates"][j]
            oa = np.asarray(ca["option_feat"], np.float32)
            ob = np.asarray(cb["option_feat"], np.float32)
            stable = (yb is not None and (yb[i] - yb[j]) * dh > 0)
            pairs.append({
                "gid": g["group_id"], "turn_band": g["turn_band"], "arch": g["arch"],
                "i": i, "j": j, "dH1": dh, "absH1": abs(dh), "stable": bool(stable),
                "td1_correct": (sc["TD1"][i] - sc["TD1"][j]) * dh > 0,
                "td0_correct": (sc["TD0"][i] - sc["TD0"][j]) * dh > 0,
                "d_td1": abs(sc["TD1"][i] - sc["TD1"][j]),
                "raw_action_dist": float(np.abs(oa - ob).sum()
                                         + (0 if ca["action_card_id"] == cb["action_card_id"]
                                            else 5.0)),
                "opt_l1": float(np.abs(oa - ob).sum()),
                "same_card": ca["action_card_id"] == cb["action_card_id"],
                "latent_dist": float(np.linalg.norm(za[i] - za[j])),
                "tags": taxonomy(ca, cb)})

    def band(a):
        return ("S0" if a < 0.05 else "S1" if a < 0.10 else "S2" if a < 0.20 else "S3")

    rep["pair_population"] = {
        "total": len(pairs),
        **{b: sum(1 for p in pairs if band(p["absH1"]) == b) for b in ("S0", "S1", "S2", "S3")},
        "td1_correct": sum(1 for p in pairs if p["td1_correct"]),
        "td1_wrong": sum(1 for p in pairs if not p["td1_correct"]),
        "stable_H1": sum(1 for p in pairs if p["stable"])}
    core = [p for p in pairs if p["absH1"] >= 0.10 and p["stable"] and not p["td1_correct"]]
    rep["core_failure_set"] = {"size": len(core),
                               "definition": "|dH1|>=0.10 かつ H1 blockA/B 一致 かつ TD1 誤り"}

    # ---- §10 taxonomy ----
    tax = {}
    allt = sorted({t for p in pairs for t in p["tags"]})
    for t in allt:
        sub = [p for p in pairs if t in p["tags"]]
        lg = [p for p in sub if p["absH1"] >= 0.10]
        tax[t] = {"pairs": len(sub),
                  "wrong_rate": round(sum(1 for p in sub if not p["td1_correct"]) / len(sub), 4),
                  "mean_H1_margin": round(st.mean([p["absH1"] for p in sub]), 4),
                  "large_margin_pairs": len(lg),
                  "large_margin_wrong_rate": (round(
                      sum(1 for p in lg if not p["td1_correct"]) / len(lg), 4) if lg else None)}
    rep["action_taxonomy"] = tax

    # ---- §11 TD0 vs TD1 ----
    lgp = [p for p in pairs if p["absH1"] >= 0.10]
    rep["td0_vs_td1"] = {
        "all": dict(Counter(("TD0ok" if p["td0_correct"] else "TD0ng") + "/" +
                            ("TD1ok" if p["td1_correct"] else "TD1ng") for p in pairs)),
        "large_margin": dict(Counter(("TD0ok" if p["td0_correct"] else "TD0ng") + "/" +
                                     ("TD1ok" if p["td1_correct"] else "TD1ng")
                                     for p in lgp))}

    # ---- §13/§14 raw input collision ----
    exact = [p for p in pairs if "model_input_identical" in p["tags"]]
    rep["raw_input_collision"] = {
        "exact_collisions": len(exact),
        "rate": round(len(exact) / len(pairs), 4),
        "mean_H1_margin": round(st.mean([p["absH1"] for p in exact]), 4) if exact else None,
        "large_margin_count": sum(1 for p in exact if p["absH1"] >= 0.10)}
    ds = sorted(p["raw_action_dist"] for p in pairs)
    qs = [ds[int(q * (len(ds) - 1))] for q in (0.25, 0.5, 0.75)]

    def dq(p):
        d = p["raw_action_dist"]
        return "Q1" if d <= qs[0] else "Q2" if d <= qs[1] else "Q3" if d <= qs[2] else "Q4"
    rep["input_distance_quartiles"] = {
        q: {"pairs": sum(1 for p in pairs if dq(p) == q),
            "mean_H1_margin": round(st.mean([p["absH1"] for p in pairs if dq(p) == q]), 4),
            "td1_wrong_rate": round(sum(1 for p in pairs if dq(p) == q
                                        and not p["td1_correct"])
                                    / max(1, sum(1 for p in pairs if dq(p) == q)), 4)}
        for q in ("Q1", "Q2", "Q3", "Q4")}

    # ---- §16 latent distance vs |dH1| ----
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((x - mb) ** 2 for x in b) ** 0.5
        return (sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db)
                if da > 0 and db > 0 else None)
    ld = [p["latent_dist"] for p in pairs]
    hh = [p["absH1"] for p in pairs]
    lo = sorted(ld)[int(0.10 * (len(ld) - 1))]
    near = [p for p in pairs if p["latent_dist"] <= lo]
    rep["latent_audit"] = {
        "corr_latent_vs_H1margin": round(corr(ld, hh), 4),
        "corr_latent_vs_scorediff": round(corr(ld, [p["d_td1"] for p in pairs]), 4),
        "lowest_decile_threshold": round(lo, 4),
        "near_collision_pairs": len(near),
        "near_collision_large_margin": sum(1 for p in near if p["absH1"] >= 0.10),
        "near_collision_wrong_rate": round(
            sum(1 for p in near if not p["td1_correct"]) / len(near), 4)}

    # ---- §21-§26 sensitivity swap tests ----
    sens = {}
    sample = [p for p in pairs if p["absH1"] >= 0.10][:400]
    gmap = {g["group_id"]: g for g in testA}
    for name in ("full_action_swap", "card_id_swap", "target_block_swap"):
        chg, hd = [], []
        for p in sample:
            g = gmap[p["gid"]]
            ca, cb = g["candidates"][p["i"]], g["candidates"][p["j"]]
            base_s = L.score_group(models["TD1"], g, mu, sd, "TD1")[p["i"]]
            if name == "full_action_swap":
                s2 = score_with_action_override(models["TD1"], g, mu, sd, "TD1", p["i"],
                                                opt_row=cb["option_feat"],
                                                card=cb["action_card_id"])
            elif name == "card_id_swap":
                s2 = score_with_action_override(models["TD1"], g, mu, sd, "TD1", p["i"],
                                                card=cb["action_card_id"])
            else:
                row = list(ca["option_feat"])
                row[TGT_PK] = list(cb["option_feat"])[TGT_PK]
                s2 = score_with_action_override(models["TD1"], g, mu, sd, "TD1", p["i"],
                                                opt_row=row)
            chg.append(abs(s2 - base_s))
            hd.append(p["absH1"])
        sens[name] = {"n": len(chg), "mean_abs_score_change": round(st.mean(chg), 4),
                      "median": round(st.median(chg), 4),
                      "corr_with_H1_margin": round(corr(chg, hd), 4) if len(chg) > 3 else None}
    rep["sensitivity"] = sens

    # ---- §27 error archetype(自動分類できる範囲)----
    arch = Counter()
    for p in pairs:
        if p["td1_correct"]:
            continue
        if not p["stable"]:
            arch["R5_H1_noise"] += 1
        elif "model_input_identical" in p["tags"]:
            arch["R1_missing_action_identity"] += 1
        elif p["latent_dist"] <= lo:
            arch["R2R3_weak_binding_or_pooling"] += 1
        else:
            arch["R4_interaction_failure"] += 1
    n_wrong = sum(arch.values())
    rep["error_archetypes"] = {k: {"count": v, "rate": round(v / n_wrong, 4)}
                               for k, v in arch.most_common()}

    # ---- §K 代表例 ----
    ex = sorted(core, key=lambda p: (p["latent_dist"], -p["absH1"]))[:15]
    rep["examples"] = [{k: p[k] for k in ("gid", "turn_band", "arch", "absH1", "dH1",
                                          "latent_dist", "raw_action_dist", "same_card",
                                          "tags", "td0_correct")} for p in ex]
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("pair_population", "core_failure_set",
                                          "raw_input_collision", "latent_audit",
                                          "sensitivity", "error_archetypes")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
