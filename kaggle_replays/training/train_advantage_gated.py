"""ISMCTS v2.8 — Advantage-Gated Search-Correction Distillation。

v2.7(HIGH_CONF visit-confidence gate)からの唯一差分: **gate を Search Q advantage ΔQ に変更**。
ΔQ = Q(a_search) - Q(a_orig)(root_me 視点、[-1,+1])。a_orig が unvisited(visit=0→Q=0 不正)の root は除外。
ADV_CHG = CHANGED かつ valid-Q かつ ΔQ >= train-only threshold(positive-ΔQ CHANGED の P75/P90)。

L = mean_ALL[KL(π_original‖student)] + λ_adv · mean_ADV_CHG[KL(π_search‖student)]（correction は ADV_CHG subset 内平均）。
dataset/split/H32/Original 初期化/Teacher は v2.5-2.7 と同一。visit-confidence gate は使わない。
`python train_advantage_gated.py [dataset.npz]`
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_ORIG_JSON = _SUB / "ptcg_ai" / "learning" / "policy_weights.json"

_LAMBDAS = (1.0, 2.0)
_EPOCHS = 40
_LR = 1e-3
_BATCH = 256
_SEED = 0
_RETENTION_GUARD = 0.90        # v2.7 の 0.95 より緩め(pre-registered)
_HC_SHARE, _HC_GAP = 0.5, 0.2  # v2.7 HIGH_CONF(overlap 診断用)


def _grp(group, split, which):
    idx = np.nonzero(split == which)[0]; by = defaultdict(list)
    for i in idx:
        by[int(group[i])].append(i)
    return list(by.values())


class Student(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def _init_orig(m):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    with torch.no_grad():
        m.fc1.weight.copy_(torch.tensor(o["layers"][0]["weight"], dtype=torch.float32))
        m.fc1.bias.copy_(torch.tensor(o["layers"][0]["bias"], dtype=torch.float32))
        m.fc2.weight.copy_(torch.tensor(o["layers"][1]["weight"], dtype=torch.float32))
        m.fc2.bias.copy_(torch.tensor(o["layers"][1]["bias"], dtype=torch.float32))
    return m


def _group_meta(groups, visit, oscore, q, changed, top1share, gap):
    """各 group の (changed, valid_Q, dQ, a_search 局所pos, hc) を返す。"""
    meta = []
    for g in groups:
        vg = visit[g].astype(np.float64); og = oscore[g].astype(np.float64); qg = q[g].astype(np.float64)
        si = int(np.argmax(vg)); oi = int(np.argmax(og))
        valid = vg[oi] > 0
        dQ = float(qg[si] - qg[oi])
        chg = int(changed[g[0]])
        f = g[0]
        hc = 1 if (chg == 1 and float(top1share[f]) >= _HC_SHARE and float(gap[f]) >= _HC_GAP) else 0
        meta.append({"changed": chg, "valid": bool(valid), "dQ": dQ, "hc": hc})
    return meta


def _pad(groups, meta_sub, X, visit, oscore, adv_flags, device):
    B = len(groups); M = max(len(g) for g in groups); D = X.shape[1]
    Xp = torch.zeros(B, M, D); vis = torch.zeros(B, M); osc = torch.full((B, M), -1e9); mask = torch.zeros(B, M)
    adv = torch.zeros(B)
    for b, (g, af) in enumerate(zip(groups, adv_flags)):
        n = len(g)
        Xp[b, :n] = torch.from_numpy(X[g]); vis[b, :n] = torch.from_numpy(visit[g].astype(np.float32))
        osc[b, :n] = torch.from_numpy(oscore[g]); mask[b, :n] = 1; adv[b] = float(af)
    ps = vis / vis.sum(1, keepdim=True).clamp_min(1e-9); po = torch.softmax(osc, dim=1)
    return Xp.to(device), ps.to(device), po.to(device), mask.to(device), adv.to(device)


def _logsm(sc, mask): return torch.log_softmax(sc.masked_fill(mask == 0, -1e9), dim=1)
def _kl(p, logq, mask): return (p * (torch.log(p.clamp_min(1e-9)) - logq) * mask).sum(1)


def _eval(model, groups, meta, X, visit, oscore, device):
    model.eval(); ov = n = 0
    ch_tot = ch_rec = un_tot = un_ret = 0
    adv_tot = adv_rec = adv_stuck = 0; rec_mass = 0.0; adv_mass = 0.0
    hc_tot = hc_rec = 0
    binrec = {"lo": [0, 0], "md": [0, 0], "hi": [0, 0], "vh": [0, 0]}
    zero = torch.zeros(1)
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH):
            gb = groups[s:s + _BATCH]; mb = meta[s:s + _BATCH]
            Xp, ps, po, mask, _ = _pad(gb, mb, X, visit, oscore, [0] * len(gb), device)
            sc = model(Xp); s_arg = sc.masked_fill(mask == 0, -1e9).argmax(1); v_arg = ps.argmax(1); o_arg = po.argmax(1)
            for bi in range(len(gb)):
                sa = int(s_arg[bi]); va = int(v_arg[bi]); oa = int(o_arg[bi]); n += 1; ov += int(sa == va)
                mt = mb[bi]
                if mt["changed"] == 1:
                    ch_tot += 1; ch_rec += int(sa == va)
                    if mt["hc"]: hc_tot += 1; hc_rec += int(sa == va)
                    if mt["valid"] and mt["dQ"] > 0:
                        adv_mass += mt["dQ"]
                        b = ("vh" if mt["dQ"] >= 0.5 else "hi" if mt["dQ"] >= 0.3 else "md" if mt["dQ"] >= 0.15 else "lo")
                        binrec[b][1] += 1; binrec[b][0] += int(sa == va)
                else:
                    un_tot += 1; un_ret += int(sa == va)
    def r(a, b): return a / b if b else float("nan")
    return {"overall": r(ov, n), "changed_recovery": r(ch_rec, ch_tot),
            "unchanged_retention": r(un_ret, un_tot), "hc_recovery": r(hc_rec, hc_tot), "hc_n": hc_tot,
            "binrec": {k: (r(v[0], v[1]), v[1]) for k, v in binrec.items()}}


def _adv_metrics(model, groups, meta, adv_flags, X, visit, oscore, device):
    """ADV_CHG subset の recovery / Original-stuck / recovered advantage mass。"""
    model.eval(); tot = rec = stuck = 0; rec_mass = 0.0; tot_mass = 0.0
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH):
            gb = groups[s:s + _BATCH]; mb = meta[s:s + _BATCH]; ab = adv_flags[s:s + _BATCH]
            Xp, ps, po, mask, _ = _pad(gb, mb, X, visit, oscore, [0] * len(gb), device)
            sc = model(Xp); s_arg = sc.masked_fill(mask == 0, -1e9).argmax(1); v_arg = ps.argmax(1); o_arg = po.argmax(1)
            for bi in range(len(gb)):
                if not ab[bi]: continue
                sa = int(s_arg[bi]); va = int(v_arg[bi]); oa = int(o_arg[bi]); dq = mb[bi]["dQ"]
                tot += 1; tot_mass += dq
                rec += int(sa == va); stuck += int(sa == oa)
                if sa == va: rec_mass += dq
    def r(a, b): return a / b if b else float("nan")
    return {"adv_recovery": r(rec, tot), "adv_stuck": r(stuck, tot), "adv_n": tot,
            "rec_mass": rec_mass, "tot_mass": tot_mass, "rec_mass_frac": r(rec_mass, tot_mass)}


def _export(model, out, lam, thr):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    o["meta"]["advantage_gated"] = {"lambda_adv": lam, "gate": "dQ>=train_P75", "threshold": thr}
    o["layers"] = [{"weight": model.fc1.weight.detach().cpu().numpy().tolist(), "bias": model.fc1.bias.detach().cpu().numpy().tolist()},
                   {"weight": model.fc2.weight.detach().cpu().numpy().tolist(), "bias": model.fc2.bias.detach().cpu().numpy().tolist()}]
    out.write_text(json.dumps(o), encoding="utf-8")


def _train(lam, tr, tr_adv, tr_meta, va, va_meta, X, visit, oscore, device):
    torch.manual_seed(_SEED); model = _init_orig(Student(X.shape[1], 32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=_LR); rng = np.random.default_rng(_SEED)
    best = -1.0; best_state = copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(_EPOCHS):
        model.train(); order = rng.permutation(len(tr))
        for s in range(0, len(order), _BATCH):
            gi = order[s:s + _BATCH]; gb = [tr[i] for i in gi]; ab = [tr_adv[i] for i in gi]
            Xp, ps, po, mask, adv = _pad(gb, None, X, visit, oscore, ab, device)
            logq = _logsm(model(Xp), mask)
            l_anchor = _kl(po, logq, mask).mean()
            l_adv = (_kl(ps, logq, mask) * adv).sum() / adv.sum().clamp_min(1.0)
            (l_anchor + lam * l_adv).backward(); opt.step(); opt.zero_grad()
        m = _eval(model, va, va_meta, X, visit, oscore, device)
        score = m["changed_recovery"] if m["unchanged_retention"] >= _RETENTION_GUARD else -1.0
        if score > best:
            best = score; best_state = copy.deepcopy(model.state_dict()); best_ep = ep + 1
        if (ep + 1) % 10 == 0:
            print(f"      [λ{lam:.0f}] ep{ep+1}: val CHG_rec={m['changed_recovery']:.3f} UNCHG_ret={m['unchanged_retention']:.3f}", flush=True)
    fin = copy.deepcopy(model.state_dict())
    _export(model, _HERE / f"adv_l{int(lam)}_final.json", lam, "P75")
    model.load_state_dict(best_state)
    return model, best_ep


def main(dataset="search_root_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(_HERE / dataset)
    X, visit, oscore, q, group, split = d["X"], d["visit"], d["oscore"], d["q"], d["group"], d["split"]
    changed, top1share, gap, entropy = d["changed"], d["top1share"], d["gap"], d["entropy"]
    tr = _grp(group, split, 0); va = _grp(group, split, 1); te = _grp(group, split, 2)
    tr_m = _group_meta(tr, visit, oscore, q, changed, top1share, gap)
    va_m = _group_meta(va, visit, oscore, q, changed, top1share, gap)
    te_m = _group_meta(te, visit, oscore, q, changed, top1share, gap)

    # ---- Phase A/B: Q / ΔQ audit ----
    dq_tr = np.array([m["dQ"] for m in tr_m if m["changed"] and m["valid"]])
    inval = sum(1 for m in tr_m if m["changed"] and not m["valid"])
    print("=== Phase A/B: Root Q / ΔQ audit(root_me 視点, [-1,1]; a_orig unvisited は除外)===")
    print(f"  train CHANGED={sum(m['changed'] for m in tr_m)}  valid-Q CHANGED={len(dq_tr)}  invalid(a_orig unvisited)={inval}")
    pos = dq_tr[dq_tr > 0]
    print(f"  ΔQ mean={dq_tr.mean():.3f} median={np.median(dq_tr):.3f} std={dq_tr.std():.3f} "
          f"min={dq_tr.min():.3f} max={dq_tr.max():.3f}")
    print(f"  positive ΔQ={len(pos)}({len(pos)/len(dq_tr):.1%})  zero/neg={len(dq_tr)-len(pos)}")
    for p in (10, 25, 50, 75, 90, 95):
        print(f"    P{p}(all ΔQ)={np.percentile(dq_tr, p):+.3f}   P{p}(positive)={np.percentile(pos, p):+.3f}" if len(pos) else "")
    # visit-confidence 相関
    ts = np.array([float(top1share[g[0]]) for g, m in zip(tr, tr_m) if m["changed"] and m["valid"]])
    gp = np.array([float(gap[g[0]]) for g, m in zip(tr, tr_m) if m["changed"] and m["valid"]])
    en = np.array([float(entropy[g[0]]) for g, m in zip(tr, tr_m) if m["changed"] and m["valid"]])
    print(f"  corr(ΔQ, share)={np.corrcoef(dq_tr, ts)[0,1]:+.3f}  corr(ΔQ, gap)={np.corrcoef(dq_tr, gp)[0,1]:+.3f}  "
          f"corr(ΔQ, entropy)={np.corrcoef(dq_tr, en)[0,1]:+.3f}")

    # ---- Phase C: train-only threshold freeze(positive ΔQ の P75 / P90)----
    THR = {"A75": float(np.percentile(pos, 75)), "A90": float(np.percentile(pos, 90))}
    print(f"\n=== Phase C: threshold freeze(train positive-ΔQ percentile)= A75 {THR['A75']:+.3f} / A90 {THR['A90']:+.3f} ===")

    def adv_flags(groups, meta, thr):
        return [1 if (m["changed"] and m["valid"] and m["dQ"] >= thr) else 0 for m in meta]

    # ---- Phase D: ADV_CHG counts + HC overlap ----
    for tag, thr in THR.items():
        af_tr = adv_flags(tr, tr_m, thr); af_va = adv_flags(va, va_m, thr); af_te = adv_flags(te, te_m, thr)
        hc_tr = [m["hc"] for m in tr_m]
        both = sum(1 for a, h in zip(af_tr, hc_tr) if a and h); aonly = sum(1 for a, h in zip(af_tr, hc_tr) if a and not h)
        honly = sum(1 for a, h in zip(af_tr, hc_tr) if h and not a)
        print(f"  [{tag}] ADV_CHG train/val/test = {sum(af_tr)}/{sum(af_va)}/{sum(af_te)}  "
              f"| overlap(train): ADV∩HC={both} ADV_only={aonly} HC_only={honly}")

    # ---- Phase E-I: train λ_adv for A75(primary threshold)----
    print(f"\n=== Phase E-I: training(gate=A75, anchor + λ_adv·ADV_CHG-subset correction)===")
    print(f"  {'λ':>3} {'overall':>8} {'CHG_rec':>8} {'ADV_rec':>8} {'ADV_stuck':>9} {'rec_mass%':>9} {'HC_rec':>7} {'UNCHG_ret':>10} {'ep*':>4}")
    thr = THR["A75"]
    tr_af = adv_flags(tr, tr_m, thr); te_af = adv_flags(te, te_m, thr)
    results = []
    for lam in _LAMBDAS:
        model, ep = _train(lam, tr, tr_af, tr_m, va, va_m, X, visit, oscore, device)
        m = _eval(model, te, te_m, X, visit, oscore, device)
        am = _adv_metrics(model, te, te_m, te_af, X, visit, oscore, device)
        _export(model, _HERE / f"adv_l{int(lam)}.json", lam, "P75")
        results.append((lam, model, m, am, ep))
        print(f"  {lam:>3.0f} {m['overall']:>8.3f} {m['changed_recovery']:>8.3f} {am['adv_recovery']:>8.3f} "
              f"{am['adv_stuck']:>9.3f} {am['rec_mass_frac']*100:>8.1f}% {m['hc_recovery']:>7.3f} {m['unchanged_retention']:>10.3f} {ep:>4d}")
        print(f"       ΔQ-bin recovery: " + " ".join(f"{k}={v[0]:.2f}(n{v[1]})" for k, v in m["binrec"].items())
              + f"  ADV_n(test)={am['adv_n']}")

    # ---- Phase H: candidate(λ=1 primary)----
    lam1 = next(r for r in results if r[0] == 1.0); _, model, m, am, ep = lam1
    _export(model, _HERE / "advantage_gated_policy.json", 1.0, "P75")
    ok = (m["unchanged_retention"] >= _RETENTION_GUARD) and (am["adv_recovery"] > 0.1)
    print(f"\n=== Phase H: candidate freeze(λ_adv=1, A75, ep{ep})===")
    print(f"  ADV_CHG recovery={am['adv_recovery']:.3f}(n={am['adv_n']}) recovered advantage mass={am['rec_mass_frac']:.1%} "
          f"UNCHANGED retention={m['unchanged_retention']:.3f}")
    print(f"  historical: v2.5 CHG_rec 0.150 / v2.6 0.085-0.099 / v2.7 HC_rec 0.17-0.22(ret 0.76-0.84)")
    print(f"  online 進行条件(retention>={_RETENTION_GUARD} & ADV_rec>0.1): {'PASS' if ok else 'FAIL'}")
    print("saved advantage_gated_policy.json (λ_adv=1, A75) + adv_l{1,2}{,_final}.json")


if __name__ == "__main__":
    main(dataset=sys.argv[1] if len(sys.argv) > 1 else "search_root_dataset.npz")
