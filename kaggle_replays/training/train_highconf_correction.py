"""ISMCTS v2.7 — High-Confidence Search-Correction Distillation。

v2.6(anchor + ALL-CHANGED correction、CHANGED recovery 改善せず Case D)からの唯一差分:
**correction 対象を HIGH_CONF CHANGED のみに絞り、correction loss を HIGH_CONF subset 内で正規化**
(v2.6 は mean-over-all で再希釈=禁止、Phase C)。

HIGH_CONF = CHANGED かつ top1 visit share>=0.5 かつ top1-top2 gap>=0.2(freeze、全 root の 8.5%)。
L = mean_over_ALL[KL(π_original‖student)] + λ_hc * mean_over_HIGH_CONF[KL(π_search‖student)]
λ_hc ∈ {1(primary), 2(diagnostic)}。dataset/split/H32/Original 初期化は v2.5/v2.6 と同一。Q 不使用。
checkpoint: UNCHANGED retention>=0.95 の下で HIGH_CONF recovery 最大。
`python train_highconf_correction.py [dataset.npz]`
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

_LAMBDAS = (1.0, 2.0)          # primary=1、diagnostic=2(online は λ を選ばない)
_EPOCHS = 40
_LR = 1e-3
_BATCH = 256
_SEED = 0
_RETENTION_GUARD = 0.95
_HC_SHARE = 0.5               # HIGH_CONF: top1 visit share >= 0.5
_HC_GAP = 0.2                #             AND top1-top2 gap >= 0.2
_V26_HC_BASELINE = 0.073     # v2.6 λ=2 high-conf recovery


def _groups(group, split, which):
    idx = np.nonzero(split == which)[0]
    by = defaultdict(list)
    for i in idx:
        by[int(group[i])].append(i)
    return list(by.values())


class Student(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def _init_original(model):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(o["layers"][0]["weight"], dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(o["layers"][0]["bias"], dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(o["layers"][1]["weight"], dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(o["layers"][1]["bias"], dtype=torch.float32))
    return model


def _is_hc(changed_v, ts_v, gp_v):
    return (changed_v == 1) and (ts_v >= _HC_SHARE) and (gp_v >= _HC_GAP)


def _pad(groups, X, visit, oscore, changed, top1share, gap, device):
    B = len(groups); M = max(len(g) for g in groups); D = X.shape[1]
    Xp = torch.zeros(B, M, D); vis = torch.zeros(B, M); osc = torch.full((B, M), -1e9); mask = torch.zeros(B, M)
    hc = torch.zeros(B)
    for b, g in enumerate(groups):
        n = len(g); f = g[0]
        Xp[b, :n] = torch.from_numpy(X[g]); vis[b, :n] = torch.from_numpy(visit[g].astype(np.float32))
        osc[b, :n] = torch.from_numpy(oscore[g]); mask[b, :n] = 1
        hc[b] = 1.0 if _is_hc(int(changed[f]), float(top1share[f]), float(gap[f])) else 0.0
    ps = vis / vis.sum(1, keepdim=True).clamp_min(1e-9)
    po = torch.softmax(osc, dim=1)
    return Xp.to(device), ps.to(device), po.to(device), mask.to(device), hc.to(device)


def _logsm(sc, mask):
    return torch.log_softmax(sc.masked_fill(mask == 0, -1e9), dim=1)


def _kl(p, logq, mask):
    return (p * (torch.log(p.clamp_min(1e-9)) - logq) * mask).sum(1)


def _eval(model, groups, X, visit, oscore, changed, top1share, gap, device):
    model.eval()
    ov = kl_s = kl_o = n = 0
    ch_tot = ch_rec = ch_stuck = 0
    un_tot = un_ret = 0
    conf = {"high": [0, 0, 0, 0], "med": [0, 0, 0, 0], "low": [0, 0, 0, 0]}  # [rec, stuck, third, total]
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH):
            gb = groups[s:s + _BATCH]
            Xp, ps, po, mask, hc = _pad(gb, X, visit, oscore, changed, top1share, gap, device)
            sc = model(Xp); logq = _logsm(sc, mask)
            kl_s += _kl(ps, logq, mask).sum().item(); kl_o += _kl(po, logq, mask).sum().item()
            s_arg = sc.masked_fill(mask == 0, -1e9).argmax(1); v_arg = ps.argmax(1); o_arg = po.argmax(1)
            for bi, g in enumerate(gb):
                sa = int(s_arg[bi]); va = int(v_arg[bi]); oa = int(o_arg[bi]); n += 1; ov += int(sa == va)
                f = g[0]
                if int(changed[f]) == 1:
                    ch_tot += 1; ch_rec += int(sa == va); ch_stuck += int(sa == oa)
                    ts = float(top1share[f]); gp = float(gap[f])
                    b = "high" if (ts >= _HC_SHARE and gp >= _HC_GAP) else ("low" if ts < 0.35 else "med")
                    conf[b][3] += 1; conf[b][0] += int(sa == va); conf[b][1] += int(sa == oa)
                    conf[b][2] += int(sa != va and sa != oa)
                else:
                    un_tot += 1; un_ret += int(sa == va)
    def r(a, b): return a / b if b else float("nan")
    return {"overall": r(ov, n), "kl_search": r(kl_s, n), "kl_orig": r(kl_o, n),
            "changed_recovery": r(ch_rec, ch_tot), "original_stuck": r(ch_stuck, ch_tot),
            "unchanged_retention": r(un_ret, un_tot), "changed_n": ch_tot, "unchanged_n": un_tot,
            "hc_recovery": r(conf["high"][0], conf["high"][3]), "hc_stuck": r(conf["high"][1], conf["high"][3]),
            "hc_third": r(conf["high"][2], conf["high"][3]), "hc_n": conf["high"][3],
            "med_recovery": r(conf["med"][0], conf["med"][3]), "med_n": conf["med"][3],
            "low_recovery": r(conf["low"][0], conf["low"][3]), "low_n": conf["low"][3]}


def _export(model, out, lam):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    o["meta"]["highconf_correction"] = {"lambda_hc": lam, "hc_def": f"CHANGED & share>={_HC_SHARE} & gap>={_HC_GAP}",
                                         "anchor": "mean-all KL to Original", "correction": "mean-HC-subset KL to Search"}
    o["layers"] = [
        {"weight": model.fc1.weight.detach().cpu().numpy().tolist(), "bias": model.fc1.bias.detach().cpu().numpy().tolist()},
        {"weight": model.fc2.weight.detach().cpu().numpy().tolist(), "bias": model.fc2.bias.detach().cpu().numpy().tolist()},
    ]
    out.write_text(json.dumps(o), encoding="utf-8")


def _train(lam, tr, X, visit, oscore, changed, top1share, gap, va, device):
    torch.manual_seed(_SEED)
    model = _init_original(Student(X.shape[1], 32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    rng = np.random.default_rng(_SEED)
    best = -1.0; best_state = copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(_EPOCHS):
        model.train(); order = rng.permutation(len(tr))
        for s in range(0, len(order), _BATCH):
            gb = [tr[i] for i in order[s:s + _BATCH]]
            Xp, ps, po, mask, hc = _pad(gb, X, visit, oscore, changed, top1share, gap, device)
            logq = _logsm(model(Xp), mask)
            l_anchor = _kl(po, logq, mask).mean()                              # mean over ALL
            kl_s = _kl(ps, logq, mask)
            l_hc = (kl_s * hc).sum() / hc.sum().clamp_min(1.0)                  # mean over HIGH_CONF subset のみ
            loss = l_anchor + lam * l_hc
            opt.zero_grad(); loss.backward(); opt.step()
        m = _eval(model, va, X, visit, oscore, changed, top1share, gap, device)
        score = m["hc_recovery"] if m["unchanged_retention"] >= _RETENTION_GUARD else -1.0   # HIGH_CONF recovery 最大
        if score > best:
            best = score; best_state = copy.deepcopy(model.state_dict()); best_ep = ep + 1
        if (ep + 1) % 10 == 0:
            print(f"      [λ{lam:.0f}] ep{ep+1}: val HC_rec={m['hc_recovery']:.3f} UNCHG_ret={m['unchanged_retention']:.3f} "
                  f"CHG_rec={m['changed_recovery']:.3f}", flush=True)
    final_m = _eval(model, va, X, visit, oscore, changed, top1share, gap, device)   # 最終 epoch(guard 無視)
    print(f"      [λ{lam:.0f}] FINAL(ep{_EPOCHS}, guard無視): HC_rec={final_m['hc_recovery']:.3f} "
          f"UNCHG_ret={final_m['unchanged_retention']:.3f}")
    _export(model, _HERE / f"highconf_l{int(lam)}_final.json", lam)   # guard 無視の final(interference の online 検証用)
    model.load_state_dict(best_state)
    return model, best_ep


def main(dataset="search_root_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(_HERE / dataset)
    X, visit, oscore, group, split = d["X"], d["visit"], d["oscore"], d["group"], d["split"]
    changed, top1share, gap = d["changed"], d["top1share"], d["gap"]
    tr = _groups(group, split, 0); va = _groups(group, split, 1); te = _groups(group, split, 2)

    # ---- Phase A/B: HIGH_CONF audit(split 別)----
    def _hc_count(gs):
        return sum(1 for g in gs if _is_hc(int(changed[g[0]]), float(top1share[g[0]]), float(gap[g[0]])))
    def _ch_count(gs):
        return sum(1 for g in gs if int(changed[g[0]]) == 1)
    print("=== Phase A/B: HIGH_CONF audit(def: CHANGED & share>=0.5 & gap>=0.2)===")
    for lab, gs in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {lab}: roots={len(gs)}  CHANGED={_ch_count(gs)}  HIGH_CONF_CHANGED={_hc_count(gs)}")
    print()

    print("=== Phase C-G: λ_hc sweep(anchor=all-mean + correction=HC-subset-mean)===")
    print(f"  {'λ_hc':>5} {'overall':>8} {'CHG_rec':>8} {'HC_rec':>7} {'HC_stuck':>8} {'HC_third':>8} "
          f"{'med_rec':>7} {'low_rec':>7} {'UNCHG_ret':>10} {'ep*':>4}")
    results = []
    for lam in _LAMBDAS:
        model, ep = _train(lam, tr, X, visit, oscore, changed, top1share, gap, va, device)
        m = _eval(model, te, X, visit, oscore, changed, top1share, gap, device)
        results.append((lam, model, m, ep))
        _export(model, _HERE / f"highconf_l{int(lam)}.json", lam)
        print(f"  {lam:>5.0f} {m['overall']:>8.3f} {m['changed_recovery']:>8.3f} {m['hc_recovery']:>7.3f} "
              f"{m['hc_stuck']:>8.3f} {m['hc_third']:>8.3f} {m['med_recovery']:>7.3f} {m['low_recovery']:>7.3f} "
              f"{m['unchanged_retention']:>10.3f} {ep:>4d}")

    # ---- Phase H: candidate freeze(primary λ=1、guard: retention>=0.95 & HC_rec > v2.6 baseline)----
    lam1 = next(r for r in results if r[0] == 1.0)
    _, model, m, ep = lam1
    _export(model, _HERE / "highconf_correction_policy.json", 1.0)
    online_ok = (m["unchanged_retention"] >= _RETENTION_GUARD) and (m["hc_recovery"] > _V26_HC_BASELINE)
    print(f"\n=== Phase H: candidate freeze(primary λ_hc=1, ep{ep})===")
    print(f"  HIGH_CONF recovery = {m['hc_recovery']:.3f} (n={m['hc_n']})  [v2.6 λ2 baseline {_V26_HC_BASELINE}]")
    print(f"  HIGH_CONF confusion: →Search {m['hc_recovery']:.3f} / →Original {m['hc_stuck']:.3f} / →third {m['hc_third']:.3f}")
    print(f"  UNCHANGED retention = {m['unchanged_retention']:.3f}  (guard {_RETENTION_GUARD})")
    print(f"  general CHANGED recovery = {m['changed_recovery']:.3f}  [v2.5 0.150 / v2.6 0.085-0.099]")
    print(f"  ONLINE 進行条件(retention>=0.95 & HC_rec>{_V26_HC_BASELINE}): {'PASS' if online_ok else 'FAIL(offline Case D)'}")
    print(f"\nsaved highconf_correction_policy.json (λ_hc=1)")


if __name__ == "__main__":
    main(dataset=sys.argv[1] if len(sys.argv) > 1 else "search_root_dataset.npz")
