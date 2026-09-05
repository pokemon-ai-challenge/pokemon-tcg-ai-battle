"""ISMCTS v2.10 — Frozen-Original Residual Correction Adapter。

Original Policy 735dd38a を完全凍結し、別容量の Residual Adapter(hidden H)を追加:
final_score = orig_score(frozen) + residual_score(trained, zero-init)。
- L_keep(V-OFF のみ): KL(π_original ‖ π_final) で Original 挙動維持。
- L_corr(V-positive=Oracle V gate のみ): KL(π_search ‖ π_final) で correction 学習。V-pos では anchor を掛けない。
- L = L_keep + λ_corr·L_corr(subset 内正規化、全 root 再希釈禁止)。λ_corr=1。
V gate = v2.7/v2.9 exact(CHANGED & share>=0.5 & gap>=0.2)。dataset は v2.5-2.8 と同一(追加収集なし)。
export: orig+adapter を単一 block-diagonal MLP(hidden 32+H)として PolicyModel JSON 化(既存推論経路で orig+residual)。
`python train_residual_adapter.py [dataset.npz]`
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

_ADP_HIDDENS = (4, 8, 16)
_EPOCHS = 60
_LR = 2e-3
_BATCH = 256
_SEED = 0
_LAMBDA = 1.0
_V_SHARE, _V_GAP = 0.5, 0.2
_RET_GUARD = 0.97          # V-OFF retention guard(pre-registered)
_REC_BAR = 0.40           # V recovery bar(v2.7 の ~2倍)


def _grp(group, split, which):
    idx = np.nonzero(split == which)[0]; by = defaultdict(list)
    for i in idx:
        by[int(group[i])].append(i)
    return list(by.values())


def _vpos(meta_first, top1share, gap, changed):
    return changed and top1share >= _V_SHARE and gap >= _V_GAP


class AdapterPolicy(nn.Module):
    """frozen Original(fc1o/fc2o)+ trainable Adapter(fc1a/fc2a、zero-init fc2a)。final=orig+adp。"""
    def __init__(self, in_dim, adp_h):
        super().__init__()
        self.fc1o = nn.Linear(in_dim, 32); self.fc2o = nn.Linear(32, 1)
        self.fc1a = nn.Linear(in_dim, adp_h); self.fc2a = nn.Linear(adp_h, 1)
        o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
        with torch.no_grad():
            self.fc1o.weight.copy_(torch.tensor(o["layers"][0]["weight"], dtype=torch.float32))
            self.fc1o.bias.copy_(torch.tensor(o["layers"][0]["bias"], dtype=torch.float32))
            self.fc2o.weight.copy_(torch.tensor(o["layers"][1]["weight"], dtype=torch.float32))
            self.fc2o.bias.copy_(torch.tensor(o["layers"][1]["bias"], dtype=torch.float32))
            self.fc2a.weight.zero_(); self.fc2a.bias.zero_()      # zero-init → step0 residual=0
        for p in (self.fc1o.weight, self.fc1o.bias, self.fc2o.weight, self.fc2o.bias):
            p.requires_grad_(False)                                # Original 凍結(gradient 0)

    def orig(self, x):
        return self.fc2o(torch.relu(self.fc1o(x))).squeeze(-1)

    def residual(self, x):
        return self.fc2a(torch.relu(self.fc1a(x))).squeeze(-1)

    def forward(self, x):
        return self.orig(x) + self.residual(x)


def _pad(groups, X, visit, oscore, vpos_flags, device):
    B = len(groups); M = max(len(g) for g in groups); D = X.shape[1]
    Xp = torch.zeros(B, M, D); vis = torch.zeros(B, M); osc = torch.full((B, M), -1e9); mask = torch.zeros(B, M)
    vp = torch.zeros(B)
    for b, (g, f) in enumerate(zip(groups, vpos_flags)):
        n = len(g)
        Xp[b, :n] = torch.from_numpy(X[g]); vis[b, :n] = torch.from_numpy(visit[g].astype(np.float32))
        osc[b, :n] = torch.from_numpy(oscore[g]); mask[b, :n] = 1; vp[b] = float(f)
    ps = vis / vis.sum(1, keepdim=True).clamp_min(1e-9)
    return Xp.to(device), ps.to(device), mask.to(device), vp.to(device)


def _logsm(sc, mask): return torch.log_softmax(sc.masked_fill(mask == 0, -1e9), dim=1)
def _klrow(p, logq, mask): return (p * (torch.log(p.clamp_min(1e-9)) - logq) * mask).sum(1)


def _eval(model, groups, vpos_flags, changed_first, X, visit, oscore, device):
    model.eval(); n = 0
    vp_tot = vp_rec = vp_stuck = vp_third = 0
    voff_tot = voff_ret = 0
    unch_tot = unch_ret = 0
    res_von = []; res_voff = []
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH):
            gb = groups[s:s + _BATCH]; vf = vpos_flags[s:s + _BATCH]; cf = changed_first[s:s + _BATCH]
            Xp, ps, mask, vp = _pad(gb, X, visit, oscore, vf, device)
            final = model(Xp); orig = model.orig(Xp); resid = model.residual(Xp)
            f_arg = final.masked_fill(mask == 0, -1e9).argmax(1)
            o_arg = orig.masked_fill(mask == 0, -1e9).argmax(1)
            v_arg = ps.argmax(1)
            rmag = (resid.abs() * mask).sum(1) / mask.sum(1).clamp_min(1)
            for bi in range(len(gb)):
                fa = int(f_arg[bi]); oa = int(o_arg[bi]); va = int(v_arg[bi]); n += 1
                if vf[bi]:
                    vp_tot += 1; vp_rec += int(fa == va); vp_stuck += int(fa == oa)
                    vp_third += int(fa != va and fa != oa); res_von.append(float(rmag[bi]))
                else:
                    voff_tot += 1; voff_ret += int(fa == oa); res_voff.append(float(rmag[bi]))
                    if cf[bi] == 0:      # UNCHANGED(historical 比較)
                        unch_tot += 1; unch_ret += int(fa == oa)
    def r(a, b): return a / b if b else float("nan")
    return {"v_recovery": r(vp_rec, vp_tot), "v_stuck": r(vp_stuck, vp_tot), "v_third": r(vp_third, vp_tot),
            "voff_retention": r(voff_ret, voff_tot), "unchanged_retention": r(unch_ret, unch_tot),
            "vpos_n": vp_tot, "voff_n": voff_tot,
            "res_von": float(np.mean(res_von)) if res_von else 0.0,
            "res_voff": float(np.mean(res_voff)) if res_voff else 0.0,
            "res_voff_p95": float(np.percentile(res_voff, 95)) if res_voff else 0.0}


def _export_combined(model, out, adp_h):
    """orig(32)+adapter(H)を単一 block-diagonal MLP(hidden 32+H)として PolicyModel JSON 化。"""
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    W1o = model.fc1o.weight.detach().cpu().numpy(); b1o = model.fc1o.bias.detach().cpu().numpy()
    W1a = model.fc1a.weight.detach().cpu().numpy(); b1a = model.fc1a.bias.detach().cpu().numpy()
    W2o = model.fc2o.weight.detach().cpu().numpy(); b2o = float(model.fc2o.bias.detach().cpu().numpy()[0])
    W2a = model.fc2a.weight.detach().cpu().numpy(); b2a = float(model.fc2a.bias.detach().cpu().numpy()[0])
    W1 = np.vstack([W1o, W1a]); b1 = np.concatenate([b1o, b1a])            # [(32+H), 239]
    W2 = np.hstack([W2o, W2a]); b2 = np.array([b2o + b2a])                 # [1, 32+H]
    o["meta"]["residual_adapter"] = {"adapter_hidden": adp_h, "frozen_original": "735dd38a",
                                     "combined_hidden": 32 + adp_h, "v_gate": "share>=0.5 & gap>=0.2"}
    o["layers"] = [{"weight": W1.tolist(), "bias": b1.tolist()}, {"weight": W2.tolist(), "bias": b2.tolist()}]
    out.write_text(json.dumps(o), encoding="utf-8")


def _train(adp_h, tr, tr_vp, va, va_vp, va_cf, X, visit, oscore, device):
    torch.manual_seed(_SEED)
    model = AdapterPolicy(X.shape[1], adp_h).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=_LR)
    rng = np.random.default_rng(_SEED)
    best = -1.0; best_state = copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(_EPOCHS):
        model.train(); order = rng.permutation(len(tr))
        for s in range(0, len(order), _BATCH):
            gi = order[s:s + _BATCH]; gb = [tr[i] for i in gi]; vf = [tr_vp[i] for i in gi]
            Xp, ps, mask, vp = _pad(gb, X, visit, oscore, vf, device)
            final = model(Xp); logq = _logsm(final, mask)
            with torch.no_grad():
                p_orig = torch.softmax(model.orig(Xp).masked_fill(mask == 0, -1e9), dim=1)
            kl_keep = _klrow(p_orig, logq, mask)          # 全 row、後で V-OFF mask
            kl_corr = _klrow(ps, logq, mask)
            voff = (1.0 - vp)
            l_keep = (kl_keep * voff).sum() / voff.sum().clamp_min(1.0)
            l_corr = (kl_corr * vp).sum() / vp.sum().clamp_min(1.0)
            (l_keep + _LAMBDA * l_corr).backward(); opt.step(); opt.zero_grad()
        m = _eval(model, va, va_vp, va_cf, X, visit, oscore, device)
        score = m["v_recovery"] if m["voff_retention"] >= _RET_GUARD else -1.0
        if score > best:
            best = score; best_state = copy.deepcopy(model.state_dict()); best_ep = ep + 1
        if (ep + 1) % 20 == 0:
            print(f"      [A{adp_h}] ep{ep+1}: val V_rec={m['v_recovery']:.3f} VOFF_ret={m['voff_retention']:.3f}", flush=True)
    _export_combined(model, _HERE / f"residual_adapter_a{adp_h}_final.json", adp_h)   # final(guard 無視)= online 検証用
    model.load_state_dict(best_state)
    return model, best_ep


def main(dataset="search_root_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(_HERE / dataset)
    X, visit, oscore, group, split = d["X"], d["visit"], d["oscore"], d["group"], d["split"]
    changed, top1share, gap = d["changed"], d["top1share"], d["gap"]
    tr = _grp(group, split, 0); va = _grp(group, split, 1); te = _grp(group, split, 2)

    def vp_flags(gs): return [_vpos(g[0], float(top1share[g[0]]), float(gap[g[0]]), int(changed[g[0]])) for g in gs]
    def cf(gs): return [int(changed[g[0]]) for g in gs]
    tr_vp, va_vp, te_vp = vp_flags(tr), vp_flags(va), vp_flags(te)
    va_cf, te_cf = cf(va), cf(te)

    # ---- Phase A: frozen correctness ----
    import hashlib
    orig_sha_before = hashlib.sha1(_ORIG_JSON.read_bytes()).hexdigest()[:12]
    m0 = AdapterPolicy(X.shape[1], 8).to(device)
    with torch.no_grad():
        xb = torch.from_numpy(X[te[0]]).to(device)
        eq = torch.allclose(m0(xb), m0.orig(xb), atol=1e-6)
    grad_iso = all(not p.requires_grad for p in (m0.fc1o.weight, m0.fc2o.weight))
    print("=== Phase A: Frozen-Original correctness ===")
    print(f"  zero-adapter == Original: {bool(eq)}  | gradient isolation(orig frozen): {grad_iso}  | orig sha {orig_sha_before}")
    print(f"  V-positive train/val/test = {sum(tr_vp)}/{sum(va_vp)}/{sum(te_vp)}  V-OFF test = {sum(1 for v in te_vp if not v)}\n")

    print("=== Phase D-I: adapter sweep(frozen Original + residual)===")
    print(f"  {'A':>4} {'V_rec':>7} {'V_stuck':>8} {'V_third':>8} {'VOFF_ret':>9} {'UNCHG_ret':>10} "
          f"{'res_VON':>8} {'res_VOFF':>9} {'ep*':>4}")
    results = []
    for adp_h in _ADP_HIDDENS:
        model, ep = _train(adp_h, tr, tr_vp, va, va_vp, va_cf, X, visit, oscore, device)
        m = _eval(model, te, te_vp, te_cf, X, visit, oscore, device)
        _export_combined(model, _HERE / f"residual_adapter_a{adp_h}.json", adp_h)
        results.append((adp_h, model, m, ep))
        print(f"  {adp_h:>4} {m['v_recovery']:>7.3f} {m['v_stuck']:>8.3f} {m['v_third']:>8.3f} {m['voff_retention']:>9.3f} "
              f"{m['unchanged_retention']:>10.3f} {m['res_von']:>8.3f} {m['res_voff']:>9.3f} {ep:>4d}")

    # ---- Phase I: candidate selection(retention>=0.97 & V_rec>=0.40、最小 adapter)----
    elig = [(a, mo, m, ep) for (a, mo, m, ep) in results if m["voff_retention"] >= _RET_GUARD and m["v_recovery"] >= _REC_BAR]
    print(f"\n=== Phase I: candidate(VOFF_ret>={_RET_GUARD} & V_rec>={_REC_BAR})===")
    if elig:
        best = min(elig, key=lambda r: r[0])   # 最小 adapter
        note = "PASS"
    else:
        best = max(results, key=lambda r: (r[2]["voff_retention"] >= _RET_GUARD, r[2]["v_recovery"]))
        note = "bar 未達→retention 満たす中で V_rec 最大を暫定"
    a, model, m, ep = best
    _export_combined(model, _HERE / "residual_adapter_policy.json", a)
    print(f"  selected A{a}(ep{ep}) {note}: V_rec={m['v_recovery']:.3f} VOFF_ret={m['voff_retention']:.3f}")
    print(f"\n=== Phase J: historical comparison ===")
    print(f"  {'model':22s} {'V_rec':>7} {'UNCHG_ret':>10} {'online':>8}")
    print(f"  {'v2.7 shared H32':22s} {'0.17-0.22':>7} {'0.76-0.84':>10} {'~0.50':>8}")
    print(f"  {'v2.10 Adapter A'+str(a):22s} {m['v_recovery']:>7.3f} {m['unchanged_retention']:>10.3f} {'(H2H)':>8}")
    print(f"  {'Oracle V':22s} {'1.00':>7} {'1.00':>10} {'0.807':>8}")
    print(f"\nsaved residual_adapter_policy.json (A{a}) + residual_adapter_a{{4,8,16}}.json")
    print(f"  orig sha after = {hashlib.sha1(_ORIG_JSON.read_bytes()).hexdigest()[:12]} (== before {orig_sha_before}: {hashlib.sha1(_ORIG_JSON.read_bytes()).hexdigest()[:12]==orig_sha_before})")


if __name__ == "__main__":
    main(dataset=sys.argv[1] if len(sys.argv) > 1 else "search_root_dataset.npz")
