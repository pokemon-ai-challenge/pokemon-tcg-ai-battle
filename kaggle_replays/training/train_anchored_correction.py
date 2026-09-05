"""ISMCTS v2.6 — Anchored Search-Correction Distillation。

v2.5 の signal dilution(全 state を Search visit へ KL → UNCHANGED 破壊 + CHANGED 回収 15%)への対処。
**全 state で Original をアンカー(KL(π_original‖student))+ CHANGED state のみ Search correction(KL(π_search‖student))を
λ_changed 倍で追加**。dataset/split/architecture(H32)/Original 初期化/Teacher は v2.5 と同一、変更は objective/weighting のみ。

L = L_anchor + λ * I_CHANGED * L_search   (λ ∈ {1,2,4})

Phase A(audit)+ Phase C-G(λ sweep + offline)+ candidate 選定(retention>=0.95 & CHANGED recovery>0.150)。
export=anchored_correction_policy.json(Original の standardization+embedding 再利用)。
`python train_anchored_correction.py [dataset.npz]`
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

_LAMBDAS = (1.0, 2.0, 4.0)
_EPOCHS = 40
_LR = 1e-3
_BATCH = 256
_SEED = 0
_RETENTION_GUARD = 0.95        # pre-registered
_CHANGED_BASELINE = 0.150      # v2.5


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


def _pad(groups, X, visit, oscore, changed, device):
    B = len(groups); M = max(len(g) for g in groups); D = X.shape[1]
    Xp = torch.zeros(B, M, D); vis = torch.zeros(B, M); osc = torch.full((B, M), -1e9); mask = torch.zeros(B, M)
    chg = torch.zeros(B)
    for b, g in enumerate(groups):
        n = len(g)
        Xp[b, :n] = torch.from_numpy(X[g]); vis[b, :n] = torch.from_numpy(visit[g].astype(np.float32))
        osc[b, :n] = torch.from_numpy(oscore[g]); mask[b, :n] = 1; chg[b] = float(changed[g[0]])
    p_search = vis / vis.sum(1, keepdim=True).clamp_min(1e-9)
    p_orig = torch.softmax(osc, dim=1)         # π_original = softmax(Original score)
    return (Xp.to(device), p_search.to(device), p_orig.to(device), mask.to(device), chg.to(device))


def _logsm(sc, mask):
    return torch.log_softmax(sc.masked_fill(mask == 0, -1e9), dim=1)


def _kl(p, logq, mask):
    return (p * (torch.log(p.clamp_min(1e-9)) - logq) * mask).sum(1)


def _eval(model, groups, X, visit, oscore, changed, top1share, gap, device):
    """overall/KL/CHANGED recovery/Original-stuck/UNCHANGED retention + confidence 別 + confusion。"""
    model.eval()
    ov = kl_s = kl_o = n = 0
    ch_tot = ch_rec = ch_stuck = ch_third = 0
    un_tot = un_ret = un_dmg = 0
    conf = {"high": [0, 0], "med": [0, 0], "low": [0, 0]}   # [recovered, total] on CHANGED
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH):
            gb = groups[s:s + _BATCH]
            Xp, ps, po, mask, chg = _pad(gb, X, visit, oscore, changed, device)
            sc = model(Xp); logq = _logsm(sc, mask)
            kl_s += _kl(ps, logq, mask).sum().item(); kl_o += _kl(po, logq, mask).sum().item()
            s_arg = sc.masked_fill(mask == 0, -1e9).argmax(1)
            v_arg = ps.argmax(1); o_arg = po.argmax(1)
            for bi, g in enumerate(gb):
                sa = int(s_arg[bi]); va = int(v_arg[bi]); oa = int(o_arg[bi]); n += 1
                ov += int(sa == va)
                if int(chg[bi]) == 1:
                    ch_tot += 1
                    ch_rec += int(sa == va); ch_stuck += int(sa == oa)
                    ch_third += int(sa != va and sa != oa)
                    ts = float(top1share[g[0]]); gp = float(gap[g[0]])
                    b = "high" if (ts >= 0.5 and gp >= 0.2) else ("low" if ts < 0.35 else "med")
                    conf[b][1] += 1; conf[b][0] += int(sa == va)
                else:
                    un_tot += 1; un_ret += int(sa == va); un_dmg += int(sa != oa)
    def r(a, b): return a / b if b else float("nan")
    return {"overall": r(ov, n), "kl_search": r(kl_s, n), "kl_orig": r(kl_o, n),
            "changed_recovery": r(ch_rec, ch_tot), "original_stuck": r(ch_stuck, ch_tot),
            "changed_third": r(ch_third, ch_tot), "unchanged_retention": r(un_ret, un_tot),
            "unchanged_damage": r(un_dmg, un_tot), "changed_n": ch_tot, "unchanged_n": un_tot,
            "conf": {k: r(v[0], v[1]) for k, v in conf.items()}, "conf_n": {k: v[1] for k, v in conf.items()}}


def _export(model, out, lam):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    o["meta"]["anchored_correction"] = {"lambda_changed": lam, "anchor": "KL to Original", "correction": "KL to Search on CHANGED"}
    o["layers"] = [
        {"weight": model.fc1.weight.detach().cpu().numpy().tolist(), "bias": model.fc1.bias.detach().cpu().numpy().tolist()},
        {"weight": model.fc2.weight.detach().cpu().numpy().tolist(), "bias": model.fc2.bias.detach().cpu().numpy().tolist()},
    ]
    out.write_text(json.dumps(o), encoding="utf-8")


def _train(lam, tr, X, visit, oscore, changed, va, top1share, gap, device):
    torch.manual_seed(_SEED)
    model = _init_original(Student(X.shape[1], 32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    rng = np.random.default_rng(_SEED)
    # pre-registered offline score = CHANGED recovery、ただし retention>=guard の epoch のみ候補
    best_score = -1.0; best_state = copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(_EPOCHS):
        model.train(); order = rng.permutation(len(tr))
        for s in range(0, len(order), _BATCH):
            gb = [tr[i] for i in order[s:s + _BATCH]]
            Xp, ps, po, mask, chg = _pad(gb, X, visit, oscore, changed, device)
            logq = _logsm(model(Xp), mask)
            l_anchor = _kl(po, logq, mask)                       # 全 state
            l_search = _kl(ps, logq, mask) * chg                 # CHANGED のみ
            loss = (l_anchor + lam * l_search).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m = _eval(model, va, X, visit, oscore, changed, top1share, gap, device)
        score = m["changed_recovery"] if m["unchanged_retention"] >= _RETENTION_GUARD else -1.0
        if score > best_score:
            best_score = score; best_state = copy.deepcopy(model.state_dict()); best_ep = ep + 1
    final_state = copy.deepcopy(model.state_dict())      # 最終 epoch(安定)= diagnostic 用
    model.load_state_dict(best_state)
    return model, best_ep, final_state


def main(dataset="search_root_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(_HERE / dataset)
    X, visit, oscore, group, split = d["X"], d["visit"], d["oscore"], d["group"], d["split"]
    changed, top1share, gap, entropy = d["changed"], d["top1share"], d["gap"], d["entropy"]
    tr = _groups(group, split, 0); va = _groups(group, split, 1); te = _groups(group, split, 2)

    # ---- Phase A: v2.5 failure audit ----
    firsts = {}
    for j, g in enumerate(group):
        firsts.setdefault(int(g), j)
    fi = list(firsts.values())
    ch = np.asarray([changed[j] for j in fi]); ts = np.asarray([top1share[j] for j in fi])
    gp = np.asarray([gap[j] for j in fi]); en = np.asarray([entropy[j] for j in fi])
    hi = (ts >= 0.5) & (gp >= 0.2)
    print("=== Phase A: dataset audit ===")
    print(f"  roots={len(fi)}  CHANGED={int(ch.sum())} ({ch.mean():.1%})  UNCHANGED={int((ch==0).sum())}")
    print(f"  high-conf CHANGED={int((hi & (ch==1)).sum())} ({(hi & (ch==1)).mean():.1%} of all)")
    print(f"  visit entropy mean={en.mean():.3f}  top1 share mean={ts.mean():.3f}  gap mean={gp.mean():.3f}")
    print(f"  CHANGED の visit share mean={ts[ch==1].mean():.3f}  gap mean={gp[ch==1].mean():.3f}\n")

    print("=== Phase C-G: λ sweep(anchor + CHANGED correction)===")
    print(f"  {'λ':>4} {'overall':>8} {'CHG_rec':>8} {'orig_stuck':>10} {'UNCHG_ret':>10} "
          f"{'hi_rec':>7} {'med_rec':>7} {'lo_rec':>7} {'KL_srch':>8} {'ep*':>4}")
    results = []
    for lam in _LAMBDAS:
        model, ep, final_state = _train(lam, tr, X, visit, oscore, changed, va, top1share, gap, device)
        m = _eval(model, te, X, visit, oscore, changed, top1share, gap, device)
        results.append((lam, model, m, ep))
        _export(model, _HERE / f"anchored_l{int(lam)}.json", lam)   # best-checkpoint(pre-reg 選定)
        fin_model = _init_original(Student(X.shape[1], 32).to(device)); fin_model.load_state_dict(final_state)
        _export(fin_model, _HERE / f"anchored_l{int(lam)}_final.json", lam)   # 最終 epoch(安定, diagnostic)
        c = m["conf"]
        print(f"  {lam:>4.0f} {m['overall']:>8.3f} {m['changed_recovery']:>8.3f} {m['original_stuck']:>10.3f} "
              f"{m['unchanged_retention']:>10.3f} {c['high']:>7.3f} {c['med']:>7.3f} {c['low']:>7.3f} "
              f"{m['kl_search']:>8.4f} {ep:>4d}")

    # ---- Phase F: candidate selection(retention guard + CHANGED recovery > baseline)----
    elig = [(lam, model, m, ep) for (lam, model, m, ep) in results
            if m["unchanged_retention"] >= _RETENTION_GUARD and m["changed_recovery"] > _CHANGED_BASELINE]
    print(f"\n=== Phase F: candidate 選定(retention>={_RETENTION_GUARD} & CHANGED_rec>{_CHANGED_BASELINE})===")
    if not elig:
        print("  pre-reg bar(retention>=guard & CHANGED_rec>baseline)を満たす候補なし = offline Case D。")
        print("  online 確認用に『学習した最大 correction』候補(retention>=0.90 & CHANGED_rec>0)を選定。")
        learned = [(lam, model, m, ep) for (lam, model, m, ep) in results
                   if m["unchanged_retention"] >= 0.90 and m["changed_recovery"] > 0.0]
        elig = sorted(learned or results,
                      key=lambda r: r[2]["changed_recovery"], reverse=True)[:1]
    best = max(elig, key=lambda r: r[2]["changed_recovery"])
    lam, model, m, ep = best
    out = _HERE / "anchored_correction_policy.json"; _export(model, out, lam)
    print(f"  selected λ={lam:.0f} (ep{ep})  CHANGED_rec={m['changed_recovery']:.3f} "
          f"UNCHANGED_ret={m['unchanged_retention']:.3f}")
    print(f"\n=== Phase E/G: correction transfer(selected λ={lam:.0f}, test)===")
    print(f"  CHANGED → Search {m['changed_recovery']:.3f} / Original {m['original_stuck']:.3f} / third {m['changed_third']:.3f}")
    print(f"  UNCHANGED retained {m['unchanged_retention']:.3f} / damaged {m['unchanged_damage']:.3f}")
    print(f"  conf recovery: high {m['conf']['high']:.3f}(n={m['conf_n']['high']}) "
          f"med {m['conf']['med']:.3f}(n={m['conf_n']['med']}) low {m['conf']['low']:.3f}(n={m['conf_n']['low']})")
    print(f"  vs v2.5: CHANGED_rec 0.150 → {m['changed_recovery']:.3f}, UNCHANGED_ret 0.879 → {m['unchanged_retention']:.3f}")
    print(f"\nsaved {out}")
    print("判定: retention 高持続 + CHANGED recovery 明確改善なら anchor が効いた。online は Phase K(Policy-only H2H)で判定。")


if __name__ == "__main__":
    main(dataset=sys.argv[1] if len(sys.argv) > 1 else "search_root_dataset.npz")
