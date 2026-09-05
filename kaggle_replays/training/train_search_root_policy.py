"""ISMCTS v2.5 Phase E/F/G — Search-to-Root Policy distillation。

search_root_dataset.npz(X=239次元入力 h、visit=Search visit N(a)、oscore=Original Policy score、changed 等)を読み、
**Original Policy 735dd38a と同一 architecture(H32)の student** を Original 重みで初期化し、
**KL(π_search || softmax(student))**(π_search=visit 分布、τ=1.0、weighting なし)で蒸留。

offline(Phase G): overall Search top1 agreement / KL / **CHANGED recovery(最重要)** / Original-stuck / UNCHANGED retention /
confidence bin 別 / context(turn 帯)別。export=search_distilled_policy.json(Original の standardization+embedding 再利用)。
`python train_search_root_policy.py [dataset.npz]`
"""
from __future__ import annotations

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

_TEMPERATURE = 1.0
_EPOCHS = 40
_LR = 1e-3            # Original init からの微調整ゆえ小さめ
_BATCH_GROUPS = 256
_SEED = 0


def _groups(group, split, which):
    idx = np.nonzero(split == which)[0]
    by = defaultdict(list)
    for i in idx:
        by[int(group[i])].append(i)
    return list(by.values())


class Student(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def _init_from_original(model):
    """Original Policy 735dd38a の layers で student を初期化(人間模倣能力を保持)。"""
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    W1 = torch.tensor(o["layers"][0]["weight"], dtype=torch.float32)
    b1 = torch.tensor(o["layers"][0]["bias"], dtype=torch.float32)
    W2 = torch.tensor(o["layers"][1]["weight"], dtype=torch.float32)
    b2 = torch.tensor(o["layers"][1]["bias"], dtype=torch.float32)
    with torch.no_grad():
        model.fc1.weight.copy_(W1); model.fc1.bias.copy_(b1)
        model.fc2.weight.copy_(W2); model.fc2.bias.copy_(b2)
    return model, (W1.shape[0], W1.shape[1])


def _pad(groups, X, visit, device):
    B = len(groups); M = max(len(g) for g in groups); D = X.shape[1]
    Xp = torch.zeros(B, M, D); vis = torch.zeros(B, M); mask = torch.zeros(B, M)
    for b, g in enumerate(groups):
        n = len(g)
        Xp[b, :n] = torch.from_numpy(X[g]); vis[b, :n] = torch.from_numpy(visit[g].astype(np.float32)); mask[b, :n] = 1
    pv = vis / vis.sum(dim=1, keepdim=True).clamp_min(1e-9)      # π_search
    return Xp.to(device), pv.to(device), mask.to(device)


def _logsm(sc, mask):
    return torch.log_softmax(sc.masked_fill(mask == 0, -1e9), dim=1)


def _recovery_metrics(model, groups, X, visit, oscore, device):
    """overall top1 / KL / CHANGED recovery / Original-stuck / UNCHANGED retention。"""
    model.eval()
    ov = kl = n = 0
    ch_tot = ch_rec = ch_stuck = un_tot = un_ret = 0
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH_GROUPS):
            gb = groups[s:s + _BATCH_GROUPS]
            Xp, pv, mask = _pad(gb, X, visit, device)
            sc = model(Xp); logp = _logsm(sc, mask)
            kl += (pv * (torch.log(pv.clamp_min(1e-9)) - logp)).sum().item()
            s_arg = sc.masked_fill(mask == 0, -1e9).argmax(1)
            v_arg = pv.argmax(1)
            for bi, g in enumerate(gb):
                sa = int(s_arg[bi]); va = int(v_arg[bi])
                osc = oscore[g]; oa = int(np.argmax(osc))
                ov += int(sa == va); n += 1
                if oa != va:      # CHANGED(Search が Original を変えた)
                    ch_tot += 1
                    ch_rec += int(sa == va)      # student が Search を回収
                    ch_stuck += int(sa == oa)    # student が Original に留まる
                else:             # UNCHANGED
                    un_tot += 1
                    un_ret += int(sa == va)
    return {"overall": ov / n, "kl": kl / n,
            "changed_recovery": ch_rec / ch_tot if ch_tot else float("nan"),
            "original_stuck": ch_stuck / ch_tot if ch_tot else float("nan"),
            "unchanged_retention": un_ret / un_tot if un_tot else float("nan"),
            "changed_n": ch_tot, "unchanged_n": un_tot}


def _export(model, out):
    o = json.loads(_ORIG_JSON.read_text(encoding="utf-8"))
    o["meta"]["distilled_from_search"] = "v2.4 H4 ISMCTS @128 iters (visit distribution)"
    o["meta"]["temperature"] = _TEMPERATURE
    o["layers"] = [
        {"weight": model.fc1.weight.detach().cpu().numpy().tolist(), "bias": model.fc1.bias.detach().cpu().numpy().tolist()},
        {"weight": model.fc2.weight.detach().cpu().numpy().tolist(), "bias": model.fc2.bias.detach().cpu().numpy().tolist()},
    ]
    out.write_text(json.dumps(o), encoding="utf-8")


def main(dataset="search_root_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(_HERE / dataset)
    X, visit, oscore, group, split, turn = d["X"], d["visit"], d["oscore"], d["group"], d["split"], d["turn"]
    print(f"dataset: rows={len(X)} h_dim={X.shape[1]} states={int(group.max())+1} device={device}")
    tr = _groups(group, split, 0); va = _groups(group, split, 1); te = _groups(group, split, 2)
    print(f"groups train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    model = Student(X.shape[1], 32).to(device)
    model, (H, D) = _init_from_original(model)
    print(f"student init from Original 735dd38a: hidden={H} in={D}")
    base = _recovery_metrics(model, te, X, visit, oscore, device)   # Original(蒸留前)= baseline
    print(f"  [Original init] test overall={base['overall']:.3f} KL={base['kl']:.4f} "
          f"CHANGED_recovery={base['changed_recovery']:.3f}(n={base['changed_n']}) "
          f"UNCHANGED_ret={base['unchanged_retention']:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    rng = np.random.default_rng(_SEED)
    import copy as _copy
    best_val_kl = float("inf"); best_state = _copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(_EPOCHS):
        model.train(); order = rng.permutation(len(tr))
        for s in range(0, len(order), _BATCH_GROUPS):
            gb = [tr[i] for i in order[s:s + _BATCH_GROUPS]]
            Xp, pv, mask = _pad(gb, X, visit, device)
            logp = _logsm(model(Xp), mask)
            loss = (pv * (torch.log(pv.clamp_min(1e-9)) - logp)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m = _recovery_metrics(model, va, X, visit, oscore, device)   # 毎 epoch val KL(early stop 用)
        if m["kl"] < best_val_kl:
            best_val_kl = m["kl"]; best_state = _copy.deepcopy(model.state_dict()); best_ep = ep + 1
        if (ep + 1) % 10 == 0:
            print(f"    ep{ep+1}: val overall={m['overall']:.3f} KL={m['kl']:.4f} "
                  f"CHANGED_rec={m['changed_recovery']:.3f} UNCHANGED_ret={m['unchanged_retention']:.3f}", flush=True)

    # pre-reg(Phase F3): validation KL 最良の checkpoint を採用(online 非依存)。
    model.load_state_dict(best_state)
    print(f"  best-val-KL checkpoint: ep{best_ep} (val KL={best_val_kl:.4f})")
    fin = _recovery_metrics(model, te, X, visit, oscore, device)
    out = _HERE / "search_distilled_policy.json"; _export(model, out)
    print(f"\n=== TEST(distilled)===")
    print(f"  overall Search agreement = {fin['overall']:.3f}  KL = {fin['kl']:.4f}")
    print(f"  ** CHANGED recovery = {fin['changed_recovery']:.3f} ** (n={fin['changed_n']})  "
          f"[Original init は {base['changed_recovery']:.3f}]")
    print(f"  Original-stuck(CHANGED で Original のまま)= {fin['original_stuck']:.3f}")
    print(f"  UNCHANGED retention = {fin['unchanged_retention']:.3f}  [Original init は {base['unchanged_retention']:.3f}]")
    # turn 帯別 CHANGED recovery
    for lo, hi, lab in [(0, 6, "early(t<6)"), (6, 11, "mid(6-10)"), (11, 99, "late(11+)")]:
        gsel = [g for g in te if lo <= int(turn[g[0]]) < hi]
        if gsel:
            mm = _recovery_metrics(model, gsel, X, visit, oscore, device)
            print(f"      {lab:12s} n={len(gsel):4d} overall={mm['overall']:.3f} "
                  f"CHANGED_rec={mm['changed_recovery']:.3f}(n={mm['changed_n']})")
    print(f"\nsaved {out}")
    print("判定: CHANGED recovery が高い(Original-stuck を大きく上回る)= Search 改善が weight へ移った兆候。")
    print("      ただし online 転移は Phase I(Policy-only H2H)で判定(offline だけで採用しない=v2.2 教訓)。")


if __name__ == "__main__":
    main(dataset=sys.argv[1] if len(sys.argv) > 1 else "search_root_dataset.npz")
