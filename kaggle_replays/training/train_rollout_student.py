"""ISMCTS v2.4 Phase D/E/F — small rollout policy の distillation training(teacher=Policy 735dd38a)。

rollout_dataset.npz(X=239次元標準化入力 h、teacher raw score、group、split)を読み、
各 group(1 state の legal options)で teacher softmax(T=1.0、_priors と同 semantics)を
student softmax へ **KL 蒸留**。student は 2 層 MLP(hidden H)。H ∈ {4,8,16} を学習し、
teacher の standardization + card_embedding を再利用して PolicyModel 形式 JSON へ export。

offline eval(Phase F): test split で top-1 / top-3 agreement・KL・rollout-step 別・seltype 別。
pre-registered: T=1.0 / loss=KL(teacher||student) / optimizer Adam / no sample weighting。
`python train_rollout_student.py [dataset.npz]`
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
_TEACHER_JSON = _SUB / "ptcg_ai" / "learning" / "policy_weights.json"

_TEMPERATURE = 1.0        # pre-registered(_priors の softmax(scores) と同一 semantics)
_HIDDENS = (4, 8, 16)
_EPOCHS = 60
_LR = 3e-3
_BATCH_GROUPS = 512
_SEED = 0


def _load_dataset(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _group_index(group, split, which):
    """split==which の group ごとに (row_indices) を返す。group は連続 int 前提でなくてよい。"""
    mask = split == which
    idx = np.nonzero(mask)[0]
    by = defaultdict(list)
    for i in idx:
        by[int(group[i])].append(i)
    return list(by.values())


class Student(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def _pad_batch(groups, X, scores, device):
    """group リスト → padded tensors。返り: (Xpad[B,M,D], teacher_p[B,M], mask[B,M])。"""
    B = len(groups)
    M = max(len(g) for g in groups)
    D = X.shape[1]
    Xpad = torch.zeros(B, M, D)
    tlogit = torch.full((B, M), -1e9)
    mask = torch.zeros(B, M)
    for b, g in enumerate(groups):
        n = len(g)
        Xpad[b, :n] = torch.from_numpy(X[g])
        tlogit[b, :n] = torch.from_numpy(scores[g] / _TEMPERATURE)
        mask[b, :n] = 1.0
    teacher_p = torch.softmax(tlogit, dim=1)
    return Xpad.to(device), teacher_p.to(device), mask.to(device)


def _masked_logsoftmax(scores, mask):
    scores = scores.masked_fill(mask == 0, -1e9)
    return torch.log_softmax(scores, dim=1)


def _eval(model, groups, X, scores, device):
    """top-1 / top-3 agreement・KL を返す。"""
    model.eval()
    top1 = top3 = kl = ntot = 0
    with torch.no_grad():
        for s in range(0, len(groups), _BATCH_GROUPS):
            gb = groups[s:s + _BATCH_GROUPS]
            Xpad, tp, mask = _pad_batch(gb, X, scores, device)
            sc = model(Xpad)
            logp = _masked_logsoftmax(sc, mask)
            sp = logp.exp()
            kl += (tp * (torch.log(tp.clamp_min(1e-9)) - logp)).sum().item()
            # argmax(masked)
            sc_m = sc.masked_fill(mask == 0, -1e9)
            t_logit = torch.where(tp > 0, torch.log(tp.clamp_min(1e-9)), torch.full_like(tp, -1e9))
            s_arg = sc_m.argmax(dim=1); t_arg = t_logit.argmax(dim=1)
            top1 += (s_arg == t_arg).sum().item()
            # top-3 of teacher contains student argmax
            t3 = t_logit.topk(min(3, t_logit.shape[1]), dim=1).indices
            top3 += (t3 == s_arg.unsqueeze(1)).any(dim=1).sum().item()
            ntot += len(gb)
    return top1 / ntot, top3 / ntot, kl / ntot


def _train_one(hidden, tr_groups, X, scores, va_groups, device):
    torch.manual_seed(_SEED)
    model = Student(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    rng = np.random.default_rng(_SEED)
    for ep in range(_EPOCHS):
        model.train()
        order = rng.permutation(len(tr_groups))
        for s in range(0, len(order), _BATCH_GROUPS):
            gb = [tr_groups[i] for i in order[s:s + _BATCH_GROUPS]]
            Xpad, tp, mask = _pad_batch(gb, X, scores, device)
            logp = _masked_logsoftmax(model(Xpad), mask)
            loss = (tp * (torch.log(tp.clamp_min(1e-9)) - logp)).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            a1, a3, kl = _eval(model, va_groups, X, scores, device)
            print(f"    H{hidden} ep{ep+1}: val top1={a1:.3f} top3={a3:.3f} KL={kl:.4f}", flush=True)
    return model


def _export(model, hidden, out_path):
    """teacher の standardization + card_embedding を再利用し、student layers を JSON 化。"""
    teacher = json.loads(_TEACHER_JSON.read_text(encoding="utf-8"))
    w1 = model.fc1.weight.detach().cpu().numpy().tolist()
    b1 = model.fc1.bias.detach().cpu().numpy().tolist()
    w2 = model.fc2.weight.detach().cpu().numpy().tolist()
    b2 = model.fc2.bias.detach().cpu().numpy().tolist()
    payload = {
        "meta": {
            "state_feature_count": teacher["meta"]["state_feature_count"],
            "option_feature_count": teacher["meta"]["option_feature_count"],
            "distilled_from": "policy_weights.json(735dd38a)",
            "student_hidden": hidden, "temperature": _TEMPERATURE, "consequence_fields": None,
        },
        "standardization": teacher["standardization"],
        "card_embedding": teacher["card_embedding"],
        "layers": [{"weight": w1, "bias": b1}, {"weight": w2, "bias": b2}],
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def main(dataset="rollout_dataset.npz"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = _load_dataset(_HERE / dataset)
    X, scores, group, split, step, seltype = d["X"], d["score"], d["group"], d["split"], d["step"], d["seltype"]
    print(f"dataset: rows={len(X)} h_dim={X.shape[1]} states={int(group.max())+1} device={device}")
    tr = _group_index(group, split, 0); va = _group_index(group, split, 1); te = _group_index(group, split, 2)
    print(f"groups train/val/test = {len(tr)}/{len(va)}/{len(te)}\n")

    # teacher self-agreement baseline は 1.0(定義上)。ここでは student を学習・評価。
    results = []
    for H in _HIDDENS:
        print(f"=== distill Student H{H} ===")
        model = _train_one(H, tr, X, scores, va, device)
        a1, a3, kl = _eval(model, te, X, scores, device)
        out = _HERE / f"rollout_student_h{H}.json"
        _export(model, H, out)
        # NN forward MACs(layer0 支配): H*239
        macs = H * X.shape[1]
        results.append((H, a1, a3, kl, macs, out.name))
        print(f"  H{H}: TEST top1={a1:.3f} top3={a3:.3f} KL={kl:.4f}  layer0 MACs={macs} (teacher {32*X.shape[1]})")
        # rollout-step 別 agreement(deployment distribution 確認)
        for lo, hi, lab in [(0, 1, "step0(自ターン開始)"), (1, 4, "step1-3"), (4, 99, "step4+相手応答含む")]:
            gsel = [g for g in te if lo <= int(step[g[0]]) < hi]
            if gsel:
                b1, b3, bkl = _eval(model, gsel, X, scores, device)
                print(f"      {lab:22s} n={len(gsel):5d} top1={b1:.3f} KL={bkl:.4f}")
        print(flush=True)

    print("=== summary(offline Pareto: top1 agreement vs NN cost)===")
    print(f"  {'H':>3} {'top1':>6} {'top3':>6} {'KL':>7} {'layer0 MACs':>12} {'vs teacher':>11}")
    for H, a1, a3, kl, macs, name in results:
        print(f"  {H:>3} {a1:>6.3f} {a3:>6.3f} {kl:>7.4f} {macs:>12} {32*X.shape[1]/macs:>9.1f}x")
    print("\n注: rollout は argmax 使用ゆえ top1 agreement が主。online strength は Phase L(H2H)で判定(offline だけで採用しない)。")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "rollout_dataset.npz"
    main(dataset=ds)
