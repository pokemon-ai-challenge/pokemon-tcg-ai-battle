"""Phase11C-2: 固定ラベル暗記テスト(学習基盤の健全性ゲート)。

拡張データの train pairwise が 0.67 前後で止まる原因が
「最適化・実装の問題」なのか「教師/入力の問題」なのかを分ける。

**固定した教師順位を暗記できるか**だけを見る。汎化は見ない。
  dropout なし / weight decay なし / early stopping なし / test 評価なし
PASS: train pairwise >= 0.95 / FAIL: < 0.90
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train_action_q as T  # noqa: E402
from eval_action_q_v2 import load  # noqa: E402


def memorize(groups, kind, epochs, lr, batch, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = np.asarray([g["state_feat"] for g in groups], dtype=np.float32)
    sm, ss = X.mean(0), X.std(0)
    ss[ss == 0] = 1.0
    D = len(groups[0]["candidates"][0]["option_feat"])
    m = T.build_model(kind, len(sm), D, use_cards=False, use_action_card=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr)     # weight_decay=0(既定)
    hist = []
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(len(groups))
        tot = 0.0
        nb = 0
        for i in range(0, len(perm), batch):
            gs = [groups[j] for j in perm[i:i + batch]]
            st, zones, of, ci, ot, mk, te, cf, sel, oc = T.make_batch(gs, sm, ss)
            q, _ = T._fwd(m, st, zones, of, ci, ot, mk)
            qi, qj = q.unsqueeze(2), q.unsqueeze(1)
            ti, tj = te.unsqueeze(2), te.unsqueeze(1)
            valid = (mk.unsqueeze(2) & mk.unsqueeze(1)) & (ti > tj)
            if not valid.any():
                continue
            loss = -torch.nn.functional.logsigmoid((qi - qj)[valid]).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
            opt.step()
            tot += float(loss)
            nb += 1
        if (ep + 1) % max(1, epochs // 8) == 0 or ep == epochs - 1:
            r = T.eval_ranker(groups, lambda g: T.score_groups(m, [g], sm, ss)[0])
            hist.append({"epoch": ep + 1, "loss": round(tot / max(1, nb), 5),
                         "train_pairwise": r["pairwise"],
                         "grad_norm": round(float(gn), 3)})
            print(f"    ep{ep+1} loss={tot/max(1,nb):.5f} train_pairwise={r['pairwise']}",
                  file=sys.stderr, flush=True)
    r = T.eval_ranker(groups, lambda g: T.score_groups(m, [g], sm, ss)[0])
    r.pop("_per_group", None)
    return r, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(_HERE.parent / "_actionq_v2.jsonl.gz"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(_HERE / "memorize_report.json"))
    args = ap.parse_args()

    v2 = load(args.v2)
    tr = [g for g in v2 if g["split"] == 0]
    rs = np.random.RandomState(0)
    sub = [tr[i] for i in rs.choice(len(tr), min(args.n, len(tr)), replace=False)]
    # ラベルは teacher_A(train は元々 teacher_A)を固定して使う
    print(f"memorize subset: {len(sub)} groups, "
          f"{sum(len(g['candidates']) for g in sub)} candidates", file=sys.stderr)

    out = {"n_groups": len(sub), "epochs": args.epochs, "lr": args.lr,
           "batch": args.batch, "arms": {}}
    for kind, name in (("pool", "Q0"), ("cset:attn", "QT")):
        t0 = time.time()
        r, hist = memorize(sub, kind, args.epochs, args.lr, args.batch)
        out["arms"][name] = {
            "final_train_pairwise": r["pairwise"], "spearman": r["spearman"],
            "top1": r["top1"], "history": hist, "sec": round(time.time() - t0, 1),
            "VERDICT": ("PASS" if (r["pairwise"] or 0) >= 0.95 else
                        "FAIL" if (r["pairwise"] or 0) < 0.90 else "MARGINAL")}
        print(f"  [{name}] final train pairwise={r['pairwise']} "
              f"=> {out['arms'][name]['VERDICT']}", file=sys.stderr, flush=True)
    out["GATE"] = ("PASS" if all(a["VERDICT"] == "PASS" for a in out["arms"].values())
                   else "FAIL" if any(a["VERDICT"] == "FAIL" for a in out["arms"].values())
                   else "MARGINAL")
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, ensure_ascii=False))
    for n, a in out["arms"].items():
        print(f"{n}: final={a['final_train_pairwise']} verdict={a['VERDICT']}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
