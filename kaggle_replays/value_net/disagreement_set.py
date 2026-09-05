"""Phase11B: Q0 と QT が **異なる行動を選ぶ**局面を固定する。

同じ行動を選ぶ局面では実力差が測れないため、`Q0 top-1 != QT top-1` を抽出する。
選択条件は **モデルの不一致だけ**(教師と一致する局面を選り好みしない §6.2)。
"""
from __future__ import annotations

import argparse, json, statistics, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train_action_q as T  # noqa: E402
from action_q import ActionQNet, CandidateSetQNet  # noqa: E402
from eval_action_q_v2 import load  # noqa: E402


def build(ck):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    kind = d["kind"]
    m = (ActionQNet(d["state_dim"], d["option_dim"], use_cards=False)
         if kind == "pool" else
         CandidateSetQNet(d["state_dim"], d["option_dim"], mode=kind.split(":", 1)[1]))
    m.load_state_dict(d["state_dict"])
    m.eval()
    return m, np.asarray(d["mean"]), np.asarray(d["std"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(_HERE.parent / "_actionq_v2.jsonl.gz"))
    ap.add_argument("--rel", default=str(_HERE.parent / "_actionq_v2_reliability.json"))
    ap.add_argument("--frozen", default=str(_HERE / "frozen"))
    ap.add_argument("--out", default=str(_HERE / "disagreement_set.json"))
    args = ap.parse_args()

    v2 = load(args.v2)
    te = [g for g in v2 if g["split"] == 2]
    rel = json.loads(Path(args.rel).read_text(encoding="utf-8"))
    tier = {r["group_id"]: r["tier"] for r in rel["records"]}

    F = Path(args.frozen)
    models = {n: build(F / f"{n}.pt") for n in ("Q0-old", "Q0-expanded", "QT", "QT-identity")}

    rows = []
    for g in te:
        sc = {}
        for n, (m, sm, ss) in models.items():
            sc[n] = T.score_groups(m, [g], sm, ss)[0]
        pol = [c["policy_score"] for c in g["candidates"]]
        tb = [c["teacher_B"]["mean"] for c in g["candidates"]]
        ta = [c["teacher_A"]["mean"] for c in g["candidates"]]
        top = lambda v: max(range(len(v)), key=lambda i: v[i])  # noqa: E731
        q0i, qti, poli, tbi, tai = (top(sc["Q0-old"]), top(sc["QT"]), top(pol),
                                    top(tb), top(ta))
        rows.append({
            "group_id": g["group_id"], "game": g["game"], "turn": g["turn"],
            "turn_band": g["turn_band"], "cand_band": g["cand_band"],
            "arch": g["opponent_archetype"], "n_cands": len(g["candidates"]),
            "tier": tier.get(g["group_id"], "other"),
            "q0_top1": q0i, "qt_top1": qti, "policy_top1": poli,
            "teacherB_top1": tbi, "teacherA_top1": tai,
            "disagree_q0_qt": q0i != qti,
            "q0_margin": round(sorted(sc["Q0-old"], reverse=True)[0]
                               - sorted(sc["Q0-old"], reverse=True)[1], 5)
            if len(sc["Q0-old"]) > 1 else 0.0,
            "teacherB_value_q0": round(tb[q0i], 6),
            "teacherB_value_qt": round(tb[qti], 6),
            "teacherB_best": round(max(tb), 6),
        })

    dis = [r for r in rows if r["disagree_q0_qt"]]
    # 不一致パターン(§6.4)
    pat = Counter()
    for r in dis:
        t, a, b, p = r["teacherB_top1"], r["q0_top1"], r["qt_top1"], r["policy_top1"]
        if t == a:
            pat["teacher=Q0≠QT"] += 1
        elif t == b:
            pat["teacher=QT≠Q0"] += 1
        else:
            pat["teacher≠both"] += 1
        if a == p and a != b:
            pat["Q0=Policy≠QT"] += 1
        if b == p and a != b:
            pat["QT=Policy≠Q0"] += 1
        if len({t, a, b}) == 3:
            pat["all_three_differ"] += 1

    # teacher_B 値での直接比較(選択の良し悪し。teacher_A は層定義専用なので使わない)
    q0v = [r["teacherB_value_q0"] for r in dis]
    qtv = [r["teacherB_value_qt"] for r in dis]
    q0_win = sum(1 for a, b in zip(q0v, qtv) if a > b)
    qt_win = sum(1 for a, b in zip(q0v, qtv) if b > a)
    tie = len(dis) - q0_win - qt_win

    out = {
        "test_groups": len(rows), "disagreement_groups": len(dis),
        "disagreement_rate": round(len(dis) / len(rows), 4),
        "patterns": dict(pat.most_common()),
        "teacherB_on_disagreements": {
            "Q0_choice_better": q0_win, "QT_choice_better": qt_win, "tie": tie,
            "Q0_win_rate": round(q0_win / max(1, q0_win + qt_win), 4),
            "mean_value_diff_Q0_minus_QT": round(
                statistics.mean([a - b for a, b in zip(q0v, qtv)]), 6),
            "note": "これは teacher_B 基準の比較。独立長期評価ではない。",
        },
        "by_turn_band": dict(Counter(r["turn_band"] for r in dis)),
        "by_cand_band": dict(Counter(r["cand_band"] for r in dis)),
        "by_arch": dict(Counter(r["arch"] for r in dis)),
        "by_tier": dict(Counter(r["tier"] for r in dis)),
        "mean_cands": round(statistics.mean([r["n_cands"] for r in dis]), 2) if dis else None,
        "rows": dis,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
