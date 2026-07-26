"""汎用 archetype offline evaluator(Phase B 再利用部品)。

--archetype <arch> で features_<arch>.npz の held-out test split に対し generic policy vs
archetype-specific BC の Top-1/NLL を比較(C1)。rank 分布・own-deck 依存(consequence_fields)も報告。
cg 不使用・production 非改変・read-only。既存 policy_net/evaluate.model_scores_per_row を再利用。

使用: python opponent_offline_eval.py --archetype archaludon_ex
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_PN = _ROOT / "kaggle_replays" / "policy_net"
_LEARN = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
sys.path.insert(0, str(_PN))
import evaluate as EV  # noqa: E402  (model_scores_per_row 再利用)


def _rbucket(r: int) -> str:
    if r < 0:
        return "unknown"
    if r <= 20:
        return "<=20"
    if r <= 50:
        return "21-50"
    if r <= 200:
        return "51-200"
    if r <= 1000:
        return "201-1000"
    return "1000+"


def _nll(scores, chosen):
    tot = 0.0
    for s, c in zip(scores, chosen):
        m = s.max()
        logZ = m + np.log(np.exp(s - m).sum())
        tot += -(s[c] - logZ)
    return tot / len(chosen)


def evaluate_archetype(arch: str) -> dict:
    feat = _PN / f"features_{arch}.npz"
    bc = _LEARN / f"policy_weights_{arch}.json"
    generic = _LEARN / "policy_weights.json"
    if not feat.exists() or not bc.exists():
        raise SystemExit(f"features or BC weights 不在: {feat.exists()=} {bc.exists()=}")

    data = np.load(feat, allow_pickle=True)
    split = data["split"].astype(np.int64)
    counts = {n: int((split == c).sum()) for n, c in (("train", 0), ("val", 1), ("test", 2))}
    test = split == 2
    state = data["state_features"][test]
    opt = data["option_features"][test]
    cids = data["option_card_ids"][test]
    chosen = data["chosen_index"][test].astype(np.int64)
    stype = data["select_type"][test].astype(np.int64)
    rank = data["rank_at_fetch"][test].astype(np.int64)

    rc = collections.Counter(_rbucket(int(r)) for r in rank)
    rank_dist = {k: rc.get(k, 0) for k in ["<=20", "21-50", "51-200", "201-1000", "1000+", "unknown"]}

    def run(wpath):
        w = json.load(open(wpath, encoding="utf-8"))
        scores = EV.model_scores_per_row(w, state, opt, cids)
        pred = np.array([int(np.argmax(s)) for s in scores])
        match = (pred == chosen).astype(np.float64)
        return match, _nll(scores, chosen), w

    g_match, g_nll, gw = run(generic)
    b_match, b_nll, bw = run(bc)
    nopt = np.array([o.shape[0] for o in opt], dtype=np.float64)

    by_stype = {}
    for st in sorted(set(stype.tolist())):
        m = stype == st
        by_stype[str(st)] = {"n": int(m.sum()), "generic": round(float(g_match[m].mean()), 4),
                             "bc": round(float(b_match[m].mean()), 4),
                             "delta": round(float(b_match[m].mean() - g_match[m].mean()), 4)}

    def _cf(w):
        return w.get("consequence_fields", w.get("meta", {}).get("consequence_fields"))

    return {
        "archetype": arch, "split_counts": counts, "test_n": int(test.sum()),
        "rank_dist_test": rank_dist,
        "offline_C1": {
            "generic_top1": round(float(g_match.mean()), 4), "bc_top1": round(float(b_match.mean()), 4),
            "delta_top1": round(float(b_match.mean() - g_match.mean()), 4),
            "generic_nll": round(g_nll, 4), "bc_nll": round(b_nll, 4),
            "uniform_random_top1": round(float((1.0 / nopt).mean()), 4),
        },
        "by_select_type": by_stype,
        "own_deck_safe": {"generic_consequence_fields": _cf(gw), "bc_consequence_fields": _cf(bw),
                          "policy_only_safe": (not _cf(gw)) and (not _cf(bw))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    r = evaluate_archetype(args.archetype)
    print(f"=== {r['archetype']} offline C1 ===")
    print(f"  split: {r['split_counts']}  test_n={r['test_n']}")
    print(f"  rank_dist(test): {r['rank_dist_test']}")
    c = r["offline_C1"]
    print(f"  generic top1={c['generic_top1']} nll={c['generic_nll']}")
    print(f"  BC      top1={c['bc_top1']} nll={c['bc_nll']}  (uniform {c['uniform_random_top1']})")
    print(f"  Δtop1 = {c['delta_top1']:+.4f}")
    print("  by select_type:")
    for st, g in r["by_select_type"].items():
        print(f"    type={st:2s} n={g['n']:5d} generic={g['generic']:.4f} bc={g['bc']:.4f} Δ={g['delta']:+.4f}")
    o = r["own_deck_safe"]
    print(f"  own-deck: generic_cf={o['generic_consequence_fields']} bc_cf={o['bc_consequence_fields']} "
          f"policy_only_safe={o['policy_only_safe']}")
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
