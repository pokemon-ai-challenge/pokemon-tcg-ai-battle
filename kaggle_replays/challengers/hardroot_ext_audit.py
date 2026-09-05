"""ISMCTS v2.13 Phase B/C/D — Hard-Root Extension offline audit(time / iteration 両モード)。

v2.11 batched H4 + v2.12 Conservative early-stop を **candidate と同一挙動**で self-play root 上に走らせ、
soft cap(既定 1350ms相当)で停止せず hard cap(3000ms相当)まで継続したときの action 変化・収束を測る。

  --time  (既定) : wall-time 基準。soft=1350ms / mid=2000ms / hard=3000ms。**マシン速度に依存**。
  --iter          : iteration 基準(machine-independent)。参照機の 1350ms≈206it / 3000ms≈458it で cap。
                    このマシンが参照機より遅くても iteration 予算が同じなので探索 regime が代表的。

hard root = soft cap までに v2.12 early-stop rule を満たさなかった root。
Phase B: soft/mid/hard の mean iterations / converged% / top1 share / gap。
Phase C: soft->hard action-change rate(hard roots)。
Phase D: convergence timing(<mid / mid-hard / unresolved@hard)。

使用: python hardroot_ext_audit.py [n_hard=100] [max_games=40] [--iter] [--soft-it 206 --hard-it 458]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import to_observation_class, SelectType  # noqa: E402
import ismcts, ismcts_v1_agent as A  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.search.leaf_eval import HandcraftedEvaluator  # noqa: E402
from batched_policy import BatchedPolicyModel  # noqa: E402

_CFG = A.load_config()
_H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_ES = {"enabled": True, "n_min": 64, "check_every": 16, "k_stable": 3, "share": 0.7, "gap": 0.4}
_COMMON = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1,
           "max_depth": 60, "iterations": 100000, "leaf_mode": "rollout", "early_stop": _ES}


def _final_share_gap(st):
    rc = st.root_children_full
    if not rc:
        return (0.0, 0.0)
    tot = sum(v for _a, v, _q, _p in rc) or 1
    n1 = rc[0][1]; n2 = rc[1][1] if len(rc) > 1 else 0
    return (round(n1 / tot, 4), round((n1 - n2) / tot, 4))


def _audit_root(obs, sel, model, ev, h4, mode, soft, mid, hard):
    """1 root を candidate 挙動で探索し、normalized record を返す(mode='time'|'iter')。"""
    det = A._determinize_factory(obs, _CFG)
    try:
        if det() is None:
            return None
    except Exception:
        return None
    cfg = dict(_COMMON)
    if mode == "time":
        cfg["soft_cap_ms"] = soft
        cfg["checkpoints_ms"] = [soft, mid, hard]
        deadline = time.perf_counter() + hard / 1000.0
        cfg["iterations"] = 100000
    else:  # iter mode: 時間は縛らず iteration で cap(参照機の予算)
        cfg["iterations"] = int(hard)
        cfg["checkpoint_iters"] = [int(soft), int(mid)]
        deadline = time.perf_counter() + 3600.0
    st = ismcts.SearchStats()
    a = ismcts.search(obs, cfg, model, ev, det, deadline, random.Random(0), st, rollout_policy=h4)
    if a is None:
        return None
    fshare, fgap = _final_share_gap(st)

    def _snap_from(cp):
        return None if cp is None else {"action": tuple(cp["action"]) if cp["action"] else None,
                                        "iter": cp["iter"], "share": cp["share"], "gap": cp["gap"]}
    hard_snap = {"action": tuple(a), "iter": st.iterations, "share": fshare, "gap": fgap}

    if mode == "time":
        cpm = {round(c["cap_ms"]): c for c in st.checkpoints}
        soft_snap = _snap_from(cpm.get(round(soft)))
        mid_snap = _snap_from(cpm.get(round(mid)))
        is_hard = st.is_hard_root
        stop_pos = st.elapsed_ms if st.early_stopped else None  # 収束位置(ms)
    else:
        cpi = {c["cap_iter"]: c for c in st.iter_checkpoints}
        soft_snap = _snap_from(cpi.get(int(soft)))
        mid_snap = _snap_from(cpi.get(int(mid)))
        # hard root = soft(iter)までに early-stop しなかった
        is_hard = not (st.early_stopped and st.stop_iter <= int(soft))
        stop_pos = st.stop_iter if st.early_stopped else None  # 収束位置(iter)
    return {
        "mode": mode, "n_options": len(sel.option), "is_hard_root": is_hard,
        "early_stopped": st.early_stopped, "stop_pos": stop_pos,
        "soft": soft_snap, "mid": mid_snap, "hard": hard_snap,
    }


def _load_deck():
    return [int(x) for x in (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")[:60]]


def main(n_hard, max_games, mode, soft, mid, hard):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB); match_context.reset()
    deck = _load_deck()
    model = mpa._get_model(_CFG); ev = HandcraftedEvaluator(); h4 = BatchedPolicyModel(weights_path=_H4)
    unit = "ms" if mode == "time" else "it"
    print(f"=== v2.13 Hard-Root Extension audit [{mode}] soft={soft}{unit} mid={mid}{unit} hard={hard}{unit} "
          f"target hard={n_hard} ===", flush=True)
    roots = []; hard_ct = 0; game = 0; t0 = time.perf_counter()
    while hard_ct < n_hard and game < max_games:
        game += 1; match_context.reset()
        od, sd = battle_start(deck, deck)
        if sd.errorType != 0:
            continue
        try:
            for _ in range(400):
                if hard_ct >= n_hard:
                    break
                obs = to_observation_class(od); cur = obs.current
                if cur is None or cur.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                try:
                    match_context.update(obs)
                except Exception:
                    pass
                if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
                    r = _audit_root(obs, sel, model, ev, h4, mode, soft, mid, hard)
                    if r is not None:
                        roots.append(r); hard_ct += int(r["is_hard_root"])
                        if len(roots) % 20 == 0:
                            print(f"  ... roots={len(roots)} hard={hard_ct} game={game} "
                                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
                factory = mpa._model_hidden_state_factory(obs, _CFG)
                if sel.maxCount == 1:
                    idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                    action = [idx if idx is not None else 0]
                else:
                    action = list(range(sel.minCount))
                od = battle_select(action)
        finally:
            battle_finish()
    _report(roots, mode, soft, mid, hard, time.perf_counter() - t0)


def _report(roots, mode, soft, mid, hard, elapsed_s):
    unit = "ms" if mode == "time" else "it"
    total = len(roots)
    if total == 0:
        print("収集失敗(root 0)"); return
    H = [r for r in roots if r["is_hard_root"]]
    nH = len(H)

    def _mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    def _col(key):
        """hard root ごとに soft/mid/hard snapshot(未記録なら hard=final で代用)。(iter,share,gap,action)。"""
        out = []
        for r in H:
            s = r.get(key) or r["hard"]   # checkpoint 未達(その前に停止)→ final で代用
            out.append((s["iter"], s["share"], s["gap"], s["action"]))
        return out
    soft_c, mid_c, hard_c = _col("soft"), _col("mid"), _col("hard")

    # convergence timing(hard roots): stop_pos(ms/iter)で分類
    def _conv(lo, hi):
        return sum(1 for r in H if r["early_stopped"] and r["stop_pos"] is not None and lo < r["stop_pos"] <= hi)
    conv_mid = sum(1 for r in H if r["early_stopped"] and r["stop_pos"] is not None and r["stop_pos"] <= mid)
    conv_mid_hard = _conv(mid, hard)
    unresolved = nH - conv_mid - conv_mid_hard

    # action change soft->hard(hard roots; soft は必ず記録済み=hard root 定義)
    chg = sum(1 for r in H if r["soft"] and r["soft"]["action"] is not None and r["soft"]["action"] != r["hard"]["action"])

    print(f"\n=== A. Hard-Root Audit [{mode}] (elapsed {elapsed_s:.0f}s)===")
    print(f"  total roots            : {total}")
    print(f"  hard roots(未収束@soft) : {nH}  (hard-root rate {nH/total*100:.1f}%)")
    print(f"  converged<soft(非hard)  : {total-nH}  ({(total-nH)/total*100:.1f}%)")

    print(f"\n=== B. Checkpoint table(hard roots, n={nH})===")
    print(f"  {'metric':12s} {f'@{soft}{unit}':>12} {f'@{mid}{unit}':>12} {f'@{hard}{unit}':>12}")
    print(f"  {'mean iters':12s} {_mean([x[0] for x in soft_c]):>12.0f} {_mean([x[0] for x in mid_c]):>12.0f} {_mean([x[0] for x in hard_c]):>12.0f}")
    print(f"  {'top1 share':12s} {_mean([x[1] for x in soft_c]):>12.3f} {_mean([x[1] for x in mid_c]):>12.3f} {_mean([x[1] for x in hard_c]):>12.3f}")
    print(f"  {'top1-2 gap':12s} {_mean([x[2] for x in soft_c]):>12.3f} {_mean([x[2] for x in mid_c]):>12.3f} {_mean([x[2] for x in hard_c]):>12.3f}")
    cvg_soft = (total-nH)/total*100
    cvg_mid = (total-nH+conv_mid)/total*100
    cvg_hard = (total-nH+conv_mid+conv_mid_hard)/total*100
    print(f"  {'converged%all':12s} {cvg_soft:>11.0f}% {cvg_mid:>11.0f}% {cvg_hard:>11.0f}%")

    print(f"\n=== C. Action change(hard roots)===")
    print(f"  soft -> hard action-change : {chg}/{nH} = {chg/nH*100 if nH else float('nan'):.1f}%")

    print(f"\n=== D. Convergence timing(hard roots, n={nH})===")
    if nH:
        print(f"  converge <{mid}{unit}        : {conv_mid}/{nH} = {conv_mid/nH*100:.1f}%")
        print(f"  converge {mid}-{hard}{unit}   : {conv_mid_hard}/{nH} = {conv_mid_hard/nH*100:.1f}%")
        print(f"  unresolved @{hard}{unit}      : {unresolved}/{nH} = {unresolved/nH*100:.1f}%")

    out = _HERE / f"_hardroot_ext_audit_{mode}_results.json"
    out.write_text(json.dumps({
        "mode": mode, "soft": soft, "mid": mid, "hard": hard, "unit": unit, "elapsed_s": round(elapsed_s, 1),
        "total_roots": total, "hard_roots": nH, "hard_rate": round(nH/total, 4),
        "action_change_soft_hard": {"n": nH, "changed": chg, "rate": round(chg/nH, 4) if nH else None},
        "convergence": {"lt_mid": conv_mid, "mid_hard": conv_mid_hard, "unresolved": unresolved},
        "checkpoint_table": {
            "soft": {"iters": _mean([x[0] for x in soft_c]), "share": _mean([x[1] for x in soft_c]), "gap": _mean([x[2] for x in soft_c])},
            "mid": {"iters": _mean([x[0] for x in mid_c]), "share": _mean([x[1] for x in mid_c]), "gap": _mean([x[2] for x in mid_c])},
            "hard": {"iters": _mean([x[0] for x in hard_c]), "share": _mean([x[1] for x in hard_c]), "gap": _mean([x[2] for x in hard_c])}},
        "roots": roots,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  results -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_hard", nargs="?", type=int, default=100)
    ap.add_argument("max_games", nargs="?", type=int, default=40)
    ap.add_argument("--iter", action="store_true", help="iteration 基準(machine-independent)")
    ap.add_argument("--soft-it", type=int, default=206); ap.add_argument("--mid-it", type=int, default=332)
    ap.add_argument("--hard-it", type=int, default=458)
    ap.add_argument("--soft-ms", type=float, default=1350.0); ap.add_argument("--mid-ms", type=float, default=2000.0)
    ap.add_argument("--hard-ms", type=float, default=3000.0)
    a = ap.parse_args()
    if a.iter:
        main(a.n_hard, a.max_games, "iter", a.soft_it, a.mid_it, a.hard_it)
    else:
        main(a.n_hard, a.max_games, "time", a.soft_ms, a.mid_ms, a.hard_ms)
