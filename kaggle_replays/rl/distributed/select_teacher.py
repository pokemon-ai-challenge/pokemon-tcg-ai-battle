"""D1: 蒸留教師checkpointの正式選定。

`kaggle_replays/rl/runs/pool_v1` の固定評価相手8体に対し、既存の
`learner.evaluate_pool()`(production同じ関数、新規実装しない)をそのまま使って
v24/v32/v40/v47を同一条件(同じ相手プール・同じ試合数・greedy・先攻後攻は
`parallel_collect`が試合indexの偶奇で自動的に均等割り)で評価する。

`collect_winrate`(学習中の温度サンプリング勝率)は使わない。ここで測るのは
`evaluate_pool`のgreedy評価のみ。

    python select_teacher.py --games 1000 --workers 16 \
        --out results/teacher_selection.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common as C
from learner import evaluate_pool
from train_v3 import wilson_lo

_RUN_DIR = _HERE.parent / "runs" / "pool_v1"
_CANDIDATES = [24, 32, 40, 47]


def winrate_diff_ci95(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """(p1-p2)の95%信頼区間(正規近似、独立2標本)。

    ``select_teacher.py``のv24〜v47比較(§D1)は同じ量(全体勝率)どうしを比べるので
    Wilson区間の単純な大小比較で足りるが、**教師とstudentの固定相手プール比較
    (§評価方法の分離)では単一の勝率ではなく「差」を見る必要がある。**
    非劣性境界(例: 教師との直接対戦の45%)は「studentの勝率が閾値を上回るか」を見る
    片側検定で、固定相手プールでの比較(「studentは教師よりどれだけ強い/弱いか」)とは
    別の問いに答えるものなので、閾値を使い回さない。
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = w1 / n1, w2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return diff - 1.96 * se, diff + 1.96 * se


def compare_pool_results(teacher_per: list[dict], student_per: list[dict],
                         teacher_overall: tuple[int, int], student_overall: tuple[int, int],
                         margin: float = -0.05) -> dict:
    """固定相手プールでの教師 vs student比較。相手ごと・全体で
    ``student win rate - teacher win rate`` の95%CIを計算し、下限が``margin``
    (既定-5%)を上回るかを判定する。**47.59%のような単一勝率の閾値は使わない**
    (それは教師との直接対戦・非劣性境界45%の文脈でのみ意味を持つ数値)。
    """
    teacher_by_id = {p["id"]: p for p in teacher_per}
    per_opponent = []
    for sp in student_per:
        tp = teacher_by_id.get(sp["id"])
        if tp is None:
            continue
        lo, hi = winrate_diff_ci95(sp["wins"], sp["valid"], tp["wins"], tp["valid"])
        per_opponent.append({
            "id": sp["id"], "student_winrate": sp["winrate"], "teacher_winrate": tp["winrate"],
            "diff": sp["winrate"] - tp["winrate"], "diff_ci95_lo": lo, "diff_ci95_hi": hi,
            "passes_margin": (lo > margin) if lo == lo else False,
        })
    sw, sv = student_overall
    tw, tv = teacher_overall
    lo, hi = winrate_diff_ci95(sw, sv, tw, tv)
    overall = {"student_winrate": sw / sv if sv else float("nan"),
              "teacher_winrate": tw / tv if tv else float("nan"),
              "diff": (sw / sv if sv else float("nan")) - (tw / tv if tv else float("nan")),
              "diff_ci95_lo": lo, "diff_ci95_hi": hi, "passes_margin": (lo > margin) if lo == lo else False}
    return {"margin": margin, "overall": overall, "per_opponent": per_opponent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--generations", default=",".join(str(g) for g in _CANDIDATES))
    ap.add_argument("--out", default=str(_HERE / "results" / "teacher_selection.json"))
    args = ap.parse_args()

    run = C.load_run(_RUN_DIR)
    generations = [int(g) for g in args.generations.split(",")]

    results = {}
    for gen in generations:
        weights = C.model_path(_RUN_DIR, gen)
        if not weights.exists():
            print(f"skip v{gen}: {weights} が無い")
            continue
        print(f"v{gen} を評価中({args.games}試合、相手プール{len(run['opponents'])}体)...", flush=True)
        t0 = time.time()
        per, avg, w, v = evaluate_pool(run, _RUN_DIR, weights, args.games, args.workers)
        dt = time.time() - t0
        for p in per:
            p["ci95_lo"] = wilson_lo(p["wins"], p["valid"])
        results[f"v{gen}"] = {
            "generation": gen, "games_requested": args.games, "seconds": round(dt, 1),
            "overall_wins": w, "overall_valid": v, "overall_winrate": avg,
            "overall_ci95_lo": wilson_lo(w, v),
            "per_opponent": per,
            "model_sha256": C.sha256_file(weights),
        }
        print(f"  v{gen}: {w}/{v} = {avg:.4f} (CI下限 {wilson_lo(w, v):.4f})  {dt:.0f}s", flush=True)

    if not results:
        raise SystemExit("評価できたcheckpointが無い")

    # 選定: 全体勝率のCI下限が最大のものを正式教師にする(単純な平均最大ではなく、
    # 400試合程度のノイズで逆転しうる差を避けるため下限で比較する)。
    best = max(results, key=lambda k: results[k]["overall_ci95_lo"])
    selection = {
        "candidates": list(results.keys()),
        "selected": best,
        "selection_rule": "overall_ci95_lo最大(Wilson区間下限)",
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n選定: {best}(全candidate: {list(results.keys())})")
    print(f"結果を保存: {out_path}")


if __name__ == "__main__":
    main()
