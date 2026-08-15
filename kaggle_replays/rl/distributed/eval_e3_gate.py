"""roadmap E3ゲート: 単腕RLの「最終イテレート vs 初期」判定。

学習側の初期重み(既定 models/model_v0.json)と最終重み(既定: run.json の現世代が指す
最新の model_v<gen>.json)を、それぞれ run["opponent_deck"] に対して独立に
--games 試合(既定1200、roadmap の下限)ぶつけ、勝率と defect#7 の Elo差
(``173.7*logit(p)``)を比較する。

learner.py の ``evaluate()`` をそのまま再利用する(温度0.01=ほぼgreedy、
EVAL_SEED_BASE を種に使う既存の評価経路と完全に同じ土俵で測るため、独自の評価ロジックは書かない)。

使い方:
    python eval_e3_gate.py --run-dir ../runs/kamitsuorochi_vs_crustle_v1
    python eval_e3_gate.py --run-dir ... --games 1200 --final-gen 25
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from learner import evaluate


def wilson_ci(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def two_proportion_z_test(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """(z, 両側p値)。プールした比率での標準的な2標本比率検定。"""
    p1, p2 = w1 / n1, w2 / n2
    p_pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p2 - p1) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p_value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--games", type=int, default=1200, help="roadmap E3 の下限(既定1200)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--initial-gen", type=int, default=0)
    ap.add_argument("--final-gen", type=int, default=None,
                    help="既定: run.json の現世代(=最後にPPO更新で作られたmodel_v<N>.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    final_gen = args.final_gen if args.final_gen is not None else run["generation"]
    workers = args.workers or (os.cpu_count() or 2)

    initial_weights = run_dir / "models" / f"model_v{args.initial_gen}.json"
    final_weights = run_dir / "models" / f"model_v{final_gen}.json"
    for p in (initial_weights, final_weights):
        if not p.is_file():
            raise SystemExit(f"重みファイルが無い: {p}")

    print(f"run_dir={run_dir}")
    print(f"opponent_deck={run['opponent_deck']}")
    print(f"initial=model_v{args.initial_gen}.json  final=model_v{final_gen}.json")
    print(f"games={args.games} (each side, independent)  workers={workers}\n")

    print(f"[initial] model_v{args.initial_gen} を {args.games} 試合評価中...", flush=True)
    wr0, w0, v0, elo0 = evaluate(run, run_dir, initial_weights, args.games, workers)
    lo0, hi0 = wilson_ci(w0, v0)
    print(f"  勝率 {wr0:.4f} ({w0}/{v0})  95%CI [{lo0:.4f}, {hi0:.4f}]  Elo差 {elo0:+.1f}\n")

    print(f"[final]   model_v{final_gen} を {args.games} 試合評価中...", flush=True)
    wr1, w1, v1, elo1 = evaluate(run, run_dir, final_weights, args.games, workers)
    lo1, hi1 = wilson_ci(w1, v1)
    print(f"  勝率 {wr1:.4f} ({w1}/{v1})  95%CI [{lo1:.4f}, {hi1:.4f}]  Elo差 {elo1:+.1f}\n")

    z, p = two_proportion_z_test(w0, v0, w1, v1)
    print("=== E3判定 ===")
    print(f"勝率差(final - initial) = {wr1 - wr0:+.4f}")
    print(f"Elo差の差(defect#7 主指標) = {elo1 - elo0:+.1f}  (initial={elo0:+.1f} -> final={elo1:+.1f})")
    print(f"2標本比率検定: z={z:.3f}  p={p:.4f}")
    if p < 0.05 and wr1 > wr0:
        print("→ 有意に改善(p<0.05 かつ final > initial)。RL続行/採用の判断材料になる。")
    elif p < 0.05 and wr1 < wr0:
        print("→ 有意に悪化(p<0.05 かつ final < initial)。RLがこの対面を壊している。")
    else:
        print("→ 有意差なし(p>=0.05)。roadmap の基準では **RLをこの対面から撤退**する側。")


if __name__ == "__main__":
    main()
