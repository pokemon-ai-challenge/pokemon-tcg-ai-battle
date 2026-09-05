"""train_pool.py で作った RL 後の重みを、初期BCと単一相手に対して直接比較する。

distributed/eval_e3_gate.py は run-dir(distributed/ アーキテクチャの世代管理)前提で
train_pool.py の出力(単一 json ファイル)とは構造が合わないため、同じ統計手法
(Wilson CI・2標本比率検定、eval_e3_gate.py からそのまま移植)で軽量版として書いた。

使い方:
    python eval_pool_matchup.py --initial-weights <BC重み>.json \
        --final-weights <RL後重み>.json --opponent <archetype> --games 400
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pools  # noqa: E402
from collect_pool import parallel_collect_pool  # noqa: E402


def wilson_ci(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def two_proportion_z_test(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    p1, p2 = w1 / n1, w2 / n2
    p_pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p2 - p1) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p_value)


def elo_diff(w: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = min(max(w / n, 1e-6), 1 - 1e-6)
    return 173.7 * math.log(p / (1 - p))


def run_side(weights_path: str, opponent_name: str, n_games: int, seed0: int, workers: int):
    opp_weights_path, deck_csv = pools.resolve_learner(opponent_name)
    from run_league import read_deck_csv_file
    deck_o = read_deck_csv_file(deck_csv)
    opponents = [(opponent_name, opp_weights_path, deck_o)]

    # learner 側デッキは kamitsuorochi_ex 固定(この比較の主目的がそれのため)。
    _, learner_deck_csv = pools.resolve_learner("kamitsuorochi_ex")
    deck_l = read_deck_csv_file(learner_deck_csv)

    trajs, stats = parallel_collect_pool(
        weights_path, opponents, deck_l,
        n_games=n_games, seed0=seed0, temperature=0.01, workers=workers,
    )
    del trajs
    total = stats["total"]
    return total["wins"], total["valid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial-weights", required=True)
    ap.add_argument("--final-weights", required=True)
    ap.add_argument("--opponent", required=True, help="pools.LEARNER_REGISTRY のキー")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed0-initial", type=int, default=900001)
    ap.add_argument("--seed0-final", type=int, default=900101)
    args = ap.parse_args()

    print(f"vs {args.opponent}: initial={Path(args.initial_weights).name} "
          f"final={Path(args.final_weights).name} games={args.games}x2 (温度0.01)", flush=True)

    w0, n0 = run_side(args.initial_weights, args.opponent, args.games, args.seed0_initial, args.workers)
    lo0, hi0 = wilson_ci(w0, n0)
    print(f"  initial: {w0}/{n0} = {w0/n0:.3f} (95%CI [{lo0:.3f},{hi0:.3f}]) "
          f"Elo {elo_diff(w0, n0):+.1f}", flush=True)

    w1, n1 = run_side(args.final_weights, args.opponent, args.games, args.seed0_final, args.workers)
    lo1, hi1 = wilson_ci(w1, n1)
    print(f"  final  : {w1}/{n1} = {w1/n1:.3f} (95%CI [{lo1:.3f},{hi1:.3f}]) "
          f"Elo {elo_diff(w1, n1):+.1f}", flush=True)

    z, p = two_proportion_z_test(w0, n0, w1, n1)
    elo_delta = elo_diff(w1, n1) - elo_diff(w0, n0)
    sig = "有意 (p<0.05)" if p < 0.05 else "有意差なし"
    verdict = "改善" if (w1 / n1) > (w0 / n0) else "非改善/悪化"
    print(f"  z={z:.3f} p={p:.4f} ({sig}) Elo差={elo_delta:+.1f} -> {verdict}", flush=True)

    result = {
        "opponent": args.opponent,
        "initial": {"wins": w0, "valid": n0, "winrate": w0 / n0, "ci": [lo0, hi0]},
        "final": {"wins": w1, "valid": n1, "winrate": w1 / n1, "ci": [lo1, hi1]},
        "z": z, "p_value": p, "elo_delta": elo_delta,
        "significant": bool(p < 0.05), "improved": bool((w1 / n1) > (w0 / n0)),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
