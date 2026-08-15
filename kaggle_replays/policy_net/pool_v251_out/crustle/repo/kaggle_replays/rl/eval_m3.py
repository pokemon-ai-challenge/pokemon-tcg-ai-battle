"""M3: RL済み相手が「強いベンチ」になったかを、full agent(overlay込み)で測る。

各アーキ<arch>について、その<arch>を操縦する相手 vs production-alakazam を run_league で対戦させ、
- baseline: 模倣重み policy_weights_<arch>.json
- rl:       RL重み  policy_weights_<arch>_rl_<tag>.json
の "相手の対alakazam勝率" を比較する。RL で上がっていれば相手が強くなった=ベンチとして有用。

注: run_league の ml_policy は lethal/attack_plan overlay込み(=本番のベンチ相手の姿)。RLは素policyを
学習したが、ここでは両者に同じ overlay をかけて比較するので公平(重みの差だけを見る)。
使い方: python kaggle_replays/rl/eval_m3.py --arch dragapult_ex --tag v3 --games 300
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def run(arch, weights_a, games, workers):
    """相手(arch, weights_a) vs alakazam(production) の相手勝率。"""
    summary = run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=str(DECKDIR / arch / "01.csv"),
        deck_b_path=str(DECKDIR / "alakazam" / "01.csv"),
        seed_start=0, progress_every=0,
        weights_a_path=weights_a, weights_b_path=None,
        config_base="ml_lethal_attackplan_v0only", workers=workers, log=lambda m: None,
    )
    ov = summary["overall"]
    return ov["win_rate"], ov["wins"], ov["games"], ov["wilson_95ci"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="dragapult_ex")
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base_w = str(WDIR / f"policy_weights_{args.arch}.json")
    rl_w = str(WDIR / f"policy_weights_{args.arch}_rl_{args.tag}.json")
    if not Path(rl_w).exists():
        print(f"RL重みが無い: {rl_w}"); return

    print(f"=== M3: {args.arch} 相手の対alakazam勝率(full agent, {args.games}試合) ===", flush=True)
    b_wr, b_w, b_n, b_ci = run(args.arch, base_w, args.games, args.workers)
    print(f"  模倣(baseline): {b_wr:.3f}  ({b_w}/{b_n})  CI[{b_ci[0]:.3f},{b_ci[1]:.3f}]", flush=True)
    r_wr, r_w, r_n, r_ci = run(args.arch, rl_w, args.games, args.workers)
    print(f"  RL({args.tag}):    {r_wr:.3f}  ({r_w}/{r_n})  CI[{r_ci[0]:.3f},{r_ci[1]:.3f}]", flush=True)
    # 2標本 z 検定
    p = (b_w + r_w) / (b_n + r_n)
    se = math.sqrt(p * (1 - p) * (1 / b_n + 1 / r_n))
    z = (r_wr - b_wr) / se if se > 0 else 0.0
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    verdict = "相手が有意に強化" if pval < 0.05 and r_wr > b_wr else "方向は正だが未達/要追加試合"
    print(f"  Δ(RL - baseline) = {(r_wr-b_wr)*100:+.1f}pt  z={z:.2f} p={pval:.3f}  -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
