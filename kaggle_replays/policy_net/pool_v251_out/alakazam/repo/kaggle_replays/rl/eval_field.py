"""方針(c)検証: alakazam(learner)の対フィールド加重勝率を full agent で測る。

単一相手(crustle)RL が他マッチを壊していないか(overfit)を検出するため、7アーキ全部に対する
alakazam の勝率を meta-share 加重で出し、production(policy_weights.json)と RL 候補を比較する。

相手は各アーキの模倣policy(policy_weights_<arch>.json)= round_robin の "field" 定義に合わせる。
alakazam 側だけ weights を差し替える。full agent(ml_lethal_attackplan_v0only overlay込み)。

使い方: python kaggle_replays/rl/eval_field.py --rl-weights policy_weights_alakazam_rl_vscrustle.json --games 150
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

# (arch, meta_share) — round_robin と同じ7アーキ。相手は各 imitation policy。
FIELD = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]


def alakazam_vs(arch, ala_weights, games, workers):
    """alakazam(ala_weights) vs arch(imitation) の alakazam 勝率。"""
    summary = run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=str(DECKDIR / "alakazam" / "01.csv"),
        deck_b_path=str(DECKDIR / arch / "01.csv"),
        seed_start=0, progress_every=0,
        weights_a_path=ala_weights,
        weights_b_path=str(WDIR / f"policy_weights_{arch}.json"),
        config_base="ml_lethal_attackplan_v0only", workers=workers, log=lambda m: None,
    )
    ov = summary["overall"]
    return ov["win_rate"], ov["wins"], ov["games"]


def field_weighted(ala_weights, games, workers, label):
    total = sum(s for _, s in FIELD)
    acc = 0.0
    print(f"--- {label} ---", flush=True)
    rows = {}
    for arch, share in FIELD:
        wr, w, n = alakazam_vs(arch, ala_weights, games, workers)
        acc += share * wr
        rows[arch] = wr
        print(f"  vs {arch:<22} {wr*100:5.1f}%  ({w}/{n})", flush=True)
    fw = acc / total
    print(f"  => 対フィールド加重勝率: {fw*100:.2f}%", flush=True)
    return fw, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-weights", required=True, help="alakazam RL 候補(WDIR内のファイル名 or 絶対)")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rlw = args.rl_weights
    if not Path(rlw).is_absolute():
        rlw = str(WDIR / rlw)
    if not Path(rlw).exists():
        print(f"RL重みが無い: {rlw}"); return

    print(f"=== 方針c: alakazam 対フィールド加重勝率(full agent, {args.games}試合/マッチ) ===", flush=True)
    base_fw, base_rows = field_weighted(None, args.games, args.workers, "production (policy_weights.json)")
    rl_fw, rl_rows = field_weighted(rlw, args.games, args.workers, f"RL ({Path(rlw).name})")
    print(f"\n=== 比較 ===", flush=True)
    print(f"  production 対フィールド {base_fw*100:.2f}%  ->  RL {rl_fw*100:.2f}%  Δ{(rl_fw-base_fw)*100:+.2f}pt")
    print(f"  マッチ別Δ:")
    for arch, _ in FIELD:
        d = (rl_rows[arch] - base_rows[arch]) * 100
        print(f"    {arch:<22} {base_rows[arch]*100:5.1f}% -> {rl_rows[arch]*100:5.1f}%  ({d:+.1f}pt)")


if __name__ == "__main__":
    main()
