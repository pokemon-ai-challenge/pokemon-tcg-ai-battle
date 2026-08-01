"""vs-crustle 縦串P0: deck_sustain(山札温存)の対crustle勝率 before/after 測定。

自分(agent A)= 本番フルエージェント(deck.csv + 既定重み)。相手(agent B)= crustle 模倣。
交絡を避けるため deck_sustain は agent A 側 config だけに乗せ、crustle 相手は常に
``abl_5_full`` 固定(config_base_b)。baseline(abl_5_full) と treatment
(abl_5_full_decksustain) の対crustle勝率を Wilson 95%CI 付きで比較する。

使い方(smoke): python kaggle_replays/_measure_decksustain_crustle.py --games 10 --workers 4
使い方(powered): python kaggle_replays/_measure_decksustain_crustle.py --games 300 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
PROD_DECK = _ROOT / "sample_submission" / "deck.csv"


def _wilson(wins: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def _run(config_a: str, games: int, workers: int) -> dict:
    """agent A = 本番(config_a) vs agent B = crustle(abl_5_full 固定)。A の勝率 summary を返す。"""
    return run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=str(PROD_DECK),
        deck_b_path=str(DECKDIR / "crustle" / "01.csv"),
        seed_start=0, progress_every=0,
        weights_a_path=None,  # 本番既定重み
        weights_b_path=str(WDIR / "policy_weights_crustle.json"),
        config_base_a=config_a,
        config_base_b="abl_5_full",  # 相手は常に固定(交絡防止)
        workers=workers, log=lambda m: None,
    )


def _two_prop_z(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """治療 vs baseline の2標本比率 z 検定(片側でなく両側 p)。(Δpt, p_two_sided)。"""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2) * 100, 1.0
    z = (p1 - p2) / se
    # 標準正規両側 p(誤差関数で)
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return (p1 - p2) * 100, p_two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="各アーム(baseline/treatment)の試合数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(_HERE / "_measure_decksustain_crustle_results.json"))
    args = ap.parse_args()

    # full agent の read_deck_csv() は cwd 相対フォールバックを持つ。deck/weights は絶対で渡すが、
    # eval_field.py に倣い main プロセスの cwd を sample_submission にそろえる。
    os.chdir(_ROOT / "sample_submission")

    print(f"=== deck_sustain vs crustle 測定 ({args.games}試合/アーム, workers={args.workers}) ===", flush=True)

    base = _run("abl_5_full", args.games, args.workers)
    bo = base["overall"]
    bw, bn = bo["wins"], bo["games"]
    bp, blo, bhi = _wilson(bw, bn)
    print(f"  baseline (abl_5_full)          {bp*100:5.1f}%  ({bw}/{bn})  CI[{blo*100:.1f},{bhi*100:.1f}]  avg_turns={bo.get('avg_turns')}", flush=True)

    treat = _run("abl_5_full_decksustain", args.games, args.workers)
    to = treat["overall"]
    tw, tn = to["wins"], to["games"]
    tp, tlo, thi = _wilson(tw, tn)
    print(f"  treatment (abl_5_full_decksustain) {tp*100:5.1f}%  ({tw}/{tn})  CI[{tlo*100:.1f},{thi*100:.1f}]  avg_turns={to.get('avg_turns')}", flush=True)

    delta, pval = _two_prop_z(tw, tn, bw, bn)
    print(f"\n  Δ = {delta:+.1f}pt   two-sided p = {pval:.4f}", flush=True)

    result = {
        "games_per_arm": args.games,
        "baseline": {"wins": bw, "games": bn, "win_rate": bp, "ci95": [blo, bhi], "avg_turns": bo.get("avg_turns")},
        "treatment": {"wins": tw, "games": tn, "win_rate": tp, "ci95": [tlo, thi], "avg_turns": to.get("avg_turns")},
        "delta_pt": delta,
        "p_two_sided": pval,
        "errors_baseline": base.get("errors"),
        "errors_treatment": treat.get("errors"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
