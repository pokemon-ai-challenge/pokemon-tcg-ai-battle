"""deck-tech Plan A の実測: 現行デッキ vs Plan A を主要3対面で比較。

Plan A = 一貫性コア復元(Kadabra3->4/Telepath2->3/RareCandy3->4/EnhancedHammer3->1/
WondrousPatch2->1)。仮説: Alakazam到達が速く安定 → 対アグロ/対Grimmsnarl/ミラーで勝率up。

各デッキを deck.csv に一時スワップ(finallyで必ず復元)して run_league(並列)で測る。相手は
imitation policy(固定 abl_5_full)。自分も abl_5_full(deck_sustain は off = デッキ変更だけを分離)。
Grimmsnarl は現在メタ上位のため最優先対面。

使い方: python kaggle_replays/_measure_planA_matchups.py --games 60 --workers 6
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
DECK_CSV = _ROOT / "sample_submission" / "deck.csv"
PLANA = _ROOT / "sample_submission" / "deck_planA.csv"
PLANB = _ROOT / "sample_submission" / "deck_planB.csv"

# (ラベル, 相手アーキデッキdir, 相手imitation重み, 試合数) — Grimmsnarl/mirror を厚く。
OPPS = [
    ("grimmsnarl", "marnie_grimmsnarl_ex", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json"), 120),
    ("mirror", "alakazam", None, 100),  # 既定(production)Alakazam policy で自ミラー
    ("lucario", "mega_lucario_ex", str(WDIR / "policy_weights_mega_lucario_ex.json"), 60),
]


def _wilson(w, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _run_vs(opp_dir, opp_w, games, workers):
    s = run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=str(DECK_CSV), deck_b_path=str(DECKDIR / opp_dir / "01.csv"),
        seed_start=0, progress_every=0,
        weights_a_path=None, weights_b_path=opp_w,
        config_base_a="abl_5_full", config_base_b="abl_5_full",
        workers=workers, log=lambda m: None,
    )
    o = s["overall"]
    return o["wins"], o["games"], o.get("avg_turns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(_HERE / "_measure_planAB_matchups_results.json"))
    args = ap.parse_args()

    os.chdir(_ROOT / "sample_submission")
    backup = DECK_CSV.read_text(encoding="utf-8")
    decks = {
        "current": backup,
        "planA": PLANA.read_text(encoding="utf-8"),
        "planB": PLANB.read_text(encoding="utf-8"),
    }

    print(f"=== deck-tech current/A/B 実測 (workers={args.workers}) ===", flush=True)
    results = {}
    try:
        for name, content in decks.items():
            DECK_CSV.write_text(content, encoding="utf-8")
            for label, opp_dir, opp_w, games in OPPS:
                w, n, turns = _run_vs(opp_dir, opp_w, games, args.workers)
                results[(name, label)] = {"wins": w, "games": n, "avg_turns": turns}
                p, lo, hi = _wilson(w, n)
                print(f"  [{name:<7}] vs {label:<10} {p*100:5.1f}% ({w}/{n}) CI[{lo*100:.0f},{hi*100:.0f}] turns={turns:.1f}", flush=True)
    finally:
        DECK_CSV.write_text(backup, encoding="utf-8")
        print("  (deck.csv restored)", flush=True)

    print("\n=== 比較 (対面ごと current / planA / planB) ===", flush=True)
    for label, _, _, _ in OPPS:
        cells = []
        for name in ("current", "planA", "planB"):
            w, n = results[(name, label)]["wins"], results[(name, label)]["games"]
            cells.append(f"{name} {w/n*100:4.1f}%({w}/{n})")
        base = results[("current", label)]["wins"] / results[("current", label)]["games"]
        da = results[("planA", label)]["wins"] / results[("planA", label)]["games"] - base
        db = results[("planB", label)]["wins"] / results[("planB", label)]["games"] - base
        print(f"  {label:<10} " + " | ".join(cells) + f"   ΔA{da*100:+.1f} ΔB{db*100:+.1f}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=1)
    print(f"\n  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
