"""現在メタ(2026-08-01, 帯別)で重み付けした対フィールド評価。

デッキ(既定 deck.csv)を各対面で測り、スコア帯別に加重勝率を出す:
  - climb_600_699 : 我々の現在地(~685)。climbで当たる相手=Lucario/Alakazam/Dragapult/Crustle中心。
  - next_800_899  : 次の壁。Alakazam/Grimmsnarl/Archaludon/Crustle。
  - ceiling_1000  : 最上位。Grimmsnarl支配(~61%)。

ローカルimitation重みが有る対面のみ(Starmie等は除外し、各帯で在る分だけ正規化)。
これが(i)v系RL・(ii)戦略NN 両方の"現在メタでの物差し"になる。

使い方: python kaggle_replays/_eval_current_meta.py --games 40 --workers 6
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

# label -> (archetype deck dir, imitation weights or None=default alakazam=mirror)
OPPS = {
    # ミラーは強いclimb(実ラダーのミラーは~820の強い方策=Noneの素の模倣では弱すぎる)。
    "alakazam_mirror": ("alakazam", str(WDIR / "policy_weights_alakazam_rl_climb.json")),
    "lucario": ("mega_lucario_ex", str(WDIR / "policy_weights_mega_lucario_ex.json")),
    "archaludon": ("archaludon_ex", str(WDIR / "policy_weights_archaludon_ex.json")),
    "crustle": ("crustle", str(WDIR / "policy_weights_crustle.json")),
    "grimmsnarl": ("marnie_grimmsnarl_ex", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json")),
    "dragapult": ("dragapult_ex", str(WDIR / "policy_weights_dragapult_ex.json")),
    "rocket_mewtwo": ("rocket_mewtwo_ex", str(WDIR / "policy_weights_rocket_mewtwo_ex.json")),
    "garchomp": ("shirona_garchomp_ex", str(WDIR / "policy_weights_shirona_garchomp_ex.json")),
}

# 帯別シェア(在る対面のみ, 各帯で正規化して加重)。値は current_meta_shares_2026-08-01.json 由来。
# real_ladder_820 は climb提出(55186283)の実ラダー62試合リプレイ実測(2026-08-03)。ミラー29%が
# 最大の対面=想定fieldがミラー0%だった乖離を是正。これがLBを予測する物差し(最優先で見る)。
# 実戦績(62試合): mirror61/archaludon75/lucario67 は問題なし。負けは crustle29%(2-5) と
# grimmsnarl45%(5-6) に集中=真の弱点。全体W36-L26=58%。
BRACKETS = {
    "real_ladder_820": {"alakazam_mirror": 29.0, "archaludon": 19.0, "grimmsnarl": 18.0,
                        "lucario": 15.0, "crustle": 11.0, "dragapult": 2.0},
    "climb_600_699": {"lucario": 24.2, "alakazam_mirror": 18.2, "dragapult": 13.0, "crustle": 12.6,
                      "grimmsnarl": 7.4, "archaludon": 3.2, "rocket_mewtwo": 3.2},
    "next_800_899": {"alakazam_mirror": 27.6, "grimmsnarl": 23.7, "archaludon": 16.2, "crustle": 12.0,
                     "lucario": 3.9, "dragapult": 3.5, "garchomp": 3.2, "rocket_mewtwo": 2.5},
    "ceiling_1000": {"grimmsnarl": 61.0, "alakazam_mirror": 9.5, "crustle": 9.5, "dragapult": 4.2,
                     "garchomp": 2.1, "rocket_mewtwo": 2.1},
}


def _wilson(w, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--deck", default=str(DECK_CSV))
    ap.add_argument("--config", default="abl_5_full", help="自分(agent A)に注入する config")
    ap.add_argument("--weights", default=None, help="自分(agent A)の policy 重み(None=既定模倣)")
    ap.add_argument("--out", default=str(_HERE / "_eval_current_meta_results.json"))
    args = ap.parse_args()

    os.chdir(_ROOT / "sample_submission")
    wlabel = Path(args.weights).name if args.weights else "default"
    print(f"=== 現在メタ評価 deck={Path(args.deck).name} config={args.config} weights={wlabel} ({args.games}試合/対面) ===", flush=True)
    wr = {}
    for label, (arch, w) in OPPS.items():
        s = run_league.run_league(
            agent_a_name="ml_policy", agent_b_name="ml_policy", games=args.games,
            deck_a_path=args.deck, deck_b_path=str(DECKDIR / arch / "01.csv"),
            seed_start=0, progress_every=0, weights_a_path=args.weights, weights_b_path=w,
            config_base_a=args.config, config_base_b="abl_5_full", workers=args.workers, log=lambda m: None,
        )
        o = s["overall"]
        p, lo, hi = _wilson(o["wins"], o["games"])
        wr[label] = p
        print(f"  vs {label:<16} {p*100:5.1f}% ({o['wins']}/{o['games']}) CI[{lo*100:.0f},{hi*100:.0f}]", flush=True)

    print("\n=== スコア帯別 加重勝率(在る対面で正規化) ===", flush=True)
    bracket_scores = {}
    for bname, shares in BRACKETS.items():
        tot = sum(shares.values())
        acc = sum(shares[l] * wr[l] for l in shares if l in wr)
        weighted = acc / tot if tot else 0.0
        bracket_scores[bname] = weighted
        cov = sum(shares.values())  # このメタでカバーしたシェア(近似)
        print(f"  {bname:<14} 加重勝率 {weighted*100:5.1f}%   (対面: {', '.join(shares.keys())})", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"deck": Path(args.deck).name, "games_per_opp": args.games,
                   "per_matchup_winrate": wr, "bracket_weighted": bracket_scores}, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
