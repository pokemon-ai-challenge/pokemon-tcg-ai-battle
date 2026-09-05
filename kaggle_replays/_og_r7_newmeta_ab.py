"""og_r7 vs 上位帯新メタ2種のローカルA/B基盤。

対面固定で「現行(baseline config)」と「修正版(candidate config)」を同条件比較する。
- lopunny_megafroslass: archetype_decks_g2/mega_froslass_ex/01.csv(rank8実デッキ、
  実ラダーの対面とほぼ同一構築) + policy_weights_mega_froslass_ex_g2.json
- yadoking: archetype_decks_g2/yadoking/01.csv(rank3実デッキ) + 代打方策
  (専用方策なし。既定 rocket_mewtwo_ex_g2 = 超ex+ピボット運用が最も近い。
   ひらめきチャレンジのループ再現は不完全な点に注意=脅威は過小評価側)

使い方:
  python kaggle_replays/_og_r7_newmeta_ab.py --config-a abl_5_full_og_r7 --games 400 --tag base
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "league"))

import run_league  # noqa: E402

DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
OWN_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
OWN_WEIGHTS = WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

MATCHUPS = {
    "lopunny_mf": dict(deck=DECKDIR / "mega_froslass_ex" / "01.csv",
                       weights=WDIR / "policy_weights_mega_froslass_ex_g2.json"),
    "yadoking": dict(deck=DECKDIR / "yadoking" / "01.csv",
                     weights=WDIR / "policy_weights_rocket_mewtwo_ex_g2.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="abl_5_full_og_r7")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=980000)
    ap.add_argument("--matchups", default="lopunny_mf,yadoking")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    out = {}
    for i, name in enumerate(args.matchups.split(",")):
        mu = MATCHUPS[name.strip()]
        s = run_league.run_league(
            agent_a_name="ml_policy", agent_b_name="ml_policy", games=args.games,
            deck_a_path=str(OWN_DECK), deck_b_path=str(mu["deck"]),
            weights_a_path=str(OWN_WEIGHTS), weights_b_path=str(mu["weights"]),
            config_base_a=args.config_a, config_base_b="abl_5_full",
            seed_start=args.seed_base + i * 10000, progress_every=100,
            workers=args.workers, log=lambda m: print(m, flush=True),
        )
        ov = s["overall"]
        out[name] = {"wins": ov["wins"], "games": ov["games"],
                     "wr": round(ov["wins"] / max(1, ov["games"]), 4)}
        print(name, json.dumps(out[name]), flush=True)
    print("FINAL", args.config_a, args.tag, json.dumps(out))


if __name__ == "__main__":
    main()
