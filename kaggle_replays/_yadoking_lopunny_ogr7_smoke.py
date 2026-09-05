"""yadoking / lopunny_megafroslass の新規BC(policy_weights_<arch>_g2.json)を、
チャンピオン og_r7(abl_5_full_og_r7 config, オーガポンデッキ)相手に少数対戦させる動作確認スモーク。

kaggle_replays/_og_r7_newmeta_ab.py と同じ構成(OWN_DECK/OWN_WEIGHTS/config-a=og_r7/
config-b=abl_5_full)を踏襲する。既存ファイルは変更せず、新規重み用に別スクリプトとして
複製する(既存スクリプトの MATCHUPS は古い代打方策を指したまま、他セッションの実験資産の
可能性があるため触らない)。

読み取り専用の対戦検証(league/run_league.run_league を再利用)。cg/ data/ sample_submission/
の中身は変更しない(policy_weights_*_g2.json は本タスクで新規追加したファイルのみ参照)。

使い方:
  PYTHONIOENCODING=utf-8 python kaggle_replays/_yadoking_lopunny_ogr7_smoke.py --games 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "league"))

import run_league  # noqa: E402

DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
OWN_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
OWN_WEIGHTS = WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

MATCHUPS = {
    "yadoking": dict(deck=DECKDIR / "yadoking" / "01.csv", weights=WDIR / "policy_weights_yadoking_g2.json"),
    "lopunny_megafroslass": dict(
        deck=DECKDIR / "lopunny_megafroslass" / "01.csv",
        weights=WDIR / "policy_weights_lopunny_megafroslass_g2.json",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="abl_5_full_og_r7", help="og_r7(チャンピオン)側の config")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=990000)
    ap.add_argument("--matchups", default="yadoking,lopunny_megafroslass")
    args = ap.parse_args()

    for path in (OWN_DECK, OWN_WEIGHTS):
        if not path.exists():
            raise SystemExit(f"必要なファイルが無い: {path}")

    out: dict[str, dict] = {}
    for i, name in enumerate(args.matchups.split(",")):
        mu = MATCHUPS[name.strip()]
        for path in (mu["deck"], mu["weights"]):
            if not path.exists():
                raise SystemExit(f"必要なファイルが無い: {path}")
        print(f"\n=== {name}: og_r7(ogerpon) vs {name}_g2 BC ({mu['deck'].name}) ===", flush=True)
        s = run_league.run_league(
            agent_a_name="ml_policy",
            agent_b_name="ml_policy",
            games=args.games,
            deck_a_path=str(OWN_DECK),
            deck_b_path=str(mu["deck"]),
            weights_a_path=str(OWN_WEIGHTS),
            weights_b_path=str(mu["weights"]),
            config_base_a=args.config_a,
            config_base_b="abl_5_full",
            seed_start=args.seed_base + i * 10000,
            progress_every=10,
            workers=args.workers,
            log=lambda m: print(m, flush=True),
        )
        ov = s["overall"]
        err = s["errors"]
        out[name] = {
            "games_run": s["games_run"],
            "valid_games": ov["games"],
            "og_r7_wins": ov["wins"],
            "og_r7_wr": ov["win_rate"],
            "og_r7_wilson_95ci": ov["wilson_95ci"],
            "errors": err,
            "raw_overall": ov,
        }
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)

    out_path = _ROOT / "kaggle_replays" / "_yadoking_lopunny_ogr7_smoke_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFINAL {json.dumps(out, ensure_ascii=False)}")
    print(f"結果を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
