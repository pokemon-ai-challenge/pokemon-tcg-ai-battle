"""壁対面(crustle=イワパレス)向け 1枚差し替え候補の A/B。

3アーム: baseline(og_v032) / bulu(+カプ・ブルル920) / aceburn(+エースバーン666)
いずれも クラッシュハンマー 4→3 で1枠を捻出(ハンマーの実効価値は実測でほぼ0)。
config は現行最良の og_r13(全10ガード)。重みは mixogerpon 共通。
--stage crustle でまず壁対面のみ、--stage regress で他対面の維持確認。
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "league"))
import run_league  # noqa: E402

DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
DECKDIR_J = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
CANDDIR = _ROOT / "kaggle_replays" / "deck_search" / "candidates_wall"
OWN_WEIGHTS = WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

ARMS = ["baseline", "bulu", "aceburn"]
CUT_ARMS = ["baseline", "cut_hammer", "cut_jumbo", "cut_pokegear", "cut_transfer", "cut_bugnet", "cut_stadium"]
STAGES = {
    # 壁対面(イワパレス3+イシズマイ3+メガガルーラex3+いしずえのめんex1)
    "crustle": [("crustle", 400, "_g2")],
    # 他対面の強さ維持確認
    # 「1枚抜いて基本草エネで埋める」= その枠の限界価値を単独で測る
    "expend": [("dragapult_ex", 150, "_g2"), ("marnie_grimmsnarl_ex", 150, "_g2"),
               ("kamitsuorochi_ex", 150, "_g2"), ("alakazam", 150, "_g2"),
               ("lopunny_megafroslass", 150, "_g2")],
    # 時間制約版: エースバーン評価を優先。主要4対面 x 120試合
    "quick": [("alakazam", 120, "_g2"), ("kamitsuorochi_ex", 120, "_g2"),
              ("lopunny_megafroslass", 120, "_g2"), ("dragapult_ex", 120, "_g2")],
    "regress": [("dragapult_ex", 200, "_g2"), ("marnie_grimmsnarl_ex", 200, "_g2"),
                ("kamitsuorochi_ex", 200, "_g2"), ("alakazam", 200, "_g2"),
                ("lopunny_megafroslass", 200, "_g2"), ("mega_lucario_ex", 200, "")],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="crustle", choices=sorted(STAGES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed-base", type=int, default=1100000)
    ap.add_argument("--config", default="abl_5_full_og_r13")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    field = STAGES[args.stage]
    out = {}
    for arm in args.arms.split(","):
        out[arm] = {}
        for i, (arch, games, gen) in enumerate(field):
            deckdir = DECKDIR if gen == "_g2" else DECKDIR_J
            s = run_league.run_league(
                agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
                deck_a_path=str(CANDDIR / f"{arm}.csv"),
                deck_b_path=str(deckdir / arch / "01.csv"),
                weights_a_path=str(OWN_WEIGHTS),
                weights_b_path=str(WDIR / f"policy_weights_{arch}{gen}.json"),
                config_base_a=args.config, config_base_b="abl_5_full",
                seed_start=args.seed_base + i * 10000,   # 全アーム同一seed列=paired
                progress_every=0, workers=args.workers, log=lambda m: None,
            )
            ov = s["overall"]
            out[arm][arch] = {"wins": ov["wins"], "games": ov["games"],
                              "wr": round(ov["wins"] / max(1, ov["games"]), 4)}
            print(arm, arch, json.dumps(out[arm][arch]), flush=True)
    dest = Path(args.out) if args.out else _ROOT / "kaggle_replays" / f"_wall_deck_ab_{args.stage}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("FINAL", json.dumps(out))


if __name__ == "__main__":
    main()
