"""og_r8 vs og_r7 のフィールドA/B(11アーキ回帰+ガード発火対面の効果測定)。

- 両アームとも自陣=og_v032デッキ+mixogerpon重み。唯一差分=config(og_r8/og_r7)。
- 相手=mixフィールド11アーキ(g2模倣重み+abl_5_full)。paired seed(同一seed列)。
- ガード(hand_damage_guard)が発火する対面(alakazam=743, mega_froslass_ex=861)は
  n を厚くする。他は回帰確認の n。
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
OWN_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
OWN_WEIGHTS = WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

# (arch, games, gen)  gen="_g2"はarchetype_decks_g2+g2重み、""は7月版
FIELD = [
    ("alakazam", 400, "_g2"),
    ("dragapult_ex", 400, "_g2"),
    ("lopunny_megafroslass", 400, "_g2"),
    ("mega_froslass_ex", 300, "_g2"),
    ("kamitsuorochi_ex", 300, "_g2"),
    ("marnie_grimmsnarl_ex", 200, "_g2"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed-base", type=int, default=1040000)
    ap.add_argument("--out", default=str(_ROOT / "kaggle_replays" / "_og_r13_final_ab_results.json"))
    args = ap.parse_args()

    out = {}
    for cfg in ("abl_5_full_og_r7_isorng", "abl_5_full_og_r13"):
        out[cfg] = {}
        for i, (arch, games, gen) in enumerate(FIELD):
            deckdir = DECKDIR if gen == "_g2" else DECKDIR_J
            w = WDIR / f"policy_weights_{arch}{gen}.json"
            s = run_league.run_league(
                agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
                deck_a_path=str(OWN_DECK), deck_b_path=str(deckdir / arch / "01.csv"),
                weights_a_path=str(OWN_WEIGHTS), weights_b_path=str(w),
                config_base_a=cfg, config_base_b="abl_5_full",
                seed_start=args.seed_base + i * 10000,  # 両アームで同一seed列=paired
                progress_every=0, workers=args.workers, log=lambda m: None,
            )
            ov = s["overall"]
            out[cfg][arch] = {"wins": ov["wins"], "games": ov["games"],
                              "wr": round(ov["wins"] / max(1, ov["games"]), 4)}
            print(cfg, arch, json.dumps(out[cfg][arch]), flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("FINAL", json.dumps(out))


if __name__ == "__main__":
    main()
