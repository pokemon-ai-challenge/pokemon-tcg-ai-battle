"""og_v032(新デッキ)× alakazam限定: ルーティング(crustle専用方策)のA/B。

旧デッキで学習した crustle 専用方策(og_vs_crustle)が、新デッキ(g2top2_v032、
2スロット差)でも効果を保つかの確認。baseline=abl_5_full vs routing08=
abl_5_full_og_routing08(threshold 0.8、提出55477812と同じ)。
Windows spawn 対策で実ファイル化(stdin実行だと worker が __main__ を再import不可)。
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "league"))

import run_league  # noqa: E402

COMMON = dict(
    agent_a_name="ml_policy", agent_b_name="ml_policy",
    deck_a_path=str(_ROOT / "kaggle_replays/deck_search/candidates_ogerpon_stage3/g2top2_v032.csv"),
    deck_b_path=str(_ROOT / "kaggle_replays/meta_analysis/archetype_decks_g2/alakazam/01.csv"),
    weights_a_path=str(_ROOT / "sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"),
    weights_b_path=str(_ROOT / "sample_submission/ptcg_ai/learning/policy_weights_alakazam_g2.json"),
    config_base_b="abl_5_full",
    games=400, progress_every=50, workers=6,
)


def main() -> None:
    out = {}
    for tag, cfg_a, seed in (("baseline", "abl_5_full", 962000),
                             ("routing08", "abl_5_full_og_routing08", 963000)):
        s = run_league.run_league(config_base_a=cfg_a, seed_start=seed, **COMMON)
        ov = s["overall"]
        out[tag] = {"wins": ov["wins"], "games": ov["games"],
                    "wr": ov["wins"] / max(1, ov["games"])}
        print(tag, json.dumps(out[tag]), flush=True)
    print("FINAL", json.dumps(out))


if __name__ == "__main__":
    main()
