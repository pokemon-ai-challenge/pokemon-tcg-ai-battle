#!/usr/bin/env python3
"""オーガポンの専用モデルが、対象相手で汎用モデル(mixogerpon)を本当に上回るかの独立検証。

学習時の eval は「学習に使った相手そのもの」なので選択バイアスが乗る。ここでは本番相当の
config で試合数を増やして測り直す。相手の重み/デッキは mix フィールドの定義と同じものを使う。
"""
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "league")); sys.path.insert(0, str(_ROOT / "sample_submission"))
import run_league  # noqa: E402
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECK_G2 = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
OG_DECK = str(DECK_G2 / "ogerpon_teal_ex" / "01.csv")
# mix フィールドでの相手定義(alakazam はミラー扱いで climb を使う)
OPP = {"crustle": (str(WDIR / "policy_weights_crustle_g2.json"), str(DECK_G2 / "crustle" / "01.csv")),
       "alakazam": (str(WDIR / "policy_weights_alakazam_rl_climb.json"), str(DECK_G2 / "alakazam" / "01.csv"))}

def run(arch, w, games, workers, cfg):
    ow, od = OPP[arch]
    s = run_league.run_league(agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=OG_DECK, deck_b_path=od, seed_start=0, progress_every=0,
        weights_a_path=str(WDIR / w), weights_b_path=ow, config_base=cfg, workers=workers,
        log=lambda m: None)
    o = s["overall"]; return o["win_rate"], o["wins"], o["games"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--config-base", default="ml_lethal_attackplan_v0only")
    a = ap.parse_args()
    os.chdir(_ROOT / "sample_submission")
    GEN = "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"
    print(f"=== 専用モデル検証({a.games}試合/条件, config={a.config_base}) ===", flush=True)
    for arch in ("crustle", "alakazam"):
        sp = f"policy_weights_ogerpon_teal_ex_rl_og_vs_{arch}.json"
        g_wr, g_w, n = run(arch, GEN, a.games, a.workers, a.config_base)
        s_wr, s_w, _ = run(arch, sp, a.games, a.workers, a.config_base)
        d = s_wr - g_wr
        se = math.sqrt(g_wr * (1 - g_wr) / n + s_wr * (1 - s_wr) / n)
        lo, hi = d - 1.96 * se, d + 1.96 * se
        print(f"  vs {arch:<10} 汎用 {g_wr*100:5.1f}% ({g_w}/{n})  ->  専用 {s_wr*100:5.1f}% ({s_w}/{n})  "
              f"Δ{d*100:+.1f}pt  95%CI [{lo*100:+.1f}, {hi*100:+.1f}]  "
              f"{'**採用可**' if lo > 0 else '判定不能(不採用)'}", flush=True)

if __name__ == "__main__":
    main()
