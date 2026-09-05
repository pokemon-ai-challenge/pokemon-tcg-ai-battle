#!/usr/bin/env python3
"""オーガポン用ルーティングの誤発火監査。

`config["routing"]` は意思決定ごとに相手デッキ予測の argmax を見て専用重みへ切り替える。
狙った相手(crustle / alakazam)で発火し、**それ以外では発火しない**ことを実測で確かめる。
発火してはいけない対面で専用モデルが動くと、その専用モデルは他対面を学習していないので
一方的に弱くなる(前回のフーディン版でも同じ監査を通してから採用した)。

_compute_route をスパイして「decision ごとにどの重みへルーティングされたか」を数える。
勝敗は見ない(それは別途 field 評価で測る)。逐次実行=遅いが監査には十分。

使い方:
  python _og_routing_audit.py --games 6
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_SUB))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# mix フィールドと同じ相手構成(アーキ, 相手重みの接尾辞, デッキdir)
FIELD = [
    ("marnie_grimmsnarl_ex", "_g2", "archetype_decks_g2"), ("alakazam", "CLIMB", "archetype_decks_g2"),
    ("mega_lucario_ex", "", "archetype_decks"), ("dragapult_ex", "_g2", "archetype_decks_g2"),
    ("mega_froslass_ex", "_g2", "archetype_decks_g2"), ("ogerpon_teal_ex", "_g2", "archetype_decks_g2"),
    ("archaludon_ex", "_g2", "archetype_decks_g2"), ("crustle", "_g2", "archetype_decks_g2"),
    ("shirona_garchomp_ex", "_g2", "archetype_decks_g2"), ("omatsuri_ondo", "_g2", "archetype_decks_g2"),
    ("rocket_mewtwo_ex", "_g2", "archetype_decks_g2"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=6, help="1相手あたりの試合数")
    ap.add_argument("--config", default="abl_5_full_og_routing")
    args = ap.parse_args()

    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.ml_policy import ml_policy_agent as MA

    wdir = _SUB / "ptcg_ai" / "learning"
    mdir = _ROOT / "kaggle_replays" / "meta_analysis"
    og_deck = runner.load_deck(mdir / "archetype_decks_g2" / "ogerpon_teal_ex" / "01.csv")
    climb = str(wdir / "policy_weights_alakazam_rl_climb.json")

    def mk(cfg_name, weights):
        c = agents.load_config_copy(cfg_name)
        c["policy_weights_path"] = weights
        return agents.make_ml_policy_agent(c)

    me = mk(args.config, str(wdir / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"))

    routed: Counter = Counter()
    original = MA._compute_route

    def spy(obs, config, _o=original):
        path = _o(obs, config)
        routed[Path(path).stem if path else "base"] += 1
        return path

    print(f"=== ルーティング発火監査 (config={args.config}, {args.games}試合/相手) ===", flush=True)
    print(f"{'相手':<24}{'発火率':>8}  内訳", flush=True)
    MA._compute_route = spy
    try:
        for arch, suffix, ddir in FIELD:
            routed.clear()
            opp_w = climb if suffix == "CLIMB" else str(wdir / f"policy_weights_{arch}{suffix}.json")
            opp = mk("abl_5_full", opp_w)
            opp_deck = runner.load_deck(mdir / ddir / arch / "01.csv")
            for g in range(args.games):  # seed引数なし(runner側で内部管理)
                try:
                    runner.play_game(me, opp, list(og_deck), list(opp_deck))
                except Exception as e:  # noqa: BLE001 -- 監査は落とさない
                    print(f"  ({arch} g{g} 失敗: {type(e).__name__})", flush=True)
            total = sum(routed.values())
            fired = total - routed.get("base", 0)
            detail = ", ".join(f"{k.replace('policy_weights_ogerpon_teal_ex_rl_', '')}×{v}"
                               for k, v in routed.most_common() if k != "base") or "なし"
            mark = ""
            if arch in ("crustle", "alakazam") and fired == 0:
                mark = "  ← 狙った相手で発火せず"
            if arch not in ("crustle", "alakazam") and fired > 0:
                mark = "  ← **誤発火**"
            print(f"  {arch:<22}{(fired / total * 100 if total else 0):7.1f}%  {detail}{mark}", flush=True)
    finally:
        MA._compute_route = original


if __name__ == "__main__":
    main()
