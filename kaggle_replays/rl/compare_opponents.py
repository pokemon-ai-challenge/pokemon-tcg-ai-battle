#!/usr/bin/env python3
"""アーキタイプごとに「7月版とgen2版のどちらを相手モデルに使うべきか」を決めるための実測。

背景: gen2 のフィールドで丸ごと学習した g2climb は climb を超えられなかった
(ローカル判定不能・LB 600.0 vs 823-831)。マッチ別に見ると rocket_mewtwo -26pt /
crustle -14.7pt と特定アーキで大きく劣化しており、「全部gen2」が最適ではない兆候が出ている。
gen2 は現行メタだがデータ量が減ったアーキがある(例 mega_lucario 1257->403デッキ、
archaludon 1078->287)一方、7月版は現行環境で絶滅したデッキを回している可能性がある。

そこで2つの軸を同時に測る:
  [強さ] (gen2重み+gen2デッキ) vs (7月重み+7月デッキ) の直接対決。強い方が
         訓練相手として厳しい = grounding として価値が高い(Strong Opponent の考え方)。
  [現行性] 両世代の代表デッキの一致枚数(60枚のmultiset intersection)。大きく違うなら
         7月版は「今は誰も使っていないデッキ」を回していることになり、強くても
         フィールドの現実性を損なう。

この2つを並べて出すだけで、採否の判断はしない(人が決める)。

使い方:
  python compare_opponents.py --games 120 --workers 14
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "league"))
sys.path.insert(0, str(_ROOT / "sample_submission"))

import run_league  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECK_JULY = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
DECK_G2 = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"

# 両世代に模倣ポリシーがあるアーキタイプ(gen2のみの ogerpon/froslass/omatsuri は比較不能)
BOTH = ["alakazam", "marnie_grimmsnarl_ex", "mega_lucario_ex", "archaludon_ex",
        "crustle", "dragapult_ex", "rocket_mewtwo_ex", "shirona_garchomp_ex"]


def deck_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as f:
        return [int(r[0]) for r in csv.reader(f) if r and r[0].strip().isdigit()]


def deck_overlap(a: Path, b: Path) -> int:
    """60枚デッキの一致枚数(multiset intersection)。"""
    from collections import Counter
    ca, cb = Counter(deck_ids(a)), Counter(deck_ids(b))
    return sum(min(v, cb[k]) for k, v in ca.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config-base", default="ml_lethal_attackplan_v0only")
    ap.add_argument("--archetypes", default=",".join(BOTH))
    args = ap.parse_args()

    import os
    os.chdir(_ROOT / "sample_submission")

    print(f"=== gen2 vs 7月 の相手モデル直接対決 ({args.games}試合, config={args.config_base}) ===", flush=True)
    print(f"{'archetype':<24}{'gen2勝率':>10}{'判定':>8}   デッキ一致", flush=True)
    rows = []
    for arch in [a for a in args.archetypes.split(",") if a]:
        # alakazam の7月版BCは本番の policy_weights.json そのもの(専用ファイル名が無い)。
        july_name = "policy_weights.json" if arch == "alakazam" else f"policy_weights_{arch}.json"
        wa, wb = WDIR / f"policy_weights_{arch}_g2.json", WDIR / july_name
        da, db = DECK_G2 / arch / "01.csv", DECK_JULY / arch / "01.csv"
        if not all(p.exists() for p in (wa, wb, da, db)):
            print(f"{arch:<24}  (両世代そろっていないのでスキップ)", flush=True)
            continue
        summary = run_league.run_league(
            agent_a_name="ml_policy", agent_b_name="ml_policy", games=args.games,
            deck_a_path=str(da), deck_b_path=str(db), seed_start=0, progress_every=0,
            weights_a_path=str(wa), weights_b_path=str(wb),
            config_base=args.config_base, workers=args.workers, log=lambda m: None,
        )
        ov = summary["overall"]
        wr = ov["win_rate"]
        ov60 = deck_overlap(da, db)
        # 120試合の95%CI半幅は約±9pt。それを超えたときだけ「強い/弱い」と言う。
        import math
        half = 1.96 * math.sqrt(wr * (1 - wr) / ov["games"])
        verdict = "gen2強" if wr - half > 0.5 else ("7月強" if wr + half < 0.5 else "差なし")
        print(f"{arch:<24}{wr*100:9.1f}%{verdict:>8}   {ov60}/60枚", flush=True)
        rows.append((arch, wr, verdict, ov60))

    print("\n--- まとめ ---", flush=True)
    print("gen2が有意に強い : " + ", ".join(a for a, _, v, _ in rows if v == "gen2強"), flush=True)
    print("7月が有意に強い  : " + ", ".join(a for a, _, v, _ in rows if v == "7月強"), flush=True)
    print("差なし           : " + ", ".join(a for a, _, v, _ in rows if v == "差なし"), flush=True)
    print("デッキが大きく変化(一致40枚未満) : "
          + ", ".join(f"{a}({o}枚)" for a, _, _, o in rows if o < 40), flush=True)


if __name__ == "__main__":
    main()
