#!/usr/bin/env python3
"""アーキタイプごとに『上位パイロットの複数デッキ』を deck.csv 形式で保存する。

将来の自己対戦(リーグ)用スパーリング相手のデッキ資産。1アーキタイプ1リストだと
その型に過学習するので、パイロット品質の高い順に distinct build を N 個取り、
ほぼ同一リスト(カードID多重集合が一致)は重複排除する。

パイロット品質 = leaderboard_score(高いほど良) を第一、rank_at_fetch(小さいほど良)を第二。

出力: meta_analysis/archetype_decks/<archetype>/NN.csv  (60行, 1行=カードID; deck.csv互換)
       meta_analysis/archetype_decks/<archetype>/manifest.json  (メタ情報)

使い方:
  python extract_archetype_decks.py [--top-decks 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_meta import (
    build_instances,
    deck_signature,
    load_episode_scores,
    load_labels,
)

_HERE = Path(__file__).parent
_OUT = _HERE / "archetype_decks"

SKIP = {"other"}


def pilot_quality(inst) -> tuple:
    """大きいほど良い並び替えキー。score優先、次に rank(小さいほど良→負号)。"""
    score = inst["score"] if inst["score"] is not None else -1e9
    rank = inst["rank"]
    rank_key = -rank if rank is not None else -1e9
    return (score, rank_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-decks", type=int, default=5)
    ap.add_argument("--min-quality-decks", type=int, default=1,
                    help="rank/score が全く付かないアーキタイプでも最低これだけは保存")
    # データ世代の切り替え口。既定(None)は7月プール = 従来と完全に同じ挙動。
    ap.add_argument("--deck-db", default=None)
    ap.add_argument("--deck-labels", default=None)
    ap.add_argument("--episodes", default=None, help="episodes_master.jsonl(rank/scoreの取得元)")
    ap.add_argument("--out", default=None, help="出力先ディレクトリ(既定 archetype_decks)")
    args = ap.parse_args()

    global _OUT
    if args.out:
        _OUT = Path(args.out)

    labels = load_labels(args.deck_labels)
    scores = load_episode_scores(args.episodes)
    instances = build_instances(labels, scores, args.deck_db)

    by_arch = {}
    for inst in instances:
        a = inst["archetype"]
        if a in SKIP:
            continue
        if len(inst["card_ids"]) != 60:
            continue  # 壊れたデッキは除外
        by_arch.setdefault(a, []).append(inst)

    _OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for a, insts in sorted(by_arch.items()):
        # パイロット品質の高い順。同一 signature は最初(=最良pilot)だけ残す。
        insts.sort(key=pilot_quality, reverse=True)
        picked = []
        seen = set()
        for inst in insts:
            sig = deck_signature(inst["card_ids"])
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(inst)
            if len(picked) >= args.top_decks:
                break

        adir = _OUT / a
        adir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, inst in enumerate(picked, 1):
            csv_path = adir / f"{i:02d}.csv"
            csv_path.write_text(
                "\n".join(str(c) for c in inst["card_ids"]) + "\n", encoding="utf-8"
            )
            manifest.append(
                {
                    "file": csv_path.name,
                    "team": inst["team"],
                    "rank_at_fetch": inst["rank"],
                    "leaderboard_score": inst["score"],
                    "episode_id": inst["episode_id"],
                    "player_index": inst["player_index"],
                }
            )
        (adir / "manifest.json").write_text(
            json.dumps({"archetype": a, "decks": manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary.append((a, len(picked), len(insts)))

    print(f"{'archetype':24s} saved  candidates")
    for a, k, tot in sorted(summary, key=lambda x: -x[2]):
        print(f"{a:24s} {k:5d}  {tot}")
    print(f"\nwrote decks under: {_OUT}")


if __name__ == "__main__":
    main()
