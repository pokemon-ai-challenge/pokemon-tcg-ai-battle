#!/usr/bin/env python3
"""アーキタイプごとの代表カードプール（card_id -> median枚数 / inclusion_rate）を集計する。

deck_db.jsonl（実際の60枚デッキ、extract_decks.py の出力）と deck_labels.jsonl
（アーキタイプラベル、label_decks.py の出力）を episode_id / player_index で結合し、
アーキタイプごとに全デッキを集計して、非公開情報推定レイヤー（Phase 2, OpponentHiddenState）が
「そのアーキタイプの代表60枚リスト」として使う card_id -> 採用枚数テーブルを作る。

deck_predictor パイプラインの手順1〜2（extract_decks.py -> label_decks.py）の直後に置く軽量集計。
同じ card_id 空間・同じアーキタイプ語彙（hybrid_predictor の classes と一致）で完結する。

出力（output/model/archetype_card_pool.json）:
    {
      "meta": {"built_at": "...", "n_decks_by_archetype": {"mega_lucario_ex": 42, ...}, ...},
      "archetypes": {
        "mega_lucario_ex": {
          "card_counts": {"678": {"median": 2, "inclusion_rate": 0.95}, ...}
        },
        ...
      }
    }

- ``median``: そのアーキタイプの全デッキにおける、その card_id の採用枚数の中央値
  （採用していないデッキは 0 枚として母数に含める）。過半数のデッキに入っていないカードは median=0。
- ``inclusion_rate``: その card_id を1枚以上採用しているデッキの割合（0.0-1.0）。
- ``other`` も含め、ラベルが付いた全アーキタイプを出力する（キー数は hybrid_predictor の classes と
  一致）。ただし ``other`` は雑多なデッキの寄せ集めで代表60枚に意味が無いため、ランタイム
  （OpponentHiddenState）側で「代表リスト無し」として特別扱いされる（本スクリプトは集計だけ行う）。

使い方:
  python build_archetype_pool.py
  python build_archetype_pool.py --deploy
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_DEPLOY_TARGET = (
    _REPO_ROOT / "sample_submission" / "ptcg_ai" / "hidden_information" / "archetype_card_pool.json"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_labels(path: Path) -> dict[tuple[str, int], str]:
    """(episode_id, player_index) -> archetype の対応表を作る。"""
    labels: dict[tuple[str, int], str] = {}
    for row in load_jsonl(path):
        labels[(row["episode_id"], row["player_index"])] = row["archetype"]
    return labels


def build_pool(
    deck_rows: list[dict], labels: dict[tuple[str, int], str]
) -> tuple[dict[str, dict], Counter]:
    """アーキタイプごとに card_id -> {median, inclusion_rate} を集計する。

    戻り値: (archetypes payload, アーキタイプ別デッキ数の Counter)。
    """
    # archetype -> list of (card_id -> 枚数) デッキ。各デッキ内の同名 card_id を Counter で数える。
    decks_by_archetype: dict[str, list[Counter]] = defaultdict(list)
    n_missing_label = 0

    for row in deck_rows:
        key = (row["episode_id"], row["player_index"])
        archetype = labels.get(key)
        if archetype is None:
            n_missing_label += 1
            continue
        deck_counts: Counter[int] = Counter(int(cid) for cid in row["deck_card_ids"])
        decks_by_archetype[archetype].append(deck_counts)

    if n_missing_label:
        print(f"警告: ラベルの見つからないデッキが {n_missing_label} 件ありました（集計から除外）", file=sys.stderr)

    archetypes_payload: dict[str, dict] = {}
    n_decks_by_archetype: Counter[str] = Counter()

    for archetype, decks in decks_by_archetype.items():
        n_decks = len(decks)
        n_decks_by_archetype[archetype] = n_decks

        # そのアーキタイプに登場する全 card_id を集める。
        all_card_ids: set[int] = set()
        for deck in decks:
            all_card_ids.update(deck.keys())

        card_counts: dict[str, dict] = {}
        for card_id in sorted(all_card_ids):
            # 全デッキでのこの card_id の採用枚数（未採用は 0）。
            per_deck_counts = [deck.get(card_id, 0) for deck in decks]
            n_including = sum(1 for c in per_deck_counts if c > 0)
            # median は 0 込みで計算する（過半数が未採用なら median=0 になり、代表リストから外れる）。
            median = int(round(statistics.median(per_deck_counts)))
            inclusion_rate = n_including / n_decks if n_decks else 0.0
            card_counts[str(card_id)] = {
                "median": median,
                "inclusion_rate": round(inclusion_rate, 4),
            }

        archetypes_payload[archetype] = {"card_counts": card_counts}

    return archetypes_payload, n_decks_by_archetype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db.jsonl"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument("--out", default=str(_HERE / "output" / "model" / "archetype_card_pool.json"))
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="sample_submission/ptcg_ai/hidden_information/archetype_card_pool.json へコピーする",
    )
    args = parser.parse_args()

    deck_db_path = Path(args.deck_db)
    labels_path = Path(args.deck_labels)
    for p in (deck_db_path, labels_path):
        if not p.exists():
            print(f"エラー: 入力ファイルが見つかりません: {p}（先に extract_decks.py / label_decks.py を実行してください）", file=sys.stderr)
            sys.exit(1)

    deck_rows = load_jsonl(deck_db_path)
    labels = load_labels(labels_path)

    archetypes_payload, n_decks_by_archetype = build_pool(deck_rows, labels)

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_decks_total": sum(n_decks_by_archetype.values()),
            "n_archetypes": len(archetypes_payload),
            "n_decks_by_archetype": dict(n_decks_by_archetype.most_common()),
            "source": {"deck_db": str(deck_db_path), "deck_labels": str(labels_path)},
        },
        "archetypes": archetypes_payload,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"アーキタイプ数: {len(archetypes_payload)}  総デッキ数: {sum(n_decks_by_archetype.values())}")
    print(f"{'アーキタイプ':28s} {'デッキ数':>6s} {'代表カード種類数':>12s}")
    for archetype, n in n_decks_by_archetype.most_common():
        n_repr = sum(1 for s in archetypes_payload[archetype]["card_counts"].values() if s["median"] > 0)
        print(f"{archetype:28s} {n:6d} {n_repr:12d}")
    print(f"\n代表カードプールを {out_path} に書き出しました")

    if args.deploy:
        _DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        _DEPLOY_TARGET.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"デプロイ用に {_DEPLOY_TARGET} にもコピーしました")


if __name__ == "__main__":
    main()
