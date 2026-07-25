#!/usr/bin/env python3
"""リプレイの60枚デッキ選択アクションを抽出し、deck_db.jsonl を作る。

kaggle_replays/replays/*.json の各リプレイについて、両プレイヤーが最初に選んだ
60枚のデッキ(steps[1][player]["action"])を取り出し、episodes_master.jsonl の
team_name / rank_at_fetch と結合して出力する。

カードID→カード名の変換は data/JP_Card_Data.csv を使う
(rough_predictor.json のカード名は日本語のため)。

使い方:
  python extract_decks.py
  python extract_decks.py --replays-dir ../replays --out ./output/deck_db.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent


def load_card_names(csv_path: Path) -> dict[int, str]:
    """data/JP_Card_Data.csv からカードID→カード名の対応表を作る。"""
    id_to_name: dict[int, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_id = row.get("カード ID")
            name = row.get("カード名")
            if card_id is None or name is None:
                continue
            try:
                id_to_name[int(card_id)] = name
            except ValueError:
                continue
    return id_to_name


def load_master_index(master_path: Path) -> dict[str, dict]:
    if not master_path.exists():
        return {}
    index: dict[str, dict] = {}
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["episode_id"]] = row
    return index


def extract_decks_from_replay(
    replay: dict,
    episode_id: str,
    master_row: dict | None,
    id_to_name: dict[int, str],
):
    steps = replay["steps"]
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    players_meta = {p["player_index"]: p for p in master_row["players"]} if master_row else {}

    if len(steps) < 2:
        raise ValueError(f"episode {episode_id}: too few steps ({len(steps)}) to contain a deck-selection action")

    for player_index in (0, 1):
        action = steps[1][player_index]["action"]
        if not action or len(action) != 60:
            raise ValueError(
                f"episode {episode_id} player {player_index}: expected a 60-card deck action, "
                f"got {len(action) if action else 0} cards"
            )
        own_meta = players_meta.get(player_index, {})
        name_counts = Counter(id_to_name.get(card_id, str(card_id)) for card_id in action)
        yield {
            "episode_id": episode_id,
            "player_index": player_index,
            "team_name": team_names[player_index] if player_index < len(team_names) else None,
            "rank_at_fetch": own_meta.get("rank_at_fetch"),
            "deck_card_ids": action,
            "deck_card_names": dict(name_counts),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_REPO_ROOT / "kaggle_replays" / "replays"))
    parser.add_argument(
        "--master-index-path",
        default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument("--card-csv", default=str(_REPO_ROOT / "data" / "JP_Card_Data.csv"))
    parser.add_argument("--out", default=str(_HERE / "output" / "deck_db.jsonl"))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    id_to_name = load_card_names(Path(args.card_csv))
    master_index = load_master_index(Path(args.master_index_path))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    replay_paths = sorted(replays_dir.glob("episode-*-replay.json"))
    n_files = 0
    n_decks = 0
    n_errors = 0
    errors: list[str] = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, replay_path in enumerate(replay_paths, start=1):
            episode_id = replay_path.stem.split("-")[1]
            master_row = master_index.get(episode_id)
            try:
                with replay_path.open(encoding="utf-8") as rf:
                    replay = json.load(rf)
                for deck_row in extract_decks_from_replay(replay, episode_id, master_row, id_to_name):
                    out_f.write(json.dumps(deck_row, ensure_ascii=False) + "\n")
                    n_decks += 1
                n_files += 1
            except Exception as exc:  # noqa: BLE001 - keep going on any per-file failure
                n_errors += 1
                errors.append(f"{replay_path.name}: {exc!r}")

            if i % 100 == 0 or i == len(replay_paths):
                print(f"  {i}/{len(replay_paths)} replays processed...", file=sys.stderr)

    print(f"{n_files}件のリプレイから{n_decks}件のデッキを {out_path} に書き出しました")
    if n_errors:
        print(f"エラー: {n_errors}件のリプレイでエラーが発生しました")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ...他 {len(errors) - 20} 件")


if __name__ == "__main__":
    main()
