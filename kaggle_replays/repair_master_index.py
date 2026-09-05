#!/usr/bin/env python3
"""取得済みだが episodes_master.jsonl に載っていないリプレイに、保存済みの
リーダーボードスナップショットから順位/スコアを付けて追記する。

背景:
  fetch_top_episodes.py はインデックスを実行の最後([5/5])にまとめて書くため、
  途中で中断するとリプレイ本体だけが replays/ に増えてインデックスが古いまま
  になることがある。このスクリプトはオフラインで復元する。

  - リプレイ本体の JSON には info.TeamNames (両プレイヤーのチーム名) が入っている。
  - リーダーボードのスナップショット(index/leaderboard_history/leaderboard-*.json)
    には teamName -> rank/score の対応がある。
  - この2つを突き合わせて episodes_master.jsonl の行を復元する。

  ここで使う「順位」はスナップショット取得時点のものであり、リプレイ対戦時点の
  順位とは限らない(fetch_top_episodes.py の通常フローと同じ制約)。

  リプレイ本体からは episode_id と TeamNames しか分からないため、
  episode_create_time / episode_end_time は常に null になる(このスクリプトでは
  Kaggle API を呼ばないため取得しようがない)。

使い方:
  # まずは dry-run で何件追記されるか確認
  python repair_master_index.py --dry-run

  # 実際に追記
  python repair_master_index.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    load_existing_episode_ids,
)

# fetch_top_episodes.py が書く通常のスナップショットのファイル名パターン。
# fetch_deep_decks.py 由来の "-deep" サフィックス付きファイルはスキーマが違う
# (rank_from/rank_to で一部の順位帯だけを持つ)ため、既定の「最新」選択からは除外する。
_SNAPSHOT_NAME_RE = re.compile(r"^leaderboard-\d{8}T\d{6}Z\.json$")


def find_latest_leaderboard(leaderboard_history_dir: Path) -> Path:
    candidates = sorted(
        p for p in leaderboard_history_dir.glob("leaderboard-*.json")
        if _SNAPSHOT_NAME_RE.match(p.name)
    )
    if not candidates:
        raise FileNotFoundError(
            f"{leaderboard_history_dir} に通常形式のリーダーボードスナップショットが見つかりません"
        )
    return candidates[-1]  # ファイル名のタイムスタンプは辞書順=時系列順


def load_name_to_context(leaderboard_path: Path) -> tuple[dict, dict]:
    """スナップショットを読み、(name_to_context, snapshot) を返す。"""
    with leaderboard_path.open(encoding="utf-8") as f:
        snapshot = json.load(f)
    name_to_context = {
        row["teamName"]: {
            "team_id": row["teamId"],
            "rank": row["rank"],
            # スナップショット上の score は文字列。既存インデックスの
            # leaderboard_score_at_fetch は数値なので型を揃える。
            "score": float(row["score"]),
        }
        for row in snapshot["leaderboard"]
    }
    return name_to_context, snapshot


def find_missing_episode_ids(replays_dir: Path, already_indexed: set[str]) -> list[str]:
    ids = []
    for replay_path in sorted(replays_dir.glob("episode-*-replay.json")):
        episode_id = replay_path.stem.split("-")[1]
        if episode_id in already_indexed:
            continue
        ids.append(episode_id)
    return ids


def summarize(rows: list[dict]) -> tuple[int, int, int]:
    """(both_ranked, one_ranked, none_ranked) を返す。"""
    both = one = none = 0
    for row in rows:
        matched = sum(1 for p in row["players"] if p["rank_at_fetch"] is not None)
        if matched == 2:
            both += 1
        elif matched == 1:
            one += 1
        else:
            none += 1
    return both, one, none


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--leaderboard", default=None,
        help="使用するリーダーボードスナップショットのパス。省略時は "
        "index/leaderboard_history/ 内の最新の通常スナップショットを使う",
    )
    parser.add_argument(
        "--replays-dir", default=str(Path(__file__).parent / "replays"),
    )
    parser.add_argument(
        "--master-index-path",
        default=str(Path(__file__).parent / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="追記せず、何件追記されるか/何件に順位が付くかを表示するだけ",
    )
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    master_index_path = Path(args.master_index_path)
    leaderboard_history_dir = Path(__file__).parent / "index" / "leaderboard_history"

    leaderboard_path = (
        Path(args.leaderboard) if args.leaderboard
        else find_latest_leaderboard(leaderboard_history_dir)
    )
    print(f"リーダーボードスナップショット: {leaderboard_path}")
    name_to_context, snapshot = load_name_to_context(leaderboard_path)
    print(
        f"  run_id={snapshot['run_id']} fetched_at={snapshot['fetched_at']} "
        f"top_n={snapshot.get('top_n')} 収録チーム数={len(name_to_context)}"
    )

    already_indexed = load_existing_episode_ids(master_index_path)
    print(f"既存インデックス: {master_index_path} ({len(already_indexed)}件)")

    missing_ids = find_missing_episode_ids(replays_dir, already_indexed)
    print(f"replays_dir 内にあるがインデックス未登録のエピソード: {len(missing_ids)}件")

    if not missing_ids:
        print("追記対象なし。終了します。")
        return

    new_rows = build_master_rows(
        run_id=snapshot["run_id"],
        fetched_at=snapshot["fetched_at"],
        competition=snapshot["competition"],
        replays_dir=replays_dir,
        episode_meta={},  # オフライン復元のため episode_create_time/episode_end_time は常に null
        name_to_context=name_to_context,
        already_indexed=already_indexed,
        episode_ids=missing_ids,
    )
    both, one, none = summarize(new_rows)
    print(
        f"episode_create_time / episode_end_time はこのスクリプトでは取得できないため "
        f"全{len(new_rows)}行で null になります"
    )
    print(
        f"追記予定: {len(new_rows)}行 "
        f"(両者に順位が付く: {both}件 / 片方のみ: {one}件 / どちらも付かない: {none}件)"
    )

    if args.dry_run:
        print("dry-run のため書き込みは行いません。")
        return

    before_lines = len(already_indexed)
    append_master_rows(master_index_path, new_rows)
    after_lines = len(load_existing_episode_ids(master_index_path))
    print(f"書き込み完了: {master_index_path}")
    print(f"インデックス件数: {before_lines} -> {after_lines}")

    print("追記されたサンプル2件:")
    for row in new_rows[:2]:
        print(f"  {json.dumps(row, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
