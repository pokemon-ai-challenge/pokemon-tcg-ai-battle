#!/usr/bin/env python3
"""保存済みのリーダーボードスナップショットから episodes_master.jsonl を復旧する。

なぜ必要か:
  fetch_top_episodes.py はダウンロード([4/5])を全部終えてからマスターインデックスを
  更新する([5/5])。そのため**ダウンロード途中で中断すると、リプレイ本体はディスクに
  あるのにインデックスに1行も載らない**。インデックスが無いと各プレイヤーの
  rank_at_fetch が null になり、
    - build_features.py --weight-scheme concentrated(rank<=20 を8倍)が実質無効になる
    - fetch_minority_archetype_episodes.py の team_id 逆引きが効かない
  という形で下流が静かに壊れる(実測: 2002件中395件しかインデックスされておらず、
  学習サンプルの rank_at_fetch 充足率が15%まで落ちていた)。

  リプレイ本体には info.TeamNames が入っており、取得時のリーダーボードは
  index_<世代>/leaderboard_history/*.json に保存済みなので、**Kaggle APIを一切叩かずに**
  team_id / rank_at_fetch / leaderboard_score_at_fetch を復元できる。

復元できないもの:
  episode_create_time / episode_end_time は episodes API のレスポンス由来で、
  スナップショットには無いため null になる(学習の重み付けには使われない)。

使い方:
  python rebuild_master_index.py --replays-dir replays_g2 \\
      --master-index-path index_g2/episodes_master.jsonl \\
      --leaderboard-history-dir index_g2/leaderboard_history
  python rebuild_master_index.py --dry-run   # 何行増えるか数えるだけ
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    load_existing_episode_ids,
)

_HERE = Path(__file__).parent


def load_leaderboard_snapshots(history_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """スナップショット群を1つの teamName -> {team_id, rank, score} に畳む。

    同じ大会の同じ時間帯のスナップショットなので、単純に「後から読んだ方で上書き」する。
    深層取得(-deep)のスナップショットは rank 201以降を含むので、上位版と合わせると
    rank 1〜2000 をカバーできる。
    """
    name_to_context: dict[str, dict] = {}
    used: list[str] = []
    for path in sorted(history_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("leaderboard", [])
        if not rows:
            continue
        for rank, row in enumerate(rows, 1):
            name_to_context[row["teamName"]] = {
                "team_id": row["teamId"],
                "rank": row.get("rank", rank),
                "score": float(row["score"]),
            }
        used.append(f"{path.name} ({len(rows)}行)")
    return name_to_context, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_HERE / "replays_g2"))
    parser.add_argument("--master-index-path", default=str(_HERE / "index_g2" / "episodes_master.jsonl"))
    parser.add_argument("--leaderboard-history-dir", default=str(_HERE / "index_g2" / "leaderboard_history"))
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--dry-run", action="store_true", help="書き込まずに件数だけ表示する")
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    master_index_path = Path(args.master_index_path)
    history_dir = Path(args.leaderboard_history_dir)

    name_to_context, used = load_leaderboard_snapshots(history_dir)
    print(f"リーダーボードスナップショット {len(used)}件から {len(name_to_context)} チームを復元")
    for u in used:
        print(f"  - {u}")

    already_indexed = load_existing_episode_ids(master_index_path)
    n_files = sum(1 for _ in replays_dir.glob("episode-*-replay.json"))
    print(f"リプレイ {n_files}件 / インデックス済み {len(already_indexed)}件 -> 未登録 {n_files - len(already_indexed)}件を復元します")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-rebuild"
    rows = build_master_rows(
        run_id=run_id,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        competition=args.competition,
        replays_dir=replays_dir,
        episode_meta={},  # 対戦日時は復元できない(null になる)
        name_to_context=name_to_context,
        already_indexed=already_indexed,
    )

    resolved = sum(1 for r in rows for p in r["players"] if p.get("team_id") is not None)
    total_players = sum(len(r["players"]) for r in rows)
    print(f"復元行数: {len(rows)}  (プレイヤー {total_players} 人中 {resolved} 人の team_id/rank を解決)")

    if args.dry_run:
        print("--dry-run のため書き込みませんでした")
        return
    append_master_rows(master_index_path, rows)
    print(f"{master_index_path} に {len(rows)} 行を追記しました(合計 {len(already_indexed) + len(rows)} 行)")


if __name__ == "__main__":
    main()
