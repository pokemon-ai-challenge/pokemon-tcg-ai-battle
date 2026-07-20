#!/usr/bin/env python3
"""Kaggleリーダーボード上位チームの対戦リプレイを取得する。

事前準備:
  pip install kaggle
  Kaggle設定ページ(https://www.kaggle.com/settings/api)で発行したトークンを
  ~/.kaggle/access_token に保存(kaggle CLI 2.x の新形式)。

使い方:
  python fetch_top_episodes.py --competition pokemon-tcg-ai-battle --top 20

自分のチームのログだけを取りたい場合は fetch_my_episodes.py を使う。

データ保存方針:
  リーダーボードの順位はKaggle上で日々変動し、実行するたびに違う結果になる。
  そのため「今回の実行(run)」ごとに以下を残す。

  - index/leaderboard_history/leaderboard-<run_id>.json
      その実行時点でのリーダーボードのスナップショット(上書きせず蓄積)。
      「いつの何位か」を後から追えるようにするための履歴。
  - index/episodes_master.jsonl
      エピソード単位の累積インデックス。各行に両プレイヤーの teamId・
      取得時点の順位/スコア・対戦が実際に行われた日時(episode_create_time)
      を記録する。同じエピソードは初めて記録されたときの情報を保持し続け、
      再実行しても上書きしない(追記のみ)。fetch_my_episodes.py とこの
      ファイルを共有するので、上位ログと自分のログが同じインデックスに載る。
      機械学習で学習データに重み付けする際は、このファイルを
      extract_training_data.py が結合してくれる。
  - replays/episode-<id>-replay.json
      リプレイ本体(全run共通のフラットなプール。episode_idで重複排除)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    download_replay,
    fetch_episodes,
    load_existing_episode_ids,
    run_kaggle_json,
)


def fetch_leaderboard(competition: str, top_n: int) -> list[dict]:
    page_size = min(max(top_n, 1), 200)
    rows = run_kaggle_json(
        ["competitions", "leaderboard", competition, "-s", "--page-size", str(page_size)]
    )
    return rows[:top_n]


def fetch_team_submission_ids(team_id: int) -> list[int]:
    rows = run_kaggle_json(["competitions", "team-submissions", str(team_id)])
    return [row["id"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--top", type=int, default=20, help="上位何チームを対象にするか")
    parser.add_argument(
        "--submissions-per-team", type=int, default=1,
        help="各チームにつき何個の提出(新しい順)を辿るか",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=300,
        help="ダウンロードするエピソード数の上限(1チームの提出だけで1000件近くあることがあるため既定値を設定。"
        "無制限にしたい場合は非常に大きい値を指定するか 0 以下を指定)",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "replays"))
    parser.add_argument(
        "--leaderboard-history-dir",
        default=str(Path(__file__).parent / "index" / "leaderboard_history"),
    )
    parser.add_argument(
        "--master-index-path",
        default=str(Path(__file__).parent / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="API呼び出し間隔(秒)。kaggle CLI自体の起動コストが1回あたり約1〜2秒あるため、"
        "この値を下げても速度への影響は限定的(体感を大きく変えたいなら --max-episodes を絞る方が効果的)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_history_dir = Path(args.leaderboard_history_dir)
    leaderboard_history_dir.mkdir(parents=True, exist_ok=True)
    master_index_path = Path(args.master_index_path)
    master_index_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fetched_at = datetime.now(timezone.utc).isoformat()

    print(f"[1/5] リーダーボード取得: {args.competition} 上位{args.top}チーム (run_id={run_id})")
    leaderboard = fetch_leaderboard(args.competition, args.top)
    for rank, row in enumerate(leaderboard, 1):
        print(f"  #{rank} {row['teamName']} (teamId={row['teamId']}, score={row['score']})")

    leaderboard_snapshot_path = leaderboard_history_dir / f"leaderboard-{run_id}.json"
    leaderboard_snapshot_path.write_text(
        json.dumps(
            {
                "competition": args.competition,
                "run_id": run_id,
                "fetched_at": fetched_at,
                "top_n": args.top,
                "leaderboard": [
                    {"rank": rank, **row} for rank, row in enumerate(leaderboard, 1)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    name_to_context = {
        row["teamName"]: {"team_id": row["teamId"], "rank": rank, "score": float(row["score"])}
        for rank, row in enumerate(leaderboard, 1)
    }

    print(f"[2/5] 各チームの提出IDを取得(チームごと最新{args.submissions_per_team}件)")
    submission_ids: list[int] = []
    for row in leaderboard:
        team_id = row["teamId"]
        try:
            subs = fetch_team_submission_ids(team_id)
        except subprocess.CalledProcessError as e:
            print(f"  警告: team {team_id} ({row['teamName']}) の提出取得に失敗: {e.stderr}", file=sys.stderr)
            continue
        submission_ids.extend(subs[: args.submissions_per_team])
        time.sleep(args.sleep)

    print(f"[3/5] エピソード一覧を取得({len(submission_ids)}件の提出から)")
    episode_meta: dict[str, dict] = {}
    for sub_id in submission_ids:
        try:
            episodes = fetch_episodes(sub_id)
        except subprocess.CalledProcessError as e:
            print(f"  警告: submission {sub_id} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
            continue
        for ep in episodes:
            episode_meta[str(ep["id"])] = ep
        time.sleep(args.sleep)

    episode_id_list = sorted(episode_meta.keys(), key=int)
    if args.max_episodes is not None and args.max_episodes > 0:
        episode_id_list = episode_id_list[-args.max_episodes:]

    print(f"[4/5] リプレイ{len(episode_id_list)}件をダウンロード -> {out_dir}")
    downloaded: list[str] = []
    for i, eid in enumerate(episode_id_list, 1):
        dest = out_dir / f"episode-{eid}-replay.json"
        if dest.exists():
            print(f"  ({i}/{len(episode_id_list)}) episode {eid} は取得済み、スキップ")
            downloaded.append(str(dest))
            continue
        try:
            path = download_replay(int(eid), out_dir)
            downloaded.append(str(path))
            print(f"  ({i}/{len(episode_id_list)}) episode {eid} -> {path.name}")
        except subprocess.CalledProcessError as e:
            print(f"  警告: episode {eid} のリプレイ取得に失敗: {e.stderr}", file=sys.stderr)
        time.sleep(args.sleep)

    print(f"[5/5] エピソードマスターインデックスを更新 -> {master_index_path}")
    already_indexed = load_existing_episode_ids(master_index_path)
    new_rows = build_master_rows(
        run_id=run_id,
        fetched_at=fetched_at,
        competition=args.competition,
        replays_dir=out_dir,
        episode_meta=episode_meta,
        name_to_context=name_to_context,
        already_indexed=already_indexed,
    )
    append_master_rows(master_index_path, new_rows)

    print(
        f"完了: リプレイ{len(downloaded)}件を保存、"
        f"マスターインデックスに{len(new_rows)}件を追記、"
        f"リーダーボードスナップショットを {leaderboard_snapshot_path} に保存しました"
    )


if __name__ == "__main__":
    main()
