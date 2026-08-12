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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    download_replay,
    download_replays_parallel,
    fetch_episodes,
    get_retry_stats,
    load_existing_episode_ids,
    run_kaggle_json,
    set_min_interval,
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
    parser.add_argument(
        "--workers", type=int, default=1,
        help="ダウンロード/メタデータ取得の並列数(既定1=逐次、従来どおり)。1件あたり実測約5秒で"
        "ほぼI/O待ちのため、数千件取得するときは6〜8程度にすると大幅に短縮できる",
    )
    parser.add_argument(
        "--min-interval", type=float, default=None,
        help="kaggle CLI 呼び出しの最小間隔(秒、全スレッド共有)。並列時に429を避けるための"
        "レート制限。既定は workers>1 のとき0.7、逐次のとき0(従来どおり)",
    )
    args = parser.parse_args()

    min_interval = args.min_interval if args.min_interval is not None else (0.7 if args.workers > 1 else 0.0)
    set_min_interval(min_interval)

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

    def _team_submissions(row: dict) -> list[int]:
        team_id = row["teamId"]
        try:
            subs = fetch_team_submission_ids(team_id)
        except subprocess.CalledProcessError as e:
            print(f"  警告: team {team_id} ({row['teamName']}) の提出取得に失敗: {e.stderr}", file=sys.stderr)
            return []
        time.sleep(args.sleep)
        return subs[: args.submissions_per_team]

    def _submission_episodes(sub_id: int) -> list[dict]:
        try:
            episodes = fetch_episodes(sub_id)
        except subprocess.CalledProcessError as e:
            print(f"  警告: submission {sub_id} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
            return []
        time.sleep(args.sleep)
        return episodes

    # メタデータ取得も kaggle CLI 1回あたり1〜2秒かかり、チーム数に比例して効いてくる
    # (200チームで約10分)。ダウンロードと同じ --workers でここも並列化する。
    print(f"[2/5] 各チームの提出IDを取得(チームごと最新{args.submissions_per_team}件, workers={args.workers})")
    submission_ids: list[int] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for subs in pool.map(_team_submissions, leaderboard):
                submission_ids.extend(subs)
    else:
        for row in leaderboard:
            submission_ids.extend(_team_submissions(row))

    print(f"[3/5] エピソード一覧を取得({len(submission_ids)}件の提出から)")
    episode_meta: dict[str, dict] = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            episode_lists = list(pool.map(_submission_episodes, submission_ids))
    else:
        episode_lists = [_submission_episodes(sub_id) for sub_id in submission_ids]
    for episodes in episode_lists:
        for ep in episodes:
            episode_meta[str(ep["id"])] = ep

    episode_id_list = sorted(episode_meta.keys(), key=int)
    if args.max_episodes is not None and args.max_episodes > 0:
        episode_id_list = episode_id_list[-args.max_episodes:]

    print(f"[4/5] リプレイ{len(episode_id_list)}件をダウンロード -> {out_dir} (workers={args.workers})")
    downloaded: list[str] = []
    if args.workers > 1:
        ok, failed = download_replays_parallel(episode_id_list, out_dir, args.workers)
        downloaded = [str(out_dir / f"episode-{eid}-replay.json") for eid in ok]
        if failed:
            print(f"  {len(failed)}件のダウンロードに失敗しました(再実行すれば続きから取得できます)")
        episode_id_list_iter: list[str] = []
    else:
        episode_id_list_iter = episode_id_list
    for i, eid in enumerate(episode_id_list_iter, 1):
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
        f"(429再試行 {get_retry_stats()['429']}回, min-interval={min_interval}s)"
    )


if __name__ == "__main__":
    main()
