#!/usr/bin/env python3
"""Kaggleリーダーボードの深い順位帯(既定: 201〜2000位)から、チームあたり少数の
リプレイを広く薄く集める(「二層データ取得戦略」の深層側)。

詳細: sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md
      の「二層データ取得戦略」節を参照。

目的はデッキリスト収集(各リプレイの step0 アクションに60枚デッキが入っている)。
1チームはほぼ同じデッキを使い続けるため、対局数を稼ぐ必要はなく
チームあたり1〜2エピソードで十分 — その代わりチーム数(=ユニークデッキ数)を
広く稼ぐ。上位〜200位を全量取得する fetch_top_episodes.py とは役割が異なる。

事前準備は fetch_top_episodes.py と同じ(kaggle CLI インストール・認証済みが前提)。

使い方:
  python fetch_deep_decks.py --rank-from 201 --rank-to 2000 --episodes-per-team 2

調査メモ(実装時に実CLIで検証した内容):
  - `kaggle competitions leaderboard <comp> -s` は 1回の呼び出しで最大200件までしか
    返せない(--page-size の上限が200)。200位を超える順位を取るには、レスポンスの
    先頭に付与される非JSON行 "Next Page Token = <token>" を次回呼び出しの
    --page-token に渡してページを進める(公式にドキュメント化されたオプションではないが、
    `kaggle competitions leaderboard --help` に --page-token が存在し、実機で
    ページが連続することを確認済み)。--format json を付けてもこの行はJSON本体の
    "前" にそのまま出力されるため、_common.run_kaggle_json_with_page_token で
    JSON本体と分離して取り出している。
  - 順位範囲の先頭(--rank-from)にジャンプする方法はない(トークンは前のページの
    続きしか指せない)。そのため 1 位から順にページを送りながら rank_from 未満の
    ページは中身を捨てて読み進める(1ページ200件なので rank_to=2000 でも
    高々10回のAPI呼び出しで済み、コストは無視できる)。
  - チームの team-submissions / episodes API は順位に関係なく同じ形式で使える
    (実機確認: 201位付近のチームでも `kaggle competitions team-submissions <id>` /
    `kaggle competitions episodes <submission_id>` は上位チームと同じレスポンスを返す)。
    そのため fetch_top_episodes.py と同じ _common.fetch_episodes() をそのまま使える。

データ保存方針は fetch_top_episodes.py と同じ(README.md 参照):
  - index/leaderboard_history/leaderboard-<run_id>.json  … このrunで見た範囲のスナップショット
  - index/episodes_master.jsonl                            … 累積インデックス(追記のみ)
  - replays/episode-<id>-replay.json                       … 全run共有のフラットプール

高速化・冪等性:
  - 既にそのチーム(team_id)のエピソードが episodes_master.jsonl に
    --episodes-per-team 件以上記録済みなら、team-submissions/episodes の
    API呼び出し自体をスキップする(count_episodes_by_team)。
  - チームを処理するたびに episodes_master.jsonl へ即時追記するため、
    Ctrl+C で中断しても再実行すれば上記スキップが効いて続きから進められる。
  - リプレイ本体のダウンロードは download_replay() が dest.exists() を見るため
    二重取得しない。
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
    count_episodes_by_team,
    download_replay,
    fetch_episodes,
    load_existing_episode_ids,
    run_kaggle_json,
    run_kaggle_json_with_page_token,
)


def fetch_leaderboard_range(competition: str, rank_from: int, rank_to: int, sleep: float) -> list[dict]:
    """rank_from 位 〜 rank_to 位(両端含む、1始まり)のリーダーボード行を返す。

    1ページ最大200件・ページトークンは「前のページの続き」しか指せないため、
    1位からページを送りながら範囲外のページは中身を捨てて読み進める。
    """
    if rank_from < 1:
        raise ValueError("rank_from は1以上を指定してください")
    if rank_to < rank_from:
        raise ValueError("rank_to は rank_from 以上を指定してください")

    page_size = 200
    token: str | None = None
    rank = 0
    collected: list[dict] = []
    page_no = 0
    while rank < rank_to:
        page_no += 1
        args = ["competitions", "leaderboard", competition, "-s", "--page-size", str(page_size)]
        if token:
            args += ["--page-token", token]
        rows, token = run_kaggle_json_with_page_token(args)
        if not rows:
            print(f"  ページ{page_no}: 空のレスポンス。リーダーボード末尾に到達したとみなして打ち切ります")
            break
        for row in rows:
            rank += 1
            if rank_from <= rank <= rank_to:
                collected.append({"rank": rank, **row})
            if rank >= rank_to:
                break
        print(
            f"  ページ{page_no}: rank {rank - len(rows) + 1}〜{rank} 取得 "
            f"(採用累計 {len(collected)}件)"
        )
        if not token:
            print("  次ページトークンなし。リーダーボード末尾に到達したとみなして打ち切ります")
            break
        time.sleep(sleep)
    return collected


def fetch_team_submission_ids(team_id: int, limit: int) -> list[int]:
    rows = run_kaggle_json(["competitions", "team-submissions", str(team_id)])
    return [row["id"] for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--rank-from", type=int, default=201, help="対象順位範囲の開始(1始まり、両端含む)")
    parser.add_argument("--rank-to", type=int, default=2000, help="対象順位範囲の終了(両端含む)")
    parser.add_argument(
        "--episodes-per-team", type=int, default=2,
        help="1チームあたり何件のエピソード(最新のものから)を取得するか。"
        "デッキリスト収集が目的なので少数でよい",
    )
    parser.add_argument(
        "--submissions-per-team", type=int, default=1,
        help="各チームにつき何個の提出(新しい順)からエピソードを探すか",
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
        "この値を下げても速度への影響は限定的",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_history_dir = Path(args.leaderboard_history_dir)
    leaderboard_history_dir.mkdir(parents=True, exist_ok=True)
    master_index_path = Path(args.master_index_path)
    master_index_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-deep"
    fetched_at = datetime.now(timezone.utc).isoformat()

    print(
        f"[1/4] リーダーボード取得: {args.competition} "
        f"{args.rank_from}位〜{args.rank_to}位 (run_id={run_id})"
    )
    leaderboard = fetch_leaderboard_range(args.competition, args.rank_from, args.rank_to, args.sleep)
    print(f"  対象チーム数: {len(leaderboard)}")

    leaderboard_snapshot_path = leaderboard_history_dir / f"leaderboard-{run_id}.json"
    leaderboard_snapshot_path.write_text(
        json.dumps(
            {
                "competition": args.competition,
                "run_id": run_id,
                "fetched_at": fetched_at,
                "rank_from": args.rank_from,
                "rank_to": args.rank_to,
                "layer": "deep",
                "leaderboard": leaderboard,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    name_to_context = {
        row["teamName"]: {"team_id": row["teamId"], "rank": row["rank"], "score": float(row["score"])}
        for row in leaderboard
    }

    print(
        f"[2/4] チームごとにエピソードを取得(最新{args.episodes_per_team}件/チーム、"
        f"既に{args.episodes_per_team}件以上プール済みのチームはAPI呼び出しをスキップ)"
    )
    already_indexed = load_existing_episode_ids(master_index_path)
    team_episode_counts = count_episodes_by_team(master_index_path)

    total_teams = len(leaderboard)
    n_skipped_teams = 0
    n_fetched_teams = 0
    n_downloaded_replays = 0
    n_failed_teams = 0
    total_new_master_rows = 0

    for i, row in enumerate(leaderboard, 1):
        team_id = row["teamId"]
        team_name = row["teamName"]
        rank = row["rank"]
        prefix = f"  ({i}/{total_teams}) rank={rank} team={team_name} (teamId={team_id})"

        if team_episode_counts.get(team_id, 0) >= args.episodes_per_team:
            n_skipped_teams += 1
            print(f"{prefix}: プールに既に{team_episode_counts[team_id]}件あり、スキップ")
            continue

        try:
            submission_ids = fetch_team_submission_ids(team_id, args.submissions_per_team)
            time.sleep(args.sleep)
        except subprocess.CalledProcessError as e:
            n_failed_teams += 1
            print(f"{prefix}: 提出取得に失敗: {e.stderr}", file=sys.stderr)
            continue

        episode_meta: dict[str, dict] = {}
        fetch_failed = False
        for sub_id in submission_ids:
            try:
                episodes = fetch_episodes(sub_id)
            except subprocess.CalledProcessError as e:
                print(f"{prefix}: submission {sub_id} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
                fetch_failed = True
                continue
            for ep in episodes:
                episode_meta[str(ep["id"])] = ep
            time.sleep(args.sleep)

        if not episode_meta:
            if fetch_failed:
                n_failed_teams += 1
            else:
                print(f"{prefix}: 完了済みエピソードなし")
            continue

        # 最新(episode_idが大きい)ものから episodes_per_team 件だけ採用
        target_episode_ids = sorted(episode_meta.keys(), key=int, reverse=True)[: args.episodes_per_team]

        downloaded_this_team = 0
        for eid in target_episode_ids:
            dest = out_dir / f"episode-{eid}-replay.json"
            if dest.exists():
                downloaded_this_team += 1
                continue
            try:
                download_replay(int(eid), out_dir)
                downloaded_this_team += 1
                n_downloaded_replays += 1
            except subprocess.CalledProcessError as e:
                print(f"{prefix}: episode {eid} のリプレイ取得に失敗: {e.stderr}", file=sys.stderr)
            time.sleep(args.sleep)

        new_rows = build_master_rows(
            run_id=run_id,
            fetched_at=fetched_at,
            competition=args.competition,
            replays_dir=out_dir,
            episode_meta=episode_meta,
            name_to_context=name_to_context,
            already_indexed=already_indexed,
            episode_ids=target_episode_ids,
        )
        if new_rows:
            append_master_rows(master_index_path, new_rows)
            for r in new_rows:
                already_indexed.add(r["episode_id"])
                for p in r["players"]:
                    tid = p.get("team_id")
                    if tid is not None:
                        team_episode_counts[tid] = team_episode_counts.get(tid, 0) + 1
            total_new_master_rows += len(new_rows)

        n_fetched_teams += 1
        print(
            f"{prefix}: エピソード{len(target_episode_ids)}件対象"
            f"(手元に{downloaded_this_team}件確認、マスターに{len(new_rows)}件追記)"
        )

    print("[3/4] 完了サマリ")
    print(f"  対象チーム数: {total_teams}")
    print(f"  スキップ(既にプール済み): {n_skipped_teams}")
    print(f"  処理したチーム: {n_fetched_teams}")
    print(f"  失敗したチーム: {n_failed_teams}")
    print(f"  ダウンロードした新規リプレイ: {n_downloaded_replays}")
    print(f"  マスターインデックスに追記した行数: {total_new_master_rows}")
    print(f"[4/4] リーダーボードスナップショット -> {leaderboard_snapshot_path}")
    print(f"完了: マスターインデックス -> {master_index_path}")


if __name__ == "__main__":
    main()
