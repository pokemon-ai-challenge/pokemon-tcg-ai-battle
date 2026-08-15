#!/usr/bin/env python3
"""自分のチームの対戦リプレイを取得する(fetch_top_episodes.py の自分版)。

上位チームではなく、自分がKaggleに提出した最新N件の submission に紐づく
エピソードを取得する。replays/ と index/episodes_master.jsonl は
fetch_top_episodes.py と共有するので、同じ episode_id は重複ダウンロードされず、
両方のスクリプトの結果が1つの学習データセットにまとまる。

自分のチームID・順位は --top のような範囲指定では届かないことが多いため、
公開リーダーボード全体(-d でダウンロードできるCSV、数千チーム分)から
自分のユーザー名が TeamMemberUserNames に含まれる行を探して特定する。
このCSVには全チームの team_id/rank/score が入っているので、対戦相手が
たまたま上位チームでなくても、判明する範囲で埋められる。

使い方:
  python fetch_my_episodes.py --submissions 5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    KAGGLE,
    append_master_rows,
    build_master_rows,
    download_replay,
    fetch_episodes,
    load_existing_episode_ids,
    run_kaggle_json,
)


def get_own_username() -> str:
    result = subprocess.run(
        [KAGGLE, "config", "view"], capture_output=True, encoding="utf-8", check=True
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.lower().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(
        "kaggle config view から username を取得できませんでした。認証設定を確認してください。"
    )


def fetch_full_leaderboard_context(competition: str) -> tuple[list[dict], dict[str, dict]]:
    """公開リーダーボード全体をCSVでダウンロードし、チーム名->{team_id,rank,score} の対応表を作る。"""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [KAGGLE, "competitions", "leaderboard", competition, "-d", "-p", tmp, "-q"],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        zip_path = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            text = zf.read(csv_name).decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    name_to_context = {
        row["TeamName"]: {
            "team_id": int(row["TeamId"]),
            "rank": int(row["Rank"]),
            "score": float(row["Score"]) if row["Score"] else None,
        }
        for row in rows
    }
    return rows, name_to_context


def find_own_team(rows: list[dict], username: str) -> dict | None:
    for row in rows:
        members = [m.strip() for m in row["TeamMemberUserNames"].split(",")]
        if username in members:
            return row
    return None


def update_submission_episode_map(path: Path, new_entries: dict[str, list[str]]) -> int:
    """提出(submission ref)→エピソードID一覧の対応表を追記更新する。

    episodes_master.jsonl にはエピソード単位の情報しか残らず、どの提出に紐づくかは
    (`fetch_episodes(submission_ref)` が提出スコープでエピソードを返す)この呼び出し
    の時点でしか分からない。そのため、ここで別ファイルとして対応を残す。

    - 既存キー(このrunで触れなかった提出)の値は変更しない。
    - 同じ提出を複数回fetchした場合はエピソードIDの和集合を取る(提出は稼働中に
      エピソードが増えていくため、和集合が正しい)。
    - ファイルが存在しない場合は新規作成する。

    戻り値: 新規に追加されたエピソードID件数(和集合で増えた分の合計、ログ表示用)。
    """
    existing: dict[str, list] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    added = 0
    for ref, eids in new_entries.items():
        before = set(str(e) for e in existing.get(ref, []))
        merged = before | set(str(e) for e in eids)
        added += len(merged) - len(before)
        existing[ref] = sorted(merged, key=int)
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--submissions", type=int, default=3, help="新しい順に何件の提出を辿るか")
    parser.add_argument(
        "--max-episodes", type=int, default=300,
        help="ダウンロードするエピソード数の上限(1件の提出だけで1000件近くあることがあるため既定値を設定。"
        "無制限にしたい場合は非常に大きい値を指定するか 0 以下を指定)",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "replays"))
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
        "--episode-map-path",
        default=str(Path(__file__).parent / "_submission_episode_map.json"),
        help="提出ref→エピソードID一覧の対応表(追記更新、既存キーは変更しない)。"
        "分析スクリプト(_archetype_winloss.py 等)が読む _submission_episode_map.json 形式",
    )
    parser.add_argument(
        "--skip-episode-map-update", action="store_true",
        help="提出→エピソード対応表の更新をスキップする(従来どおりepisodes_master.jsonlのみ更新)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_index_path = Path(args.master_index_path)
    master_index_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-self"
    fetched_at = datetime.now(timezone.utc).isoformat()

    print("[1/6] 自分のユーザー名を確認")
    username = get_own_username()
    print(f"  username={username}")

    print(f"[2/6] 公開リーダーボード全体を取得し、自分のチームを特定 ({args.competition})")
    all_rows, name_to_context = fetch_full_leaderboard_context(args.competition)
    self_row = find_own_team(all_rows, username)
    if self_row is None:
        print(
            f"  警告: リーダーボード上に自分のチーム(username={username})が見つかりませんでした。"
            "順位情報なしで続行します。",
            file=sys.stderr,
        )
    else:
        print(
            f"  自チーム: {self_row['TeamName']} "
            f"(teamId={self_row['TeamId']}, rank={self_row['Rank']}, score={self_row['Score']})"
        )

    print(f"[3/6] 自分の提出一覧を取得(新しい順に最大{args.submissions}件)")
    own_submissions = run_kaggle_json(["competitions", "submissions", args.competition])
    completed = [s for s in own_submissions if s.get("status") == "SubmissionStatus.COMPLETE"]
    target_submissions = completed[: args.submissions]
    for s in target_submissions:
        print(f"  submission {s['ref']} ({s['date']}, publicScore={s['publicScore']})")

    print("[4/6] エピソード一覧を取得・リプレイをダウンロード")
    episode_meta: dict[str, dict] = {}
    # fetch_episodes(ref) は提出スコープでエピソードを返すため、この時点でしか
    # 「どの提出にどのエピソードが属するか」は分からない(下のepisode_metaは全提出を
    # フラットにマージしてしまうため対応が失われる)。[6/6]の対応表更新用に別途残す。
    submission_episode_ids: dict[str, list[str]] = {}
    for s in target_submissions:
        try:
            episodes = fetch_episodes(s["ref"])
        except subprocess.CalledProcessError as e:
            print(f"  警告: submission {s['ref']} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
            continue
        submission_episode_ids[str(s["ref"])] = [str(ep["id"]) for ep in episodes]
        for ep in episodes:
            episode_meta[str(ep["id"])] = ep
        time.sleep(args.sleep)

    episode_id_list = sorted(episode_meta.keys(), key=int)
    if args.max_episodes is not None and args.max_episodes > 0:
        episode_id_list = episode_id_list[-args.max_episodes:]

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

    print(f"[5/6] エピソードマスターインデックスを更新 -> {master_index_path}")
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

    added_map_entries = 0
    if args.skip_episode_map_update:
        print("[6/6] 提出→エピソード対応表の更新はスキップ(--skip-episode-map-update)")
    else:
        episode_map_path = Path(args.episode_map_path)
        print(f"[6/6] 提出→エピソード対応表を更新 -> {episode_map_path}")
        added_map_entries = update_submission_episode_map(episode_map_path, submission_episode_ids)

    print(
        f"完了: リプレイ{len(downloaded)}件を保存、マスターインデックスに{len(new_rows)}件を追記、"
        f"提出対応表に新規エピソードID{added_map_entries}件を追記しました"
    )


if __name__ == "__main__":
    main()
