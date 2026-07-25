"""fetch_top_episodes.py / fetch_my_episodes.py で共有するヘルパー。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

KAGGLE = "kaggle"


def run_kaggle_json(args: list[str]):
    obj, _token = run_kaggle_json_with_page_token(args)
    return obj


def run_kaggle_json_with_page_token(args: list[str]) -> tuple[object, str | None]:
    """`--format json` の出力をパースし、あわせて "Next Page Token = ..." 行があれば返す。

    `kaggle competitions leaderboard -s` 等、ページネーション対応のサブコマンドは
    JSON本体の前に "Next Page Token = <token>" という非JSON行を出力する
    (--page-token に渡すことで次ページを取得できる)。トークンが出力されない
    (最終ページ等)場合は None を返す。
    """
    result = subprocess.run(
        [KAGGLE, *args, "--format", "json"],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    # 一部のサブコマンドは JSON の前後に "Next Page Token = ..." や
    # 使い方ヒント等の非JSON行を stdout に混ぜて出力するため、
    # JSON開始位置から raw_decode して末尾の余計な文字列は無視する。
    stdout = result.stdout
    token = None
    m = re.match(r"Next Page Token = (\S+)", stdout)
    if m:
        token = m.group(1)
    start = min((i for i in (stdout.find("["), stdout.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"kaggle CLI の出力からJSONを検出できませんでした: {stdout!r}")
    obj, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return obj, token


def fetch_episodes(submission_id) -> list[dict]:
    """指定した提出に紐づく完了済みエピソードのメタ情報(id/createTime/endTime)を返す。"""
    rows = run_kaggle_json(["competitions", "episodes", str(submission_id)])
    return [row for row in rows if row.get("state") == "EpisodeState.COMPLETED"]


def download_replay(episode_id: int, out_dir: Path) -> Path:
    dest = out_dir / f"episode-{episode_id}-replay.json"
    if dest.exists():
        return dest
    subprocess.run(
        [KAGGLE, "competitions", "replay", str(episode_id), "-p", str(out_dir), "-q"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return dest


def load_existing_episode_ids(master_path: Path) -> set[str]:
    if not master_path.exists():
        return set()
    ids: set[str] = set()
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["episode_id"])
    return ids


def count_episodes_by_team(master_path: Path) -> dict[int, int]:
    """episodes_master.jsonl から team_id ごとの累積エピソード件数を数える。

    深層取得(fetch_deep_decks.py)で「そのチームは既に episodes-per-team 件以上
    プールにあるからAPI呼び出し自体をスキップする」高速化に使う。team_id が
    null の行はカウントしない。
    """
    counts: dict[int, int] = {}
    if not master_path.exists():
        return counts
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for player in row.get("players", []):
                team_id = player.get("team_id")
                if team_id is None:
                    continue
                counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def build_master_rows(
    run_id: str,
    fetched_at: str,
    competition: str,
    replays_dir: Path,
    episode_meta: dict[str, dict],
    name_to_context: dict[str, dict],
    already_indexed: set[str],
    episode_ids: list[str] | None = None,
) -> list[dict]:
    """未インデックスのリプレイから episodes_master.jsonl の行を作る。

    name_to_context: チーム名 -> {"team_id", "rank", "score"}。
    対応表に存在しないチーム(取得範囲外だった相手)は team_id/rank/score が null になる。

    episode_ids を指定した場合はその一覧だけを対象にする(replays_dir 全体を
    毎回スキャンするコストを避けたい呼び出し元向け)。省略時は従来どおり
    replays_dir 内の episode-*-replay.json を全走査する。
    """
    if episode_ids is not None:
        candidates = [
            (eid, replays_dir / f"episode-{eid}-replay.json") for eid in episode_ids
        ]
    else:
        candidates = [
            (replay_path.stem.split("-")[1], replay_path)
            for replay_path in sorted(replays_dir.glob("episode-*-replay.json"))
        ]
    rows = []
    for episode_id, replay_path in candidates:
        if episode_id in already_indexed:
            continue
        if not replay_path.exists():
            continue
        with replay_path.open(encoding="utf-8") as f:
            replay = json.load(f)
        team_names = replay.get("info", {}).get("TeamNames", [None, None])
        meta = episode_meta.get(episode_id, {})
        players = []
        for player_index, team_name in enumerate(team_names):
            ctx = name_to_context.get(team_name, {})
            players.append(
                {
                    "player_index": player_index,
                    "team_name": team_name,
                    "team_id": ctx.get("team_id"),
                    "rank_at_fetch": ctx.get("rank"),
                    "leaderboard_score_at_fetch": ctx.get("score"),
                }
            )
        rows.append(
            {
                "episode_id": episode_id,
                "competition": competition,
                "episode_create_time": meta.get("createTime"),
                "episode_end_time": meta.get("endTime"),
                "discovered_run_id": run_id,
                "discovered_at": fetched_at,
                "players": players,
            }
        )
    return rows


def append_master_rows(master_index_path: Path, rows: list[dict]) -> None:
    with master_index_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
