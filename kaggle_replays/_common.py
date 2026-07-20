"""fetch_top_episodes.py / fetch_my_episodes.py で共有するヘルパー。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

KAGGLE = "kaggle"


def run_kaggle_json(args: list[str]):
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
    start = min((i for i in (stdout.find("["), stdout.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"kaggle CLI の出力からJSONを検出できませんでした: {stdout!r}")
    obj, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return obj


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


def build_master_rows(
    run_id: str,
    fetched_at: str,
    competition: str,
    replays_dir: Path,
    episode_meta: dict[str, dict],
    name_to_context: dict[str, dict],
    already_indexed: set[str],
) -> list[dict]:
    """replays_dir にある未インデックスのリプレイから episodes_master.jsonl の行を作る。

    name_to_context: チーム名 -> {"team_id", "rank", "score"}。
    対応表に存在しないチーム(取得範囲外だった相手)は team_id/rank/score が null になる。
    """
    rows = []
    for replay_path in sorted(replays_dir.glob("episode-*-replay.json")):
        episode_id = replay_path.stem.split("-")[1]
        if episode_id in already_indexed:
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
