"""adjust_prior.py と evaluate.py が共有する、episodes_master.jsonl とのジョイン・
ウィンドウ判定(直近 N 日 x 上位 R 位以内)・ランク帯バケットのユーティリティ。

deck_labels.jsonl / dataset.jsonl の各行は episode_id (+ 対象プレイヤーの player_index) しか
持たないが、「そのプレイヤーの相手ランク」「エピソード作成日時」は
kaggle_replays/index/episodes_master.jsonl 側にしかない。adjust_prior.py(ターゲット分布の算出)と
evaluate.py(validation のウィンドウ絞り込み・ランク帯別/期間別集計)の両方で同じジョイン・
フィルタ・バケット定義を使うため、ここに切り出す。

詳細は sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md
の「評価プロトコル」節を参照。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 相手ランク帯のバケット定義・表示順(不明 = rank_at_fetch が None、主に相手が自チームでない場合)。
RANK_BUCKETS = ["1-50", "51-200", "201-1000", "1001+", "不明"]


def parse_episode_time(value: str | None) -> datetime | None:
    """episode_create_time の ISO8601 文字列を naive UTC datetime にパースする。パース不可/None は None。"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class EpisodeIndex:
    """episode_id -> {episode_create_time, player_ranks: {0: rank|None, 1: rank|None}} の索引。"""

    def __init__(self, entries: dict[str, dict]):
        self._entries = entries

    @classmethod
    def load(cls, master_path: str | Path) -> "EpisodeIndex":
        entries: dict[str, dict] = {}
        with Path(master_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                player_ranks = {p["player_index"]: p.get("rank_at_fetch") for p in row.get("players", [])}
                entries[row["episode_id"]] = {
                    "episode_create_time": parse_episode_time(row.get("episode_create_time")),
                    "player_ranks": player_ranks,
                }
        return cls(entries)

    def time_of(self, episode_id: str) -> datetime | None:
        entry = self._entries.get(episode_id)
        if entry is None:
            return None
        return entry["episode_create_time"]

    def rank_of(self, episode_id: str, player_index: int) -> int | None:
        entry = self._entries.get(episode_id)
        if entry is None:
            return None
        return entry["player_ranks"].get(player_index)


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "不明"
    if rank <= 50:
        return "1-50"
    if rank <= 200:
        return "51-200"
    if rank <= 1000:
        return "201-1000"
    return "1001+"


def in_recent_top_rank_window(
    episode_time: datetime | None,
    rank: int | None,
    now: datetime,
    recent_days: int,
    top_rank: int,
) -> bool:
    """「エピソード作成が直近 N 日以内」かつ「対象プレイヤーの rank_at_fetch が上位 R 位以内
    (rank 不明は除外)」を満たすか判定する。"""
    if episode_time is None:
        return False
    if now - episode_time > timedelta(days=recent_days):
        return False
    if rank is None:
        return False
    return rank <= top_rank
