#!/usr/bin/env python3
"""ダウンロード済みのリプレイJSONを (observation, action) の学習データ(JSONL)に変換する。

Kaggleシミュレーションのリプレイ形式では、steps[i][player]["observation"] が
そのプレイヤーに提示された観測、steps[i+1][player]["action"] がその観測に対して
実際に選んだ行動(sample_submission/main.py の agent() が返す選択インデックス列)に
対応する。steps[i][player]["status"] が "ACTIVE" のときだけ、そのプレイヤーは
実際に意思決定を行っている(INACTIVE側は空アクションのプレースホルダ)。

fetch_top_episodes.py が作る index/episodes_master.jsonl を結合し、各ペアに
「取得時点でそのプレイヤーが何位・何点だったか」を rank_at_fetch /
leaderboard_score_at_fetch として付与する。学習時にこれをサンプルの重みとして
使う想定(例: 上位ほど重みを大きくする、点数をそのまま重みにする、など)。
マスターインデックスに存在しないエピソード(--top 範囲外だった相手など)は
該当フィールドが null になる。

使い方:
  python extract_training_data.py --replays-dir ./replays --out ./training_data/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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


def extract_pairs(replay: dict, episode_id: str, master_row: dict | None):
    steps = replay["steps"]
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    players_meta = {p["player_index"]: p for p in master_row["players"]} if master_row else {}

    for player_index in (0, 1):
        opponent_index = 1 - player_index
        own_meta = players_meta.get(player_index, {})
        opp_meta = players_meta.get(opponent_index, {})
        for i in range(len(steps) - 1):
            agent_step = steps[i][player_index]
            if agent_step["status"] != "ACTIVE":
                continue
            action = steps[i + 1][player_index]["action"]
            yield {
                "episode_id": episode_id,
                "episode_create_time": master_row.get("episode_create_time") if master_row else None,
                "player_index": player_index,
                "team_name": team_names[player_index] if player_index < len(team_names) else None,
                "rank_at_fetch": own_meta.get("rank_at_fetch"),
                "leaderboard_score_at_fetch": own_meta.get("leaderboard_score_at_fetch"),
                "opponent_team_name": team_names[opponent_index] if opponent_index < len(team_names) else None,
                "opponent_rank_at_fetch": opp_meta.get("rank_at_fetch"),
                "opponent_leaderboard_score_at_fetch": opp_meta.get("leaderboard_score_at_fetch"),
                "step": i,
                "observation": agent_step["observation"],
                "action": action,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(Path(__file__).parent / "replays"))
    parser.add_argument(
        "--master-index-path",
        default=str(Path(__file__).parent / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument("--out", default=str(Path(__file__).parent / "training_data" / "pairs.jsonl"))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    master_index = load_master_index(Path(args.master_index_path))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_files = 0
    n_without_rank = 0
    with out_path.open("w", encoding="utf-8") as f:
        for replay_path in sorted(replays_dir.glob("episode-*-replay.json")):
            episode_id = replay_path.stem.split("-")[1]
            master_row = master_index.get(episode_id)
            if master_row is None:
                n_without_rank += 1
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
            for pair in extract_pairs(replay, episode_id, master_row):
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_pairs += 1
            n_files += 1

    print(f"{n_files}件のリプレイから{n_pairs}件の(observation, action)ペアを {out_path} に書き出しました")
    if n_without_rank:
        print(
            f"注意: {n_without_rank}件のエピソードはマスターインデックスに未登録のため "
            "rank_at_fetch 等が null になっています(fetch_top_episodes.py の再実行で解消できます)"
        )


if __name__ == "__main__":
    main()
