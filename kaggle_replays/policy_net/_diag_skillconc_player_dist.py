#!/usr/bin/env python3
"""Diagnostic (throwaway, not shipped): rank<=200 データのプレイヤー(team_id)分布を確認する。

policy-skill-concentration-implementation-plan.md Step1:
「≤200位は335エピソード。特定少数プレイヤーに偏っていないか(単一スタイルへの過適合リスク)は
Step1で確認する」に対応する。policy_positions.jsonl.gz の各行(episode_id, player_index,
rank_at_fetch)を kaggle_replays/index/episodes_master.jsonl の players[].team_id と突き合わせ、
rank_at_fetch<=200 の意思決定点・エピソードが何人のプレイヤーに由来するか、上位プレイヤーへの
偏重度合いを集計する。

出力: 標準出力にサマリを表示するのみ(結果は
sample_submission/results/2026-07-21_skillconc_configs.md に手動で転記する)。

実行: python kaggle_replays/policy_net/_diag_skillconc_player_dist.py
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_POLICY_POSITIONS = _REPO_ROOT / "kaggle_replays" / "training_data" / "policy_positions.jsonl.gz"
_EPISODES_MASTER = _REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl"

_RANK_MAX = 200


def load_episode_team_map() -> dict[str, dict[int, tuple]]:
    """episode_id -> {player_index: (team_id, team_name)}"""
    out: dict[str, dict[int, tuple]] = {}
    with _EPISODES_MASTER.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["episode_id"]] = {
                p["player_index"]: (p.get("team_id"), p.get("team_name"))
                for p in row.get("players", [])
            }
    return out


def main() -> None:
    episode_team_map = load_episode_team_map()
    print(f"episodes_master.jsonl エピソード数: {len(episode_team_map)}")

    n_total = 0
    n_top200 = 0
    top200_episode_ids: set[str] = set()
    team_decision_counts: Counter = Counter()
    team_episode_ids: dict[tuple, set[str]] = {}
    n_missing_join = 0

    with gzip.open(_POLICY_POSITIONS, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1
            rank = row.get("rank_at_fetch")
            if rank is None or rank > _RANK_MAX:
                continue
            n_top200 += 1
            episode_id = row["episode_id"]
            player_index = row["player_index"]
            top200_episode_ids.add(episode_id)

            team_info = episode_team_map.get(episode_id, {}).get(player_index)
            if team_info is None:
                n_missing_join += 1
                team_key = ("__missing__", None)
            else:
                team_key = team_info
            team_decision_counts[team_key] += 1
            team_episode_ids.setdefault(team_key, set()).add(episode_id)

    print(f"\n全意思決定点: {n_total}")
    print(f"rank_at_fetch<=200 の意思決定点: {n_top200} ({n_top200/n_total*100:.1f}%)")
    print(f"rank_at_fetch<=200 のユニークエピソード数: {len(top200_episode_ids)}")
    print(f"episodes_master.jsonl との突き合わせ失敗(missing join): {n_missing_join}")

    n_players = len([k for k in team_decision_counts if k[0] != "__missing__"])
    print(f"\nrank<=200 データに寄与しているユニークプレイヤー(team_id)数: {n_players}")

    print("\n=== プレイヤー別内訳(意思決定点数の多い順、上位20) ===")
    print(f"{'team_id':>12} {'team_name':30} {'decisions':>10} {'episodes':>9} {'share':>7}")
    for team_key, cnt in team_decision_counts.most_common(20):
        team_id, team_name = team_key
        n_eps = len(team_episode_ids[team_key])
        share = cnt / n_top200 * 100
        name_disp = (team_name or "?")[:30]
        print(f"{str(team_id):>12} {name_disp:30} {cnt:>10} {n_eps:>9} {share:>6.1f}%")

    # 集中度指標: 上位1/3/5プレイヤーの意思決定点シェア。
    sorted_counts = [cnt for _, cnt in team_decision_counts.most_common()]
    for top_n in (1, 3, 5, 10):
        share = sum(sorted_counts[:top_n]) / n_top200 * 100
        print(f"\n上位{top_n}プレイヤーの意思決定点シェア: {share:.1f}%")

    # HHI (Herfindahl-Hirschman Index) で集中度を定量化(0に近いほど分散、1に近いほど集中)。
    hhi = sum((cnt / n_top200) ** 2 for cnt in sorted_counts)
    print(f"\nHHI(意思決定点シェアの二乗和): {hhi:.4f}  (参考: N人均等なら 1/N = {1/max(n_players,1):.4f})")


if __name__ == "__main__":
    main()
