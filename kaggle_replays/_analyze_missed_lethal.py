#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

「ハイブリッド化(確定リーサル探索の統合)より前の提出(submission 54857624、
imitation-only)の実戦の負け試合で、確定リーサルを見逃していたか」を
lethal_simple.search() を実際に呼んで直接検証する。

このスクリプトは ptcg_ai / league / configs を一切変更しない。分析専用。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
KAGGLE_REPLAYS_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = KAGGLE_REPLAYS_DIR / "replays"

# cg.api の DLL ロードは __file__ 相対なので cwd に依存しないが、
# read_deck_csv() は相対パス "deck.csv" を見るので cwd を sample_submission にする。
sys.path.insert(0, str(SAMPLE_SUB))
sys.path.insert(0, str(KAGGLE_REPLAYS_DIR))
os.chdir(SAMPLE_SUB)

from cg.api import to_observation_class, SelectType  # noqa: E402
from ptcg_ai.search import lethal_simple  # noqa: E402
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state  # noqa: E402
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv  # noqa: E402

from _common import fetch_episodes  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"
SUBMISSION_ID = 54857624

# ptcg_ai_agent が実戦で使った configs/ml_lethal.json の lethal_search 節と同一。
LETHAL_CONFIG = {
    "enabled": True,
    "module": "lethal_simple",
    "max_remaining_prizes": 3,
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000,
    "max_combinations_per_select": 128,
    "verify_shuffles": 1,
}


def find_last_known_turn(steps: list) -> int | None:
    """リプレイ末尾から遡って、いずれかのプレイヤーの observation.current.turn を取る。"""
    for i in range(len(steps) - 1, -1, -1):
        for p in (0, 1):
            obs_dict = steps[i][p].get("observation")
            if obs_dict and obs_dict.get("current"):
                return obs_dict["current"].get("turn")
    return None


def analyze_game(episode_id: int, replay: dict, my_idx: int, full_deck: list[int]) -> dict:
    steps = replay["steps"]
    missed_points = []
    n_main_decisions = 0
    errors = 0

    for i in range(len(steps) - 1):
        agent_step = steps[i][my_idx]
        if agent_step.get("status") != "ACTIVE":
            continue
        obs_dict = agent_step.get("observation")
        if not obs_dict:
            continue
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            errors += 1
            continue
        if obs.select is None or obs.current is None:
            continue
        if obs.select.type != SelectType.MAIN:
            continue

        n_main_decisions += 1
        turn = obs.current.turn

        context = {
            "observation": obs,
            "config": LETHAL_CONFIG,
            "hidden_state_factory": lambda o=obs: build_dummy_search_state(o, full_deck),
        }
        try:
            result = lethal_simple.search(obs.current, obs.select.option, context)
        except Exception:
            errors += 1
            result = None

        if result is not None:
            missed_points.append({"step": i, "turn": turn})

    return {
        "episode_id": episode_id,
        "n_main_decisions": n_main_decisions,
        "missed_lethal_points": missed_points,
        "last_turn": find_last_known_turn(steps),
        "errors": errors,
    }


def main() -> None:
    full_deck = read_deck_csv()
    print(f"deck.csv loaded: {len(full_deck)} cards")

    episodes = fetch_episodes(SUBMISSION_ID)
    print(f"submission {SUBMISSION_ID}: {len(episodes)} completed episodes")

    loss_games = []
    missing_replays = []
    not_my_team = []
    for ep in episodes:
        eid = ep["id"]
        path = REPLAYS_DIR / f"episode-{eid}-replay.json"
        if not path.exists():
            missing_replays.append(eid)
            continue
        with path.open(encoding="utf-8") as f:
            replay = json.load(f)
        team_names = replay.get("info", {}).get("TeamNames", [None, None])
        if TEAM_NAME not in team_names:
            not_my_team.append(eid)
            continue
        my_idx = team_names.index(TEAM_NAME)
        opp_idx = 1 - my_idx
        rewards = replay.get("rewards", [None, None])
        if rewards[my_idx] is None or rewards[opp_idx] is None:
            continue
        if rewards[my_idx] >= rewards[opp_idx]:
            continue  # 勝ち/引き分けはスキップ
        loss_games.append((eid, replay, my_idx))

    print(f"missing replay files: {len(missing_replays)} {missing_replays}")
    print(f"episodes not involving {TEAM_NAME}: {len(not_my_team)}")
    print(f"loss games analyzed: {len(loss_games)}")
    print()

    results = []
    for eid, replay, my_idx in loss_games:
        r = analyze_game(eid, replay, my_idx, full_deck)
        results.append(r)
        flag = "MISSED LETHAL" if r["missed_lethal_points"] else "-"
        print(
            f"episode {eid}: MAIN decisions={r['n_main_decisions']:3d}  "
            f"last_turn={r['last_turn']}  errors={r['errors']}  {flag}"
        )
        if r["missed_lethal_points"]:
            for p in r["missed_lethal_points"]:
                print(f"    -> step {p['step']}, turn {p['turn']}")

    n_losses = len(results)
    n_with_missed = sum(1 for r in results if r["missed_lethal_points"])
    total_errors = sum(r["errors"] for r in results)

    print()
    print("=== SUMMARY ===")
    print(f"total loss games analyzed: {n_losses}")
    print(f"loss games with >=1 missed certain lethal: {n_with_missed}")
    if n_losses:
        print(f"missed-lethal rate among losses: {n_with_missed / n_losses * 100:.1f}%")
    print(f"total search()/to_observation_class() errors (caught, non-fatal): {total_errors}")

    print()
    print("=== Turn gap (earliest missed-lethal turn vs actual game-end turn) ===")
    for r in results:
        if not r["missed_lethal_points"]:
            continue
        turns = [p["turn"] for p in r["missed_lethal_points"]]
        earliest = min(turns)
        last_turn = r["last_turn"]
        gap = (last_turn - earliest) if last_turn is not None else None
        print(
            f"episode {r['episode_id']}: earliest missed-lethal turn={earliest}, "
            f"game ended turn={last_turn}, gap(turns)={gap}, "
            f"all missed turns={turns}"
        )

    out_path = KAGGLE_REPLAYS_DIR / "_missed_lethal_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
