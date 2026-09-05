#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

pimc-null-diagnosis-implementation-plan.md Step4: 山札切れ敗因の拡大標本分類。

[[project_deckout_loss_cause]] (n=28、29%山札切れ)を、既に大量ダウンロード済みの
自チーム("MORIOKA Tsuoi")リプレイ(kaggle_replays/replays/、364試合)を使って拡大標本で
再確認し、山札切れが (a) 回避可能なエージェント判断 (b) デッキ構築(ドロー/サーチ過多)
(c) 強制・回避不能 のどれに主に起因するかを判定する。

**データの制約**: Kaggle の配布リプレイ形式には LogType.RESULT(cg/api.py の reason フィールド、
本来なら 1=サイド完投/2=ターン開始時山札0枚/3=バトル場ポケモン不在/4=カード効果、を直接示す)
が一度も出現しないことを確認済み(200件スキャンして0件)。そのため本分析は「自分の deckCount
が試合中に一度でも0を記録したか」を山札切れの代理指標として使う(cg/api.py の
LogType.RESULT reason=2 の定義「ターン開始時に山札0枚」と整合する、最も直接的に観測可能な
代理)。

山札切れ負けと判定した試合について、以下を集計して (a)/(b)/(c) を判定する材料とする:
- 試合の長さ(自分のターン数)。全体平均との比較で「単に長引いた試合か」を見る((c)の材料)。
- 自分の LogType.DRAW 件数(通常のターン開始ドロー含む全ドロー)と、山札からの
  サーチ(LogType.MOVE_CARD, fromArea=DECK, toArea in {HAND,BENCH,ACTIVE})件数。
  「本来の1ターン1ドローを超える消費」がどれだけ大きいかを見る((a)の材料)。
- deck.csv の構成(サーチ効果を持つカードの枚数)。デッキの構造的なドロー/サーチ密度が
  そもそも高いかを見る((b)の材料)。

このスクリプトは ptcg_ai / league / configs / deck.csv を一切変更しない。分析専用。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
KAGGLE_REPLAYS_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = KAGGLE_REPLAYS_DIR / "replays"
MASTER_INDEX = KAGGLE_REPLAYS_DIR / "index" / "episodes_master.jsonl"

sys.path.insert(0, str(SAMPLE_SUB))
sys.path.insert(0, str(KAGGLE_REPLAYS_DIR))

from cg.api import AreaType, LogType, to_observation_class  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"
MAX_EPISODES = 400

_DECK_TARGET_AREAS = {AreaType.HAND, AreaType.BENCH, AreaType.ACTIVE}


def find_own_episode_ids(limit: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with MASTER_INDEX.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            eid = row["episode_id"]
            if eid in seen:
                continue
            if any(p.get("team_name") == TEAM_NAME for p in row.get("players", [])):
                path = REPLAYS_DIR / f"episode-{eid}-replay.json"
                if path.exists():
                    ids.append(eid)
                    seen.add(eid)
            if len(ids) >= limit:
                break
    return ids


def load_my_idx(replay: dict) -> int | None:
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    if TEAM_NAME not in team_names:
        return None
    return team_names.index(TEAM_NAME)


def game_outcome(replay: dict, my_idx: int) -> str:
    rewards = replay.get("rewards", [None, None])
    opp_idx = 1 - my_idx
    if rewards[my_idx] is None or rewards[opp_idx] is None:
        return "unknown"
    if rewards[my_idx] > rewards[opp_idx]:
        return "win"
    if rewards[my_idx] < rewards[opp_idx]:
        return "loss"
    return "draw"


def analyze_game(replay: dict, my_idx: int) -> dict | None:
    """自分視点で1試合を通しで見て、deckCount推移・ログ集計を返す。"""
    steps = replay["steps"]
    deck_counts: list[int] = []
    last_turn = None
    draws = 0
    deck_searches = 0
    my_turns_seen: set[int] = set()
    first_player = None

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
            continue
        if obs.current is None:
            continue
        state = obs.current
        me = state.yourIndex
        deck_counts.append(state.players[me].deckCount)
        last_turn = state.turn
        if first_player is None and state.firstPlayer in (0, 1):
            first_player = state.firstPlayer

        for log in obs.logs or []:
            if log.playerIndex != me:
                continue
            if log.type == LogType.DRAW:
                draws += 1
            elif log.type == LogType.MOVE_CARD and log.fromArea == AreaType.DECK and log.toArea in _DECK_TARGET_AREAS:
                deck_searches += 1
            elif log.type == LogType.TURN_START:
                my_turns_seen.add(last_turn)

    if not deck_counts:
        return None

    n_my_turns = len(my_turns_seen) if my_turns_seen else None
    expected_mandatory_draws = None
    if n_my_turns is not None and first_player is not None:
        expected_mandatory_draws = n_my_turns - (1 if first_player == my_idx else 0)

    return {
        "min_deck_count": min(deck_counts),
        "last_deck_count": deck_counts[-1],
        "final_turn": last_turn,
        "draws": draws,
        "deck_searches": deck_searches,
        "n_my_turns": n_my_turns,
        "expected_mandatory_draws": expected_mandatory_draws,
        "extra_draws": (draws - expected_mandatory_draws) if expected_mandatory_draws is not None else None,
    }


def main() -> None:
    episode_ids = find_own_episode_ids(MAX_EPISODES)
    print(f"自チームエピソード候補: {len(episode_ids)}件(上限{MAX_EPISODES}件スキャン)")

    all_games = []
    for eid in episode_ids:
        path = REPLAYS_DIR / f"episode-{eid}-replay.json"
        try:
            with path.open(encoding="utf-8") as f:
                replay = json.load(f)
        except Exception:
            continue
        my_idx = load_my_idx(replay)
        if my_idx is None:
            continue
        outcome = game_outcome(replay, my_idx)
        if outcome not in ("win", "loss"):
            continue
        stats = analyze_game(replay, my_idx)
        if stats is None:
            continue
        stats["episode_id"] = eid
        stats["outcome"] = outcome
        stats["deck_out"] = stats["min_deck_count"] == 0
        all_games.append(stats)

    print(f"分析対象試合(win/lossのみ): {len(all_games)}件")

    wins = [g for g in all_games if g["outcome"] == "win"]
    losses = [g for g in all_games if g["outcome"] == "loss"]
    deckout_losses = [g for g in losses if g["deck_out"]]
    other_losses = [g for g in losses if not g["deck_out"]]

    print(f"内訳: win={len(wins)}, loss={len(losses)}, "
          f"loss中の山札切れ={len(deckout_losses)}({len(deckout_losses) / len(losses) * 100:.1f}%)" if losses else "loss=0")

    def turn_stats(games):
        turns = [g["final_turn"] for g in games if g["final_turn"] is not None]
        return (statistics.mean(turns), len(turns)) if turns else (None, 0)

    def extra_draw_stats(games):
        vals = [g["extra_draws"] for g in games if g["extra_draws"] is not None]
        return (statistics.mean(vals), len(vals)) if vals else (None, 0)

    print("\n=== 試合の長さ(自分のターン数)===")
    for label, games in [("win", wins), ("loss(全体)", losses),
                         ("loss(山札切れ)", deckout_losses), ("loss(非山札切れ)", other_losses)]:
        mean_turn, n = turn_stats(games)
        if mean_turn is not None:
            print(f"  {label}: n={n}, 平均最終ターン(自分視点)={mean_turn:.1f}")
        else:
            print(f"  {label}: n=0")

    print("\n=== 余剰ドロー数(実ドロー数 - 理論上の必須ドロー数)===")
    for label, games in [("win", wins), ("loss(全体)", losses),
                         ("loss(山札切れ)", deckout_losses), ("loss(非山札切れ)", other_losses)]:
        mean_extra, n = extra_draw_stats(games)
        if mean_extra is not None:
            print(f"  {label}: n={n}, 平均余剰ドロー数={mean_extra:.2f}")
        else:
            print(f"  {label}: n=0")

    print("\n=== 山札からのサーチ回数(Buddy-Buddy Poffin/Hilda/Dawn/Poké Pad等)===")
    for label, games in [("win", wins), ("loss(全体)", losses),
                         ("loss(山札切れ)", deckout_losses), ("loss(非山札切れ)", other_losses)]:
        vals = [g["deck_searches"] for g in games]
        if vals:
            print(f"  {label}: n={len(vals)}, 平均サーチ回数={statistics.mean(vals):.2f}")

    out_path = KAGGLE_REPLAYS_DIR / "_diag_deckout_classification_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(all_games),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "n_deckout_losses": len(deckout_losses),
            "deckout_loss_rate_among_losses": (len(deckout_losses) / len(losses)) if losses else None,
            "games": all_games,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
