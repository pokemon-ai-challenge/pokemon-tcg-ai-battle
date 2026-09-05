#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

deckout-piloting-diagnosis-implementation-plan.md Step3: 自チーム山札切れ負けの
決定レベル・トレース。

[2026-07-21_diag_deckout_deck_inventory.md](../sample_submission/results/2026-07-21_diag_deckout_deck_inventory.md)
(Step1)で特定した2系統を、勝ち試合・非山札切れ負け・山札切れ負けの3群で比較する:

- (a) sustain札(Sacred Ash=1129, Night Stretcher=1097, Lana's Aid=1184)の使用回数。
  「デッキに残り枚数がほぼ無いカードを、手札にあるのに使わなかった」頻度の直接測定は
  リプレイの手札スナップショットと PLAY ログの突合で可能だが、本スクリプトでは
  まず素直に「使用回数(自分のターン数で正規化)」を見る。deckCountが逼迫していた
  ターンに手札にあったのに未使用だったケースも別途カウントする。
- (b) 詰めの速さ: 自分の残りサイド(prize)の減り方(サイド1枚あたりのターン数)、
  Boss's Orders(1182)の使用回数(自分のターン数で正規化)。

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

from cg.api import LogType, to_observation_class  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"
MAX_EPISODES = 400

SACRED_ASH = 1129
NIGHT_STRETCHER = 1097
LANAS_AID = 1184
SUSTAIN_IDS = {SACRED_ASH, NIGHT_STRETCHER, LANAS_AID}
BOSSS_ORDERS = 1182

DECK_CRITICAL_THRESHOLD = 8  # 「残り僅か」とみなすdeckCountのしきい値


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
    steps = replay["steps"]
    deck_counts: list[int] = []
    initial_prize = None
    final_prize = None
    final_turn = None
    n_my_turns = 0
    sustain_plays = 0
    boss_plays = 0
    critical_turns_with_sustain_in_hand = 0
    critical_turns_with_sustain_played = 0

    seen_first_state = False

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
        player = state.players[me]

        # turn=0 の最初の1-2件はプライズ配置が完了する前のスナップショットで
        # len(prize)==0 になる既知の癖があるため、turn>=1 になってから initial_prize を取る。
        if not seen_first_state and state.turn >= 1:
            initial_prize = len(player.prize or [])
            seen_first_state = True
        if state.turn >= 1:
            final_prize = len(player.prize or [])
        final_turn = state.turn
        deck_counts.append(player.deckCount)

        hand_ids = {c.id for c in (player.hand or []) if c is not None}
        has_sustain_in_hand = bool(hand_ids & SUSTAIN_IDS)
        is_critical = player.deckCount <= DECK_CRITICAL_THRESHOLD

        played_this_turn = set()
        for log in obs.logs or []:
            if log.playerIndex != me:
                continue
            if log.type == LogType.TURN_START:
                n_my_turns += 1
            elif log.type == LogType.PLAY:
                if log.cardId in SUSTAIN_IDS:
                    sustain_plays += 1
                    played_this_turn.add(log.cardId)
                elif log.cardId == BOSSS_ORDERS:
                    boss_plays += 1

        if is_critical and has_sustain_in_hand:
            critical_turns_with_sustain_in_hand += 1
            if hand_ids & SUSTAIN_IDS & played_this_turn:
                critical_turns_with_sustain_played += 1

    if not deck_counts or initial_prize is None or final_prize is None:
        return None

    prizes_taken = initial_prize - final_prize
    turns_per_prize = (final_turn / prizes_taken) if prizes_taken > 0 else None

    return {
        "min_deck_count": min(deck_counts),
        "deck_out": min(deck_counts) == 0,
        "final_turn": final_turn,
        "n_my_turns": n_my_turns,
        "initial_prize": initial_prize,
        "final_prize": final_prize,
        "prizes_taken": prizes_taken,
        "turns_per_prize": turns_per_prize,
        "sustain_plays": sustain_plays,
        "sustain_plays_per_turn": (sustain_plays / n_my_turns) if n_my_turns else None,
        "boss_plays": boss_plays,
        "boss_plays_per_turn": (boss_plays / n_my_turns) if n_my_turns else None,
        "critical_turns_with_sustain_in_hand": critical_turns_with_sustain_in_hand,
        "critical_turns_with_sustain_played": critical_turns_with_sustain_played,
        "sustain_use_rate_when_critical_and_in_hand": (
            critical_turns_with_sustain_played / critical_turns_with_sustain_in_hand
        ) if critical_turns_with_sustain_in_hand else None,
    }


def summarize(label: str, games: list[dict]) -> None:
    if not games:
        print(f"  {label}: n=0")
        return
    n = len(games)

    def avg(key):
        vals = [g[key] for g in games if g[key] is not None]
        return (statistics.mean(vals), len(vals)) if vals else (None, 0)

    tpp, n_tpp = avg("turns_per_prize")
    spt, n_spt = avg("sustain_plays_per_turn")
    bpt, n_bpt = avg("boss_plays_per_turn")
    sur, n_sur = avg("sustain_use_rate_when_critical_and_in_hand")

    print(f"  {label}: n={n}")
    print(f"    サイド1枚あたりターン数(低いほど詰めが速い): "
          f"{tpp:.2f}" if tpp is not None else "    サイド1枚あたりターン数: N/A", f"(n={n_tpp})")
    print(f"    sustain札 使用/自ターン: {spt:.3f} (n={n_spt})" if spt is not None else "    sustain札使用率: N/A")
    print(f"    Boss's Orders 使用/自ターン: {bpt:.3f} (n={n_bpt})" if bpt is not None else "    Boss使用率: N/A")
    print(f"    山札逼迫時のsustain札使用率(手札にある場合): "
          f"{sur * 100:.1f}% (n={n_sur}機会)" if sur is not None else "    山札逼迫時sustain使用率: 機会なし")


def main() -> None:
    episode_ids = find_own_episode_ids(MAX_EPISODES)
    print(f"自チームエピソード候補: {len(episode_ids)}件")

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
        all_games.append(stats)

    print(f"分析対象試合: {len(all_games)}件")

    wins = [g for g in all_games if g["outcome"] == "win"]
    losses = [g for g in all_games if g["outcome"] == "loss"]
    deckout_losses = [g for g in losses if g["deck_out"]]
    other_losses = [g for g in losses if not g["deck_out"]]

    print(f"\n内訳: win={len(wins)}, loss(全体)={len(losses)}, "
          f"loss(山札切れ)={len(deckout_losses)}, loss(非山札切れ)={len(other_losses)}")

    print("\n=== (b) 詰めの速さ・Boss's Orders使用率 ===")
    summarize("win", wins)
    summarize("loss(全体)", losses)
    summarize("loss(山札切れ)", deckout_losses)
    summarize("loss(非山札切れ)", other_losses)

    out_path = KAGGLE_REPLAYS_DIR / "_diag_deckout_decision_trace_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"games": all_games}, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
