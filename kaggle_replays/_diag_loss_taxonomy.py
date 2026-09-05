#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

loss-taxonomy-implementation-plan.md Step1-3: 実戦敗因の網羅的分類、試合長/最終サイド差との
クロス集計、相手アーキタイプとのクロス集計をまとめて行う。

分類は優先順位つきの相互排他ルール(詳細は
sample_submission/results/2026-07-21_loss_taxonomy_definitions.md 参照):

1. timeout/error: replay["statuses"][my_idx] != "DONE"
2. 山札切れ: 自分の deckCount が試合中に一度でも0を記録(既存の代理指標、diag_deckout_classification と同一)
3. ベンチ壊滅: 山札切れでなく、相手の最終残りサイドが0でない(=相手はサイド完投で勝ったのではない)
   かつ最終状態で自分の active/bench が両方とも空
4. サイドレース負け: 相手の最終残りサイドが0(相手はサイド完投で勝った)。
   自分の最終残りサイドで 接戦(<=2, 4枚以上獲得) / 大差(>=3) に分岐。
   大差側はさらに「序盤展開の停滞」(Alakazam(743)を一度も場に出せなかった)を副次フラグとして記録。
5. その他/分類不能: 上記のいずれにも当てはまらない残り

相手アーキタイプは ptcg_ai.opponent_modeling.OpponentKnowledge で試合を通じて観測した
相手の公開カードを蓄積し、HybridDeckPredictor.predict() で最終ターン時点の確率分布を得て
最尤クラスを採用する(事後同定、既存の学習済み重みをそのまま使う。追加学習はしない)。

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

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402
from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"
MAX_EPISODES = 400
ALAKAZAM_ID = 743
CLOSE_MARGIN_THRESHOLD = 2  # 自分の最終残りサイドがこれ以下なら「接戦」
EARLY_TURN_THRESHOLD = 6    # この時点までにAlakazamが場に出ていなければ「展開停滞」


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


_predictor: HybridDeckPredictor | None = None


def get_predictor() -> HybridDeckPredictor:
    global _predictor
    if _predictor is None:
        _predictor = HybridDeckPredictor()
    return _predictor


def analyze_loss(replay: dict, my_idx: int) -> dict | None:
    steps = replay["steps"]
    opp_idx = 1 - my_idx
    knowledge = OpponentKnowledge(opponent_index=opp_idx)

    deck_counts: list[int] = []
    final_turn = None
    my_final_prize = None
    opp_final_prize = None
    my_final_active_empty = None
    my_final_bench_empty = None
    alakazam_seen_turn = None
    seen_initial = False

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

        try:
            knowledge.update_from_logs(obs.logs or [])
            knowledge.update_from_state(state)
        except Exception:
            pass

        if state.turn >= 1:
            deck_counts.append(state.players[me].deckCount)
            final_turn = state.turn
            my_final_prize = len(state.players[me].prize or [])
            opp_final_prize = len(state.players[1 - me].prize or [])
            active = state.players[me].active or []
            bench = state.players[me].bench or []
            my_final_active_empty = not (active and active[0] is not None)
            my_final_bench_empty = len(bench) == 0

            if alakazam_seen_turn is None and state.turn <= EARLY_TURN_THRESHOLD:
                hand_ids = {c.id for c in (state.players[me].hand or []) if c is not None}
                board_ids = {p.id for p in active if p is not None} | {p.id for p in bench if p is not None}
                if ALAKAZAM_ID in hand_ids or ALAKAZAM_ID in board_ids:
                    alakazam_seen_turn = state.turn

    if not deck_counts or final_turn is None:
        return None

    deck_out = min(deck_counts) == 0

    # 注意(重要な既知の癖): リプレイの最終スナップショットは、相手が自分のターンで
    # 勝利条件を満たした瞬間を捉えられない(自分視点の観測は自分の最後の手番までしか
    # 進まないため、相手の最終ターンの結果は反映されない)。実測でも opp_final_prize が
    # 0になるケースは皆無で、多くは1(=あと1枚で完投という、まさに勝利直前の状態)
    # だった。そのため opp_final_prize==0 を条件にする判定はほぼ機能しない。
    # 代わりに「timeout/error でも山札切れでもベンチ壊滅でもない」という消去法で
    # サイドレース負け(相手のサイド完投)と判定する(このゲームの勝利条件は
    # サイド完投/ベンチ壊滅/山札切れ/まれなカード効果、の実質4通りしかないため)。
    category = None
    detail = {}
    if deck_out:
        category = "deck_out"
    elif my_final_active_empty and my_final_bench_empty:
        category = "bench_wipe"
    else:
        if my_final_prize is not None and my_final_prize <= CLOSE_MARGIN_THRESHOLD:
            category = "side_race_close"
        else:
            category = "side_race_blowout"
        detail["development_stalled"] = alakazam_seen_turn is None

    # アーキタイプ推定(最終観測カードで事後推論)。
    archetype = "unknown"
    try:
        features = knowledge.get_prediction_features()
        observed_cards = features.get("observed_cards", {})
        predictor = get_predictor()
        if predictor.is_ready and observed_cards:
            top = predictor.predict_top(observed_cards, final_turn, n=1)
            if top:
                archetype = top[0][0]
    except Exception:
        pass

    return {
        "category": category,
        "detail": detail,
        "final_turn": final_turn,
        "my_final_prize": my_final_prize,
        "opp_final_prize": opp_final_prize,
        "opp_archetype": archetype,
    }


def main() -> None:
    episode_ids = find_own_episode_ids(MAX_EPISODES)
    print(f"自チームエピソード候補: {len(episode_ids)}件")

    losses = []
    n_total = 0
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
        n_total += 1
        if outcome != "loss":
            continue

        statuses = replay.get("statuses", [])
        if len(statuses) > my_idx and statuses[my_idx] != "DONE":
            losses.append({
                "episode_id": eid, "category": "timeout_error",
                "detail": {"status": statuses[my_idx]},
                "final_turn": None, "my_final_prize": None, "opp_final_prize": None,
                "opp_archetype": "unknown",
            })
            continue

        result = analyze_loss(replay, my_idx)
        if result is None:
            result = {"category": "other", "detail": {"reason": "no_states_parsed"},
                       "final_turn": None, "my_final_prize": None, "opp_final_prize": None,
                       "opp_archetype": "unknown"}
        result["episode_id"] = eid
        losses.append(result)

    print(f"分析対象: 試合総数={n_total}, 負け={len(losses)}件")

    # --- カテゴリ別内訳 ---
    by_cat: dict[str, list[dict]] = {}
    for g in losses:
        by_cat.setdefault(g["category"], []).append(g)

    print("\n=== カテゴリ別内訳 ===")
    for cat, games in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        pct = len(games) / len(losses) * 100
        print(f"  {cat}: {len(games)}件 ({pct:.1f}%)")
        if cat.startswith("side_race"):
            stalled = sum(1 for g in games if g["detail"].get("development_stalled"))
            print(f"    うち序盤展開停滞(ターン{EARLY_TURN_THRESHOLD}までにAlakazam未展開): "
                  f"{stalled}件 ({stalled / len(games) * 100:.1f}%)")

    print("\n=== 試合長・最終サイド差(カテゴリ別平均) ===")
    for cat, games in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        turns = [g["final_turn"] for g in games if g["final_turn"] is not None]
        my_prize = [g["my_final_prize"] for g in games if g["my_final_prize"] is not None]
        if turns:
            print(f"  {cat}: n={len(games)}, 平均最終ターン={statistics.mean(turns):.1f}, "
                  f"平均自分の最終残りサイド={statistics.mean(my_prize):.2f}" if my_prize else "")

    # --- 相手アーキタイプ×カテゴリ クロス集計 ---
    print("\n=== 相手アーキタイプ別 内訳(上位10) ===")
    by_archetype: dict[str, list[dict]] = {}
    for g in losses:
        by_archetype.setdefault(g["opp_archetype"], []).append(g)
    for arch, games in sorted(by_archetype.items(), key=lambda kv: -len(kv[1]))[:10]:
        cat_counts = {}
        for g in games:
            cat_counts[g["category"]] = cat_counts.get(g["category"], 0) + 1
        print(f"  {arch}: {len(games)}件負け  内訳={cat_counts}")

    out_path = KAGGLE_REPLAYS_DIR / "_diag_loss_taxonomy_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "n_total_games": n_total,
            "n_losses": len(losses),
            "by_category_count": {k: len(v) for k, v in by_cat.items()},
            "by_archetype_count": {k: len(v) for k, v in by_archetype.items()},
            "losses": losses,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
