#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

blowout-piloting-validation-implementation-plan.md Step1-3: 「ブローアウト負け
(サイドレース大差)は piloting で減らせるのか、それとも構造的天井か」を、山札切れ検証
(_diag_deckout_topplayer_rate.py)と同じ手法で検証する。

自チーム(MORIOKA Tsuoi)以外の各プレイヤー視点の観測から、同アーキタイプ(Alakazam:
card_id 743 Alakazam + 742 Kadabra の両方を一度でも観測)を特定し、
- Step1: 全体勝率(自チーム44.6%と比較)
- Step2: 負けのうちブローアウト(サイドレース大差)が占める率(自チーム69.2%と比較)
         を、loss taxonomy(_diag_loss_taxonomy.py)と同一の優先順位つき相互排他ルールで判定
- Step3: チーム別集計から「勝率上位」「上位数チーム除外」の部分集団で頑健性を補正

このスクリプトは ptcg_ai / league / configs / deck.csv を一切変更しない。分析専用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
KAGGLE_REPLAYS_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = KAGGLE_REPLAYS_DIR / "replays"

sys.path.insert(0, str(SAMPLE_SUB))
sys.path.insert(0, str(KAGGLE_REPLAYS_DIR))

from cg.api import to_observation_class  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"
ALAKAZAM_ID = 743
KADABRA_ID = 742
CLOSE_MARGIN_THRESHOLD = 2  # 自分の最終残りサイドがこれ以下なら「接戦」
MAX_FILES = 6000  # 全リプレイ(4763件)をカバー

# 自チーム基準値(2026-07-21_loss_taxonomy_breakdown.md)
OWN_WIN_RATE = 162 / 363
OWN_LOSS_TOTAL = 201
OWN_BLOWOUT_RATE = 139 / 201
OWN_CLOSE_RATE = 10 / 201
OWN_DECKOUT_RATE = 52 / 201

# 上位部分集団の頑健性チェック用しきい値
MIN_GAMES_FOR_TEAM_STATS = 5
TOP_TEAMS_EXCLUDED_FOR_ROBUSTNESS = 5


def all_cards_seen(state, side: int) -> set[int]:
    player = state.players[side]
    ids = set()
    for p in (player.active or []):
        if p is not None:
            ids.add(p.id)
    for p in (player.bench or []):
        if p is not None:
            ids.add(p.id)
    for c in (player.hand or []):
        if c is not None:
            ids.add(c.id)
    for c in (player.discard or []):
        if c is not None:
            ids.add(c.id)
    return ids


def analyze_replay(path: Path) -> list[dict]:
    """1リプレイの両陣営(自チーム除く)について、勝敗とブローアウト分類を返す。"""
    with path.open(encoding="utf-8") as f:
        replay = json.load(f)
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    rewards = replay.get("rewards", [None, None])
    statuses = replay.get("statuses", [])
    steps = replay["steps"]

    results = []
    for side in (0, 1):
        if team_names[side] == TEAM_NAME:
            continue
        seen_ids: set[int] = set()
        deck_counts: list[int] = []
        final_prize = None
        final_active_empty = None
        final_bench_empty = None

        for i in range(len(steps) - 1):
            step = steps[i][side]
            if step.get("status") != "ACTIVE":
                continue
            obs_dict = step.get("observation")
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
            seen_ids |= all_cards_seen(state, me)
            deck_counts.append(state.players[me].deckCount)
            active = state.players[me].active or []
            bench = state.players[me].bench or []
            final_active_empty = not (active and active[0] is not None)
            final_bench_empty = len(bench) == 0
            final_prize = len(state.players[me].prize or [])

        if not deck_counts:
            continue
        is_alakazam = ALAKAZAM_ID in seen_ids and KADABRA_ID in seen_ids
        if not is_alakazam:
            continue

        opp_side = 1 - side
        if rewards[side] is None or rewards[opp_side] is None:
            outcome = "unknown"
        elif rewards[side] > rewards[opp_side]:
            outcome = "win"
        elif rewards[side] < rewards[opp_side]:
            outcome = "loss"
        else:
            outcome = "draw"

        category = None
        if outcome == "loss":
            # loss taxonomy と同一の優先順位つき相互排他ルール(2026-07-21_loss_taxonomy_definitions.md)
            if len(statuses) > side and statuses[side] != "DONE":
                category = "timeout_error"
            elif min(deck_counts) == 0:
                category = "deck_out"
            elif final_active_empty and final_bench_empty:
                category = "bench_wipe"
            elif final_prize is not None and final_prize <= CLOSE_MARGIN_THRESHOLD:
                category = "side_race_close"
            else:
                category = "side_race_blowout"

        results.append({
            "episode_id": replay.get("id"),
            "team_name": team_names[side],
            "outcome": outcome,
            "category": category,
        })
    return results


def summarize(matches: list[dict], label: str) -> dict:
    wins = [m for m in matches if m["outcome"] == "win"]
    losses = [m for m in matches if m["outcome"] == "loss"]
    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided) if decided else None

    by_cat: dict[str, list[dict]] = {}
    for m in losses:
        by_cat.setdefault(m["category"], []).append(m)
    blowout = by_cat.get("side_race_blowout", [])
    close = by_cat.get("side_race_close", [])
    deckout = by_cat.get("deck_out", [])
    bench_wipe = by_cat.get("bench_wipe", [])
    timeout = by_cat.get("timeout_error", [])

    blowout_rate = (len(blowout) / len(losses)) if losses else None

    print(f"\n=== {label} ===")
    print(f"decided={decided} (win={len(wins)}, loss={len(losses)}), "
          f"win_rate={win_rate * 100:.1f}%" if win_rate is not None else f"decided=0")
    if losses:
        print(f"  負け内訳: blowout={len(blowout)} ({len(blowout)/len(losses)*100:.1f}%), "
              f"close={len(close)} ({len(close)/len(losses)*100:.1f}%), "
              f"deck_out={len(deckout)} ({len(deckout)/len(losses)*100:.1f}%), "
              f"bench_wipe={len(bench_wipe)} ({len(bench_wipe)/len(losses)*100:.1f}%), "
              f"timeout={len(timeout)} ({len(timeout)/len(losses)*100:.1f}%)")

    return {
        "label": label,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_decided": decided,
        "win_rate": win_rate,
        "n_blowout": len(blowout),
        "n_close": len(close),
        "n_deckout": len(deckout),
        "n_bench_wipe": len(bench_wipe),
        "n_timeout": len(timeout),
        "blowout_rate_among_losses": blowout_rate,
    }


def main() -> None:
    all_files = sorted(REPLAYS_DIR.glob("episode-*-replay.json"))
    print(f"リプレイファイル総数: {len(all_files)}件、上限{MAX_FILES}件をスキャン")

    matches: list[dict] = []
    scanned = 0
    for path in all_files:
        if scanned >= MAX_FILES:
            break
        scanned += 1
        try:
            results = analyze_replay(path)
        except Exception:
            continue
        matches.extend(results)
        if scanned % 1000 == 0:
            print(f"  ...{scanned}件スキャン済み、Alakazam該当{len(matches)}件")

    print(f"\nスキャン完了: {scanned}件、Alakazamアーキタイプ該当(自チーム除く): "
          f"{len(matches)}件(延べプレイヤー×試合)")

    teams: dict[str, list[dict]] = {}
    for m in matches:
        teams.setdefault(m["team_name"], []).append(m)
    print(f"該当ユニークチーム数: {len(teams)}")

    # --- チーム別勝率(decided試合のみ) ---
    team_stats = []
    for name, ms in teams.items():
        w = sum(1 for m in ms if m["outcome"] == "win")
        l = sum(1 for m in ms if m["outcome"] == "loss")
        decided = w + l
        wr = (w / decided) if decided else None
        team_stats.append({"team_name": name, "n_games": len(ms), "n_wins": w,
                            "n_losses": l, "win_rate": wr})
    team_stats.sort(key=lambda t: -t["n_games"])
    print("\nチーム別内訳(試合数上位20):")
    for t in team_stats[:20]:
        wr_str = f"{t['win_rate']*100:.1f}%" if t["win_rate"] is not None else "n/a"
        print(f"  {t['team_name']}: {t['n_games']}試合 (win={t['n_wins']}, loss={t['n_losses']}, win_rate={wr_str})")

    # === Step1+2: 全体 ===
    overall = summarize(matches, "全体(第三者Alakazamパイロット)")
    print(f"\n[比較] 自チーム: win_rate={OWN_WIN_RATE*100:.1f}%, "
          f"blowout_rate(負け中)={OWN_BLOWOUT_RATE*100:.1f}%")

    # === Step3a: 勝率上位チーム(>=MIN_GAMES_FOR_TEAM_STATS試合、win_rate>=50%)の部分集団 ===
    eligible = [t for t in team_stats if t["n_games"] >= MIN_GAMES_FOR_TEAM_STATS and t["win_rate"] is not None]
    top_tier_names = {t["team_name"] for t in eligible if t["win_rate"] >= 0.5}
    top_tier_matches = [m for m in matches if m["team_name"] in top_tier_names]
    print(f"\n勝率上位チーム(>= {MIN_GAMES_FOR_TEAM_STATS}試合 かつ win_rate>=50%): "
          f"{len(top_tier_names)}チーム, {len(top_tier_matches)}試合")
    top_tier_summary = summarize(top_tier_matches, "Step3a: 勝率上位チームの部分集団")

    # === Step3b: 試合数上位チームを除いた頑健性チェック ===
    excluded_names = {t["team_name"] for t in team_stats[:TOP_TEAMS_EXCLUDED_FOR_ROBUSTNESS]}
    robust_matches = [m for m in matches if m["team_name"] not in excluded_names]
    print(f"\n試合数上位{TOP_TEAMS_EXCLUDED_FOR_ROBUSTNESS}チームを除外: "
          f"{len(excluded_names)}チーム除外, 残り{len(robust_matches)}試合")
    robust_summary = summarize(robust_matches, "Step3b: 試合数上位チーム除外(頑健性チェック)")

    out_path = KAGGLE_REPLAYS_DIR / "_diag_blowout_valid_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "scanned_files": scanned,
            "n_matches": len(matches),
            "n_unique_teams": len(teams),
            "team_stats": team_stats,
            "overall": overall,
            "top_tier": {"team_names": sorted(top_tier_names), "summary": top_tier_summary},
            "robust_excl_top_teams": {"excluded_team_names": sorted(excluded_names), "summary": robust_summary},
            "own_team_baseline": {
                "win_rate": OWN_WIN_RATE,
                "loss_total": OWN_LOSS_TOTAL,
                "blowout_rate_among_losses": OWN_BLOWOUT_RATE,
                "close_rate_among_losses": OWN_CLOSE_RATE,
                "deckout_rate_among_losses": OWN_DECKOUT_RATE,
            },
            "matches": matches,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
