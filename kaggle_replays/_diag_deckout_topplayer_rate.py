#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

deckout-piloting-diagnosis-implementation-plan.md Step2: 「デッキは無罪」の裏取り。

自チーム以外(第三者)のリプレイから、同アーキタイプ(フーディン/Alakazam: card_id 743
Alakazam + 742 Kadabra の両方をそのプレイヤーの手札/ベンチ/バトル場/捨札のいずれかで
一度でも観測できたプレイヤー)を特定し、その山札切れ率(自チームの分析と同じ「自分の
deckCountが試合中に一度でも0を記録したか」の代理指標)を測る。自チームの25.9%より
明確に低ければ、「デッキ構築ではなく回し方が犯人」という前提訂正を裏取りできる。

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
MAX_FILES = 5000


def all_cards_seen(state, side: int) -> set[int]:
    player = state.players[side]
    ids = set()
    active = player.active or []
    for p in active:
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
    """1リプレイの両陣営について、(is_alakazam, outcome, deck_out) を返す。"""
    with path.open(encoding="utf-8") as f:
        replay = json.load(f)
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    rewards = replay.get("rewards", [None, None])
    steps = replay["steps"]

    results = []
    for side in (0, 1):
        if team_names[side] == TEAM_NAME:
            continue  # 自チーム側は既存の分析(diag_deckout_classification)で扱い済み
        seen_ids: set[int] = set()
        deck_counts: list[int] = []
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

        results.append({
            "episode_id": replay.get("id"),
            "team_name": team_names[side],
            "outcome": outcome,
            "deck_out": min(deck_counts) == 0,
        })
    return results


def main() -> None:
    all_files = sorted(REPLAYS_DIR.glob("episode-*-replay.json"))
    print(f"リプレイファイル総数: {len(all_files)}件、上限{MAX_FILES}件をスキャン")

    matches = []
    scanned = 0
    for path in all_files:
        if scanned >= MAX_FILES:
            break
        scanned += 1
        try:
            results = analyze_replay(path)
        except Exception as e:
            continue
        matches.extend(results)
        if scanned % 500 == 0:
            print(f"  ...{scanned}件スキャン済み、Alakazam該当{len(matches)}件")

    print(f"\nスキャン完了: {scanned}件、Alakazamアーキタイプ該当(自チーム除く): {len(matches)}件(延べプレイヤー×試合)")

    teams = {}
    for m in matches:
        teams.setdefault(m["team_name"], []).append(m)
    print(f"該当ユニークチーム数: {len(teams)}")
    for name, ms in sorted(teams.items(), key=lambda kv: -len(kv[1]))[:20]:
        print(f"  {name}: {len(ms)}試合")

    wins = [m for m in matches if m["outcome"] == "win"]
    losses = [m for m in matches if m["outcome"] == "loss"]
    deckout_losses = [m for m in losses if m["deck_out"]]

    print(f"\n=== 第三者Alakazamパイロットの山札切れ率 ===")
    print(f"win={len(wins)}, loss={len(losses)}, loss中の山札切れ={len(deckout_losses)}")
    if losses:
        rate = len(deckout_losses) / len(losses) * 100
        print(f"山札切れ率(loss中): {rate:.1f}%  (自チーム: 25.9%)")
    else:
        print("負け試合が見つからず、率を計算できません。")

    out_path = KAGGLE_REPLAYS_DIR / "_diag_deckout_topplayer_rate_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "scanned_files": scanned,
            "n_matches": len(matches),
            "n_unique_teams": len(teams),
            "teams": {k: len(v) for k, v in teams.items()},
            "n_wins": len(wins),
            "n_losses": len(losses),
            "n_deckout_losses": len(deckout_losses),
            "deckout_rate_among_losses": (len(deckout_losses) / len(losses)) if losses else None,
            "matches": matches,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
