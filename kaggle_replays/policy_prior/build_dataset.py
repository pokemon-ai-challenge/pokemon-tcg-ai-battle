#!/usr/bin/env python3
"""上位エージェントのリプレイから Behavior Cloning 用データセット(JSONL)を作る。

要件定義 v2 §4（Semantic Action）/ §7（観測と特権的教師の分離）/ §10（分割）に対応。

## 1行 = 1 decision

`steps[i][player]["observation"]["select"]` が提示した選択肢集合と、
`steps[i+1][player]["action"]` が示す実際の選択を対にする
(extract_training_data.py と同じ規約)。選択肢は semantic_action.resolve_option()
で解決済みの形で格納するため、学習側は index の意味を知らなくてよい。

## 既定のスコープ（MVP）

実測(ptcg_replay_schema_report.md)にもとづき、既定では以下に絞る。

- `SelectContext.MAIN` のみ … 全 decision の 52.4%。最も判断が重い
- 単一選択のみ … 複数選択は 5.1%。pointwise ranking で表現できない
- 強制手を除外 … 選択肢1個は 8.1%。方策の情報を持たず精度を水増しする

`--all-contexts` / `--include-multi` / `--include-forced` で解除できる。

## リーク防止（v2 §7 FR-LEAK-001）

格納する状態は「その手番のプレイヤーに提示された observation」に限る。
リプレイには両プレイヤーの observation が入っているが、相手側の観測は参照しない。
相手の手札は null、サイドは [null]*6 として渡ってくるため、observation を
そのまま使う限り非公開情報は混入しない。

## 使い方

  python policy_prior/build_dataset.py \
      --replays-dir ./replays --out ./policy_prior/output/main_decisions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_action import (  # noqa: E402
    ResolutionStats,
    action_label,
    check_invariants,
    resolve_option,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAIN_CONTEXT = 0


def load_master_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                index[row["episode_id"]] = row
    return index


def assign_split(episode_id: str, ratios=(0.8, 0.1, 0.1)) -> str:
    """エピソード単位で train/val/test を決める（v2 §10）。

    decision 単位で分けると同一エピソード内の強い相関でリークするため、
    必ずエピソード単位で分ける。乱数シードではなく episode_id のハッシュを使うのは、
    エピソードが追加されても既存の割り当てが動かないようにするため
    （シード + shuffle だと母集合が変わるたびに全件の割り当てが変わる）。
    """
    h = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "val"
    return "test"


def observable_state(current: dict, me: int) -> dict:
    """その手番のプレイヤーが見えている盤面だけを取り出す。"""
    players = current.get("players") or [{}, {}]
    own, opp = players[me], players[1 - me]

    def in_play(entity):
        if not isinstance(entity, dict):
            return None
        return {
            "card_id": entity.get("id"),
            "hp": entity.get("hp"),
            "max_hp": entity.get("maxHp"),
            "n_energy": len(entity.get("energies") or []),
            "n_tool": len(entity.get("tools") or []),
            "appear_this_turn": entity.get("appearThisTurn"),
        }

    def side(player: dict) -> dict:
        return {
            "active": in_play((player.get("active") or [None])[0]),
            "bench": [in_play(e) for e in (player.get("bench") or [])],
            "n_prize": len(player.get("prize") or []),
            "n_hand": player.get("handCount"),
            "n_deck": player.get("deckCount"),
            # トラッシュは公開領域。カード種の列として持つ
            "discard": [e.get("id") for e in (player.get("discard") or []) if isinstance(e, dict)],
            "asleep": player.get("asleep"),
            "confused": player.get("confused"),
            "paralyzed": player.get("paralyzed"),
            "poisoned": player.get("poisoned"),
            "burned": player.get("burned"),
        }

    return {
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "first_player": current.get("firstPlayer"),
        "energy_attached": current.get("energyAttached"),
        "retreated": current.get("retreated"),
        "supporter_played": current.get("supporterPlayed"),
        "stadium_played": current.get("stadiumPlayed"),
        "stadium": [e.get("id") for e in (current.get("stadium") or []) if isinstance(e, dict)],
        "own": side(own),
        "opponent": side(opp),
    }


def iter_decisions(replay: dict, episode_id: str, master_row: dict | None, args):
    steps = replay["steps"]
    meta = {p["player_index"]: p for p in (master_row or {}).get("players", [])}
    stats: ResolutionStats = args._stats
    skipped: Counter = args._skipped

    for i in range(len(steps) - 1):
        for player_index in (0, 1):
            step = steps[i][player_index]
            if step.get("status") != "ACTIVE":
                continue
            obs = step.get("observation") or {}
            select, current = obs.get("select"), obs.get("current")
            action = steps[i + 1][player_index].get("action")

            if not select or not current:
                skipped["no_select_or_current"] += 1   # デッキ選択ターン等
                continue
            if not args.all_contexts and select.get("context") != MAIN_CONTEXT:
                skipped["other_context"] += 1
                continue
            options = select.get("option") or []
            if not action or not isinstance(action, list):
                skipped["no_action"] += 1
                continue
            max_count = select.get("maxCount", 1)
            if not args.include_multi and max_count != 1:
                skipped["multi_select"] += 1
                continue
            forced = len(options) < 2
            if not args.include_forced and forced:
                skipped["forced"] += 1
                continue

            check_invariants(select, stats)
            me = current.get("yourIndex", player_index)
            actions = [resolve_option(o, current, me, stats) for o in options]
            player_meta = meta.get(player_index, {})

            yield {
                "episode_id": episode_id,
                "player_index": player_index,
                "step_index": i,
                "split": assign_split(episode_id),
                # --- 教師の出所（サンプル重み付けに使う） ---
                "agent": player_meta.get("team_name"),
                "rank_at_fetch": player_meta.get("rank_at_fetch"),
                "score_at_fetch": player_meta.get("leaderboard_score_at_fetch"),
                # --- decision ---
                "select_context": select.get("context"),
                "select_type": select.get("type"),
                "min_count": select.get("minCount", 1),
                "max_count": max_count,
                "n_options": len(options),
                "forced": forced,
                "actions": actions,
                "chosen": action,
                "chosen_label": list(action_label(actions[action[0]])) if action else None,
                # --- 観測（v2 §7 Observable Features のみ） ---
                "state": observable_state(current, me),
                "remaining_overage_time": obs.get("remainingOverageTime"),
            }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replays-dir", type=Path, default=here.parent / "replays")
    parser.add_argument("--master", type=Path, default=here.parent / "index" / "episodes_master.jsonl")
    parser.add_argument("--out", type=Path, default=here / "output" / "main_decisions.jsonl")
    parser.add_argument("--all-contexts", action="store_true", help="MAIN 以外も含める")
    parser.add_argument("--include-multi", action="store_true", help="複数選択も含める")
    parser.add_argument("--include-forced", action="store_true", help="強制手も含める")
    parser.add_argument("--limit", type=int, default=None, help="先頭N件のリプレイのみ")
    args = parser.parse_args()

    args._stats = ResolutionStats()
    args._skipped = Counter()

    master = load_master_index(args.master)
    paths = sorted(args.replays_dir.glob("episode-*-replay.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"リプレイが見つかりません: {args.replays_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_counts: Counter = Counter()
    agents: Counter = Counter()
    n_rows = n_bad = 0

    with args.out.open("w", encoding="utf-8") as f:
        for path in paths:
            episode_id = path.stem.split("-")[1]
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 -- 1件の破損で全体を止めない
                print(f"  ! 読込失敗 {path.name}: {exc}", file=sys.stderr)
                n_bad += 1
                continue
            for row in iter_decisions(replay, episode_id, master.get(episode_id), args):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                split_counts[row["split"]] += 1
                agents[row["agent"]] += 1
                n_rows += 1

    stats = args._stats
    print(f"リプレイ {len(paths)} 件 (読込失敗 {n_bad})")
    print(f"decision {n_rows:,} 行を書き出し: {args.out}")
    print(f"  split: {dict(split_counts)}")
    print(f"  エージェント {len(agents)} 種 (上位5: {agents.most_common(5)})")
    print(f"  除外: {dict(args._skipped)}")
    print(f"  Option 解決: 成功 {sum(stats.ok.values()):,} / 失敗 "
          f"{sum(stats.fail.values()):,} ({stats.failure_rate:.2%})")
    if stats.fail:
        print(f"  ! 解決失敗の内訳: {dict(stats.fail)}")
    if stats.invariant_violations:
        print(f"  ! 不変条件の破れ: {dict(stats.invariant_violations)}")

    summary = args.out.with_suffix(".summary.json")
    summary.write_text(
        json.dumps(
            {
                "n_replays": len(paths),
                "n_rows": n_rows,
                "splits": dict(split_counts),
                "n_agents": len(agents),
                "skipped": dict(args._skipped),
                "resolution": stats.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  サマリ: {summary}")


if __name__ == "__main__":
    main()
