"""コーパスを拡張スキーマで再ラベルし、Phase 2 コストを分解する（Step 1-18 第 2 回）。

局面は採り直さない。保存済みの `obs` / `hidden_stub` に対して
拡張したオラクルを再実行し、以下を分けて記録する。

  oracle_win / searched_to_depth / proof_complete   （指示 5）
  minimum_depth_kind = exact | upper_bound | unknown（指示 6）
  chance_free / outcome_complete                    （指示 7）
  near miss の不足量タグ / negative の細分類        （指示 8）
  has_root_deck_reveal / has_midturn_deck_reveal /
  has_opponent_choice / has_multi_opponent_choice   （指示 16）
  ground_truth = EXACT | CHANCE_FREE |
                 DETERMINIZATION_ONLY | DEPTH_LIMITED（指示 17）

さらに Phase 2 のコストを outcome 数で分解する（指示 11）。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession
from tools import lethal_oracle

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def near_miss_deficit(oracle: dict) -> str:
    """指示 8: 何が足りなかったのかをタグ付けする（測れる範囲だけ）。"""
    if oracle["oracle_win"]:
        return "-"
    remaining = oracle.get("min_opponent_active_hp")
    if oracle.get("can_knock_out_active") and oracle.get("opponent_bench_count", 0) > 0:
        return "bench_remaining"     # 倒せるが相手のベンチが残っている
    if remaining is not None and remaining <= 60:
        return f"damage_short_{remaining}"
    if remaining is not None:
        return "damage_short_large"
    if oracle.get("attach_still_available"):
        return "energy_or_target_short"
    return "unknown_deficit"


def negative_subtype(oracle: dict) -> str:
    """指示 8: negative を 1 カテゴリで済ませない。"""
    if oracle["oracle_win"]:
        return "-"
    if oracle["truncated"]:
        return "depth_or_budget_limited"
    if oracle.get("has_opponent_choice"):
        return "opponent_choice_involved"
    if oracle.get("can_knock_out_active"):
        return "near_miss"
    return "clearly_impossible"


def relabel(rows) -> list[dict]:
    out = []
    for index, row in enumerate(rows):
        observation = to_observation_class(row["obs"])
        hidden = HiddenState.from_stub(row["hidden_stub"])
        try:
            oracle = lethal_oracle.solve(
                observation, hidden, 0,
                max_depth=6, time_limit_ms=6_000.0, node_limit=20_000,
            ).as_dict()
        except Exception:  # noqa: BLE001
            oracle = dict(row["oracle"])
            oracle["relabel_error"] = True
        oracle["near_miss_deficit"] = near_miss_deficit(oracle)
        oracle["negative_subtype"] = negative_subtype(oracle)
        oracle["has_multi_opponent_choice"] = oracle.get("opponent_choice_count", 0) > 1
        row = dict(row)
        row["oracle"] = oracle
        out.append(row)
        if index % 40 == 39:
            print(f"  relabel {index + 1}/{len(rows)}", flush=True)
    return out


def phase2_cost(rows, deck) -> None:
    """指示 11: Phase 2 のコストを outcome 数で分解する。"""
    print("\n## Phase 2 コスト分解")
    buckets = defaultdict(list)
    detail = []
    for row in rows:
        observation = to_observation_class(row["obs"])
        hidden = HiddenState.from_stub(row["hidden_stub"])
        started = time.perf_counter()
        try:
            with SearchSession(observation, hidden) as session:
                backend = CgBackend(
                    session, observation, hidden,
                    deck_composition=KnownDeckComposition.from_card_ids(deck),
                )
                if backend.is_win(backend.root()):
                    continue
                result = phase2.search(
                    backend,
                    Budget(time_limit_ms=500.0, max_nodes=20_000,
                           max_depth=8, max_chance_depth=1),
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                acquired = getattr(session, "acquired", 0)
        except Exception:  # noqa: BLE001
            continue
        detail.append({
            "id": row["id"],
            "proof": result.proof.name,
            "nodes": result.nodes,
            "search_begin": acquired,
            "ms": elapsed,
            "stop": [s.name for s in result.stop_reasons],
        })
        # search_begin 回数 = materialization 回数の実測プロキシ
        key = ("1", "2-5", "6-20", "21-100", ">100")[
            min(4, 0 if acquired <= 1 else 1 if acquired <= 5 else
                2 if acquired <= 20 else 3 if acquired <= 100 else 4)
        ]
        buckets[key].append((elapsed, result.nodes, result.proof.name))
    print(f"{'search_begin 回数':>18}{'件数':>6}{'p50 ms':>9}{'p95 ms':>9}"
          f"{'max ms':>9}{'nodes p50':>11}  proof 内訳")
    for key in ("1", "2-5", "6-20", "21-100", ">100"):
        group = buckets.get(key, [])
        if not group:
            continue
        times = [g[0] for g in group]
        nodes = [g[1] for g in group]
        proofs = Counter(g[2] for g in group)
        print(f"{key:>18}{len(group):>6}{percentile(times, 0.5):>9.1f}"
              f"{percentile(times, 0.95):>9.1f}{max(times):>9.1f}"
              f"{percentile(nodes, 0.5):>11.0f}  {dict(proofs)}")
    worst = sorted(detail, key=lambda d: -d["ms"])[:5]
    print("  最も遅い 5 件:")
    for item in worst:
        print(f"    {item['id']} {item['ms']:8.1f}ms  search_begin={item['search_begin']:>4} "
              f"nodes={item['nodes']:>5} {item['proof']} {item['stop']}")


def false_positive_by_capability(rows, deck) -> None:
    """指示 13/14: false PROVEN_WIN を能力別に集計する。"""
    print("\n## false PROVEN_WIN（能力別）")
    counts = Counter()
    violations = []
    for row in rows:
        oracle = row["oracle"]
        if oracle["oracle_win"] or oracle["truncated"]:
            continue  # 「勝てない」と言い切れない局面は判定に使わない
        observation = to_observation_class(row["obs"])
        hidden = HiddenState.from_stub(row["hidden_stub"])
        try:
            with SearchSession(observation, hidden) as session:
                backend = CgBackend(
                    session, observation, hidden,
                    deck_composition=KnownDeckComposition.from_card_ids(deck),
                )
                if backend.is_win(backend.root()):
                    continue
                result = phase1.search(
                    backend, Budget(time_limit_ms=500.0, max_nodes=20_000, max_depth=8)
                )
        except Exception:  # noqa: BLE001
            continue
        tags = set(oracle.get("capabilities") or []) or {"(no_line)"}
        if oracle.get("has_opponent_choice"):
            tags.add("OPPONENT_CHOICE")
        if row["category"] == "near_miss":
            tags.add("NEAR_MISS")
        for tag in tags:
            counts[f"{tag}_checked"] += 1
            if result.proof.name == "PROVEN_WIN":
                counts[f"{tag}_FALSE_WIN"] += 1
        if result.proof.name == "PROVEN_WIN":
            violations.append(row["id"])
    for name in sorted({k.rsplit("_", 1)[0] for k in counts if k.endswith("checked")}):
        checked = counts[f"{name}_checked"]
        false_win = counts.get(f"{name}_FALSE_WIN", 0)
        print(f"  {name:20} 検査 {checked:>4} 件  false PROVEN_WIN {false_win}")
    print(f"  合計 false PROVEN_WIN: {len(violations)} {violations[:10]}")


def main() -> None:
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    deck = read_deck_csv()
    print(f"再ラベル対象 {len(rows)} 件")
    relabeled = relabel(rows)
    CORPUS.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in relabeled) + "\n",
        encoding="utf-8",
    )
    print(f"書き戻し: {CORPUS}")

    print("\n## Ground truth confidence")
    for name, count in Counter(r["oracle"]["ground_truth"] for r in relabeled).most_common():
        print(f"  {name:28} {count}")
    print("\n## minimum_depth の厳密さ")
    for name, count in Counter(
        r["oracle"]["minimum_depth_kind"] for r in relabeled
    ).most_common():
        print(f"  {name:28} {count}")
    exact = [r for r in relabeled if r["oracle"]["minimum_depth_kind"] == "exact"]
    print("  exact な minimum_depth 分布:",
          dict(sorted(Counter(r["oracle"]["minimum_depth"] for r in exact).items())))
    print("\n## B1 / B4 タグ")
    for key in ("has_root_deck_reveal", "has_midturn_deck_reveal",
                "has_opponent_choice", "has_multi_opponent_choice",
                "opponent_choice_after_ko"):
        print(f"  {key:28} {sum(1 for r in relabeled if r['oracle'].get(key))}")
    print("\n## near miss の不足量")
    for name, count in Counter(
        r["oracle"]["near_miss_deficit"] for r in relabeled
        if r["oracle"]["near_miss_deficit"] != "-"
    ).most_common():
        print(f"  {name:28} {count}")
    print("\n## negative の細分類")
    for name, count in Counter(
        r["oracle"]["negative_subtype"] for r in relabeled
        if r["oracle"]["negative_subtype"] != "-"
    ).most_common():
        print(f"  {name:28} {count}")

    phase2_cost(relabeled, deck)
    false_positive_by_capability(relabeled, deck)


if __name__ == "__main__":
    main()
