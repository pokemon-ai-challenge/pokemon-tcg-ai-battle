"""能力評価コーパスの分析（Step 1-18 指示 6/7/8/9/14/15）。

solver は一切変更しない。保存コーパスに対して測るだけ。
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

from cg.api import OptionType, to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# 実装コストは推測で工数を書かない。low/medium/high/unknown のみ。
IMPL_COST = {
    "ATTACK": "low（実装済み）",
    "ATTACH": "low（実装済み）",
    "GUST": "unknown",
    "EVOLVE": "low（実装済み）",
    "SEARCH": "unknown",
    "RETREAT": "low（実装済み）",
    "DRAW": "medium（Phase 2 依存）",
    "COIN": "medium（B8）",
}


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def load():
    if len(sys.argv) > 1:
        globals()["CORPUS"] = Path(sys.argv[1])
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows


def run(module, row, deck, *, depth=8, ms=500.0, nodes=20_000, chance_depth=1):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    started = time.perf_counter()
    with SearchSession(observation, hidden) as session:
        backend = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        if backend.is_win(backend.root()):
            return None, 0.0
        result = module.search(
            backend,
            Budget(time_limit_ms=ms, max_nodes=nodes, max_depth=depth,
                   max_chance_depth=chance_depth),
        )
    return result, (time.perf_counter() - started) * 1000.0


def capability_tags(row) -> set[str]:
    """この fixture が要求する能力。positive は勝ち筋から、negative は root から。"""
    oracle = row["oracle"]
    tags = set(oracle["capabilities"])
    if oracle["expected_lethal"] and not oracle["chance_free"]:
        tags.add("DRAW")  # 近似: chance を含む勝ち筋はほぼドロー由来
    if not tags:
        observation = to_observation_class(row["obs"])
        if observation.select is not None:
            tags = {OptionType(int(o.type)).name for o in observation.select.option}
    return tags


def section_fixture_summary(rows):
    print("\n## Fixture")
    print(f"  総数: {len(rows)}")
    by_category = Counter(row["category"] for row in rows)
    for name, count in by_category.most_common():
        print(f"    {name:24} {count}")
    positive = [r for r in rows if r["oracle"]["expected_lethal"]]
    near = [r for r in rows if r["category"] == "near_miss"]
    negative = [r for r in rows if r["category"] in ("negative", "opponent_choice")]
    print(f"  positive={len(positive)}  near_miss={len(near)}  negative={len(negative)}")
    depths = Counter(r["oracle"]["minimum_depth"] for r in positive)
    print("  minimum_depth 分布:", {k: depths[k] for k in sorted(depths)})
    print(f"  chance_free positive: {sum(1 for r in positive if r['oracle']['chance_free'])}"
          f" / {len(positive)}")
    print(f"  ターン中の相手選択を含む: "
          f"{sum(1 for r in rows if r['oracle']['has_opponent_choice'])}")
    print(f"  サイド以外の勝利: {sum(1 for r in rows if r['oracle']['non_prize_win'])}")


def section_capability_matrix(rows, deck, cache):
    print("\n## Capability demand matrix")
    header = (f"{'Capability':<12}{'fixture':>8}{'lethal':>8}{'解けた':>8}"
              f"{'UNKNOWN':>9}{'平均depth':>10}  実装コスト")
    print(header)
    buckets = defaultdict(list)
    for row in rows:
        for tag in capability_tags(row):
            buckets[tag].append(row)
    for name in ("ATTACK", "ATTACH", "GUST", "EVOLVE", "SEARCH", "RETREAT",
                 "DRAW", "COIN", "CARD", "PLAY", "ABILITY"):
        group = buckets.get(name, [])
        if not group and name not in IMPL_COST:
            continue
        lethal = [r for r in group if r["oracle"]["expected_lethal"]]
        solved = sum(1 for r in lethal if cache[r["id"]][0] == "PROVEN_WIN")
        unknown = sum(1 for r in group if cache[r["id"]][0] == "UNKNOWN")
        depths = [r["oracle"]["minimum_depth"] for r in lethal
                  if r["oracle"]["minimum_depth"]]
        average = f"{sum(depths) / len(depths):.1f}" if depths else "-"
        print(f"{name:<12}{len(group):>8}{len(lethal):>8}{solved:>8}"
              f"{unknown:>9}{average:>10}  {IMPL_COST.get(name, 'unknown')}")


def section_phase_comparison(rows, deck):
    print("\n## Phase 1 vs Phase 2")
    cache = {}
    summary = {}
    for label, module, chance_depth in (("Phase 1", phase1, 0), ("Phase 2", phase2, 1)):
        counts = Counter()
        durations = []
        per_row = {}
        for row in rows:
            try:
                result, elapsed = run(module, row, deck, chance_depth=chance_depth)
            except Exception:  # noqa: BLE001
                counts["ERROR"] += 1
                per_row[row["id"]] = "ERROR"
                continue
            if result is None:
                counts["ALREADY_WON"] += 1
                per_row[row["id"]] = "ALREADY_WON"
                continue
            counts[result.proof.name] += 1
            durations.append(elapsed)
            per_row[row["id"]] = result.proof.name
        summary[label] = (counts, durations, per_row)
        print(f"  {label}: WIN={counts['PROVEN_WIN']} NO_WIN={counts['PROVEN_NO_WIN']} "
              f"UNKNOWN={counts['UNKNOWN']} ERROR={counts['ERROR']} "
              f"p50={percentile(durations, 0.5):.1f} p95={percentile(durations, 0.95):.1f} "
              f"p99={percentile(durations, 0.99):.1f} max={max(durations or [0]):.1f}")
    p1 = summary["Phase 1"][2]
    p2 = summary["Phase 2"][2]
    gained = [rid for rid in p1
              if p1[rid] != "PROVEN_WIN" and p2.get(rid) == "PROVEN_WIN"]
    lost = [rid for rid in p1
            if p1[rid] == "PROVEN_WIN" and p2.get(rid) != "PROVEN_WIN"]
    print(f"  Phase 1 で解けず Phase 2 で確定できた: {len(gained)}")
    print(f"  Phase 1 で解けたが Phase 2 で落ちた  : {len(lost)}")
    for rid in p1:
        cache[rid] = (p1[rid],)
    return cache


def section_depth_sweep(rows, deck):
    print("\n## Phase 1 depth sweep（予算 100ms / 10k nodes）")
    positives = [r for r in rows if r["oracle"]["expected_lethal"]]
    print(f"{'depth':>6}{'WIN':>6}{'NO_WIN':>8}{'UNKNOWN':>9}"
          f"{'md<=d を拾えた':>16}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for depth in (1, 2, 3, 4, 6, 8):
        counts = Counter()
        durations = []
        caught = 0
        reachable = 0
        for row in rows:
            try:
                result, elapsed = run(row=row, module=phase1, deck=deck, depth=depth,
                                      ms=100.0, nodes=10_000, chance_depth=0)
            except Exception:  # noqa: BLE001
                counts["ERROR"] += 1
                continue
            if result is None:
                continue
            counts[result.proof.name] += 1
            durations.append(elapsed)
            minimum = row["oracle"]["minimum_depth"]
            if row in positives and minimum is not None and minimum <= depth:
                reachable += 1
                if result.proof.name == "PROVEN_WIN":
                    caught += 1
        print(f"{depth:>6}{counts['PROVEN_WIN']:>6}{counts['PROVEN_NO_WIN']:>8}"
              f"{counts['UNKNOWN']:>9}{f'{caught}/{reachable}':>16}"
              f"{percentile(durations, 0.5):>8.1f}{percentile(durations, 0.95):>8.1f}"
              f"{percentile(durations, 0.99):>8.1f}{max(durations or [0]):>8.1f}")


def section_attach_analysis(rows, deck):
    """指示 6: 「ATTACH が 48%」の意味を確認する。"""
    print("\n## ATTACH の実態")
    positives = [r for r in rows if r["oracle"]["expected_lethal"]]
    needs = [r for r in positives if "ATTACH" in r["oracle"]["capabilities"]]
    print(f"  ATTACH が**本当に必要**なリーサル: {len(needs)} / {len(positives)}")
    candidate_only = 0
    total_options = Counter()
    for row in positives:
        observation = to_observation_class(row["obs"])
        if observation.select is None:
            continue
        kinds = [OptionType(int(o.type)).name for o in observation.select.option]
        total_options.update(kinds)
        if "ATTACH" in kinds and "ATTACH" not in row["oracle"]["capabilities"]:
            candidate_only += 1
    print(f"  ATTACH が候補に出るが不要だったリーサル: {candidate_only}")
    print(f"  positive 局面の root 選択肢の内訳: "
          f"{dict(total_options.most_common(6))}")
    if needs:
        depths = [r["oracle"]["minimum_depth"] for r in needs]
        print(f"  ATTACH を使う確定リーサルの minimum_depth: "
              f"min={min(depths)} 中央={sorted(depths)[len(depths) // 2]} max={max(depths)}")
        after = Counter()
        for row in needs:
            sequence = row["oracle"]["capabilities"]
            after.update(k for k in sequence if k != "ATTACH")
        print(f"  ATTACH の後に使われた能力: {dict(after.most_common(6))}")


def main() -> None:
    rows = load()
    deck = read_deck_csv()
    section_fixture_summary(rows)
    cache = section_phase_comparison(rows, deck)
    section_capability_matrix(rows, deck, cache)
    section_attach_analysis(rows, deck)
    section_depth_sweep(rows, deck)


if __name__ == "__main__":
    main()
