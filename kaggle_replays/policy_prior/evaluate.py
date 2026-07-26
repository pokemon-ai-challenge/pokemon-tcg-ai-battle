#!/usr/bin/env python3
"""学習済み Policy Prior(``policy_weights.json``)を評価する。

必ず以下を報告する:

- test 一致率(主要な数字)
- **lookup が当たらない decision に限定した一致率**(汎化能力の本体)
- 自明ベースライン(type 事前分布順)
- lookup ベースライン(バックオフ付き)
- context/選択肢数別の内訳

推論には ``ptcg_ai.learning.policy_model.PolicyModel`` をそのまま使う(train.py が
学習時に計算したスコアの再実装をしない。ここで使うのは常に「重みJSONを読み込んで
計算した」本番と同じスコア)。これにより学習/推論のパリティが自動的に保証される
(パリティの明示的なテストは kaggle_replays/tests/test_policy_parity.py 参照)。

使い方:
  python policy_prior/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.learning.semantic_action import action_label  # noqa: E402

_DEFAULT_DATASET = _HERE / "output" / "main_decisions.jsonl"
_DEFAULT_WEIGHTS = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _valid_rows(rows: list[dict]) -> list[dict]:
    """chosen が単一選択で actions 範囲内にある行だけを使う(train.py の build_examples と同条件)。"""
    out = []
    for row in rows:
        chosen = row.get("chosen")
        if chosen and len(chosen) == 1 and 0 <= chosen[0] < len(row["actions"]):
            out.append(row)
    return out


# ============================================================
# 自明ベースライン: type(option_type) 事前分布順に選ぶ
# ============================================================

def build_type_priority(train_rows: list[dict]) -> list[int]:
    """train で「選ばれた側」の option_type 頻度が高い順のリストを返す。"""
    counts: Counter[int] = Counter()
    for row in train_rows:
        chosen_action = row["actions"][row["chosen"][0]]
        counts[chosen_action.get("option_type")] += 1
    return [t for t, _ in counts.most_common()]


def predict_type_priority(actions: list[dict], priority: list[int]) -> int:
    """利用可能な選択肢の中で、優先順位が最も高い option_type を持つものを選ぶ。
    同じ option_type が複数あれば先頭(最初の出現)を選ぶ。優先順位に無い type は最下位扱い。
    """
    rank = {t: i for i, t in enumerate(priority)}
    best_idx = 0
    best_rank = rank.get(actions[0].get("option_type"), len(priority))
    for i in range(1, len(actions)):
        r = rank.get(actions[i].get("option_type"), len(priority))
        if r < best_rank:
            best_rank = r
            best_idx = i
    return best_idx


# ============================================================
# lookup ベースライン(バックオフ付き)
# README / test_plan/ptcg_replay_schema_report.md §6.2 の定義:
#   train の (行動集合, 自分の場ポケモンのcard_id, 自分のサイド枚数, 相手のサイド枚数)
#   -> 最頻ラベル の表を作り、バックオフして引く。
# ============================================================

def _action_set(actions: list[dict]) -> frozenset:
    return frozenset(action_label(a) for a in actions)


def _own_active_card_id(state: dict) -> int | None:
    active = (state.get("own") or {}).get("active")
    return active.get("card_id") if active else None


def _own_opp_prize(state: dict) -> tuple[int | None, int | None]:
    own = state.get("own") or {}
    opp = state.get("opponent") or {}
    return own.get("n_prize"), opp.get("n_prize")


class LookupBaseline:
    """4段バックオフの lookup 方策。

    Level A: (行動集合, own_active_card_id, own_n_prize, opp_n_prize) -> 最頻ラベル
    Level B: (行動集合, own_active_card_id) -> 最頻ラベル
    Level C: (行動集合,) -> 最頻ラベル
    Level D(最終フォールバック): type(option_type)事前分布順(自明ベースライン)

    Level D を欠くと「lookup が一切当たらない decision」が常にハズレ扱いになり、
    lookup ベースラインの一致率が不当に低く出る(要件どおり、必ず4段目まで実装する)。
    """

    def __init__(self, train_rows: list[dict]):
        table_a: dict[tuple, Counter] = defaultdict(Counter)
        table_b: dict[tuple, Counter] = defaultdict(Counter)
        table_c: dict[tuple, Counter] = defaultdict(Counter)

        for row in train_rows:
            actions = row["actions"]
            chosen_label = tuple(row["chosen_label"]) if row.get("chosen_label") else None
            if chosen_label is None:
                continue
            aset = _action_set(actions)
            own_active = _own_active_card_id(row["state"])
            own_prize, opp_prize = _own_opp_prize(row["state"])

            table_a[(aset, own_active, own_prize, opp_prize)][chosen_label] += 1
            table_b[(aset, own_active)][chosen_label] += 1
            table_c[(aset,)][chosen_label] += 1

        self._table_a = {k: v.most_common(1)[0][0] for k, v in table_a.items()}
        self._table_b = {k: v.most_common(1)[0][0] for k, v in table_b.items()}
        self._table_c = {k: v.most_common(1)[0][0] for k, v in table_c.items()}
        self._type_priority = build_type_priority(train_rows)

    def predict(self, row: dict) -> tuple[int, str]:
        """予測した option index と、当たったバックオフ段("A"/"B"/"C"/"miss")を返す。"""
        actions = row["actions"]
        aset = _action_set(actions)
        own_active = _own_active_card_id(row["state"])
        own_prize, opp_prize = _own_opp_prize(row["state"])

        label = self._table_a.get((aset, own_active, own_prize, opp_prize))
        level = "A"
        if label is None:
            label = self._table_b.get((aset, own_active))
            level = "B"
        if label is None:
            label = self._table_c.get((aset,))
            level = "C"
        if label is None:
            level = "miss"

        if label is not None:
            for i, action in enumerate(actions):
                if action_label(action) == label:
                    return i, level

        # 最終フォールバック: lookup が一切当たらない(ラベルすら引けない、
        # または引けたラベルが今回の選択肢集合に存在しない)場合は type 事前分布順。
        return predict_type_priority(actions, self._type_priority), "miss"


# ============================================================
# 一致率の計測
# ============================================================

def model_accuracy(model: PolicyModel, rows: list[dict]) -> tuple[int, int]:
    correct = 0
    for row in rows:
        idx = model.select(row["state"], row["actions"])
        if idx == row["chosen"][0]:
            correct += 1
    return correct, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument(
        "--weights", type=Path, nargs="+", default=[_DEFAULT_WEIGHTS],
        help="評価する重みJSON。複数指定すると同じ test 集合上で比較する"
             "(例: --weights .../policy_weights.json ./output/policy_weights_archetype1.json)",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="--weights に対応する表示名(省略時はファイル名から自動生成)",
    )
    parser.add_argument(
        "--archetype", type=str, default=None,
        help="指定したアーキタイプ(archetypes.py / build_dataset.py の archetype フィールド)"
             "の行だけに絞って評価する(train.py --archetype と同じ絞り込み)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.weights):
        raise SystemExit("--labels の個数は --weights の個数と揃えてください")
    labels = args.labels or [w.stem for w in args.weights]

    print(f"データセット読み込み中: {args.dataset}")
    rows = load_dataset(args.dataset)
    print(f"  decision 総数: {len(rows):,}")
    if args.archetype:
        before = len(rows)
        rows = [r for r in rows if r.get("archetype") == args.archetype]
        print(f"  --archetype={args.archetype} で絞り込み: {before:,} -> {len(rows):,} 行")

    train_rows = _valid_rows([r for r in rows if r["split"] == "train"])
    val_rows = _valid_rows([r for r in rows if r["split"] == "val"])
    test_rows = _valid_rows([r for r in rows if r["split"] == "test"])
    scope = f"(アーキタイプ={args.archetype})" if args.archetype else "(全アーキタイプ)"
    print(f"  train={len(train_rows):,} val={len(val_rows):,} test={len(test_rows):,} {scope}")
    n = len(test_rows)
    if n == 0:
        raise SystemExit("test 行が0件のため評価できません(--archetype の指定を確認してください)")

    # --- ベースライン(比較対象の全モデルに共通。train_rows から一度だけ計算) --------------
    print()
    print("=" * 70)
    print(f"[baseline] test 集合の件数: {n:,} 件 {scope}")

    type_priority = build_type_priority(train_rows)
    naive_correct = sum(
        1 for row in test_rows
        if predict_type_priority(row["actions"], type_priority) == row["chosen"][0]
    )
    naive_acc = naive_correct / n
    print(f"[baseline] 自明ベースライン(type事前分布順)test 一致率: {naive_acc:.4f} "
          f"({naive_correct:,}/{n:,})  [全体参考値: README記載 32.2%]")

    # lookup ベースラインは4段バックオフ(Level A/B/C/D)。Level D は type 事前分布順
    # (自明ベースライン)で、これが無いと lookup が一切当たらない decision の予測が
    # 常にハズレ扱いになり、lookup ベースラインの数字が不当に低くなる。
    lookup = LookupBaseline(train_rows)
    lookup_correct = 0
    lookup_levels: Counter[str] = Counter()
    lookup_hit_correct = 0
    lookup_hit_total = 0
    miss_test_rows: list[dict] = []
    for row in test_rows:
        pred_idx, level = lookup.predict(row)
        lookup_levels[level] += 1
        is_correct = pred_idx == row["chosen"][0]
        if is_correct:
            lookup_correct += 1
        if level == "miss":
            miss_test_rows.append(row)
        else:
            lookup_hit_total += 1
            if is_correct:
                lookup_hit_correct += 1
    lookup_acc = lookup_correct / n
    print(f"[baseline] lookup ベースライン(4段バックオフ A/B/C/D)test 一致率: {lookup_acc:.4f} "
          f"({lookup_correct:,}/{n:,})  [全体参考値: README記載 41.8%]")
    print(f"    バックオフ段の内訳: {dict(lookup_levels)}")
    if lookup_hit_total:
        print(f"    (参考) lookup がヒットした decision に限った lookup 自身の一致率: "
              f"{lookup_hit_correct / lookup_hit_total:.4f} ({lookup_hit_total:,} decisions)")
    print(f"    ★lookup非ヒット(miss) decision: {len(miss_test_rows):,} 件"
          f" (test全体の {len(miss_test_rows) / n:.1%})。汎化能力を見るならここに限定した一致率を見る。")

    naive_miss_acc = None
    if miss_test_rows:
        miss_naive_correct = sum(
            1 for row in miss_test_rows
            if predict_type_priority(row["actions"], type_priority) == row["chosen"][0]
        )
        naive_miss_acc = miss_naive_correct / len(miss_test_rows)
        print(f"    (参考) 自明ベースラインの一致率(miss限定): "
              f"{naive_miss_acc:.4f} ({miss_naive_correct:,}/{len(miss_test_rows):,})")

    # --- 各モデルを同じ test 集合(と miss 部分集合)で評価 ---------------------------
    results: list[dict] = []
    for weights_path, label in zip(args.weights, labels):
        print()
        print("-" * 70)
        print(f"モデル [{label}]  重み: {weights_path}")
        model = PolicyModel(weights_path=weights_path)
        if not model.is_ready:
            print(f"  ! 重みJSONの読み込みに失敗しました: {weights_path} (このモデルは比較から除外)")
            continue

        correct, n_test = model_accuracy(model, test_rows)
        test_acc = correct / n_test if n_test else 0.0
        print(f"  test 一致率: {test_acc:.4f} ({correct:,}/{n_test:,})")

        miss_acc = miss_n = miss_correct = None
        if miss_test_rows:
            miss_correct, miss_n = model_accuracy(model, miss_test_rows)
            miss_acc = miss_correct / miss_n
            print(f"  ★lookup非ヒット(miss)限定 一致率: {miss_acc:.4f} ({miss_correct:,}/{miss_n:,})")

        if len(args.weights) == 1:
            # 単一モデルのときは従来どおり選択肢数/contextの内訳も出す。
            print()
            print("  [選択肢数(n_options)別の内訳(test)]")
            by_n: dict[int, list[dict]] = defaultdict(list)
            for row in test_rows:
                by_n[row["n_options"]].append(row)
            for n_opt in sorted(by_n):
                sub = by_n[n_opt]
                c, t = model_accuracy(model, sub)
                print(f"    n_options={n_opt:>2}: {c:>4}/{t:<4} = {c / t:.3f}  (test中 {t:,} 件)")

            print("  [select_context 別の内訳(test)]")
            by_ctx: dict[int, list[dict]] = defaultdict(list)
            for row in test_rows:
                by_ctx[row.get("select_context")].append(row)
            for ctx in sorted(by_ctx, key=lambda k: (k is None, k)):
                sub = by_ctx[ctx]
                c, t = model_accuracy(model, sub)
                print(f"    select_context={ctx}: {c:>4}/{t:<4} = {c / t:.3f}")

        results.append({
            "label": label, "weights": str(weights_path),
            "test_acc": test_acc, "n_test": n_test,
            "miss_acc": miss_acc, "miss_n": miss_n,
        })

    # --- 比較サマリ(汎用モデル vs アーキタイプ専用モデル等) -----------------------------
    print()
    print("=" * 70)
    print("サマリ")
    print(f"  test 集合件数           : {n:,} {scope}")
    print(f"  自明ベースライン(test)  : {naive_acc:.4f}")
    print(f"  lookupベースライン(test): {lookup_acc:.4f}  (miss {len(miss_test_rows):,}件"
          f" / {len(miss_test_rows) / n:.1%})")
    if naive_miss_acc is not None:
        print(f"  自明ベースライン(miss限定): {naive_miss_acc:.4f} (n={len(miss_test_rows):,})")
    print(f"  {'モデル':20s} {'test一致率':>12s} {'test件数':>8s} {'miss限定一致率':>14s} {'miss件数':>8s}")
    for r in sorted(results, key=lambda r: -r["test_acc"]):
        miss_str = f"{r['miss_acc']:.4f}" if r["miss_acc"] is not None else "-"
        miss_n_str = f"{r['miss_n']:,}" if r["miss_n"] is not None else "-"
        print(f"  {r['label']:20s} {r['test_acc']:>12.4f} {r['n_test']:>8,} {miss_str:>14s} {miss_n_str:>8s}")
        beat_lookup = r["test_acc"] > lookup_acc
        print(f"    ({'lookupベースラインを超えた' if beat_lookup else 'lookupベースラインを超えられなかった'})")

    if len(results) >= 2:
        best = max(results, key=lambda r: r["test_acc"])
        print(f"  -> test一致率が最も高いモデル: [{best['label']}] {best['test_acc']:.4f}")


if __name__ == "__main__":
    main()
