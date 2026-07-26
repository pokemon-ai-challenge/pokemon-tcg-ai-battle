#!/usr/bin/env python3
"""``main_decisions.jsonl`` から Policy Prior（pointwise 線形スコアリング）を学習する。

1 decision(状態 + 選択肢集合 + 選ばれた1つ)を「選択肢数ぶんの事例」に展開する
pointwise ranking として学習する。選ばれた選択肢を正例(y=1)、他の選択肢を負例(y=0)とし、
二値ロジスティック回帰で分離平面を学習する。推論(decision内のargmax)は
``policy_model.PolicyModel.select()`` と同じロジックをここでも使い、学習時に測った
一致率が本番と一致することを保証する。

特徴抽出は ``ptcg_ai.learning.policy_features``（唯一の実装）を import するだけで、
ここでは一切再実装しない。学習側の責務は「特徴 dict のリスト -> 重みJSON」だけ。

出力: ``sample_submission/ptcg_ai/learning/policy_weights.json``
(``ptcg_ai.learning.policy_model.PolicyModel`` が読む契約どおりのスキーマ)

使い方:
  python policy_prior/train.py
  python policy_prior/train.py --max-frequent-cards 200 --C 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.learning import policy_features  # noqa: E402

sys.path.insert(0, str(_HERE))
from build_card_attributes import build_attributes  # noqa: E402

_DEFAULT_DATASET = _HERE / "output" / "main_decisions.jsonl"
_DEFAULT_CARD_CSV = _REPO_ROOT / "data" / "EN_Card_Data.csv"
_DEFAULT_WEIGHTS_OUT = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"

# --- 頻出カード上限Nの既定値 -------------------------------------------------
# card_id one-hot は次元過多を避けるため上位N種のみに絞る(policy_features.py の設計)。
# 既定値はこのスクリプトが毎回 train split で実測するカバレッジ曲線から選ぶ
# (下の _log_frequent_card_rationale を参照)。決め打ちにしないのは、リプレイが
# 追加されて出現カードの分布が変わっても妥当な値を保てるようにするため。
_DEFAULT_MAX_FREQUENT_CARDS = 150


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _log_frequent_card_rationale(train_rows: list[dict], chosen_n: int) -> None:
    """train split における card_id 出現頻度のカバレッジ曲線をログに出す。

    ``card_id`` / ``target_card_id`` は state の card_attributes 経由でも渡るが、
    one-hot(``frequent_card_ids``)は「このカードそのものが選ばれやすいか」という
    カード種固有のバイアスを線形項として拾うためのもの。次元を無限に増やせないので
    上位N種に絞る、その N の妥当性をここで可視化する。
    """
    counts: Counter[int] = Counter()
    for row in train_rows:
        for action in row["actions"]:
            for key in ("card_id", "target_card_id"):
                cid = action.get(key)
                if cid is not None:
                    counts[cid] += 1

    total = sum(counts.values())
    ranked = counts.most_common()
    print(f"  train中の card_id/target_card_id 出現: 延べ {total:,} 件 / 種類数 {len(ranked):,}")
    for n in (50, 100, 150, 200, 300, 500):
        covered = sum(c for _, c in ranked[:n])
        pct = covered / total if total else 0.0
        marker = " <- 既定値" if n == _DEFAULT_MAX_FREQUENT_CARDS else ""
        print(f"    上位{n:>4}種でカバレッジ {pct:.1%}{marker}")
    print(f"  既定値 N={_DEFAULT_MAX_FREQUENT_CARDS}: 出現頻度の裾が長く(上位150種超でも"
          f"1%未満/50種の緩やかな伸びしか無い)、150種は主要カードをほぼ押さえつつ"
          f"次元を抑えるバランス点として選んだ。--max-frequent-cards で変更可能。")
    print(f"  今回使用する N = {chosen_n}")


def build_examples(
    rows: list[dict],
    card_attributes: dict[str, dict[str, float]],
    frequent_card_ids: list[int],
) -> tuple[list[dict[str, float]], list[int], list[int]]:
    """decision の集合から (特徴dictのリスト, ラベルのリスト, group_idのリスト) を作る。

    group_id は同じ decision に属する選択肢をまとめてargmax評価するためのキー
    (連番。decision の出現順)。
    """
    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []
    group_ids: list[int] = []

    group_id = 0
    for row in rows:
        chosen = row.get("chosen")
        if not chosen or len(chosen) != 1:
            continue
        chosen_index = chosen[0]
        actions = row["actions"]
        if not (0 <= chosen_index < len(actions)):
            continue
        state = row["state"]
        for i, action in enumerate(actions):
            feats = policy_features.extract_features(state, action, card_attributes, frequent_card_ids)
            feature_dicts.append(feats)
            labels.append(1 if i == chosen_index else 0)
            group_ids.append(group_id)
        group_id += 1

    return feature_dicts, labels, group_ids


def decision_level_accuracy(scores, labels: list[int], group_ids: list[int]) -> tuple[float, int]:
    """group_id ごとに argmax(score) を取り、正解ラベル(y=1)の選択肢と一致するかを見る。

    ``policy_model.PolicyModel.select()`` と同じ「同点なら先勝ち(最初に見つかった最大値)」規則。
    """
    groups: dict[int, list[tuple[float, int]]] = {}
    for score, label, gid in zip(scores, labels, group_ids):
        groups.setdefault(gid, []).append((score, label))

    correct = 0
    for items in groups.values():
        best_idx = 0
        best_score = items[0][0]
        for i in range(1, len(items)):
            if items[i][0] > best_score:
                best_score = items[i][0]
                best_idx = i
        if items[best_idx][1] == 1:
            correct += 1
    n = len(groups)
    return (correct / n if n else 0.0), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--card-csv", type=Path, default=_DEFAULT_CARD_CSV)
    parser.add_argument("--out", type=Path, default=_DEFAULT_WEIGHTS_OUT)
    parser.add_argument("--max-frequent-cards", type=int, default=_DEFAULT_MAX_FREQUENT_CARDS)
    parser.add_argument("--C", type=float, default=1.0, help="L2正則化の逆数(sklearn LogisticRegression)")
    parser.add_argument("--balanced", action="store_true", default=True,
                         help="class_weight='balanced' を使う(既定: 有効)")
    parser.add_argument("--no-balanced", dest="balanced", action="store_false")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    print(f"データセット読み込み中: {args.dataset}")
    rows = load_dataset(args.dataset)
    print(f"  decision 総数: {len(rows):,}")

    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    print(f"  split: train={len(train_rows):,} val={len(val_rows):,} test={len(test_rows):,}"
          f" (既存の split フィールドをそのまま使用。再分割はしない)")

    print(f"カード静的属性を構築中: {args.card_csv}")
    card_attributes = build_attributes(args.card_csv)
    print(f"  カード {len(card_attributes):,} 種")

    _log_frequent_card_rationale(train_rows, args.max_frequent_cards)
    counts: Counter[int] = Counter()
    for row in train_rows:
        for action in row["actions"]:
            for key in ("card_id", "target_card_id"):
                cid = action.get(key)
                if cid is not None:
                    counts[cid] += 1
    frequent_card_ids = [cid for cid, _ in counts.most_common(args.max_frequent_cards)]

    print("特徴抽出中(train)...")
    X_train_dicts, y_train, group_train = build_examples(train_rows, card_attributes, frequent_card_ids)
    print(f"  展開後の事例数(train, 選択肢単位): {len(X_train_dicts):,} "
          f"(decision数 {len(set(group_train)):,})")

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    feature_names = list(vectorizer.feature_names_)
    print(f"  特徴次元数: {len(feature_names):,}")

    class_weight = "balanced" if args.balanced else None
    print(f"学習中... (LogisticRegression, C={args.C}, class_weight={class_weight}, "
          f"max_iter={args.max_iter})")
    model = LogisticRegression(
        C=args.C, class_weight=class_weight, max_iter=args.max_iter, solver="lbfgs"
    )
    model.fit(X_train, y_train)

    weights = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])

    # --- 学習時スコアでの decision-level 一致率(train / val / test) ---
    train_scores = model.decision_function(X_train)
    train_acc, n_train_decisions = decision_level_accuracy(train_scores, y_train, group_train)
    print(f"  train 一致率(decision単位): {train_acc:.4f} ({n_train_decisions:,} decisions)")

    def eval_split(split_rows: list[dict], name: str) -> float:
        feats, labels, groups = build_examples(split_rows, card_attributes, frequent_card_ids)
        if not feats:
            return 0.0
        X = vectorizer.transform(feats)
        scores = model.decision_function(X)
        acc, n = decision_level_accuracy(scores, labels, groups)
        print(f"  {name} 一致率(decision単位, 学習時スコア): {acc:.4f} ({n:,} decisions)")
        return acc

    val_acc = eval_split(val_rows, "val")
    test_acc = eval_split(test_rows, "test")

    payload = {
        "schema_version": 1,
        "model": "linear_pointwise",
        "feature_names": feature_names,
        "weights": weights,
        "intercept": intercept,
        "frequent_card_ids": frequent_card_ids,
        "card_attributes": card_attributes,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_rows": len(X_train_dicts),
            "train_decisions": n_train_decisions,
            "val_decisions": len(val_rows),
            "test_decisions": len(test_rows),
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
            "test_accuracy": round(test_acc, 6),
            "max_frequent_cards": args.max_frequent_cards,
            "C": args.C,
            "class_weight": class_weight,
            "backend": "scikit-learn LogisticRegression",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"重みJSONを書き出しました: {args.out}")
    print("  (詳細な評価・ベースライン比較は policy_prior/evaluate.py を実行してください)")


if __name__ == "__main__":
    main()
