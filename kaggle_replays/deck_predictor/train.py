#!/usr/bin/env python3
"""dataset.jsonl から多クラスロジスティック回帰(softmax回帰)を学習し、ベース重みJSONを出力する。

「ベースモデル + チューニング層」構成(ml-predictor-phase2-scaling.md 参照)のベース側。
ここで出力する重みJSONはそのまま提出には使わない。事前分布(intercept)の頻繁な補正は
adjust_prior.py の責務なので、このスクリプトは sample_submission/ へのコピーは行わない。

特徴量: feature_names = ["__turn__"] + 学習データに出現したカード名(ソート済み)。
値は __turn__ がターン数、カード名が観測枚数(dataset.jsonl の observed_cards の値そのまま)。

train/valid 分割はエピソード単位(episode_id で 80/20)。同一エピソードの2視点
(player_index 0/1)は必ず同じ側に入る(リーク防止)。

scikit-learn の LogisticRegression が使えればそれを使う。使えない場合は
numpy の勾配降下(L2正則化つき softmax 回帰)を自前実装したフォールバックを使う。

weights JSON の meta には、adjust_prior.py が事前分布シフト補正の基準として使う
class_priors(学習サンプルベースのクラス頻度)と、参考情報として class_deck_counts
(ラベル付きデッキ単位でのクラス頻度、train split に含まれる分のみ)を記録する。

出力:
  - output/model/deck_predictor_weights_base.json (スキーマは ml-predictor-plan.md 準拠)
  - output/model/split.json (evaluate.py / adjust_prior.py が同じ train/valid 分割を再利用するため)

使い方:
  python train.py
  python train.py --dataset ./output/dataset.jsonl --out-dir ./output/model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent

_TURN_FEATURE_NAME = "__turn__"
_SPLIT_SEED = 42
_VALID_FRACTION = 0.2


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_deck_labels(path: Path) -> list[dict]:
    """deck_labels.jsonl を読む(class_deck_counts の参考集計用)。存在しない場合は空リスト。"""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_class_priors(labels: list[str]) -> dict[str, float]:
    """学習サンプル(dataset.jsonl の行)ベースのクラス頻度。adjust_prior.py の pi_c になる。"""
    counts = Counter(labels)
    n = len(labels)
    if n == 0:
        return {}
    return {label: count / n for label, count in sorted(counts.items())}


def compute_class_deck_counts(deck_label_rows: list[dict], train_episode_ids: set[str]) -> dict[str, int]:
    """train split に含まれるエピソードのみを対象に、ラベル付きデッキ単位でのクラス頻度を数える
    (dataset.jsonl の行はエピソード内の複数意思決定時点に重複カウントされるため、こちらは
    「デッキが何本あったか」の参考値として別途記録する)。"""
    counts = Counter(
        row["archetype"] for row in deck_label_rows if row["episode_id"] in train_episode_ids
    )
    return {label: count for label, count in sorted(counts.items())}


def _episode_hash_bucket(episode_id: str, seed: int, modulus: int = 1_000_000) -> int:
    """episode_id を [0, modulus) の決定的なバケットへ写像する。

    他のエピソードの有無に一切依存しない(episode_id + seed だけで決まる)ため、
    データプールにエピソードが追加/削除されても既存エピソードのバケット値は変わらない。
    """
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % modulus


def split_episodes(episode_ids: list[str], valid_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """episode_id ごとに決定的に train/valid を割り当てる。

    旧実装は `random.Random(seed).shuffle(unique_ids)` でエピソードID一覧全体を
    Fisher-Yates シャッフルしていたため、データプールに新規エピソードが1件増減しただけで
    既存エピソードの train/valid 割り当てまで総入れ替えされてしまっていた(shuffle は
    リストの並び全体に依存するため)。ここでは各 episode_id を「episode_id 自身のハッシュ値」
    だけでバケット分けするので、他のエピソードが増えても減っても、あるエピソードが
    train/valid どちらに入るかは変化しない(単調: valid_fraction を大きくしたときに
    train 側だったエピソードが valid 側に移ることはあっても、その逆は起きない)。
    """
    unique_ids = sorted(set(episode_ids))
    modulus = 1_000_000
    threshold = int(round(valid_fraction * modulus))
    valid_ids = {eid for eid in unique_ids if _episode_hash_bucket(eid, seed, modulus) < threshold}
    train_ids = set(unique_ids) - valid_ids

    # 極端に小さいプール(バケット運悪く全員 train または全員 valid)向けの安全弁。
    # 旧実装は "n_valid = max(1, round(...))" で valid が必ず1件以上入るようにしていたので、
    # その保証だけは維持する(実運用の数千エピソード規模では発動しない想定)。
    if unique_ids and not valid_ids:
        forced = min(unique_ids, key=lambda eid: _episode_hash_bucket(eid, seed, modulus))
        valid_ids = {forced}
        train_ids = set(unique_ids) - valid_ids
    elif unique_ids and not train_ids:
        forced = max(unique_ids, key=lambda eid: _episode_hash_bucket(eid, seed, modulus))
        train_ids = {forced}
        valid_ids = set(unique_ids) - train_ids

    return train_ids, valid_ids


def build_feature_names(rows: list[dict]) -> list[str]:
    card_names: set[str] = set()
    for row in rows:
        card_names.update(row["observed_cards"].keys())
    return [_TURN_FEATURE_NAME] + sorted(card_names)


def build_feature_vector(row: dict, feature_index: dict[str, int], n_features: int) -> list[float]:
    x = [0.0] * n_features
    x[feature_index[_TURN_FEATURE_NAME]] = float(row["turn"])
    for name, count in row["observed_cards"].items():
        idx = feature_index.get(name)
        if idx is not None:
            x[idx] = float(count)
    return x


def train_with_sklearn(x_train, y_train, feature_names: list[str]):
    from sklearn.linear_model import LogisticRegression

    # sklearn >= 1.5 では multi_class 引数が廃止され、lbfgs ソルバーは
    # クラス数 > 2 のとき自動的に多項(softmax)ロジスティック回帰になる。
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        C=1.0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    classes = list(model.classes_)
    coef = model.coef_.tolist()
    intercept = model.intercept_.tolist()
    return classes, coef, intercept


def train_with_numpy_fallback(
    x_train, y_train, feature_names: list[str], epochs: int = 500, lr: float = 0.1, l2: float = 1e-3
):
    """scikit-learn が使えない場合の自前 softmax 回帰(勾配降下 + L2正則化)。"""
    import numpy as np

    classes = sorted(set(y_train))
    class_index = {c: i for i, c in enumerate(classes)}
    n_samples = len(x_train)
    n_features = len(feature_names)
    n_classes = len(classes)

    X = np.asarray(x_train, dtype=np.float64)
    # 特徴量のスケールを揃える(ターン数とカード枚数でスケールが違うため学習が安定しやすい)。
    scale = np.maximum(X.std(axis=0), 1e-6)
    X_scaled = X / scale

    Y = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_train):
        Y[i, class_index[label]] = 1.0

    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)

    for epoch in range(epochs):
        Z = X_scaled @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        P = expZ / expZ.sum(axis=1, keepdims=True)
        grad_z = (P - Y) / n_samples
        grad_W = X_scaled.T @ grad_z + l2 * W
        grad_b = grad_z.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
        if (epoch + 1) % 100 == 0:
            loss = -np.sum(Y * np.log(np.clip(P, 1e-12, 1.0))) / n_samples
            print(f"  epoch {epoch + 1}/{epochs}: loss={loss:.4f}", file=sys.stderr)

    # 特徴量スケーリングを重みに畳み込み、推論側では生の特徴量をそのまま使えるようにする。
    W_unscaled = W / scale[:, None]
    coef = W_unscaled.T.tolist()  # [n_classes][n_features]
    intercept = b.tolist()
    return classes, coef, intercept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(_HERE / "output" / "dataset.jsonl"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument("--out-dir", default=str(_HERE / "output" / "model"))
    parser.add_argument("--valid-fraction", type=float, default=_VALID_FRACTION)
    parser.add_argument("--seed", type=int, default=_SPLIT_SEED)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"データセット読み込み中: {dataset_path}")
    rows = load_dataset(dataset_path)
    print(f"  {len(rows)} 件のサンプルを読み込みました")

    episode_ids = [row["episode_id"] for row in rows]
    train_ids, valid_ids = split_episodes(episode_ids, args.valid_fraction, args.seed)
    print(f"  エピソード数: train={len(train_ids)}, valid={len(valid_ids)}")

    split_path = out_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {"seed": args.seed, "valid_fraction": args.valid_fraction,
             "train_episode_ids": sorted(train_ids), "valid_episode_ids": sorted(valid_ids)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    train_rows = [row for row in rows if row["episode_id"] in train_ids]
    valid_rows = [row for row in rows if row["episode_id"] in valid_ids]
    print(f"  サンプル数: train={len(train_rows)}, valid={len(valid_rows)}")

    feature_names = build_feature_names(train_rows)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    n_features = len(feature_names)
    print(f"  特徴量数: {n_features} (__turn__ + カード名 {n_features - 1})")

    x_train = [build_feature_vector(row, feature_index, n_features) for row in train_rows]
    y_train = [row["label"] for row in train_rows]

    print("学習中...")
    try:
        classes, coef, intercept = train_with_sklearn(x_train, y_train, feature_names)
        used_backend = "scikit-learn LogisticRegression (multinomial, L2)"
    except ImportError:
        print("scikit-learn が利用できないため numpy フォールバックで学習します", file=sys.stderr)
        classes, coef, intercept = train_with_numpy_fallback(x_train, y_train, feature_names)
        used_backend = "numpy gradient descent (softmax + L2, fallback)"
    print(f"  学習バックエンド: {used_backend}")
    print(f"  クラス数: {len(classes)} -> {classes}")

    class_priors = compute_class_priors(y_train)
    deck_label_rows = load_deck_labels(Path(args.deck_labels))
    class_deck_counts = compute_class_deck_counts(deck_label_rows, train_ids)
    print(f"  class_priors (学習サンプルベース): {class_priors}")
    print(f"  class_deck_counts (train split・ラベル付きデッキ単位、参考): {class_deck_counts}")

    weights_payload = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_episodes": len(train_ids) + len(valid_ids),
            "n_samples": len(rows),
            "n_train_samples": len(train_rows),
            "n_valid_samples": len(valid_rows),
            "feature_type": "card_name_counts+turn",
            "backend": used_backend,
            "version": 1,
            # adjust_prior.py が事前分布シフト補正 (b'_c = b_c + log(pi'_c / pi_c)) の
            # pi_c として使う、学習サンプルベースのクラス事前分布。
            "class_priors": class_priors,
            # 参考情報: エピソード(ラベル付きデッキ)単位でのクラス頻度。dataset.jsonl の行は
            # 1エピソードにつき複数の意思決定時点が重複カウントされるため、class_priors とは
            # 分布が異なりうる(ターン数が多い対局ほど重みが増える)。
            "class_deck_counts": class_deck_counts,
        },
        "classes": classes,
        "feature_names": feature_names,
        "coef": coef,
        "intercept": intercept,
    }

    weights_path = out_dir / "deck_predictor_weights_base.json"
    weights_path.write_text(json.dumps(weights_payload, ensure_ascii=False), encoding="utf-8")
    print(f"ベース重みを {weights_path} に書き出しました")
    print("  (デプロイ用の重みは adjust_prior.py で事前分布シフト補正をかけてから生成してください)")


if __name__ == "__main__":
    main()
