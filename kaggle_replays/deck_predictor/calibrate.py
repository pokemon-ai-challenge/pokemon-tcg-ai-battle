#!/usr/bin/env python3
"""adjust_prior.py が出力した重み(事前分布シフト補正済み)に、観測エビデンス数
(observed_cards のうちユニークなカード名の個数)のバケットごとの温度スケーリングによる
キャリブレーション補正を追加する「ベースモデル + チューニング層」構成の追加チューニング層。

  probs = softmax(z / T)

  z は softmax 直前の生ロジット(coef @ x + intercept、事前分布シフト補正込み)。
  T はバケットごとに、validation データ上でこのバケットに属するサンプルの多クラス log loss
  (negative log likelihood) を最小化するように scipy.optimize.minimize_scalar
  (bounds=(1.0, 15.0), method="bounded") で1次元最適化して求める。

  **T は 1.0 を下回らない(慎重側に倒す設計判断)**: T<1.0 はモデルの生ロジットより
  「シャープ化」する(確信度を強める)方向の補正であり、「エビデンスが薄いときに
  断定しすぎない」というキャリブレーション導入の目的そのものに反する。実際、
  evidence_count=1 バケットでは無制約最適化だと T*≈0.78(シャープ化)になり、
  検証したところ「Snorunt 1枚しか見えていない曖昧な場面」のような誤答の確信度が
  63.9%→71.8% に悪化する具体例が見つかった(evidence_count はカードの判別力を
  考慮しないユニーク枚数カウントに過ぎず、少数の高判別力アンカーカード
  (例: Abra→ほぼ確定でalakazam)が同じバケット内の多くのサンプルを占めるため、
  バケット全体の集計 log loss 最小化では曖昧なサンプルの方が「割を食う」ことがある)。
  そのため探索範囲の下限を post-hoc なクランプではなく `minimize_scalar` の
  `bounds` 自体に組み込み、無制約最適解が 1.0 未満のバケットは境界の T=1.0
  (実質的に無補正)に張り付くようにしている。

なぜ必要か: 観測カードが少ない(エビデンスが薄い)序盤ほど、線形分類器の生ロジットは
過信(overconfidence)しやすく、確率がいきなり90%以上に張り付くことがある。系統判定
(argmax)自体は合っていても、根拠が薄いのに断定しすぎているのは問題。同一サンプル内の
全ロジットを同じ正の T で割るのは単調変換なので、argmax(top-1予測クラス)は T を変えても
絶対に変わらない。変わるのは確率の「強さ」(reliability)だけ。

coef / intercept は一切変更しない。ランタイム(ml_predictor.py)は meta.calibration.buckets を
読み込み、observed_cards の evidence_count に応じた T を動的に適用する
(重みJSONのスキーマは同一で、meta に calibration が追記されるだけ)。

**fit と評価を同じ validation split で行う点について**: 通常の機械学習では fit 用と評価用を
分けるべきだが、本キャリブレーションはバケットあたり温度1パラメータのみの極めて単純なモデル
(自由度がバケット数程度しかない)であり、過学習のリスクは実質的に無視できるため許容する。
train 側は一切使わない(train/valid 分割は split.json のものをそのまま維持する)。

推論はランタイムと同じ sample_submission/ptcg_ai/opponent_modeling/ml_predictor.py の
MLDeckPredictor._raw_logits() / .evidence_count() を再利用する(学習パイプラインとランタイムで
ロジット計算・エビデンス数の定義がズレないようにするため)。

自己検証: 最も薄いエビデンスのバケット(evidence_count=0 または 1)について、実際の
validation データから observed_cards の例をサンプリングし、キャリブレーション前後の
predict() の出力(top1クラスと確率)を並べて表示する。argmax は同じで確率だけ下がって
いることを目視確認できる。

詳細方針は
sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-plan.md
の「deck_predictor_weights.json」節(meta.calibration)を参照。

使い方:
  python calibrate.py
  python calibrate.py --min-bucket-samples 300 --deploy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scipy.optimize import minimize_scalar

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor, _softmax  # noqa: E402

_DEPLOY_TARGET = (
    _REPO_ROOT / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "deck_predictor_weights.json"
)
_DEFAULT_MIN_BUCKET_SAMPLES = 300
# バケット設計の出発点(min_evidence, max_evidence)。max_evidence=None は上限なし。
# 実際の validation 分布を集計したうえで、--min-bucket-samples 未満のバケットは
# merge_small_buckets() で隣接バケットと統合する(恣意的に決めない)。
_DEFAULT_BUCKET_CANDIDATES: list[tuple[int, int | None]] = [(0, 0), (1, 1), (2, 3), (4, None)]
# T=1.0 未満(モデルの生の自信よりシャープ化する方向)は絶対に許さない、慎重側に倒すポリシー。
# 緩和(T>1、確信度を弱める)方向のみを探索範囲とする。
_T_BOUNDS = (1.0, 15.0)
_SELF_CHECK_N_EXAMPLES = 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def in_bucket(evidence_count: int, min_evidence: int, max_evidence: int | None) -> bool:
    if evidence_count < min_evidence:
        return False
    if max_evidence is not None and evidence_count > max_evidence:
        return False
    return True


def merge_small_buckets(
    counts_by_evidence: Counter, candidates: list[tuple[int, int | None]], min_samples: int
) -> list[tuple[int, int | None]]:
    """candidates ((min_evidence, max_evidence) の min_evidence 昇順・連続・網羅的なリスト) のうち、
    サンプル数が min_samples 未満のバケットを隣接バケットとマージする。
    マージ後も昇順・連続・網羅的が維持される。"""

    def bucket_count(b: tuple[int, int | None]) -> int:
        lo, hi = b
        return sum(c for ev, c in counts_by_evidence.items() if in_bucket(ev, lo, hi))

    buckets = list(candidates)
    changed = True
    while changed and len(buckets) > 1:
        changed = False
        for i, b in enumerate(buckets):
            if bucket_count(b) >= min_samples:
                continue
            if i < len(buckets) - 1:
                merged = (b[0], buckets[i + 1][1])
                buckets = buckets[:i] + [merged] + buckets[i + 2 :]
            else:
                merged = (buckets[i - 1][0], b[1])
                buckets = buckets[: i - 1] + [merged]
            changed = True
            break
    return buckets


def neg_log_likelihood(temperature: float, logits: list[list[float]], y_indices: list[int]) -> float:
    total = 0.0
    for z, y in zip(logits, y_indices):
        z_t = [v / temperature for v in z]
        probs = _softmax(z_t)
        total += -math.log(max(probs[y], 1e-12))
    return total / len(logits)


def top1_accuracy(logits: list[list[float]], y_indices: list[int]) -> float:
    # argmax は T に依存しない(正の単調スケーリングのため)ので T=1 相当の生ロジットで判定してよい。
    correct = sum(1 for z, y in zip(logits, y_indices) if max(range(len(z)), key=lambda i: z[i]) == y)
    return correct / len(logits) if logits else 0.0


def fit_bucket_temperature(logits: list[list[float]], y_indices: list[int]) -> tuple[float, float, float]:
    """(T*, logloss_before(T=1), logloss_after(T=T*)) を返す。"""
    logloss_before = neg_log_likelihood(1.0, logits, y_indices)
    result = minimize_scalar(
        lambda t: neg_log_likelihood(t, logits, y_indices), bounds=_T_BOUNDS, method="bounded"
    )
    t_star = float(result.x)
    logloss_after = neg_log_likelihood(t_star, logits, y_indices)
    # T=1.0 を初期比較値として、最適化後に改善しているか確認する。改善していなければ T=1.0 にフォールバック。
    if logloss_after > logloss_before:
        print(
            f"    警告: 最適化後の logloss({logloss_after:.4f}) が T=1.0 の logloss({logloss_before:.4f}) "
            "より悪化したため、このバケットは T=1.0 にフォールバックします",
            file=sys.stderr,
        )
        t_star = 1.0
        logloss_after = logloss_before
    return t_star, logloss_before, logloss_after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(_HERE / "output" / "model" / "deck_predictor_weights.json"))
    parser.add_argument("--dataset", default=str(_HERE / "output" / "dataset.jsonl"))
    parser.add_argument("--split", default=str(_HERE / "output" / "model" / "split.json"))
    parser.add_argument("--out", default=None, help="省略時は --weights と同じパス(上書き)")
    parser.add_argument("--min-bucket-samples", type=int, default=_DEFAULT_MIN_BUCKET_SAMPLES)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="sample_submission/ptcg_ai/opponent_modeling/deck_predictor_weights.json へコピーする",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    out_path = Path(args.out) if args.out is not None else weights_path

    if not weights_path.exists():
        print(
            f"エラー: weights が見つかりません: {weights_path} (先に train.py / adjust_prior.py を実行してください)",
            file=sys.stderr,
        )
        sys.exit(1)

    predictor = MLDeckPredictor(weights_path=str(weights_path))
    if not predictor.is_ready:
        print(f"エラー: 重みファイル {weights_path} を読み込めませんでした", file=sys.stderr)
        sys.exit(1)

    classes: list[str] = list(json.loads(weights_path.read_text(encoding="utf-8"))["classes"])
    class_index = {c: i for i, c in enumerate(classes)}
    print(f"重み: {weights_path}")
    print(f"  クラス数: {len(classes)} -> {classes}")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    valid_ids = set(split["valid_episode_ids"])
    print(f"  validation エピソード数(split.json): {len(valid_ids)}")

    rows = load_jsonl(Path(args.dataset))
    valid_rows = [row for row in rows if row["episode_id"] in valid_ids]
    print(f"  dataset 全体行数: {len(rows)} / validation 側行数: {len(valid_rows)}(train 側は使わない)")

    # --- validation サンプルの z, evidence_count, y を集計 ---
    sample_logits: list[list[float]] = []
    sample_evidence: list[int] = []
    sample_y: list[int] = []
    sample_observed: list[dict] = []
    sample_turn: list[int] = []
    n_skipped_unknown_label = 0

    for row in valid_rows:
        label = row["label"]
        y = class_index.get(label)
        if y is None:
            n_skipped_unknown_label += 1
            continue
        observed_cards = row["observed_cards"]
        turn = row["turn"]
        z = predictor._raw_logits(observed_cards, turn)
        ec = predictor.evidence_count(observed_cards)
        sample_logits.append(z)
        sample_evidence.append(ec)
        sample_y.append(y)
        sample_observed.append(observed_cards)
        sample_turn.append(turn)

    if n_skipped_unknown_label:
        print(
            f"  警告: label が classes に無い validation サンプルを {n_skipped_unknown_label} 件スキップしました",
            file=sys.stderr,
        )
    print(f"  キャリブレーション対象サンプル数: {len(sample_logits)}")

    # --- evidence_count の分布を集計・表示 ---
    counts_by_evidence: Counter = Counter(sample_evidence)
    print("\nevidence_count 分布(validation):")
    for ec in sorted(counts_by_evidence):
        print(f"  evidence_count={ec:3d}: {counts_by_evidence[ec]:6d} 件")

    # --- バケット境界の確定(閾値未満は隣接バケットとマージ) ---
    buckets_range = merge_small_buckets(counts_by_evidence, _DEFAULT_BUCKET_CANDIDATES, args.min_bucket_samples)
    print(f"\nバケット候補: {_DEFAULT_BUCKET_CANDIDATES}")
    print(f"--min-bucket-samples={args.min_bucket_samples} 未満のバケットを隣接マージした結果:")
    for lo, hi in buckets_range:
        n = sum(c for ev, c in counts_by_evidence.items() if in_bucket(ev, lo, hi))
        hi_str = "∞" if hi is None else str(hi)
        print(f"  [{lo}, {hi_str}]: {n} 件")

    # --- バケットごとに T* を最適化 ---
    bucket_meta: list[dict] = []
    bucket_indices: list[list[int]] = []  # 自己検証用にサンプルの元インデックスを保持
    print("\nバケットごとの温度最適化:")
    for lo, hi in buckets_range:
        indices = [i for i, ec in enumerate(sample_evidence) if in_bucket(ec, lo, hi)]
        bucket_logits = [sample_logits[i] for i in indices]
        bucket_y = [sample_y[i] for i in indices]

        t_star, logloss_before, logloss_after = fit_bucket_temperature(bucket_logits, bucket_y)
        acc = top1_accuracy(bucket_logits, bucket_y)

        hi_str = "∞" if hi is None else str(hi)
        print(
            f"  [{lo}, {hi_str}] n={len(indices)} T*={t_star:.3f} "
            f"logloss: {logloss_before:.4f} -> {logloss_after:.4f} top1_acc={acc * 100:.2f}%"
        )

        bucket_meta.append(
            {
                "min_evidence": lo,
                "max_evidence": hi,
                "temperature": t_star,
                "n_valid_samples": len(indices),
                "logloss_before": logloss_before,
                "logloss_after": logloss_after,
                "top1_acc": acc,
            }
        )
        bucket_indices.append(indices)

    # --- 重みJSONへの書き出し ---
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    meta = dict(payload.get("meta", {}))
    meta["calibration"] = {
        "method": "evidence_count_bucketed_temperature_scaling",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fitted_on": "validation split (output/model/split.json)",
        "buckets": bucket_meta,
    }
    payload["meta"] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nキャリブレーション済み重みを {out_path} に書き出しました")

    if args.deploy:
        _DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        _DEPLOY_TARGET.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"デプロイ用に {_DEPLOY_TARGET} にもコピーしました")

    # --- 自己検証: 最も薄いエビデンスのバケットで、キャリブレーション前後の predict() を比較 ---
    # `predictor` は書き出し前にロード済み(meta.calibration が無い状態 = T=1相当)なのでそのまま
    # 「補正前」として使う。--out が --weights と同じパスの場合、ここで weights_path を
    # 読み直すとキャリブレーション追記後の内容を読んでしまうため、ディスクから再ロードしない。
    predictor_before = predictor
    predictor_after = MLDeckPredictor(weights_path=str(out_path))  # calibration 追記後

    thinnest_bucket_i = 0  # buckets_range は min_evidence 昇順なので先頭が最も薄い
    thinnest_indices = bucket_indices[thinnest_bucket_i]
    lo, hi = buckets_range[thinnest_bucket_i]
    hi_str = "∞" if hi is None else str(hi)
    print(f"\n自己検証: 最も薄いエビデンスのバケット [{lo}, {hi_str}] のサンプル例(キャリブレーション前後比較)")

    rng = random.Random(42)
    sample_pick = rng.sample(thinnest_indices, min(_SELF_CHECK_N_EXAMPLES, len(thinnest_indices)))
    for i in sample_pick:
        observed_cards = sample_observed[i]
        turn = sample_turn[i]
        true_label = classes[sample_y[i]]

        probs_before = predictor_before.predict(observed_cards, turn)
        probs_after = predictor_after.predict(observed_cards, turn)
        top_before = max(probs_before.items(), key=lambda kv: kv[1])
        top_after = max(probs_after.items(), key=lambda kv: kv[1])

        print(f"  observed_cards={observed_cards} turn={turn} true_label={true_label}")
        print(f"    補正前: top1={top_before[0]} prob={top_before[1]:.4f}")
        print(f"    補正後: top1={top_after[0]} prob={top_after[1]:.4f}")
        if top_before[0] != top_after[0]:
            print(
                "    エラー: argmax がキャリブレーション前後で変化しています(実装バグの可能性)",
                file=sys.stderr,
            )
        elif top_after[1] > top_before[1]:
            print(
                "    警告: このバケットでは確率が下がるはずですが、上がっています(T > 1 か確認)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
