#!/usr/bin/env python3
"""LR(``MLDeckPredictor``, adjusted)と NB(``NBDeckPredictor``)の予測分布を、evidence
(観測エビデンス数)バケットごとに **log-space の幾何ブレンド**で混ぜる重み ``weight_nb``
(NB 側の重み w、0.0〜1.0)を validation データ上でフィットし、
``sample_submission/ptcg_ai/opponent_modeling/hybrid_predictor.HybridDeckPredictor`` が読み込む
``deck_predictor_hybrid.json`` を出力する。

背景: ``compare_nb.py`` の比較(``output/nb_compare_report.md``)で、evidence 0(何も見えていない
序盤)は NB が LR より大幅に log loss で優れる一方、evidence 7 以上では NB の「観測カードごとの
尤度が独立」という仮定が崩れて大幅に劣化することが分かった。1バケット1パラメータ(w)の
極小さなモデルなので、evidence バケットごとに最適な w を選ぶのが本スクリプトの役割。
詳細方針は
``sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md``
を参照。

## ブレンド式

    log p_blend(c) = (1 - w) * log p_LR(c) + w * log p_NB(c)

を log-sum-exp で正規化して確率分布に戻す。w=0 は LR 単体、w=1 は NB 単体と一致する。

## フィット対象データ

``compare_nb.py`` と全く同じ「固定 validation」ウィンドウ(既定: 直近14日 x 相手上位200位以内、
``episode_window.py`` 経由)の validation サンプル(``output/model/split.json`` の
``valid_episode_ids``)を使う。evidence バケットの定義(0 / 1 / 2-3 / 4-6 / 7-10 / 11+)・
evidence 数の算出(``MLDeckPredictor.evidence_count()``)も ``compare_nb.py`` と共通(同モジュールから
import して二重実装を避ける)。

## 推論のキャッシュ

LR・NB それぞれの予測確率分布は各 validation サンプルにつき1回だけ計算してキャッシュする
(w のグリッド探索(21点)のたびに再推論すると純Pythonの softmax 回帰/ナイーブベイズ推論を
21倍繰り返すことになり無駄なため)。ブレンドは log 値の線形結合 + log-sum-exp のみなので、
キャッシュ済みの log p_LR / log p_NB からグリッド全体を高速に評価できる。

## 安定性チェック

validation サンプルをエピソード単位で(サンプル単位ではなく)50/50 に分割し(同一エピソードの
サンプルが train/valid のように split されるとリーク的になるため)、片方の半分でフィットした
バケット別 w を、もう片方の半分の log loss で評価する。全データでフィットした w と比べて
選択された w が大きくズレるバケットがあれば警告する。

## 採用方針(--baseline、慎重側の原則)

``--baseline`` に現行デプロイ済みの ``deck_predictor_hybrid.json`` を渡すと、50/50 安定性
チェックを**通過したバケットだけ**新しいフィット値 w を採用し、不安定なバケットは baseline
の w をそのまま維持する(自動補正)。さらに evidence>=4 のバケット(``4-6`` / ``7-10`` /
``11+``)は、安定化しているだけでは不十分で、baseline の w で評価した log loss より
``--min-logloss-improvement`` 以上明確に改善していない限り baseline の w を維持する
(確信度は強気にしない原則。NBの独立性仮定が evidence が多いほど崩れやすいため)。
``--baseline`` を省略した場合は従来どおり全バケットでフィット値をそのまま採用する
(後方互換)。どのバケットが「採用」/「据え置き」だったかは標準出力と出力JSONの
``meta.adoption`` に記録する。

使い方:
  python fit_hybrid.py
  python fit_hybrid.py --baseline ../../sample_submission/ptcg_ai/opponent_modeling/deck_predictor_hybrid.json
  python fit_hybrid.py --deploy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.nb_predictor import NBDeckPredictor  # noqa: E402

from compare_nb import _EVIDENCE_BUCKET_ORDER, evidence_bucket  # noqa: E402
from episode_window import EpisodeIndex, in_recent_top_rank_window  # noqa: E402
from evaluate import load_jsonl, opponent_ref, parse_now  # noqa: E402

_DEPLOY_TARGET = (
    _REPO_ROOT / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "deck_predictor_hybrid.json"
)
_DEFAULT_VALID_RECENT_DAYS = 14
_DEFAULT_VALID_TOP_RANK = 200
_DEFAULT_SEED = 42
_DEFAULT_STABILITY_WARN_THRESHOLD = 0.2
_DEFAULT_MIN_LOGLOSS_IMPROVEMENT_HIGH_EVIDENCE = 0.01
_HIGH_EVIDENCE_MIN_EVIDENCE = 4  # このmin_evidence以上のバケットは「安定 かつ 明確な改善」を要求する
_EPS = 1e-12

# evidence バケット文字列(compare_nb.py と共通) -> (min_evidence, max_evidence) の対応。
# deck_predictor_hybrid.json の "buckets" に出力する際の境界値として使う。
_BUCKET_RANGES: dict[str, tuple[int, int | None]] = {
    "0": (0, 0),
    "1": (1, 1),
    "2-3": (2, 3),
    "4-6": (4, 6),
    "7-10": (7, 10),
    "11+": (11, None),
}

# w のグリッド: 0.00, 0.05, ..., 1.00(21点)。
_WEIGHT_GRID: list[float] = [round(i * 0.05, 2) for i in range(21)]


def _log_prob_vector(probs: dict[str, float], classes: list[str]) -> list[float]:
    return [math.log(max(probs.get(c, 0.0), _EPS)) for c in classes]


def blended_log_loss(log_lr: list[float], log_nb: list[float], true_idx: int, w: float) -> float:
    """1サンプル分の -log p_blend(true) を計算する(log-sum-exp で正規化)。"""
    blend = [(1.0 - w) * a + w * b for a, b in zip(log_lr, log_nb)]
    m = max(blend)
    lse = m + math.log(sum(math.exp(v - m) for v in blend))
    return lse - blend[true_idx]


def avg_logloss(items: list[dict], w: float) -> float:
    if not items:
        return float("nan")
    return sum(blended_log_loss(it["log_lr"], it["log_nb"], it["true_idx"], w) for it in items) / len(items)


def best_weight(items: list[dict], grid: list[float]) -> tuple[float, float, dict[float, float]]:
    """(最良の w, そのときの平均log loss, w -> 平均log loss の全グリッド結果) を返す。
    タイの場合は grid の先頭(=w=0.0 に近い方)を優先する(浮動小数点誤差で不必要に
    w>0 側へブレるのを避けるため)。"""
    losses: dict[float, float] = {}
    best_w = grid[0]
    best_ll = float("inf")
    for w in grid:
        ll = avg_logloss(items, w)
        losses[w] = ll
        if ll < best_ll - 1e-12:
            best_ll = ll
            best_w = w
    return best_w, best_ll, losses


def prepare_rows(
    rows: list[dict], lr: MLDeckPredictor, nb: NBDeckPredictor, classes: list[str], class_index: dict[str, int]
) -> tuple[list[dict], int]:
    """各 validation 行について LR/NB の log 確率ベクトル・evidence バケット・正解クラス index
    をキャッシュした dict のリストに変換する。ラベルが classes に無い行はスキップする。"""
    prepared: list[dict] = []
    n_skipped = 0
    for row in rows:
        true_idx = class_index.get(row["label"])
        if true_idx is None:
            n_skipped += 1
            continue
        observed_cards = row["observed_cards"]
        turn = row["turn"]
        lr_probs = lr.predict(observed_cards, turn)
        nb_probs = nb.predict(observed_cards, turn)
        evidence_count = lr.evidence_count(observed_cards)
        prepared.append(
            {
                "episode_id": row["episode_id"],
                "bucket": evidence_bucket(evidence_count),
                "evidence_count": evidence_count,
                "true_idx": true_idx,
                "log_lr": _log_prob_vector(lr_probs, classes),
                "log_nb": _log_prob_vector(nb_probs, classes),
            }
        )
    return prepared, n_skipped


def group_by_bucket(items: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        grouped[it["bucket"]].append(it)
    return grouped


def split_episodes_half(episode_ids: list[str], seed: int) -> tuple[set[str], set[str]]:
    shuffled = list(episode_ids)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return set(shuffled[:half]), set(shuffled[half:])


def load_baseline_weights(path: str | None) -> dict[str, float] | None:
    """--baseline のJSONから bucket文字列("0"/"1"/"2-3"/...) -> weight_nb を読む。

    baseline側のbucket境界(min_evidence/max_evidence)を _BUCKET_RANGES と突き合わせて
    キーを揃える。境界がどのバケットとも一致しない行は無視する(想定外フォーマットへの
    防御。呼び出し側は match しなかったバケットに対して安全にフォールバックする)。
    """
    if not path:
        return None
    range_to_bucket = {v: k for k, v in _BUCKET_RANGES.items()}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weights: dict[str, float] = {}
    for entry in data.get("buckets", []):
        key = (entry.get("min_evidence"), entry.get("max_evidence"))
        bucket = range_to_bucket.get(key)
        if bucket is None:
            continue
        weights[bucket] = float(entry["weight_nb"])
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(_HERE / "output" / "dataset.jsonl"))
    parser.add_argument("--lr-weights", default=str(_HERE / "output" / "model" / "deck_predictor_weights.json"))
    parser.add_argument("--nb-weights", default=str(_HERE / "output" / "model" / "deck_predictor_nb.json"))
    parser.add_argument("--split", default=str(_HERE / "output" / "model" / "split.json"))
    parser.add_argument(
        "--master-index", default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl")
    )
    parser.add_argument("--out", default=str(_HERE / "output" / "model" / "deck_predictor_hybrid.json"))
    parser.add_argument("--valid-recent-days", type=int, default=_DEFAULT_VALID_RECENT_DAYS)
    parser.add_argument("--valid-top-rank", type=int, default=_DEFAULT_VALID_TOP_RANK)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="50/50安定性チェックの分割シード")
    parser.add_argument(
        "--stability-warn-threshold",
        type=float,
        default=_DEFAULT_STABILITY_WARN_THRESHOLD,
        help="半分フィットのwが全データフィットのwからこの値以上ズレたら警告する"
        "(このしきい値以上を「不安定」と判定し、--baseline 採用方針でも使う)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="現行デプロイ済みの deck_predictor_hybrid.json パス。指定すると、50/50安定性チェックを"
        "通過したバケットだけ新フィット値wを採用し、不安定なバケットはこのbaselineのwを維持する"
        "(evidence>=4バケットはさらに明確なlog loss改善が無い限りbaseline維持)。省略時は従来どおり"
        "全バケットでフィット値をそのまま採用する。",
    )
    parser.add_argument(
        "--min-logloss-improvement",
        type=float,
        default=_DEFAULT_MIN_LOGLOSS_IMPROVEMENT_HIGH_EVIDENCE,
        help=f"--baseline 指定時、evidence>={_HIGH_EVIDENCE_MIN_EVIDENCE} のバケットで新フィット値wを"
        "採用するために必要な baseline比でのlog loss改善量(この値以上小さくないとbaseline維持)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help=f"{_DEPLOY_TARGET} へもコピーする",
    )
    parser.add_argument("--now", default=None, help="基準時刻をISO8601で上書き(再現性確認用)")
    args = parser.parse_args()

    now = parse_now(args.now)

    lr_predictor = MLDeckPredictor(weights_path=args.lr_weights)
    if not lr_predictor.is_ready:
        print(f"エラー: LR重みファイル {args.lr_weights} を読み込めませんでした", file=sys.stderr)
        sys.exit(1)

    nb_predictor = NBDeckPredictor(weights_path=args.nb_weights)
    if not nb_predictor.is_ready:
        print(f"エラー: NB重みファイル {args.nb_weights} を読み込めませんでした", file=sys.stderr)
        sys.exit(1)

    lr_classes = list(lr_predictor._classes)
    nb_classes = list(nb_predictor._classes)
    if set(lr_classes) != set(nb_classes):
        print(
            f"エラー: LR({lr_classes}) と NB({nb_classes}) のクラスリストが一致しません。"
            "同じ学習パイプライン成果物を指しているか確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    classes = lr_classes
    class_index = {c: i for i, c in enumerate(classes)}
    print(f"クラス数: {len(classes)} -> {classes}")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    valid_ids = set(split["valid_episode_ids"])
    print(f"validation エピソード数(split.json): {len(valid_ids)}")

    rows = load_jsonl(Path(args.dataset))
    full_valid_rows = [row for row in rows if row["episode_id"] in valid_ids]
    print(f"validation サンプル数(全体、フィルタなし): {len(full_valid_rows)}")

    episode_index = EpisodeIndex.load(Path(args.master_index))

    def in_window(row: dict) -> bool:
        episode_id, opp_index = opponent_ref(row)
        ep_time = episode_index.time_of(episode_id)
        rank = episode_index.rank_of(episode_id, opp_index)
        return in_recent_top_rank_window(ep_time, rank, now, args.valid_recent_days, args.valid_top_rank)

    windowed_rows = [row for row in full_valid_rows if in_window(row)]
    print(
        f"validation サンプル数(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内、"
        f"フィット対象の固定validation): {len(windowed_rows)} / {len(full_valid_rows)}"
    )
    if not windowed_rows:
        print("エラー: 固定 validation ウィンドウのサンプルが0件です。フィットできません。", file=sys.stderr)
        sys.exit(1)

    prepared, n_skipped = prepare_rows(windowed_rows, lr_predictor, nb_predictor, classes, class_index)
    if n_skipped:
        print(f"警告: label が classes に無いサンプルを {n_skipped} 件スキップしました", file=sys.stderr)
    print(f"フィット対象サンプル数(LR/NB推論キャッシュ済み): {len(prepared)}")

    grouped_full = group_by_bucket(prepared)

    # --- 安定性チェック用のエピソード単位 50/50 分割 ---
    episode_ids = sorted({it["episode_id"] for it in prepared})
    half_a_ids, half_b_ids = split_episodes_half(episode_ids, args.seed)
    items_a = [it for it in prepared if it["episode_id"] in half_a_ids]
    items_b = [it for it in prepared if it["episode_id"] in half_b_ids]
    print(
        f"安定性チェック用分割(seed={args.seed}): エピソード数 A={len(half_a_ids)} / B={len(half_b_ids)}"
        f", サンプル数 A={len(items_a)} / B={len(items_b)}"
    )
    grouped_a = group_by_bucket(items_a)
    grouped_b = group_by_bucket(items_b)

    baseline_weights = load_baseline_weights(args.baseline)
    if args.baseline:
        if baseline_weights is None:
            print(f"エラー: baseline {args.baseline} からバケット重みを読み込めませんでした", file=sys.stderr)
            sys.exit(1)
        print(f"\nbaseline({args.baseline})のバケット別w: {baseline_weights}")
        print(
            f"採用方針: 安定バケットのみ新フィット値wを採用、不安定バケットはbaseline維持。"
            f"evidence>={_HIGH_EVIDENCE_MIN_EVIDENCE}バケットはさらに、baseline比でlog lossが"
            f"{args.min_logloss_improvement}以上改善しない限りbaseline維持。"
        )

    # --- バケットごとに w をフィット(全データ)+ 安定性チェック(50/50)+ 採用判定 ---
    bucket_meta: list[dict] = []
    output_buckets: list[dict] = []
    n_unstable = 0
    n_adopted_fitted = 0
    n_kept_baseline = 0
    adoption_log: list[dict] = []

    print("\nバケットごとのブレンド重みフィット:")
    print(
        f"  {'bucket':10s} {'n':>7s} {'w_full':>8s} {'ll@w=0(LR)':>12s} {'ll@w=1(NB)':>12s} "
        f"{'ll@best':>10s} {'w_A':>6s} {'w_B':>6s} {'dev':>6s} {'adopted':>8s} {'reason':>28s}"
    )
    for bucket in _EVIDENCE_BUCKET_ORDER:
        items = grouped_full.get(bucket, [])
        min_ev, max_ev = _BUCKET_RANGES[bucket]
        baseline_w = baseline_weights.get(bucket) if baseline_weights is not None else None

        if not items:
            if baseline_weights is not None:
                adopted_w = baseline_w if baseline_w is not None else 0.0
                reason = "no_samples_baseline_kept"
            else:
                adopted_w = 0.0
                reason = "no_samples_zero_fallback"
            print(f"  {bucket:10s} サンプルなし。w={adopted_w:.2f}({reason})")
            n_kept_baseline += 1
            bucket_meta.append(
                {
                    "bucket": bucket,
                    "min_evidence": min_ev,
                    "max_evidence": max_ev,
                    "n_samples": 0,
                    "weight_nb_fitted": None,
                    "weight_nb_baseline": baseline_w,
                    "weight_nb": adopted_w,
                    "adoption_reason": reason,
                    "logloss_lr_only": None,
                    "logloss_nb_only": None,
                    "logloss_best": None,
                    "logloss_at_baseline_w": None,
                    "stability": None,
                }
            )
            output_buckets.append({"min_evidence": min_ev, "max_evidence": max_ev, "weight_nb": adopted_w})
            adoption_log.append({"bucket": bucket, "adopted": adopted_w, "reason": reason})
            continue

        w_full, ll_best, losses_full = best_weight(items, _WEIGHT_GRID)
        ll_lr_only = losses_full[0.0]
        ll_nb_only = losses_full[1.0]

        # --- 安定性チェック: 半分でフィット、もう半分で評価 ---
        items_a_bucket = grouped_a.get(bucket, [])
        items_b_bucket = grouped_b.get(bucket, [])
        stability = None
        stable = False
        if items_a_bucket and items_b_bucket:
            w_a, _, _ = best_weight(items_a_bucket, _WEIGHT_GRID)
            w_b, _, _ = best_weight(items_b_bucket, _WEIGHT_GRID)
            ll_a_fit_on_b = avg_logloss(items_b_bucket, w_a)
            ll_b_fit_on_a = avg_logloss(items_a_bucket, w_b)
            deviation = max(abs(w_a - w_full), abs(w_b - w_full), abs(w_a - w_b))
            unstable = deviation >= args.stability_warn_threshold
            stable = not unstable
            if unstable:
                n_unstable += 1
            stability = {
                "n_half_a": len(items_a_bucket),
                "n_half_b": len(items_b_bucket),
                "weight_nb_fit_on_half_a": w_a,
                "weight_nb_fit_on_half_b": w_b,
                "logloss_half_a_fit_evaluated_on_half_b": ll_a_fit_on_b,
                "logloss_half_b_fit_evaluated_on_half_a": ll_b_fit_on_a,
                "max_deviation_from_full_fit": deviation,
                "unstable": unstable,
            }
            w_a_str, w_b_str, dev_str = f"{w_a:.2f}", f"{w_b:.2f}", f"{deviation:.2f}"
        else:
            w_a_str, w_b_str, dev_str = " n/a", " n/a", " n/a"
            print(
                f"    警告: バケット {bucket} は半分の一方にサンプルが無いため安定性チェックをスキップします",
                file=sys.stderr,
            )

        # --- 採用判定 ---
        ll_at_baseline_w = None
        if baseline_weights is None:
            adopted_w = w_full
            reason = "fitted_no_baseline"
        elif not stable:
            adopted_w = baseline_w if baseline_w is not None else 0.0
            reason = "baseline_kept_unstable"
        elif min_ev >= _HIGH_EVIDENCE_MIN_EVIDENCE:
            ll_at_baseline_w = avg_logloss(items, baseline_w if baseline_w is not None else 0.0)
            improvement = ll_at_baseline_w - ll_best
            if improvement > args.min_logloss_improvement:
                adopted_w = w_full
                reason = "fitted_stable_high_evidence_improved"
            else:
                adopted_w = baseline_w if baseline_w is not None else 0.0
                reason = "baseline_kept_high_evidence_no_improvement"
        else:
            adopted_w = w_full
            reason = "fitted_stable"

        if reason.startswith("fitted"):
            n_adopted_fitted += 1
        else:
            n_kept_baseline += 1

        print(
            f"  {bucket:10s} {len(items):>7d} {w_full:>8.2f} {ll_lr_only:>12.4f} {ll_nb_only:>12.4f} "
            f"{ll_best:>10.4f} {w_a_str:>6s} {w_b_str:>6s} {dev_str:>6s} {adopted_w:>8.2f} {reason:>28s}"
        )
        if stability and stability["unstable"]:
            print(
                f"    警告: バケット {bucket} は50/50分割間で選択wが大きくズレています"
                f"(deviation={stability['max_deviation_from_full_fit']:.2f} >= "
                f"{args.stability_warn_threshold})。フィット結果を目視確認してください。",
                file=sys.stderr,
            )

        bucket_meta.append(
            {
                "bucket": bucket,
                "min_evidence": min_ev,
                "max_evidence": max_ev,
                "n_samples": len(items),
                "weight_nb_fitted": w_full,
                "weight_nb_baseline": baseline_w,
                "weight_nb": adopted_w,
                "adoption_reason": reason,
                "logloss_lr_only": ll_lr_only,
                "logloss_nb_only": ll_nb_only,
                "logloss_best": ll_best,
                "logloss_at_baseline_w": ll_at_baseline_w,
                "stability": stability,
            }
        )
        output_buckets.append({"min_evidence": min_ev, "max_evidence": max_ev, "weight_nb": adopted_w})
        adoption_log.append({"bucket": bucket, "fitted": w_full, "baseline": baseline_w, "adopted": adopted_w, "reason": reason})

    overall_ll_lr_only = avg_logloss(prepared, 0.0)
    overall_ll_hybrid = sum(
        blended_log_loss(
            it["log_lr"], it["log_nb"], it["true_idx"], next(b["weight_nb"] for b in bucket_meta if b["bucket"] == it["bucket"])
        )
        for it in prepared
    ) / len(prepared)
    print(f"\n全体 log loss: LR単体={overall_ll_lr_only:.4f} -> hybrid(採用後)={overall_ll_hybrid:.4f}")

    if baseline_weights is not None:
        print(f"\n採用方針サマリ: 新フィット値を採用={n_adopted_fitted}バケット, baseline維持={n_kept_baseline}バケット")
        for entry in adoption_log:
            print(f"  {entry['bucket']:10s} -> {entry['reason']}")

    if n_unstable:
        print(
            f"\n警告: {n_unstable} 個のバケットで50/50安定性チェックの deviation が閾値"
            f"({args.stability_warn_threshold})以上でした。上記の警告ログを確認してください。",
            file=sys.stderr,
        )
    else:
        print("\n安定性チェック: 全バケットで deviation は閾値未満でした。")

    payload = {
        "buckets": output_buckets,
        "meta": {
            "method": "evidence_count_bucketed_log_space_geometric_blend",
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "fitted_on": (
                f"validation split (output/model/split.json), 固定validationウィンドウ"
                f"(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内)"
            ),
            "n_samples": len(prepared),
            "n_episodes": len(episode_ids),
            "classes": classes,
            "weight_grid": _WEIGHT_GRID,
            "overall_logloss_lr_only": overall_ll_lr_only,
            "overall_logloss_hybrid": overall_ll_hybrid,
            "stability_check": {
                "seed": args.seed,
                "warn_threshold": args.stability_warn_threshold,
                "n_episodes_half_a": len(half_a_ids),
                "n_episodes_half_b": len(half_b_ids),
                "n_unstable_buckets": n_unstable,
            },
            "adoption": {
                "baseline_path": args.baseline,
                "min_logloss_improvement_high_evidence": args.min_logloss_improvement,
                "high_evidence_min_evidence": _HIGH_EVIDENCE_MIN_EVIDENCE,
                "n_buckets_adopted_fitted": n_adopted_fitted,
                "n_buckets_kept_baseline": n_kept_baseline,
                "per_bucket": adoption_log,
            },
            "buckets_detail": bucket_meta,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nハイブリッド重みを {out_path} に書き出しました")

    if args.deploy:
        _DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        _DEPLOY_TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"デプロイ用に {_DEPLOY_TARGET} にもコピーしました")


if __name__ == "__main__":
    main()
