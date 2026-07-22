#!/usr/bin/env python3
"""validation セットで top-1 正解率・log loss・ランク帯別/期間別正解率・クラス別 precision/recall、
および base(補正前)vs adjusted(補正後)の比較を計算する。

train.py が書き出した output/model/split.json の valid_episode_ids を使い、dataset.jsonl から
validation サンプルだけを抽出する(train/valid の分割はエピソード単位で train.py と共有)。

「固定 validation」(ml-predictor-phase2-scaling.md の評価プロトコル)として、既定では
validation をさらに「直近 --valid-recent-days 日以内」かつ「相手の rank_at_fetch が
--valid-top-rank 位以内」に絞り込む(実戦で当たる相手の分布に合わせるため)。
--all を付けると、絞り込みなしの従来の全体評価もあわせて実行する。

相手の作成日時・ランクは episodes_master.jsonl と結合して求める(episode_window.py)。

推論はランタイムと同じ sample_submission/ptcg_ai/opponent_modeling/ml_predictor.py の
MLDeckPredictor を使う(学習パイプラインとランタイムで推論ロジックがズレないようにするため)。

序盤(見えているカードが少ない場面)の予測が過信気味という問題を定量化するため、
evidence(観測エビデンス)数別バケット評価・reliability テーブル(確信度キャリブレーション)・
誤答ダンプ(--error-dump)も出力する。詳細は
sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md
の「フェーズA」節を参照。

出力: 標準出力 + output/eval_report.md
      (--error-dump 指定時) output/error_dump.jsonl + output/error_summary.md

使い方:
  python evaluate.py
  python evaluate.py --valid-recent-days 30 --valid-top-rank 500
  python evaluate.py --all
  python evaluate.py --error-dump
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor  # noqa: E402

from episode_window import (  # noqa: E402
    RANK_BUCKETS,
    EpisodeIndex,
    in_recent_top_rank_window,
    rank_bucket,
)

_TURN_BUCKET_ORDER = ["0"] + [str(t) for t in range(1, 11)] + ["11+"]
_PERIOD_ORDER = ["古い半分", "新しい半分", "不明"]

_DEFAULT_VALID_RECENT_DAYS = 14
_DEFAULT_VALID_TOP_RANK = 200

# evidence 数(見えているユニークカード種類数)のバケット定義・表示順。
_EVIDENCE_BUCKET_ORDER = ["0", "1", "2-3", "4-6", "7-10", "11+"]

# top-1 確信度ビン(reliability テーブル用)。境界は [lo, hi) で、最後だけ [0.9, 1.0] 閉区間。
_CONFIDENCE_BINS = [
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 0.6, "0.5-0.6"),
    (0.6, 0.7, "0.6-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 1.0 + 1e-9, "0.9-1.0"),
]

# ECE 計算用の10ビン(0.0-0.1, ..., 0.9-1.0)。
_ECE_BIN_COUNT = 10

# 誤答ダンプ(--error-dump)の設定。
_ERROR_DUMP_ALL_LIMIT = 2000
_HIGH_CONFIDENCE_THRESHOLD = 0.8
_CONFUSION_PAIR_TOP_N = 20
_ERROR_CARD_TOP_N = 30


def turn_bucket(turn: int) -> str:
    if turn <= 0:
        return "0"
    if turn >= 11:
        return "11+"
    return str(turn)


def evidence_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return "11+"


def confidence_bin(confidence: float) -> str | None:
    for lo, hi, label in _CONFIDENCE_BINS:
        if lo <= confidence < hi:
            return label
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def opponent_ref(row: dict) -> tuple[str, int]:
    """dataset.jsonl の1行から「相手」を特定する (episode_id, 相手の player_index)。

    dataset.jsonl の player_index は観測している側(視点側)なので、相手は 1 - player_index。
    """
    return row["episode_id"], 1 - row["player_index"]


class PredictionResult(NamedTuple):
    """1サンプル分の推論結果。1回の predict() 呼び出しから得られる値をまとめて持ち回し、
    evidence バケット評価・reliability テーブル・誤答ダンプで使い回す(同じ行を二度推論しない)。"""

    true_label: str
    pred_label: str
    log_loss: float
    evidence_count: int
    top1_confidence: float
    top3: list[tuple[str, float]]


def predict_rows(rows: list[dict], predictor: MLDeckPredictor) -> list[PredictionResult]:
    """各行を1回だけ推論し、正解率・log loss・evidence バケット・reliability・誤答ダンプに
    必要な値をまとめて PredictionResult として返す。"""
    results: list[PredictionResult] = []
    for row in rows:
        observed_cards = row["observed_cards"]
        probs = predictor.predict(observed_cards, row["turn"])
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        pred_label, top1_confidence = ranked[0]
        true_label = row["label"]
        p_true = probs.get(true_label, 0.0)
        results.append(
            PredictionResult(
                true_label=true_label,
                pred_label=pred_label,
                log_loss=-math.log(max(p_true, 1e-12)),
                evidence_count=predictor.evidence_count(observed_cards),
                top1_confidence=top1_confidence,
                top3=ranked[:3],
            )
        )
    return results


def _unpack(results: list[PredictionResult]) -> tuple[list[str], list[str], list[float]]:
    """既存の overall_metrics / bucket_accuracy / class_prf 用に (y_true, y_pred, log_losses) を取り出す。"""
    y_true = [r.true_label for r in results]
    y_pred = [r.pred_label for r in results]
    log_losses = [r.log_loss for r in results]
    return y_true, y_pred, log_losses


def compute_ece(results: list[PredictionResult], n_bins: int = _ECE_BIN_COUNT) -> float:
    """ECE(expected calibration error)。top-1 確信度を n_bins 個の等幅ビンに分け、
    Σ (|ビン内平均確信度 - ビン内正解率| × ビン内サンプル数 / 総数)。"""
    if not results:
        return float("nan")
    bins: list[list[PredictionResult]] = [[] for _ in range(n_bins)]
    for r in results:
        idx = min(int(r.top1_confidence * n_bins), n_bins - 1)
        bins[idx].append(r)
    n = len(results)
    ece = 0.0
    for bucket_results in bins:
        if not bucket_results:
            continue
        bucket_n = len(bucket_results)
        avg_conf = sum(r.top1_confidence for r in bucket_results) / bucket_n
        acc = sum(1 for r in bucket_results if r.true_label == r.pred_label) / bucket_n
        ece += abs(avg_conf - acc) * bucket_n / n
    return ece


def overall_metrics(y_true: list[str], y_pred: list[str], log_losses: list[float]) -> tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return 0.0, float("nan")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / n, sum(log_losses) / n


def bucket_accuracy(rows: list[dict], y_true: list[str], y_pred: list[str], key_fn) -> dict[str, tuple[int, int]]:
    """key_fn(row) -> バケット名。戻り値は {バケット名: (正解数, 総数)}。"""
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for row, t, p in zip(rows, y_true, y_pred):
        bucket = key_fn(row)
        total[bucket] += 1
        if t == p:
            correct[bucket] += 1
    return {b: (correct[b], total[b]) for b in total}


def class_prf(y_true: list[str], y_pred: list[str]) -> list[tuple[str, float, float, float, int]]:
    classes = sorted(set(y_true) | set(y_pred))
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1
    result = []
    for cls in classes:
        p_count = tp[cls] + fp[cls]
        r_count = tp[cls] + fn[cls]
        precision = tp[cls] / p_count if p_count else 0.0
        recall = tp[cls] / r_count if r_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result.append((cls, precision, recall, f1, support[cls]))
    return result


def parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def report_section(
    lines: list[str],
    title: str,
    target_rows: list[dict],
    predictor: MLDeckPredictor,
    base_predictor: MLDeckPredictor | None,
    episode_index: EpisodeIndex,
) -> list[PredictionResult]:
    """target_rows を評価し、レポート本文を lines に追記する。呼び出し側(--error-dump)が
    誤答サンプルを特定できるよう、target_rows と同じ順序の PredictionResult 一覧を返す
    (サンプルなしの場合は空リスト)。"""
    lines.append(f"## {title}")
    lines.append("")
    if not target_rows:
        lines.append("(サンプルなし)")
        lines.append("")
        print(f"\n[{title}] サンプルなし")
        return []

    results = predict_rows(target_rows, predictor)
    y_true, y_pred, log_losses = _unpack(results)
    acc, ll = overall_metrics(y_true, y_pred, log_losses)
    lines.append(f"- サンプル数: {len(target_rows)}")
    lines.append(f"- top-1 正解率: {acc * 100:.2f}%")
    lines.append(f"- log loss: {ll:.4f}")
    lines.append("")
    print(f"\n[{title}] サンプル数={len(target_rows)} top-1正解率={acc * 100:.2f}% log loss={ll:.4f}")

    # ターン別
    lines.append("### ターン別 top-1 正解率")
    lines.append("")
    lines.append("| ターン | サンプル数 | 正解数 | 正解率 |")
    lines.append("|---|---:|---:|---:|")
    tb = bucket_accuracy(target_rows, y_true, y_pred, lambda r: turn_bucket(r["turn"]))
    for b in _TURN_BUCKET_ORDER:
        if b not in tb:
            continue
        c, t = tb[b]
        label = f"turn {b}" if b != "0" else "turn 0(選択直後)"
        lines.append(f"| {label} | {t} | {c} | {c / t * 100:.2f}% |")
    lines.append("")

    # 相手ランク帯別
    lines.append("### 相手ランク帯別 top-1 正解率")
    lines.append("")
    lines.append("| ランク帯 | サンプル数 | 正解数 | 正解率 |")
    lines.append("|---|---:|---:|---:|")

    def rank_key(row: dict) -> str:
        episode_id, opp_index = opponent_ref(row)
        return rank_bucket(episode_index.rank_of(episode_id, opp_index))

    rb = bucket_accuracy(target_rows, y_true, y_pred, rank_key)
    for b in RANK_BUCKETS:
        if b not in rb:
            continue
        c, t = rb[b]
        lines.append(f"| {b} | {t} | {c} | {c / t * 100:.2f}% |")
    lines.append("")

    # 期間別(エピソード日付で古い半分 vs 新しい半分)
    lines.append("### 期間別 top-1 正解率(エピソード日付で古い半分 vs 新しい半分)")
    lines.append("")
    known_times = sorted(
        t for t in (episode_index.time_of(opponent_ref(row)[0]) for row in target_rows) if t is not None
    )
    if known_times:
        median_time = known_times[len(known_times) // 2]

        def period_key(row: dict) -> str:
            episode_id, _ = opponent_ref(row)
            t_ep = episode_index.time_of(episode_id)
            if t_ep is None:
                return "不明"
            return "新しい半分" if t_ep >= median_time else "古い半分"

        pb = bucket_accuracy(target_rows, y_true, y_pred, period_key)
        lines.append("| 期間 | サンプル数 | 正解数 | 正解率 |")
        lines.append("|---|---:|---:|---:|")
        for b in _PERIOD_ORDER:
            if b not in pb:
                continue
            c, t = pb[b]
            lines.append(f"| {b} | {t} | {c} | {c / t * 100:.2f}% |")
    else:
        lines.append("(エピソード日時が不明のためスキップ)")
    lines.append("")

    # クラス別 precision/recall/F1
    lines.append("### クラス別 precision / recall / F1")
    lines.append("")
    lines.append("| クラス | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls, precision, recall, f1, support in class_prf(y_true, y_pred):
        lines.append(f"| {cls} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} |")
    lines.append("")

    # evidence 数別バケット評価(序盤過信の定量化。フェーズA)
    lines.append("### evidence 数別バケット評価")
    lines.append("")
    lines.append(
        "- evidence 数 = 観測されているユニークカード種類数(`MLDeckPredictor.evidence_count()`、"
        "fit/infer と同一定義)。"
    )
    lines.append("- 平均確信度 = 各サンプルの予測分布の最大確率(top-1確信度)の平均。")
    lines.append(
        "- ECE(expected calibration error) = top-1確信度を10ビンに分け、"
        "Σ |ビン内平均確信度 - ビン内正解率| × ビン内サンプル数 / 総数。"
    )
    lines.append("")
    lines.append("| evidence数 | サンプル数 | top-1正解率 | log loss | 平均確信度 | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    evidence_groups: dict[str, list[PredictionResult]] = {b: [] for b in _EVIDENCE_BUCKET_ORDER}
    for r in results:
        evidence_groups[evidence_bucket(r.evidence_count)].append(r)
    for b in _EVIDENCE_BUCKET_ORDER:
        group = evidence_groups[b]
        if not group:
            continue
        n = len(group)
        g_acc = sum(1 for r in group if r.true_label == r.pred_label) / n
        g_ll = sum(r.log_loss for r in group) / n
        g_conf = sum(r.top1_confidence for r in group) / n
        g_ece = compute_ece(group)
        lines.append(f"| {b} | {n} | {g_acc * 100:.2f}% | {g_ll:.4f} | {g_conf * 100:.2f}% | {g_ece:.4f} |")
    lines.append("")
    print("  evidence数別バケット評価:")
    for b in _EVIDENCE_BUCKET_ORDER:
        group = evidence_groups[b]
        if not group:
            continue
        n = len(group)
        g_acc = sum(1 for r in group if r.true_label == r.pred_label) / n
        g_ece = compute_ece(group)
        print(f"    evidence={b}: n={n} top-1正解率={g_acc * 100:.2f}% ECE={g_ece:.4f}")

    # reliability テーブル(予測X%のとき本当にX%当たっているか)
    lines.append("### reliability テーブル(top-1確信度 vs 実際の正解率)")
    lines.append("")
    lines.append("| 確信度ビン | サンプル数 | 平均確信度 | 実際の正解率 |")
    lines.append("|---|---:|---:|---:|")
    confidence_groups: dict[str, list[PredictionResult]] = {label: [] for _, _, label in _CONFIDENCE_BINS}
    for r in results:
        cbin = confidence_bin(r.top1_confidence)
        if cbin is not None:
            confidence_groups[cbin].append(r)
    for _, _, label in _CONFIDENCE_BINS:
        group = confidence_groups[label]
        if not group:
            continue
        n = len(group)
        g_conf = sum(r.top1_confidence for r in group) / n
        g_acc = sum(1 for r in group if r.true_label == r.pred_label) / n
        lines.append(f"| {label} | {n} | {g_conf * 100:.2f}% | {g_acc * 100:.2f}% |")
    lines.append("")

    # base vs adjusted 比較
    if base_predictor is not None:
        b_true, b_pred, b_ll = _unpack(predict_rows(target_rows, base_predictor))
        b_acc, b_loss = overall_metrics(b_true, b_pred, b_ll)
        lines.append("### base(補正前) vs adjusted(補正後) 比較")
        lines.append("")
        lines.append("| 重み | top-1 正解率 | log loss |")
        lines.append("|---|---:|---:|")
        lines.append(f"| base | {b_acc * 100:.2f}% | {b_loss:.4f} |")
        lines.append(f"| adjusted | {acc * 100:.2f}% | {ll:.4f} |")
        lines.append(f"| 差分(adjusted - base) | {(acc - b_acc) * 100:+.2f}pt | {ll - b_loss:+.4f} |")
        lines.append("")
        print(
            f"  base vs adjusted: base_acc={b_acc * 100:.2f}% adjusted_acc={acc * 100:.2f}% "
            f"base_logloss={b_loss:.4f} adjusted_logloss={ll:.4f}"
        )

    return results


class ErrorSection(NamedTuple):
    """誤答ダンプ・誤答サマリで扱う1セクション分のデータ(report_section の結果を再利用)。"""

    title: str
    rows: list[dict]
    results: list[PredictionResult]
    truncate: bool  # True の場合、誤答が多ければ _ERROR_DUMP_ALL_LIMIT 件で打ち切る。


def _error_dump_record(row: dict, result: PredictionResult) -> dict:
    return {
        "episode_id": row["episode_id"],
        "player_index": row["player_index"],
        "turn": row["turn"],
        "step": row.get("step"),
        "evidence_count": result.evidence_count,
        "observed_cards": row["observed_cards"],
        "top3": [[label, prob] for label, prob in result.top3],
        "true_label": result.true_label,
    }


def write_error_dump_and_summary(
    sections: list[ErrorSection],
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    """誤答サンプルを output/error_dump.jsonl に、集計を output/error_summary.md に書き出す。"""
    summary_lines = ["# 相手デッキ予測器 誤答サマリ", ""]

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for section in sections:
            # 集計(混同ペア・カード頻度・evidence バケット別など)は打ち切りに関係なく全誤答を対象にする。
            all_errors = [
                (row, r) for row, r in zip(section.rows, section.results) if r.true_label != r.pred_label
            ]
            total_errors = len(all_errors)

            # jsonl への書き出しだけ --all 側(truncate=True)は _ERROR_DUMP_ALL_LIMIT 件で打ち切る。
            dump_errors = all_errors
            truncated = False
            if section.truncate and total_errors > _ERROR_DUMP_ALL_LIMIT:
                dump_errors = all_errors[:_ERROR_DUMP_ALL_LIMIT]
                truncated = True

            for row, r in dump_errors:
                jf.write(json.dumps(_error_dump_record(row, r), ensure_ascii=False) + "\n")

            print(
                f"\n[誤答ダンプ: {section.title}] 誤答 {total_errors}件"
                + (f"(先頭{_ERROR_DUMP_ALL_LIMIT}件のみ書き出し。打ち切りました)" if truncated else "を書き出しました")
            )

            # --- サマリ集計 ---
            summary_lines.append(f"## {section.title}")
            summary_lines.append("")
            summary_lines.append(f"- 誤答サンプル数: {total_errors}")
            if truncated:
                summary_lines.append(
                    f"- **error_dump.jsonl への書き出しは先頭 {_ERROR_DUMP_ALL_LIMIT} 件で打ち切り**"
                    f"(全 {total_errors} 件のうち)。集計(以下)は打ち切り前の全誤答が対象。"
                )
            summary_lines.append("")

            if total_errors == 0:
                summary_lines.append("(誤答なし)")
                summary_lines.append("")
                continue

            # 高確信誤答(top-1確信度 >= 0.8 なのに誤答)
            high_conf_errors = [
                (row, r) for row, r in all_errors if r.top1_confidence >= _HIGH_CONFIDENCE_THRESHOLD
            ]
            summary_lines.append(
                f"- 高確信誤答数(top-1確信度 >= {_HIGH_CONFIDENCE_THRESHOLD:.0%} なのに誤答): "
                f"{len(high_conf_errors)} / {total_errors}"
                f"({len(high_conf_errors) / total_errors * 100:.1f}%)"
            )
            summary_lines.append("")
            print(
                f"  高確信誤答(confidence>={_HIGH_CONFIDENCE_THRESHOLD:.0%}): "
                f"{len(high_conf_errors)} / {total_errors}"
            )

            # 混同ペア((予測, 正解) の頻度上位20) + 代表例1件
            pair_counts: Counter[tuple[str, str]] = Counter()
            pair_example: dict[tuple[str, str], dict] = {}
            for row, r in all_errors:
                pair = (r.pred_label, r.true_label)
                pair_counts[pair] += 1
                if pair not in pair_example:
                    pair_example[pair] = row["observed_cards"]

            summary_lines.append("### 混同ペア((予測クラス, 正解クラス)) 頻度上位")
            summary_lines.append("")
            summary_lines.append("| 予測クラス | 正解クラス | 件数 | 代表例(observed_cards) |")
            summary_lines.append("|---|---|---:|---|")
            for (pred, true), cnt in pair_counts.most_common(_CONFUSION_PAIR_TOP_N):
                example = json.dumps(pair_example[(pred, true)], ensure_ascii=False)
                summary_lines.append(f"| {pred} | {true} | {cnt} | `{example}` |")
            summary_lines.append("")

            # 誤答時に観測されていたカード名の頻度上位30(誤答中の出現回数 vs 全体中の出現回数)
            error_card_counts: Counter[str] = Counter()
            for row, _r in all_errors:
                for name, count in row["observed_cards"].items():
                    if count > 0:
                        error_card_counts[name] += 1
            all_card_counts: Counter[str] = Counter()
            for row in section.rows:
                for name, count in row["observed_cards"].items():
                    if count > 0:
                        all_card_counts[name] += 1

            summary_lines.append("### 誤答時に観測されていたカード名の頻度上位")
            summary_lines.append("")
            summary_lines.append("| カード名 | 誤答中の出現回数 | 全体中の出現回数 |")
            summary_lines.append("|---|---:|---:|")
            for name, cnt in error_card_counts.most_common(_ERROR_CARD_TOP_N):
                summary_lines.append(f"| {name} | {cnt} | {all_card_counts[name]} |")
            summary_lines.append("")

            # evidence 数バケット別の誤答件数
            evidence_error_counts: Counter[str] = Counter()
            for _row, r in all_errors:
                evidence_error_counts[evidence_bucket(r.evidence_count)] += 1
            summary_lines.append("### evidence 数バケット別の誤答件数")
            summary_lines.append("")
            summary_lines.append("| evidence数 | 誤答件数 |")
            summary_lines.append("|---|---:|")
            for b in _EVIDENCE_BUCKET_ORDER:
                if b not in evidence_error_counts:
                    continue
                summary_lines.append(f"| {b} | {evidence_error_counts[b]} |")
            summary_lines.append("")

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\n誤答ダンプを {jsonl_path} に、誤答サマリを {summary_path} に書き出しました")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(_HERE / "output" / "dataset.jsonl"))
    parser.add_argument("--weights", default=str(_HERE / "output" / "model" / "deck_predictor_weights.json"))
    parser.add_argument(
        "--base-weights", default=str(_HERE / "output" / "model" / "deck_predictor_weights_base.json")
    )
    parser.add_argument("--split", default=str(_HERE / "output" / "model" / "split.json"))
    parser.add_argument(
        "--master-index", default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl")
    )
    parser.add_argument("--report", default=str(_HERE / "output" / "eval_report.md"))
    parser.add_argument("--valid-recent-days", type=int, default=_DEFAULT_VALID_RECENT_DAYS)
    parser.add_argument("--valid-top-rank", type=int, default=_DEFAULT_VALID_TOP_RANK)
    parser.add_argument(
        "--all", action="store_true", help="ウィンドウ絞り込みなしの従来の全体評価もあわせて実行する"
    )
    parser.add_argument("--now", default=None, help="基準時刻をISO8601で上書き(再現性確認用)")
    parser.add_argument(
        "--error-dump",
        action="store_true",
        help=(
            "誤答サンプルを output/error_dump.jsonl に、集計を output/error_summary.md に書き出す"
            "(固定 validation の全誤答 + --all 時は全体評価の誤答も、こちらは最大2000件で打ち切り)"
        ),
    )
    parser.add_argument("--error-dump-jsonl", default=str(_HERE / "output" / "error_dump.jsonl"))
    parser.add_argument("--error-summary", default=str(_HERE / "output" / "error_summary.md"))
    args = parser.parse_args()

    now = parse_now(args.now)

    predictor = MLDeckPredictor(weights_path=args.weights)
    if not predictor.is_ready:
        print(f"エラー: 重みファイル {args.weights} を読み込めませんでした", file=sys.stderr)
        sys.exit(1)

    base_predictor: MLDeckPredictor | None = None
    base_path = Path(args.base_weights)
    if base_path.exists():
        candidate = MLDeckPredictor(weights_path=str(base_path))
        if candidate.is_ready:
            base_predictor = candidate
        else:
            print(f"警告: base weights {base_path} を読み込めませんでした。base比較はスキップします", file=sys.stderr)
    else:
        print(f"注記: base weights {base_path} が見つからないため、base比較はスキップします")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    valid_ids = set(split["valid_episode_ids"])

    rows = load_jsonl(Path(args.dataset))
    full_valid_rows = [row for row in rows if row["episode_id"] in valid_ids]
    print(f"validation サンプル数(全体、フィルタなし): {len(full_valid_rows)} (エピソード数: {len(valid_ids)})")

    episode_index = EpisodeIndex.load(Path(args.master_index))

    def in_window(row: dict) -> bool:
        episode_id, opp_index = opponent_ref(row)
        ep_time = episode_index.time_of(episode_id)
        rank = episode_index.rank_of(episode_id, opp_index)
        return in_recent_top_rank_window(ep_time, rank, now, args.valid_recent_days, args.valid_top_rank)

    windowed_valid_rows = [row for row in full_valid_rows if in_window(row)]
    print(
        f"validation サンプル数(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内): "
        f"{len(windowed_valid_rows)} / {len(full_valid_rows)}"
    )
    if not windowed_valid_rows:
        print(
            "警告: ウィンドウ条件を満たす validation サンプルが0件です。"
            "--valid-recent-days / --valid-top-rank を緩めるか、--all で従来の全体評価を確認してください。",
            file=sys.stderr,
        )

    lines = ["# 相手デッキ予測器 evaluation レポート", ""]
    lines.append(f"- 評価対象重み(adjusted): `{args.weights}`")
    if base_predictor is not None:
        lines.append(f"- 比較用ベース重み(base): `{args.base_weights}`")
    lines.append(f"- now(基準時刻): {now.isoformat()}")
    lines.append(f"- validation エピソード数(split.json): {len(valid_ids)}")
    lines.append(f"- validation サンプル数(全体、フィルタなし): {len(full_valid_rows)}")
    lines.append(
        f"- validation サンプル数(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内): "
        f"{len(windowed_valid_rows)}"
    )
    lines.append("")

    windowed_title = f"固定 validation(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内)"
    windowed_results = report_section(
        lines,
        windowed_title,
        windowed_valid_rows,
        predictor,
        base_predictor,
        episode_index,
    )

    error_sections = [ErrorSection(windowed_title, windowed_valid_rows, windowed_results, truncate=False)]

    if args.all:
        all_title = "従来評価(全体、ウィンドウ絞り込みなし)"
        full_results = report_section(
            lines,
            all_title,
            full_valid_rows,
            predictor,
            base_predictor,
            episode_index,
        )
        error_sections.append(ErrorSection(all_title, full_valid_rows, full_results, truncate=True))

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {args.report} に書き出しました")

    if args.error_dump:
        write_error_dump_and_summary(error_sections, Path(args.error_dump_jsonl), Path(args.error_summary))


if __name__ == "__main__":
    main()
