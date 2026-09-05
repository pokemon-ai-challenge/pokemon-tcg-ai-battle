#!/usr/bin/env python3
"""LR(MLDeckPredictor, adjusted)版と NB(NBDeckPredictor)版の相手デッキ予測器を、
同一の validation セット・同一の evidence バケット定義で比較する。

evaluate.py と同じ validation セット(train.py が書き出した output/model/split.json の
valid_episode_ids)・同じ「固定 validation」ウィンドウ(既定: 直近14日 x 相手上位200位以内、
episode_window.py 経由)を使う。evidence(観測エビデンス数)バケットは
0 / 1 / 2-3 / 4-6 / 7-10 / 11+ の6段階(early-confidence-improvement-plan.md のフェーズA節と
同じ粒度)。evidence 数の定義はモデル間で共通にする必要があるため、LR / NB / hybrid どの評価でも
``MLDeckPredictor.evidence_count()``(fit/infer時と同一定義、LRの feature_names 語彙基準)を使う。

evaluate.py は変更禁止のため、再利用したいユーティリティ(load_jsonl / opponent_ref / parse_now
/ episode_window 経由のウィンドウ判定)は evaluate.py からそのまま import する(コピーしても
よいとされているが、ズレを避けるため import で済ませる)。

``fit_hybrid.py`` が出力する ``deck_predictor_hybrid.json``(既定パス
``output/model/deck_predictor_hybrid.json``)が存在する場合、``HybridDeckPredictor`` も
あわせて読み込み、比較テーブルに hybrid 列(top-1 / log loss)を追加した3列比較にする
(``fit_hybrid.py`` 実行前に本スクリプトを動かした場合など、ファイルが無ければ従来通り
LR/NB の2列比較のまま動く。後方互換)。

出力: 標準出力 + output/nb_compare_report.md

使い方:
  python compare_nb.py                 # 固定 validation(直近14日 x 上位200位以内)のみ
  python compare_nb.py --all           # 全体(ウィンドウ絞り込みなし)もあわせて評価
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.nb_predictor import NBDeckPredictor  # noqa: E402

from episode_window import EpisodeIndex, in_recent_top_rank_window  # noqa: E402
from evaluate import load_jsonl, opponent_ref, parse_now  # noqa: E402

_EVIDENCE_BUCKET_ORDER = ["0", "1", "2-3", "4-6", "7-10", "11+"]

_DEFAULT_VALID_RECENT_DAYS = 14
_DEFAULT_VALID_TOP_RANK = 200


def evidence_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def evaluate_rows(
    rows: list[dict],
    lr_predictor: MLDeckPredictor,
    nb_predictor: NBDeckPredictor,
    hybrid_predictor: HybridDeckPredictor | None = None,
) -> list[dict]:
    """各行について evidence バケット・各モデルの top-1 予測・log loss を計算する。

    ``hybrid_predictor`` が None の場合は LR/NB の2モデルのみ評価する(後方互換)。
    """
    results = []
    for row in rows:
        observed_cards = row["observed_cards"]
        turn = row["turn"]
        true_label = row["label"]

        # evidence 数の定義は LR の語彙(feature_names)を全モデル共通の基準として使う
        # (compare_nb.py の要件: どのモデルを評価する場合も同じバケットに入れて公平に比較する)。
        evidence_count = lr_predictor.evidence_count(observed_cards)
        bucket = evidence_bucket(evidence_count)

        lr_probs = lr_predictor.predict(observed_cards, turn)
        nb_probs = nb_predictor.predict(observed_cards, turn)

        lr_pred = max(lr_probs.items(), key=lambda kv: kv[1])[0]
        nb_pred = max(nb_probs.items(), key=lambda kv: kv[1])[0]

        lr_logloss = -math.log(max(lr_probs.get(true_label, 0.0), 1e-12))
        nb_logloss = -math.log(max(nb_probs.get(true_label, 0.0), 1e-12))

        result = {
            "bucket": bucket,
            "evidence_count": evidence_count,
            "true_label": true_label,
            "lr_correct": lr_pred == true_label,
            "nb_correct": nb_pred == true_label,
            "lr_logloss": lr_logloss,
            "nb_logloss": nb_logloss,
        }

        if hybrid_predictor is not None:
            hybrid_probs = hybrid_predictor.predict(observed_cards, turn)
            hybrid_pred = max(hybrid_probs.items(), key=lambda kv: kv[1])[0]
            result["hybrid_correct"] = hybrid_pred == true_label
            result["hybrid_logloss"] = -math.log(max(hybrid_probs.get(true_label, 0.0), 1e-12))

        results.append(result)
    return results


def aggregate_by_bucket(results: list[dict], has_hybrid: bool = False) -> dict[str, dict]:
    """bucket -> {n, lr_acc, lr_logloss, nb_acc, nb_logloss, [hybrid_acc, hybrid_logloss]} に
    集計する(全体集計は '__all__' キー)。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        grouped[r["bucket"]].append(r)
        grouped["__all__"].append(r)

    summary: dict[str, dict] = {}
    for bucket, items in grouped.items():
        n = len(items)
        lr_acc = sum(1 for i in items if i["lr_correct"]) / n if n else 0.0
        nb_acc = sum(1 for i in items if i["nb_correct"]) / n if n else 0.0
        lr_ll = sum(i["lr_logloss"] for i in items) / n if n else float("nan")
        nb_ll = sum(i["nb_logloss"] for i in items) / n if n else float("nan")
        entry = {
            "n": n,
            "lr_acc": lr_acc,
            "nb_acc": nb_acc,
            "lr_logloss": lr_ll,
            "nb_logloss": nb_ll,
        }
        if has_hybrid:
            entry["hybrid_acc"] = sum(1 for i in items if i["hybrid_correct"]) / n if n else 0.0
            entry["hybrid_logloss"] = sum(i["hybrid_logloss"] for i in items) / n if n else float("nan")
        summary[bucket] = entry
    return summary


def render_section(
    lines: list[str],
    title: str,
    rows: list[dict],
    lr_predictor,
    nb_predictor,
    hybrid_predictor: HybridDeckPredictor | None = None,
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if not rows:
        lines.append("(サンプルなし)")
        lines.append("")
        print(f"\n[{title}] サンプルなし")
        return

    has_hybrid = hybrid_predictor is not None
    results = evaluate_rows(rows, lr_predictor, nb_predictor, hybrid_predictor)
    summary = aggregate_by_bucket(results, has_hybrid=has_hybrid)

    lines.append(f"- サンプル数: {len(rows)}")
    lines.append("")
    if has_hybrid:
        lines.append(
            "| evidence バケット | サンプル数 | LR top-1 | NB top-1 | hybrid top-1 | "
            "LR log loss | NB log loss | hybrid log loss |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        print(f"\n[{title}] サンプル数={len(rows)}")
        print(
            f"  {'bucket':10s} {'n':>8s} {'LR top1':>10s} {'NB top1':>10s} {'hyb top1':>10s} "
            f"{'LR logloss':>12s} {'NB logloss':>12s} {'hyb logloss':>12s}"
        )
    else:
        lines.append("| evidence バケット | サンプル数 | LR top-1 | NB top-1 | LR log loss | NB log loss |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        print(f"\n[{title}] サンプル数={len(rows)}")
        print(f"  {'bucket':10s} {'n':>8s} {'LR top1':>10s} {'NB top1':>10s} {'LR logloss':>12s} {'NB logloss':>12s}")

    for bucket in _EVIDENCE_BUCKET_ORDER:
        if bucket not in summary:
            continue
        s = summary[bucket]
        if has_hybrid:
            lines.append(
                f"| {bucket} | {s['n']} | {s['lr_acc'] * 100:.2f}% | {s['nb_acc'] * 100:.2f}% | "
                f"{s['hybrid_acc'] * 100:.2f}% | {s['lr_logloss']:.4f} | {s['nb_logloss']:.4f} | "
                f"{s['hybrid_logloss']:.4f} |"
            )
            print(
                f"  {bucket:10s} {s['n']:>8d} {s['lr_acc'] * 100:>9.2f}% {s['nb_acc'] * 100:>9.2f}% "
                f"{s['hybrid_acc'] * 100:>9.2f}% {s['lr_logloss']:>12.4f} {s['nb_logloss']:>12.4f} "
                f"{s['hybrid_logloss']:>12.4f}"
            )
        else:
            lines.append(
                f"| {bucket} | {s['n']} | {s['lr_acc'] * 100:.2f}% | {s['nb_acc'] * 100:.2f}% | "
                f"{s['lr_logloss']:.4f} | {s['nb_logloss']:.4f} |"
            )
            print(
                f"  {bucket:10s} {s['n']:>8d} {s['lr_acc'] * 100:>9.2f}% {s['nb_acc'] * 100:>9.2f}% "
                f"{s['lr_logloss']:>12.4f} {s['nb_logloss']:>12.4f}"
            )

    overall = summary["__all__"]
    if has_hybrid:
        lines.append(
            f"| **全体** | {overall['n']} | {overall['lr_acc'] * 100:.2f}% | {overall['nb_acc'] * 100:.2f}% | "
            f"{overall['hybrid_acc'] * 100:.2f}% | {overall['lr_logloss']:.4f} | {overall['nb_logloss']:.4f} | "
            f"{overall['hybrid_logloss']:.4f} |"
        )
        print(
            f"  {'全体':10s} {overall['n']:>8d} {overall['lr_acc'] * 100:>9.2f}% {overall['nb_acc'] * 100:>9.2f}% "
            f"{overall['hybrid_acc'] * 100:>9.2f}% {overall['lr_logloss']:>12.4f} {overall['nb_logloss']:>12.4f} "
            f"{overall['hybrid_logloss']:>12.4f}"
        )
    else:
        lines.append(
            f"| **全体** | {overall['n']} | {overall['lr_acc'] * 100:.2f}% | {overall['nb_acc'] * 100:.2f}% | "
            f"{overall['lr_logloss']:.4f} | {overall['nb_logloss']:.4f} |"
        )
        print(
            f"  {'全体':10s} {overall['n']:>8d} {overall['lr_acc'] * 100:>9.2f}% {overall['nb_acc'] * 100:>9.2f}% "
            f"{overall['lr_logloss']:>12.4f} {overall['nb_logloss']:>12.4f}"
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(_HERE / "output" / "dataset.jsonl"))
    parser.add_argument("--lr-weights", default=str(_HERE / "output" / "model" / "deck_predictor_weights.json"))
    parser.add_argument("--nb-weights", default=str(_HERE / "output" / "model" / "deck_predictor_nb.json"))
    parser.add_argument("--split", default=str(_HERE / "output" / "model" / "split.json"))
    parser.add_argument(
        "--master-index", default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl")
    )
    parser.add_argument(
        "--hybrid-weights", default=str(_HERE / "output" / "model" / "deck_predictor_hybrid.json")
    )
    parser.add_argument("--report", default=str(_HERE / "output" / "nb_compare_report.md"))
    parser.add_argument("--valid-recent-days", type=int, default=_DEFAULT_VALID_RECENT_DAYS)
    parser.add_argument("--valid-top-rank", type=int, default=_DEFAULT_VALID_TOP_RANK)
    parser.add_argument(
        "--all", action="store_true", help="ウィンドウ絞り込みなしの全体評価もあわせて実行する"
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

    # ハイブリッド重み(fit_hybrid.py の出力)が存在する場合のみ3列比較にする(後方互換)。
    hybrid_predictor: HybridDeckPredictor | None = None
    if Path(args.hybrid_weights).exists():
        hybrid_predictor = HybridDeckPredictor(
            lr_weights_path=args.lr_weights,
            nb_weights_path=args.nb_weights,
            hybrid_config_path=args.hybrid_weights,
        )
        print(f"ハイブリッド重み {args.hybrid_weights} を読み込みました(3列比較モード)")
    else:
        print(f"ハイブリッド重み {args.hybrid_weights} が見つからないため、LR/NBの2列比較のみ実行します")

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
            "--valid-recent-days / --valid-top-rank を緩めるか、--all で全体評価を確認してください。",
            file=sys.stderr,
        )

    lines = ["# 相手デッキ予測器 LR vs NB 比較レポート", ""]
    lines.append(f"- LR重み(adjusted): `{args.lr_weights}`")
    lines.append(f"- NB重み: `{args.nb_weights}`")
    if hybrid_predictor is not None:
        lines.append(f"- hybrid重み: `{args.hybrid_weights}`")
    lines.append(f"- now(基準時刻): {now.isoformat()}")
    lines.append(f"- validation エピソード数(split.json): {len(valid_ids)}")
    lines.append(f"- validation サンプル数(全体、フィルタなし): {len(full_valid_rows)}")
    lines.append(
        f"- validation サンプル数(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内): "
        f"{len(windowed_valid_rows)}"
    )
    lines.append("")
    lines.append(
        "evidence 数(観測エビデンス数)は `MLDeckPredictor.evidence_count()`"
        "(LRの feature_names 語彙基準)を両モデル共通で使用している。"
    )
    lines.append("")

    render_section(
        lines,
        f"固定 validation(直近{args.valid_recent_days}日 x 相手上位{args.valid_top_rank}位以内)",
        windowed_valid_rows,
        lr_predictor,
        nb_predictor,
        hybrid_predictor,
    )

    if args.all:
        render_section(
            lines,
            "全体(ウィンドウ絞り込みなし)",
            full_valid_rows,
            lr_predictor,
            nb_predictor,
            hybrid_predictor,
        )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {out_path} に書き出しました")


if __name__ == "__main__":
    main()
