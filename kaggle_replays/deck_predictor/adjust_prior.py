#!/usr/bin/env python3
"""ベース重み(train.py が出力する deck_predictor_weights_base.json)に事前分布シフト補正
(intercept 補正)をかけ、デプロイ用の deck_predictor_weights.json を出力する
「ベースモデル + チューニング層」構成のチューニング層。

  b'_c = b_c + log(pi'_c / pi_c)

  - pi_c  = ベース学習時のクラス事前分布(base weights の meta.class_priors)
  - pi'_c = ターゲット分布 = 「直近 N 日以内」かつ「そのデッキの持ち主チームの
            rank_at_fetch が上位 R 位以内(rank 不明は除外)」のデッキラベル分布。
            deck_labels.jsonl(アーキタイプラベル)と episodes_master.jsonl(作成日時・ランク)を
            episode_id / player_index で結合して算出する。ゼロ件クラスはラプラススムージング
            (加算 alpha、既定 1)で潰さない。

coef はベースのまま変更しない。ランタイム(ml_predictor.py)は無変更 — スキーマは同一で
intercept の値だけが違う。

自己検証: 補正後の重みで observed_cards={}, turn=1 の予測分布を計算し、pi' との全変動距離
(TVD)を表示する。近くなければ補正のバグ(ml-predictor-phase2-scaling.md の
「自己検証則」参照)。

詳細方針は
sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md
の「ベースモデル + チューニング層」節を参照。

使い方:
  python adjust_prior.py
  python adjust_prior.py --recent-days 30 --top-rank 500 --deploy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor  # noqa: E402

from episode_window import EpisodeIndex, in_recent_top_rank_window  # noqa: E402

_DEPLOY_TARGET = (
    _REPO_ROOT / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "deck_predictor_weights.json"
)
_DEFAULT_ALPHA = 1.0
_DEFAULT_RECENT_DAYS = 14
_DEFAULT_TOP_RANK = 200
_DEFAULT_MIN_TARGET_DECKS = 30
_DEFAULT_SELF_CHECK_TURN = 1


def load_deck_labels(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_target_distribution(
    deck_label_rows: list[dict],
    episode_index: EpisodeIndex,
    classes: list[str],
    now: datetime,
    recent_days: int,
    top_rank: int,
    alpha: float,
) -> tuple[dict[str, float], int, Counter]:
    """pi'_c(ターゲット分布)、対象デッキ件数、想定外クラスの内訳を返す。"""
    counts: Counter[str] = Counter()
    unknown_classes: Counter[str] = Counter()

    for row in deck_label_rows:
        episode_id = row["episode_id"]
        player_index = row["player_index"]
        archetype = row["archetype"]

        episode_time = episode_index.time_of(episode_id)
        rank = episode_index.rank_of(episode_id, player_index)
        if not in_recent_top_rank_window(episode_time, rank, now, recent_days, top_rank):
            continue

        if archetype not in classes:
            # base weights の classes に無いアーキタイプ(通常は label_decks.py の定義変更後の
            # 再ラベリング忘れ等でのみ起こる)。ターゲット分布の母数には含めない。
            unknown_classes[archetype] += 1
            continue
        counts[archetype] += 1

    n_target = sum(counts.values())
    n_classes = len(classes)
    denom = n_target + alpha * n_classes
    pi_prime = {c: (counts.get(c, 0) + alpha) / denom for c in classes} if denom > 0 else {c: 0.0 for c in classes}
    return pi_prime, n_target, unknown_classes


def parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument(
        "--master-index", default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl")
    )
    parser.add_argument("--base", default=str(_HERE / "output" / "model" / "deck_predictor_weights_base.json"))
    parser.add_argument("--out", default=str(_HERE / "output" / "model" / "deck_predictor_weights.json"))
    parser.add_argument("--recent-days", type=int, default=_DEFAULT_RECENT_DAYS)
    parser.add_argument("--top-rank", type=int, default=_DEFAULT_TOP_RANK)
    parser.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA, help="ラプラススムージングの加算値")
    parser.add_argument("--min-target-decks", type=int, default=_DEFAULT_MIN_TARGET_DECKS)
    parser.add_argument(
        "--now", default=None, help="基準時刻をISO8601で上書き(再現性確認用)。省略時は実行時のUTC時刻"
    )
    parser.add_argument("--self-check-turn", type=int, default=_DEFAULT_SELF_CHECK_TURN)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="sample_submission/ptcg_ai/opponent_modeling/deck_predictor_weights.json へコピーする",
    )
    args = parser.parse_args()

    now = parse_now(args.now)

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"エラー: base weights が見つかりません: {base_path} (先に train.py を実行してください)", file=sys.stderr)
        sys.exit(1)
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    classes: list[str] = list(base_payload["classes"])
    base_meta: dict = base_payload.get("meta", {})
    pi: dict | None = base_meta.get("class_priors")
    if not pi:
        print(
            f"エラー: {base_path} の meta.class_priors がありません。"
            "train.py を改修版で再実行してベース重みを作り直してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"ベース重み: {base_path}")
    print(f"  trained_at: {base_meta.get('trained_at')}")
    print(f"  クラス数: {len(classes)} -> {classes}")

    deck_label_rows = load_deck_labels(Path(args.deck_labels))
    episode_index = EpisodeIndex.load(Path(args.master_index))

    pi_prime, n_target, unknown = compute_target_distribution(
        deck_label_rows, episode_index, classes, now, args.recent_days, args.top_rank, args.alpha
    )

    print(f"\nターゲット窓: 直近{args.recent_days}日 x 上位{args.top_rank}位以内 (now={now.isoformat()})")
    print(f"  対象デッキ件数: {n_target}")
    if unknown:
        print(
            f"  警告: base のクラスに無いアーキタイプが {sum(unknown.values())} 件ありました(ターゲット分布から除外): "
            f"{dict(unknown)}",
            file=sys.stderr,
        )
    if n_target < args.min_target_decks:
        print(
            f"警告: ターゲット窓のデッキ件数が {n_target} 件と少なすぎます(閾値 {args.min_target_decks})。"
            "--recent-days を増やす、または --top-rank を緩めることを検討してください。",
            file=sys.stderr,
        )

    base_intercept = list(base_payload["intercept"])
    adjusted_intercept: list[float] = []
    print(f"\n{'クラス':30s} {'pi(base)':>10s} {'pi_prime':>10s} {'shift(log)':>10s}")
    for i, c in enumerate(classes):
        pi_c = float(pi.get(c, 0.0))
        pi_prime_c = pi_prime[c]
        if pi_c <= 0.0:
            shift = 0.0
            print(f"警告: クラス '{c}' はベースの class_priors で0のため、シフトをスキップします(補正なし)", file=sys.stderr)
        else:
            # alpha>0 の Laplace スムージングにより pi_prime_c は常に > 0 になる。
            shift = math.log(pi_prime_c / pi_c)
        adjusted_intercept.append(base_intercept[i] + shift)
        print(f"{c:30s} {pi_c:10.4f} {pi_prime_c:10.4f} {shift:10.4f}")

    adjusted_payload = dict(base_payload)
    adjusted_payload["intercept"] = adjusted_intercept
    adjusted_meta = dict(base_meta)
    adjusted_meta["base_trained_at"] = base_meta.get("trained_at")
    adjusted_meta["adjusted_at"] = datetime.now(timezone.utc).isoformat()
    adjusted_meta["target_window"] = {
        "now": now.isoformat(),
        "recent_days": args.recent_days,
        "top_rank": args.top_rank,
        "n_target_decks": n_target,
        "alpha": args.alpha,
        "pi_prime": pi_prime,
    }
    adjusted_payload["meta"] = adjusted_meta

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(adjusted_payload, ensure_ascii=False), encoding="utf-8")
    print(f"\n補正済み重みを {out_path} に書き出しました")

    if args.deploy:
        _DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        _DEPLOY_TARGET.write_text(json.dumps(adjusted_payload, ensure_ascii=False), encoding="utf-8")
        print(f"デプロイ用に {_DEPLOY_TARGET} にもコピーしました")

    # --- 自己検証: observed_cards={}, turn=self-check-turn の予測分布 vs pi' ---
    predictor = MLDeckPredictor(weights_path=str(out_path))
    predicted = predictor.predict({}, args.self_check_turn)
    tvd = 0.5 * sum(abs(predicted.get(c, 0.0) - pi_prime[c]) for c in classes)

    print(f"\n自己検証: observed_cards={{}}, turn={args.self_check_turn} の予測分布 vs pi_prime")
    print(f"{'クラス':30s} {'pi_prime':>10s} {'predicted':>10s} {'diff':>10s}")
    for c in classes:
        p = predicted.get(c, 0.0)
        print(f"{c:30s} {pi_prime[c]:10.4f} {p:10.4f} {p - pi_prime[c]:10.4f}")

    if tvd < 0.05:
        verdict = "OK: pi' に近い"
    elif tvd < 0.15:
        verdict = "要確認: やや乖離あり"
    else:
        verdict = "NG: 乖離が大きい(補正のバグの可能性)"
    print(f"\n全変動距離(TVD) = {tvd:.4f}  [{verdict}]")


if __name__ == "__main__":
    main()
