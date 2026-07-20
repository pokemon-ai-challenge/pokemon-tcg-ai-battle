#!/usr/bin/env python3
"""value_positions.jsonl.gz を読み、バリューネットワーク学習用の特徴量アレイ(features.npz)を作る。

各行の observation を ``ptcg_ai.learning.encoder.encode_obs_dict`` で固定長ベクトル化し、
サンプル重み(rank_at_fetch バケット)・train/val/test split(episode_id の md5 ハッシュ)を
付与して numpy の savez で書き出す。

設計は固定(このファイル単体の都合で変更しないこと):
- サンプル重み: rank_bucket() の結果 -> {1-50: 1.5, 51-200: 1.3, 201-1000: 1.1, 1001+: 1.0, 不明: 1.0}
- split: h = md5(episode_id) % 100 -> h<80: train(0) / h<90: val(1) / else: test(2)

encode_obs_dict が例外を投げる行はスキップし、件数をログに出す(全体は止めない)。

player_index / step_index も追加で保存する(スキーマの必須列ではないが、
train.py が sample_predictions.json を作る際に元の value_positions.jsonl.gz の該当行
(observation の生データ)を一意に再特定するために必要)。

使い方:
  python build_features.py
  PYTHONIOENCODING=utf-8 python build_features.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"

# 既存の repo -> sample_submission import パターン(kaggle_replays/deck_predictor/build_dataset.py 踏襲)。
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
sys.path.insert(0, str(_HERE.parent / "deck_predictor"))

from ptcg_ai.learning.encoder import BASE_FEATURE_COUNT, encode_obs_dict  # noqa: E402
from episode_window import rank_bucket  # noqa: E402

_DEFAULT_IN = _HERE.parent / "training_data" / "value_positions.jsonl.gz"
_DEFAULT_OUT = _HERE / "features.npz"

# rank_bucket() の出力(日本語バケット名)->サンプル重み。この対応表は固定(変更しないこと)。
_WEIGHT_BY_RANK_BUCKET: dict[str, float] = {
    "1-50": 1.5,
    "51-200": 1.3,
    "201-1000": 1.1,
    "1001+": 1.0,
    "不明": 1.0,
}

_PROGRESS_EVERY = 50_000


def split_for_episode(episode_id: str) -> int:
    """episode_id の md5 ハッシュから train(0)/val(1)/test(2) を決める(固定式)。"""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return 0
    if h < 90:
        return 1
    return 2


def weight_for_rank(rank_at_fetch: int | None) -> float:
    return _WEIGHT_BY_RANK_BUCKET[rank_bucket(rank_at_fetch)]


def main() -> None:
    in_path = _DEFAULT_IN
    out_path = _DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    turn_rows: list[int] = []
    weight_rows: list[float] = []
    split_rows: list[int] = []
    episode_id_rows: list[str] = []
    player_index_rows: list[int] = []
    step_index_rows: list[int] = []

    n_total = 0
    n_ok = 0
    n_encode_error = 0
    n_other_error = 0
    t0 = time.time()

    print(f"読み込み開始: {in_path}", file=sys.stderr)

    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001 - 1行の壊れたJSONで全体を止めない
                n_other_error += 1
                print(f"  警告: JSON パース失敗(行 {n_total}): {exc!r}", file=sys.stderr)
                continue

            try:
                vec = encode_obs_dict(row["observation"])
            except Exception as exc:  # noqa: BLE001 - encode_obs_dict の例外は行単位でスキップ
                n_encode_error += 1
                if n_encode_error <= 20:
                    print(
                        f"  警告: encode_obs_dict 失敗(行 {n_total}, "
                        f"episode_id={row.get('episode_id')!r}): {exc!r}",
                        file=sys.stderr,
                    )
                continue

            if len(vec) != BASE_FEATURE_COUNT:
                n_encode_error += 1
                print(
                    f"  警告: 特徴ベクトル長不正(行 {n_total}): {len(vec)} != {BASE_FEATURE_COUNT}",
                    file=sys.stderr,
                )
                continue

            episode_id = str(row["episode_id"])
            turn = row.get("turn")
            turn_val = int(turn) if turn is not None else -1

            X_rows.append(vec)
            y_rows.append(int(row["label"]))
            turn_rows.append(turn_val)
            weight_rows.append(weight_for_rank(row.get("rank_at_fetch")))
            split_rows.append(split_for_episode(episode_id))
            episode_id_rows.append(episode_id)
            player_index_rows.append(int(row.get("player_index", -1)))
            step_index_rows.append(int(row.get("step_index", -1)))

            n_ok += 1
            if n_total % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed if elapsed > 0 else 0.0
                print(
                    f"  {n_total} 行処理済み(採用 {n_ok} 件、経過 {elapsed:.1f}s、{rate:.0f} rows/sec)",
                    file=sys.stderr,
                )

    elapsed = time.time() - t0
    print(
        f"完了: 総行数 {n_total} / 採用 {n_ok} / "
        f"encode失敗 {n_encode_error} / その他失敗 {n_other_error}(経過 {elapsed:.1f}s)",
        file=sys.stderr,
    )

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int8)
    turn = np.asarray(turn_rows, dtype=np.int16)
    weight = np.asarray(weight_rows, dtype=np.float32)
    split = np.asarray(split_rows, dtype=np.int8)
    episode_id = np.asarray(episode_id_rows)  # 固定長 unicode dtype(pickle 不要)
    player_index = np.asarray(player_index_rows, dtype=np.int8)
    step_index = np.asarray(step_index_rows, dtype=np.int32)

    print(f"X shape={X.shape} dtype={X.dtype}", file=sys.stderr)
    print(
        f"split 内訳: train={int((split == 0).sum())} "
        f"val={int((split == 1).sum())} test={int((split == 2).sum())}",
        file=sys.stderr,
    )
    print(f"label 内訳: 1={int((y == 1).sum())} 0={int((y == 0).sum())}", file=sys.stderr)

    np.savez(
        out_path,
        X=X,
        y=y,
        turn=turn,
        weight=weight,
        split=split,
        episode_id=episode_id,
        player_index=player_index,
        step_index=step_index,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"書き出し完了: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
