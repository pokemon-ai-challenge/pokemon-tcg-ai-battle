#!/usr/bin/env python3
"""policy_positions.jsonl.gz を読み、模倣ポリシー学習用の特徴量アレイ(features.npz)を作る。

各行(1意思決定点)について:
  - obs = to_observation_class({**row["observation"], "logs": []})
  - state_features = encoder.encode_state(obs)  (長さ BASE_FEATURE_COUNT)
  - option_features = encoder.encode_options(obs)  (長さ n_options のリスト。各要素は
    長さ OPTION_FEATURE_COUNT)
  - option_card_ids = encoder.encode_option_card_ids(obs.current, obs.select)  (長さ n_options の
    int リスト。card_embedding 用の生の card_id キー列。2026-07-20 追加、policy_model.py の
    card_embedding 対応)
  - split = split_for_episode(row["episode_id"])  (value_net/build_features.py と同じ md5 式)
  - weight = weight_for_rank(row["rank_at_fetch"])  (value_net と同じ rank_bucket テーブル)

card_id_max は ``cg.api.all_card_data()`` から動的に計算し(ハードコードしない。デッキ・
カードデータが変わっても再学習だけで使い回せるという policy_model.py 側の設計要件)、
features.npz にスカラーとして保存する。card_embedding テーブルのサイズ(card_id_max + 1)を
train.py 側が決めるのに使う。

設計は固定(このファイル単体の都合で変更しないこと。step2-design.md §4 参照):
- サンプル重み: rank_bucket() の結果 -> {1-50: 1.5, 51-200: 1.3, 201-1000: 1.1, 1001+: 1.0, 不明: 1.0}
- split: h = md5(episode_id) % 100 -> h<80: train(0) / h<90: val(1) / else: test(2)

重要な制約: 行の順序は変えない(シャッフルしない)。features.npz の行 i は
policy_positions.jsonl.gz の i 行目(0-indexed、空行を除く)に対応する。evaluate.py が
後で rule_based_agent の再実行のために元の observation(同じ行番号)を突き合わせるのに
必要。したがって、value_net/build_features.py と異なり **encode 失敗行を黙ってスキップ
しない**(スキップすると row_index の対応関係が崩れる)。encode_state/encode_options は
仕様上「完成済みで正しく動作確認済み」の前提のため、失敗した場合は詳細をログした上で
例外を再送出しビルドを止める(fail-fast)。

使い方:
  python build_features.py                  # 全件処理
  python build_features.py --limit 2000      # 先頭2000行のみ(動作確認用)
  python build_features.py --in ... --out ... --limit ...

skill-concentration実験用フラグ(policymodel-skill-concentration-implementation-plan.md Step2。
既定値は上記の固定設計のまま = フラグを付けなければ既存の挙動と完全に同じ):
  --rank-max N        rank_at_fetch が N を超える行(不明=Noneも含む)を除外するハードフィルタ。
                       既定は None(フィルタなし)。
  --weight-scheme {default,concentrated}
                       default(既定)は上記の _WEIGHT_BY_RANK_BUCKET をそのまま使う。
                       concentrated は rank_at_fetch の生値に基づく専用スキーム
                       (weight_for_rank_concentrated、rank<=20:8.0 / <=50:3.0 / <=200:1.5 /
                       <=1000:1.0 / 1000超:0.5 / 不明:0.2)。既存の rank_bucket() より粒度が
                       細かく、上位への勾配集中を強める実験用(design.md §4 の固定表は変更しない
                       別関数として追加)。
"""

from __future__ import annotations

import argparse
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

# 既存の repo -> sample_submission import パターン(kaggle_replays/value_net/build_features.py 踏襲)。
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
sys.path.insert(0, str(_HERE.parent / "deck_predictor"))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from ptcg_ai.learning.encoder import (  # noqa: E402
    BASE_FEATURE_COUNT,
    OPTION_FEATURE_COUNT,
    encode_option_card_ids,
    encode_options,
    encode_state,
)
from episode_window import rank_bucket  # noqa: E402

_DEFAULT_IN = _HERE.parent / "training_data" / "policy_positions.jsonl.gz"
_DEFAULT_OUT = _HERE / "features.npz"

# rank_bucket() の出力(日本語バケット名)->サンプル重み。value_net と同じ対応表(固定)。
_WEIGHT_BY_RANK_BUCKET: dict[str, float] = {
    "1-50": 1.5,
    "51-200": 1.3,
    "201-1000": 1.1,
    "1001+": 1.0,
    "不明": 1.0,
}

_PROGRESS_EVERY = 5_000


def split_for_episode(episode_id: str) -> int:
    """episode_id の md5 ハッシュから train(0)/val(1)/test(2) を決める(value_net と同じ固定式)。"""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return 0
    if h < 90:
        return 1
    return 2


def weight_for_rank(rank_at_fetch: int | None) -> float:
    return _WEIGHT_BY_RANK_BUCKET[rank_bucket(rank_at_fetch)]


def weight_for_rank_concentrated(rank_at_fetch: int | None) -> float:
    """--weight-scheme concentrated 用。rank_bucket() より粒度の細かい専用テーブル
    (skill-concentration実験専用。既定の weight_for_rank は変更しない)。"""
    if rank_at_fetch is None:
        return 0.2
    if rank_at_fetch <= 20:
        return 8.0
    if rank_at_fetch <= 50:
        return 3.0
    if rank_at_fetch <= 200:
        return 1.5
    if rank_at_fetch <= 1000:
        return 1.0
    return 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default=str(_DEFAULT_IN))
    parser.add_argument("--out", dest="out_path", default=str(_DEFAULT_OUT))
    parser.add_argument(
        "--limit", type=int, default=None, help="先頭N行のみ処理する(動作確認モード)"
    )
    parser.add_argument(
        "--rank-max", type=int, default=None,
        help="rank_at_fetch がこの値を超える行(不明含む)を除外するハードフィルタ(既定: フィルタなし)",
    )
    parser.add_argument(
        "--weight-scheme", choices=["default", "concentrated"], default="default",
        help="サンプル重みスキーム(既定: default = 固定の rank_bucket テーブル)",
    )
    args = parser.parse_args()
    weight_fn = weight_for_rank_concentrated if args.weight_scheme == "concentrated" else weight_for_rank

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    card_id_max = max(c.cardId for c in all_card_data())
    print(f"card_id_max = {card_id_max}(all_card_data() から動的に計算)", file=sys.stderr)

    state_feature_rows: list[list[float]] = []
    option_feature_rows: list[np.ndarray] = []
    option_card_id_rows: list[np.ndarray] = []
    chosen_index_rows: list[int] = []
    split_rows: list[int] = []
    weight_rows: list[float] = []
    turn_rows: list[int] = []
    select_type_rows: list[int] = []
    select_context_rows: list[int] = []
    rank_rows: list[int] = []
    row_index_rows: list[int] = []

    n_total = 0  # フィルタ後の採用行数(= 出力行数)
    n_seen_lines = 0
    line_no = -1  # 空行を除いた0-indexed行番号(evaluate.pyの数え方と同じ。フィルタの影響を受けない)
    t0 = time.time()

    print(f"読み込み開始: {in_path}", file=sys.stderr)
    if args.limit is not None:
        print(f"動作確認モード: 先頭 {args.limit} 行のみ処理", file=sys.stderr)
    if args.rank_max is not None:
        print(f"rank フィルタ: rank_at_fetch<= {args.rank_max}(不明含め超過行は除外)", file=sys.stderr)
    if args.weight_scheme != "default":
        print(f"weight スキーム: {args.weight_scheme}", file=sys.stderr)

    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_seen_lines += 1
            line = line.strip()
            if not line:
                continue
            line_no += 1
            if args.limit is not None and n_total >= args.limit:
                break

            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"エラー: JSON パース失敗(生ファイル行 {n_seen_lines}): {exc!r}",
                    file=sys.stderr,
                )
                raise

            rank_at_fetch = row.get("rank_at_fetch")
            if args.rank_max is not None and (rank_at_fetch is None or rank_at_fetch > args.rank_max):
                continue  # フィルタで除外(line_no は既にインクリメント済みなので元ファイルの行番号との対応は崩れない)

            row_index = line_no  # 元ファイル(policy_positions.jsonl.gz)での0-indexed行番号

            try:
                obs_dict = {**row["observation"], "logs": []}
                obs = to_observation_class(obs_dict)
                state_feats = encode_state(obs)
                option_feats = encode_options(obs)
                option_card_ids = encode_option_card_ids(obs.current, obs.select)
            except Exception as exc:  # noqa: BLE001 - fail-fast(row_index 対応関係を壊さない)
                print(
                    f"エラー: encode 失敗(row_index={row_index}, "
                    f"episode_id={row.get('episode_id')!r}, step_index={row.get('step_index')!r}): "
                    f"{exc!r}",
                    file=sys.stderr,
                )
                raise

            if len(state_feats) != BASE_FEATURE_COUNT:
                raise ValueError(
                    f"state特徴ベクトル長不正(row_index={row_index}): "
                    f"{len(state_feats)} != {BASE_FEATURE_COUNT}"
                )
            n_options = row["n_options"]
            if len(option_feats) != n_options:
                raise ValueError(
                    f"選択肢数不一致(row_index={row_index}): "
                    f"encode_options={len(option_feats)} != row.n_options={n_options}"
                )
            for opt_vec in option_feats:
                if len(opt_vec) != OPTION_FEATURE_COUNT:
                    raise ValueError(
                        f"option特徴ベクトル長不正(row_index={row_index}): "
                        f"{len(opt_vec)} != {OPTION_FEATURE_COUNT}"
                    )
            if len(option_card_ids) != n_options:
                raise ValueError(
                    f"option_card_ids 長不正(row_index={row_index}): "
                    f"{len(option_card_ids)} != n_options={n_options}"
                )

            chosen_index = int(row["chosen_index"])
            if not (0 <= chosen_index < n_options):
                raise ValueError(
                    f"chosen_index が範囲外(row_index={row_index}): "
                    f"{chosen_index} not in [0, {n_options})"
                )

            episode_id = str(row["episode_id"])

            state_feature_rows.append(state_feats)
            option_feature_rows.append(np.asarray(option_feats, dtype=np.float32))
            option_card_id_rows.append(np.asarray(option_card_ids, dtype=np.int32))
            chosen_index_rows.append(chosen_index)
            split_rows.append(split_for_episode(episode_id))
            weight_rows.append(weight_fn(rank_at_fetch))
            turn_rows.append(int(row.get("turn", -1)))
            select_type_rows.append(int(row["select_type"]))
            select_context_rows.append(int(row["select_context"]))
            rank_rows.append(rank_at_fetch if rank_at_fetch is not None else -1)
            row_index_rows.append(row_index)

            n_total += 1
            if n_total % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed if elapsed > 0 else 0.0
                print(
                    f"  {n_total} 件処理済み(経過 {elapsed:.1f}s、{rate:.1f} rows/sec)",
                    file=sys.stderr,
                )

    elapsed = time.time() - t0
    print(f"完了: 採用 {n_total} 件(経過 {elapsed:.1f}s)", file=sys.stderr)

    state_features = np.asarray(state_feature_rows, dtype=np.float32)
    # ragged なので object 配列(dtype=object)にする。読み込み側は allow_pickle=True が必要。
    option_features = np.empty(n_total, dtype=object)
    for i, arr in enumerate(option_feature_rows):
        option_features[i] = arr
    option_card_ids_arr = np.empty(n_total, dtype=object)
    for i, arr in enumerate(option_card_id_rows):
        option_card_ids_arr[i] = arr
    chosen_index = np.asarray(chosen_index_rows, dtype=np.int32)
    split = np.asarray(split_rows, dtype=np.int8)
    weight = np.asarray(weight_rows, dtype=np.float32)
    turn = np.asarray(turn_rows, dtype=np.int32)
    select_type = np.asarray(select_type_rows, dtype=np.int32)
    select_context = np.asarray(select_context_rows, dtype=np.int32)
    rank_at_fetch_arr = np.asarray(rank_rows, dtype=np.int32)  # -1 = 不明(rank_at_fetch is None)
    row_index = np.asarray(row_index_rows, dtype=np.int32)  # 元ファイルでの0-indexed行番号(rank-maxフィルタ時はarangeと異なる)

    print(f"state_features shape={state_features.shape} dtype={state_features.dtype}", file=sys.stderr)
    print(
        f"split 内訳: train={int((split == 0).sum())} "
        f"val={int((split == 1).sum())} test={int((split == 2).sum())}",
        file=sys.stderr,
    )

    np.savez_compressed(
        out_path,
        state_features=state_features,
        option_features=option_features,
        option_card_ids=option_card_ids_arr,
        card_id_max=np.array(card_id_max),
        chosen_index=chosen_index,
        split=split,
        weight=weight,
        turn=turn,
        select_type=select_type,
        select_context=select_context,
        row_index=row_index,
        rank_at_fetch=rank_at_fetch_arr,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"書き出し完了: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    print(
        "注意: option_features/option_card_ids は object 配列(ragged)。読み込みは "
        "np.load(path, allow_pickle=True) を使うこと。card_id_max はスカラー"
        "(data['card_id_max'].item() で取り出す)。",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
