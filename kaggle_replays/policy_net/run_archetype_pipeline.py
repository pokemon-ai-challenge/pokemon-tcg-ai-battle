#!/usr/bin/env python3
"""1アーキタイプ分の模倣ポリシー学習パイプラインを end-to-end で回すドライバ。

フーディン(alakazam)で確立した本番レシピをそのまま他アーキタイプへ適用するための
薄いオーケストレータ。各アーキタイプについて次の3ステップを順に実行する:

  1. extract_policy_dataset.py --archetype <ARCH>
       -> training_data/policy_positions_<ARCH>.jsonl.gz
  2. build_features.py --weight-scheme concentrated   (= 本番=構成C と同じレシピ)
       -> policy_net/features_<ARCH>.npz
  3. train.py --features features_<ARCH>.npz
       -> sample_submission/ptcg_ai/learning/policy_weights_<ARCH>.json

生成した policy_weights_<ARCH>.json は本番の policy_weights.json を上書きしない。
ml_policy_agent は config["policy_weights_path"] にこのパスを渡すことで、混線なく
アーキタイプ別ポリシーを読み込める(既存の Step4 注入点を再利用)。

--tag を付けると全成果物の名前に接尾辞が付き(policy_positions_<ARCH>_<TAG> /
features_<ARCH>_<TAG> / policy_weights_<ARCH>_<TAG>.json)、リプレイプール・
マスターインデックス・デッキラベルの参照先も切り替えられる。これにより
「以前の模倣学習の成果物を1つも上書きせずに、新しく取得したデータで学習し直す」
(データ世代を並走させる)ことができる。--tag 省略時は従来と完全に同じ挙動。

使い方:
  python run_archetype_pipeline.py --archetype mega_lucario_ex
  python run_archetype_pipeline.py --archetype crustle --skip-extract   # 抽出済みを再利用
  python run_archetype_pipeline.py --archetype dragapult_ex --train-args "--max-epochs 50"
  # 2026-08取得の別データ世代(gen2)で学習し、旧成果物は一切触らない:
  python run_archetype_pipeline.py --archetype alakazam --tag g2 \
      --replays-dir ../replays_g2 --master-index-path ../index_g2/episodes_master.jsonl \
      --deck-labels-path ../deck_predictor/output/deck_labels_g2.jsonl

複数アーキタイプをまとめて回す場合はシェル側でループする(1件失敗しても他に波及
させないため、あえてこのスクリプトは1アーキタイプ=1プロセスに保つ)。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent  # kaggle_replays/policy_net
_KAGGLE_REPLAYS = _HERE.parent
_REPO_ROOT = _KAGGLE_REPLAYS.parent
_LEARNING_DIR = _REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning"


def _run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    """1ステップを実行し、標準出力/エラーをログにも残す。失敗したら例外で止める。"""
    print(f"\n$ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        ret = proc.wait()
    elapsed = time.time() - t0
    if ret != 0:
        raise SystemExit(f"ステップ失敗(exit={ret}, {elapsed:.1f}s): {' '.join(cmd)}\nログ: {log_path}")
    print(f"  完了 ({elapsed:.1f}s) ログ: {log_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archetype", required=True, help="対象アーキタイプ(deck_labels の archetype 値)")
    parser.add_argument("--skip-extract", action="store_true", help="抽出をスキップ(既存の policy_positions_<ARCH> を再利用)")
    parser.add_argument("--skip-features", action="store_true", help="特徴量ビルドをスキップ(既存の features_<ARCH> を再利用)")
    parser.add_argument("--train-args", default="", help="train.py に渡す追加引数(例: '--max-epochs 50')")
    parser.add_argument("--python", default=sys.executable, help="使う python 実行ファイル")
    parser.add_argument(
        "--tag", default="",
        help="データ世代の接尾辞(例: g2)。指定すると成果物が policy_positions_<ARCH>_<TAG> / "
        "features_<ARCH>_<TAG> / policy_weights_<ARCH>_<TAG>.json になり、既存の成果物を上書きしない",
    )
    parser.add_argument("--replays-dir", default=None, help="extract に渡すリプレイプール(既定=extract 側の既定)")
    parser.add_argument("--master-index-path", default=None, help="extract に渡す episodes_master.jsonl")
    parser.add_argument("--deck-labels-path", default=None, help="extract に渡す deck_labels.jsonl")
    args = parser.parse_args()

    arch = args.archetype
    py = args.python
    suffix = f"_{args.tag}" if args.tag else ""

    positions_path = _KAGGLE_REPLAYS / "training_data" / f"policy_positions_{arch}{suffix}.jsonl.gz"
    features_path = _HERE / f"features_{arch}{suffix}.npz"
    weights_path = _LEARNING_DIR / f"policy_weights_{arch}{suffix}.json"
    log_dir = _HERE / "archetype_runs"
    log_dir.mkdir(exist_ok=True)

    print(f"=== アーキタイプ '{arch}'{f' (tag={args.tag})' if args.tag else ''} の模倣学習パイプライン ===")
    print(f"  positions: {positions_path}")
    print(f"  features : {features_path}")
    print(f"  weights  : {weights_path}")

    # 1. 抽出
    if not args.skip_extract:
        extract_cmd = [py, "extract_policy_dataset.py", "--archetype", arch, "--out", str(positions_path)]
        for flag, value in (
            ("--replays-dir", args.replays_dir),
            ("--master-index-path", args.master_index_path),
            ("--deck-labels-path", args.deck_labels_path),
        ):
            if value:
                extract_cmd += [flag, value]
        _run(
            extract_cmd,
            cwd=_KAGGLE_REPLAYS, log_path=log_dir / f"{arch}{suffix}_extract.log",
        )
    else:
        print(f"[skip-extract] {positions_path} を再利用")
    if not positions_path.exists():
        raise SystemExit(f"positions が無い: {positions_path}")

    # 2. 特徴量(本番=構成C と同じ concentrated 重みスキーム)
    if not args.skip_features:
        _run(
            [py, "build_features.py", "--in", str(positions_path),
             "--weight-scheme", "concentrated", "--out", str(features_path)],
            cwd=_HERE, log_path=log_dir / f"{arch}{suffix}_features.log",
        )
    else:
        print(f"[skip-features] {features_path} を再利用")
    if not features_path.exists():
        raise SystemExit(f"features が無い: {features_path}")

    # 3. 学習(重みは policy_weights_<ARCH>.json へ。本番の policy_weights.json は上書きしない)
    train_cmd = [py, "train.py", "--features", str(features_path), "--out-weights", str(weights_path)]
    if args.train_args.strip():
        train_cmd += args.train_args.split()
    _run(train_cmd, cwd=_HERE, log_path=log_dir / f"{arch}{suffix}_train.log")

    print(f"\n=== 完了: {arch} -> {weights_path} ===")


if __name__ == "__main__":
    main()
