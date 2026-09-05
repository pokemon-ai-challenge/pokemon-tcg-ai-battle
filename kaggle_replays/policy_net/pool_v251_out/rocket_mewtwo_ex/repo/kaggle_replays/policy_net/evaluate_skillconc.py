#!/usr/bin/env python3
"""Skill Concentration(Step3)専用オフライン評価。

policymodel-skill-concentration-implementation-plan.md Step3:「上位(<=200位) held-out
決定に対する Top-1一致率」を主指標に、複数の候補 `policy_weights_<config>.json` と現行
`policy_weights.json`(ベースライン)を比較する。

評価対象行の定義: 既定(フィルタなし)の `features.npz`(build_features.py の既定実行、
Step2で `rank_at_fetch` フィールドを追加済み)の **test split(split==2)** のうち
`rank_at_fetch` が 1〜200 の行。split は `episode_id` の md5 ハッシュのみで決まる
(build_features.py の `split_for_episode`)ため、A(<=200フィルタ)/B(<=1000フィルタ)/
C(全データ+リウェイト)のどの構成で学習したモデルでも、この一つの held-out 集合で
公平に比較できる(＝各候補固有の `features_config*.npz` からではなく、既定の
`features.npz` から評価用の行を取る)。

各候補の推論は `evaluate.py` の `model_scores_per_row`(numpy、policy_model.py の
`_forward` と同じ計算)をそのまま再利用する(独立実装を重複させない)。

出力: `kaggle_replays/policy_net/evaluate_skillconc_results.json` に全指標を保存。

実行:
    PYTHONIOENCODING=utf-8 python kaggle_replays/policy_net/evaluate_skillconc.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate import model_scores_per_row  # noqa: E402 (既定features.npzのフォワードパス実装を再利用)

_FEATURES = _HERE / "features.npz"
_OUT_PATH = _HERE / "evaluate_skillconc_results.json"

_RANK_MAX = 200
_TEST = 2

# 候補: 名前 -> 重みJSONパス。baseline は本番の既定重み(上書きしない、読むだけ)。
_CANDIDATES = {
    "baseline_current": _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json",
    "configA_rankmax200": _HERE / "policy_weights_configA.json",
    "configB_rankmax1000": _HERE / "policy_weights_configB.json",
    "configC_concentrated_reweight": _HERE / "policy_weights_configC.json",
}


def load_weights(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def top1_accuracy(
    weights: dict,
    state_features: np.ndarray,
    option_features: np.ndarray,
    option_card_ids: np.ndarray,
    chosen_index: np.ndarray,
) -> float:
    scores_per_row = model_scores_per_row(weights, state_features, option_features, option_card_ids)
    pred = np.array([int(np.argmax(s)) for s in scores_per_row], dtype=np.int64)
    return float((pred == chosen_index).mean())


def main() -> None:
    t0 = time.time()
    print(f"features.npz を読み込み: {_FEATURES}")
    data = np.load(_FEATURES, allow_pickle=True)

    if "rank_at_fetch" not in data.files:
        print(
            "エラー: features.npz に rank_at_fetch が無い。"
            "build_features.py(既定引数)で再ビルドしてから実行すること。",
            file=sys.stderr,
        )
        sys.exit(1)

    split = data["split"].astype(np.int64)
    rank_at_fetch = data["rank_at_fetch"].astype(np.int64)
    test_mask = split == _TEST
    top200_mask = test_mask & (rank_at_fetch >= 1) & (rank_at_fetch <= _RANK_MAX)

    n_test = int(test_mask.sum())
    n_top200 = int(top200_mask.sum())
    print(f"test split 全体: {n_test} 件 / うち rank_at_fetch<=200(held-out上位): {n_top200} 件")

    chosen_index_all = data["chosen_index"].astype(np.int64)

    results: dict[str, dict] = {}
    for name, path in _CANDIDATES.items():
        if not path.exists():
            print(f"  [{name}] 重みファイル無し、スキップ: {path}")
            results[name] = {"available": False, "path": str(path)}
            continue
        weights = load_weights(path)

        top200_acc = top1_accuracy(
            weights,
            data["state_features"][top200_mask],
            data["option_features"][top200_mask],
            data["option_card_ids"][top200_mask],
            chosen_index_all[top200_mask],
        )
        pool_acc = top1_accuracy(
            weights,
            data["state_features"][test_mask],
            data["option_features"][test_mask],
            data["option_card_ids"][test_mask],
            chosen_index_all[test_mask],
        )
        results[name] = {
            "available": True,
            "path": str(path),
            "top200_holdout_top1_accuracy": top200_acc,
            "top200_holdout_n": n_top200,
            "pool_test_top1_accuracy": pool_acc,
            "pool_test_n": n_test,
            "meta_test_metrics": weights.get("meta", {}).get("test_metrics"),
        }
        print(
            f"  [{name}] top200_holdout_top1={top200_acc:.4f}  pool_test_top1={pool_acc:.4f}"
        )

    out = {
        "features_path": str(_FEATURES),
        "test_n": n_test,
        "top200_holdout_n": n_top200,
        "candidates": results,
        "elapsed_seconds": time.time() - t0,
    }
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n結果を書き出しました: {_OUT_PATH}")
    print(f"総経過時間: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
