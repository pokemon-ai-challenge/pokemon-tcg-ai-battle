#!/usr/bin/env python3
"""``sample_predictions.json``(ゴールデンフィクスチャ)の ``expected_win_prob`` を再生成する。

背景: ``board_evaluation/attack_features.py`` の打点計算へ防御側の防壁を配線した
(``resolve_damage`` に ``defender``/``damage_is_effect`` を渡す)ことで特徴量エンコードの
結果が変わり、``sample_predictions.json`` の ``expected_win_prob`` が古くなった。このスクリプトは
その期待値だけを更新し、``observation``/``episode_id``/``turn`` 等の入力側は一切変更しない。

**独立性の担保**: ``ValueModel``(推論側, ``sample_submission/ptcg_ai/learning/value_model.py``)
自身には期待値を作らせない(それをやると ``test_golden_predictions_match`` の
``got == expected`` が定義上必ず成立し、テストが実装のズレを何も検出しなくなる)。
代わりに ``kaggle_replays/value_net/evaluate.py`` が持つ、推論側とは別に書かれた
numpy ベクトル化フォワード(``vectorized_raw_proba`` + ``calibrate``)を再利用して期待値を計算する。

手順:
  1. ``sample_predictions.json`` の各行の ``observation``(生 dict)を、現在の
     ``ptcg_ai.learning.encoder.encode_obs_dict`` で特徴ベクトル化する
     (推論側の ``ValueModel.predict_win_prob_from_dict`` と同じエンコード経路。
     エンコード自体は両者で共有されるコードなので独立性を損なわない。独立なのは
     「エンコード後の特徴ベクトルから確率を出すフォワードパスの実装」の方)。
  2. ``evaluate.py`` の ``vectorized_raw_proba`` + ``calibrate``(numpy 実装)で
     較正後確率を計算する。
  3. ``expected_win_prob`` を更新する(他フィールドは不変)。

使用する重みは常に本番 ``sample_submission/ptcg_ai/learning/value_weights.json``
(読み込むだけで書き換えない)。

実行:
    python kaggle_replays/value_net/regenerate_sample_predictions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION = _REPO_ROOT / "sample_submission"
if str(_SAMPLE_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_SUBMISSION))

# evaluate.py の numpy フォワード(vectorized_raw_proba/calibrate)を再利用する。
# 推論側(ValueModel)には期待値を作らせない、というこのスクリプトの根幹。
from evaluate import calibrate, load_weights, vectorized_raw_proba  # noqa: E402

from ptcg_ai.learning import encoder  # noqa: E402

_SAMPLE_PREDICTIONS = _HERE / "sample_predictions.json"
_WEIGHTS = _SAMPLE_SUBMISSION / "ptcg_ai" / "learning" / "value_weights.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="書き込まず差分だけ表示する")
    parser.add_argument("--predictions", default=str(_SAMPLE_PREDICTIONS))
    parser.add_argument("--weights", default=str(_WEIGHTS))
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    weights_path = Path(args.weights)

    with predictions_path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)

    weights = load_weights(weights_path)

    # 各行の observation を現在のエンコーダで特徴ベクトル化(X)し、ターンを取り出す。
    X = np.array(
        [encoder.encode_obs_dict(row["observation"]) for row in rows],
        dtype=np.float64,
    )
    turns = np.array([int(row["turn"]) for row in rows], dtype=np.int64)

    p_raw = vectorized_raw_proba(X, weights)
    p_cal = calibrate(p_raw, turns, weights)

    diffs = []
    for row, new_p in zip(rows, p_cal):
        old_p = row["expected_win_prob"]
        diffs.append((row.get("episode_id"), row.get("turn"), old_p, float(new_p), float(new_p) - old_p))
        row["expected_win_prob"] = float(new_p)

    print(f"{'episode_id':<12}{'turn':>5}  {'old':>10}  {'new':>10}  {'diff':>10}")
    n_changed = 0
    for ep, turn, old_p, new_p, d in diffs:
        changed = abs(d) > 1e-9
        n_changed += changed
        marker = "*" if changed else " "
        print(f"{ep!s:<12}{turn:>5}  {old_p:>10.6f}  {new_p:>10.6f}  {d:>+10.6f} {marker}")
    print(f"\n{n_changed}/{len(diffs)} 件が変化")

    if args.dry_run:
        print("\n--dry-run のため書き込みなし")
        return

    with predictions_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n書き込み完了: {predictions_path}")


if __name__ == "__main__":
    main()
