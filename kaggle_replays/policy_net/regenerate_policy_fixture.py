#!/usr/bin/env python3
"""``policy_model_predictions.json``(ゴールデンフィクスチャ)の ``expected_scores`` を再生成する。

背景: ``board_evaluation/attack_features.py`` の打点計算へ防御側の防壁を配線したことで
特徴量エンコードの結果が変わり、``sample_submission/tests/fixtures/policy_model_predictions.json``
の ``expected_scores`` が古くなった。このスクリプトはその期待値だけを更新し、
``observation``/``episode_id`` 等の入力側は一切変更しない。

**独立性の担保**: ``PolicyModel``(推論側, ``sample_submission/ptcg_ai/learning/policy_model.py``)
自身には期待値を作らせない(それをやると ``test_golden_scores_match`` の ``got == expected`` が
定義上必ず成立し、テストが実装のズレを何も検出しなくなる)。代わりに
``kaggle_replays/policy_net/recall_at_k.py`` が持つ、推論側とは別に書かれた numpy ベクトル化
フォワード(``score_all``)を再利用して期待値を計算する。

手順(各 fixture 行について):
  1. ``observation`` を ``cg.api.to_observation_class`` で ``Observation`` にする
     (``test_policy_model.py`` の ``test_golden_scores_match`` と同じ組み立て方: ``logs=[]`` を補う)。
  2. 現在の ``ptcg_ai.learning.encoder`` で state_features / option_features / option_card_ids
     を作る(``PolicyModel.score_options`` と同じエンコード経路。エンコード自体は両者で共有される
     コードなので独立性を損なわない。独立なのは「エンコード後の特徴からスコアを出す
     フォワードパスの実装」の方)。
  3. ``recall_at_k.py`` の ``score_all()``(numpy 実装)でスコアを計算する。
  4. ``expected_scores`` を更新する(他フィールドは不変)。

使用する重みは常に本番 ``sample_submission/ptcg_ai/learning/policy_weights.json``
(読み込むだけで書き換えない)。この重みは ``meta.consequence_fields`` を持たないため
(2026-08-06 時点)、``score_all()`` をそのまま使える。将来 consequence 特徴つきの重みに
差し替える場合は ``score_all()`` 側に連結ロジックを追加する必要がある。

実行:
    python kaggle_replays/policy_net/regenerate_policy_fixture.py [--dry-run]
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

# recall_at_k.py の numpy フォワード(score_all)を再利用する。
# 推論側(PolicyModel)には期待値を作らせない、というこのスクリプトの根幹。
from recall_at_k import score_all  # noqa: E402

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402

_FIXTURE = _SAMPLE_SUBMISSION / "tests" / "fixtures" / "policy_model_predictions.json"
_WEIGHTS = _SAMPLE_SUBMISSION / "ptcg_ai" / "learning" / "policy_weights.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="書き込まず差分だけ表示する")
    parser.add_argument("--fixture", default=str(_FIXTURE))
    parser.add_argument("--weights", default=str(_WEIGHTS))
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    weights_path = Path(args.weights)

    with fixture_path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)

    with weights_path.open("r", encoding="utf-8") as fh:
        weights = json.load(fh)

    consequence_fields = weights.get("meta", {}).get("consequence_fields") or []
    if consequence_fields:
        raise SystemExit(
            "policy_weights.json が meta.consequence_fields を持っています。"
            "score_all() は consequence 特徴の連結に未対応のため、このスクリプトを"
            "拡張してから再実行してください。"
        )

    # C2/T1 の card_id カウント特徴は語彙(meta.hand_card_vocab / meta.opponent_card_vocab)を
    # 渡さないと全0になる。PolicyModel.score_options は meta の語彙を渡してエンコードするので、
    # ここで渡さないと「期待値だけ card_id 特徴が死んだ状態」になり、ゴールデンテストが
    # 実装のズレではなく生成側の手落ちで落ちる(2026-08-11 に実際に発生)。
    hand_card_vocab = weights.get("meta", {}).get("hand_card_vocab") or None
    opponent_card_vocab = weights.get("meta", {}).get("opponent_card_vocab") or None
    if weights.get("meta", {}).get("use_board_set"):
        raise SystemExit(
            "policy_weights.json が meta.use_board_set=true です。score_all() は盤面 Deep Sets "
            "(T2段階1)のプーリングに未対応のため、このスクリプトを拡張してから再実行してください。"
        )

    n_changed = 0
    print(f"{'episode_id':<12}{'n_opt':>6}  {'max|diff|':>12}  変化した選択肢")
    for row in rows:
        obs = to_observation_class({**row["observation"], "logs": []})
        state_features = np.asarray(
            encoder.encode_state_from_state(
                obs.current,
                hand_card_vocab=hand_card_vocab,
                opponent_card_vocab=opponent_card_vocab,
            ),
            dtype=np.float64,
        )
        option_rows = np.asarray(encoder.encode_options_from_state(obs.current, obs.select), dtype=np.float64)
        card_ids = np.asarray(encoder.encode_option_card_ids(obs.current, obs.select), dtype=np.int64)

        new_scores = score_all(weights, state_features, option_rows, card_ids)
        old_scores = row["expected_scores"]

        if len(new_scores) != len(old_scores):
            raise SystemExit(
                f"episode {row.get('episode_id')}: 選択肢数が一致しません "
                f"(旧 {len(old_scores)} 件 vs 新 {len(new_scores)} 件)。入力データの前提が崩れています。"
            )

        diffs = [float(n) - float(o) for n, o in zip(new_scores, old_scores)]
        max_abs = max(abs(d) for d in diffs) if diffs else 0.0
        changed = sum(1 for d in diffs if abs(d) > 1e-9)
        n_changed += changed > 0
        marker = "*" if changed else " "
        print(f"{row.get('episode_id')!s:<12}{len(old_scores):>6}  {max_abs:>12.6f}  {changed}/{len(diffs)} {marker}")

        row["expected_scores"] = [float(s) for s in new_scores]

    print(f"\n{n_changed}/{len(rows)} 局面でスコアが変化")

    if args.dry_run:
        print("\n--dry-run のため書き込みなし")
        return

    with fixture_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n書き込み完了: {fixture_path}")


if __name__ == "__main__":
    main()
