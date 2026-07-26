"""学習パイプライン(policy_prior/train.py)が計算するスコアと、
``ptcg_ai.learning.policy_model.PolicyModel`` が生成済み重みJSONを読んで計算する
スコアが一致することを検証する(オフライン学習側 CLAUDE.md の必須項目)。

ここがズレる典型例:
  - DictVectorizer の feature_names_ の順序と coef_ の列の対応がズレる
  - JSON への書き出し/読み込みで特徴名や重みの対応が崩れる
  - PolicyModel.score() が feature_names に無いキーをどう扱うかの解釈違い

train.py と全く同じ関数(build_examples / DictVectorizer / LogisticRegression / 出力
スキーマ)を小さな合成データセットで実際に動かし、その decision_function のスコアと、
書き出したJSONを PolicyModel でロードして計算したスコアを突き合わせる。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
_POLICY_PRIOR_DIR = _REPO_ROOT / "kaggle_replays" / "policy_prior"
for p in (_SAMPLE_SUBMISSION_DIR, _POLICY_PRIOR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.learning import policy_features  # noqa: E402

sklearn = pytest.importorskip("sklearn")
from sklearn.feature_extraction import DictVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import train as train_module  # noqa: E402

_TOL = 1e-9


def _pokemon(card_id: int, hp: int = 100) -> dict:
    return {"card_id": card_id, "hp": hp, "max_hp": hp, "n_energy": 1, "n_tool": 0}


def _side(active_card_id: int, n_prize: int) -> dict:
    return {
        "active": _pokemon(active_card_id),
        "bench": [],
        "n_prize": n_prize,
        "n_hand": 4,
        "n_deck": 20,
        "discard": [],
        "asleep": False,
        "confused": False,
        "paralyzed": False,
        "poisoned": False,
        "burned": False,
    }


def _state(turn: int, own_active: int, opp_active: int, own_prize: int, opp_prize: int) -> dict:
    return {
        "turn": turn,
        "energy_attached": False,
        "retreated": False,
        "supporter_played": False,
        "stadium_played": False,
        "own": _side(own_active, own_prize),
        "opponent": _side(opp_active, opp_prize),
    }


def _action(option_type: int, card_id: int | None, target_card_id: int | None = None) -> dict:
    return {
        "option_type": option_type,
        "from_area": None,
        "from_player": None,
        "count": None,
        "number": None,
        "attack_id": None,
        "card_id": card_id,
        "target_card_id": target_card_id,
    }


def _make_decision(chosen_index: int, actions: list[dict], state: dict) -> dict:
    return {"chosen": [chosen_index], "actions": actions, "state": state}


def _synthetic_rows() -> list[dict]:
    """20件程度の合成 decision。option_type/card_id が偏るように選ばれた側を作る
    (学習が意味のある係数を持つようにするため。真の精度は問わない、パリティだけ見る)。
    """
    rows = []
    end_action = _action(14, None)
    for i in range(20):
        own_active = 100 + (i % 3)
        opp_active = 200 + (i % 2)
        state = _state(turn=i + 1, own_active=own_active, opp_active=opp_active,
                        own_prize=6 - (i % 6), opp_prize=6 - ((i + 2) % 6))
        play_action = _action(7, card_id=300 + (i % 5))
        attach_action = _action(8, card_id=1, target_card_id=own_active)
        actions = [play_action, attach_action, end_action]
        # 偶数回目は PLAY を選ぶ、奇数回目は ATTACH を選ぶ(学習可能なパターンを作る)
        chosen_index = 0 if i % 2 == 0 else 1
        rows.append(_make_decision(chosen_index, actions, state))
    return rows


def _card_attributes() -> dict[str, dict[str, float]]:
    attrs = {}
    for cid in list(range(100, 103)) + list(range(200, 202)) + list(range(300, 305)) + [1]:
        attrs[str(cid)] = {"hp": float(60 + cid % 40), "stage": float(cid % 3)}
    return attrs


def test_train_time_score_matches_policy_model_loaded_score(tmp_path):
    rows = _synthetic_rows()
    card_attributes = _card_attributes()
    frequent_card_ids = [300, 301, 302, 303, 304, 1]

    feature_dicts, labels, group_ids = train_module.build_examples(
        rows, card_attributes, frequent_card_ids
    )
    assert feature_dicts, "合成データから特徴が作れていない"

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(feature_dicts)
    feature_names = list(vectorizer.feature_names_)

    model = LogisticRegression(max_iter=500)
    model.fit(X, labels)

    # --- 学習時に計算したスコア(decision_function。intercept + w・x そのもの) ---
    train_time_scores = model.decision_function(X)

    payload = {
        "schema_version": 1,
        "model": "linear_pointwise",
        "feature_names": feature_names,
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "frequent_card_ids": frequent_card_ids,
        "card_attributes": card_attributes,
        "meta": {},
    }
    weights_path = tmp_path / "policy_weights.json"
    weights_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # --- PolicyModel が同じJSONを読んで計算したスコア ---
    loaded_model = PolicyModel(weights_path=weights_path)
    assert loaded_model.is_ready

    for feats, expected_score in zip(feature_dicts, train_time_scores):
        actual_score = loaded_model.score(feats)
        assert actual_score == pytest.approx(float(expected_score), abs=_TOL), (
            f"train-time score {expected_score} != PolicyModel score {actual_score}"
        )


def test_select_matches_argmax_of_train_time_scores(tmp_path):
    """decision内のargmax選択も train 側と PolicyModel 側で一致することを確認する。"""
    rows = _synthetic_rows()
    card_attributes = _card_attributes()
    frequent_card_ids = [300, 301, 302, 303, 304, 1]

    feature_dicts, labels, group_ids = train_module.build_examples(
        rows, card_attributes, frequent_card_ids
    )
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(feature_dicts)
    model = LogisticRegression(max_iter=500)
    model.fit(X, labels)

    payload = {
        "schema_version": 1,
        "model": "linear_pointwise",
        "feature_names": list(vectorizer.feature_names_),
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "frequent_card_ids": frequent_card_ids,
        "card_attributes": card_attributes,
        "meta": {},
    }
    weights_path = tmp_path / "policy_weights.json"
    weights_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded_model = PolicyModel(weights_path=weights_path)

    train_time_scores = model.decision_function(X)

    # group_id ごとに train-time の argmax を計算し、PolicyModel.select() の結果と比べる
    per_group: dict[int, list[int]] = {}
    for i, gid in enumerate(group_ids):
        per_group.setdefault(gid, []).append(i)

    for gid, row in zip(sorted(per_group), rows):
        indices = per_group[gid]
        best_local = max(range(len(indices)), key=lambda k: train_time_scores[indices[k]])

        state = row["state"]
        actions = row["actions"]
        selected = loaded_model.select(state, actions)
        assert selected == best_local, (
            f"decision {gid}: train-time argmax={best_local} != PolicyModel.select()={selected}"
        )
