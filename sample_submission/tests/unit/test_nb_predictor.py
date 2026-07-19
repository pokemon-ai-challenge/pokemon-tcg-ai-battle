"""nb_predictor.NBDeckPredictor のユニットテスト。

deck_predictor_nb.json は学習パイプライン(kaggle_replays/deck_predictor/train_nb.py)の
成果物であり、このリポジトリにはまだ配置されていない。そのため各テストは tmp_path に
合成した小さな重みJSONを書き出し、それを weights_path として渡す形で検証する
(test_ml_predictor.py と同じスタイル)。
"""

from pathlib import Path
import json
import math
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.opponent_modeling.nb_predictor import NBDeckPredictor


# 合成語彙: 3クラス x 3カード名。
#   - SignatureA: mega_lucario_ex 専用アンカー(他クラスではほぼ0)
#   - SignatureB: alakazam 専用アンカー(他クラスではほぼ0)
#   - CommonCard: 全クラスで採用率が全く同じ(汎用カード)
CLASSES = ["mega_lucario_ex", "alakazam", "other"]
PRIORS = [0.5, 0.3, 0.2]
CARD_NAMES = ["SignatureA", "SignatureB", "CommonCard"]

_COMMON_PROBS = [0.5, 0.4, 0.3, 0.2]  # 全クラスで同一の [P(>=1), P(>=2), P(>=3), P(>=4)]

LIKELIHOODS = {
    "SignatureA": {
        "mega_lucario_ex": [0.9, 0.85, 0.8, 0.75],
        "alakazam": [0.02, 0.01, 0.005, 0.001],
        "other": [0.02, 0.01, 0.005, 0.001],
    },
    "SignatureB": {
        "mega_lucario_ex": [0.02, 0.01, 0.005, 0.001],
        "alakazam": [0.9, 0.85, 0.8, 0.75],
        "other": [0.02, 0.01, 0.005, 0.001],
    },
    "CommonCard": {
        "mega_lucario_ex": list(_COMMON_PROBS),
        "alakazam": list(_COMMON_PROBS),
        "other": list(_COMMON_PROBS),
    },
}


def _write_weights(
    tmp_path: Path,
    *,
    classes=None,
    priors=None,
    card_names=None,
    likelihoods=None,
    max_k=4,
    filename="deck_predictor_nb.json",
) -> Path:
    payload = {
        "classes": classes if classes is not None else CLASSES,
        "priors": priors if priors is not None else PRIORS,
        "card_names": card_names if card_names is not None else CARD_NAMES,
        "likelihoods": likelihoods if likelihoods is not None else LIKELIHOODS,
        "meta": {
            "trained_at": "test",
            "n_decks_per_class": {"mega_lucario_ex": 10, "alakazam": 10, "other": 10},
            "alpha": 1.0,
            "max_k": max_k,
            "prior_window": {"recent_days": 14, "top_rank": 200, "n_window_decks": 10},
            "version": 1,
        },
    }
    weights_path = tmp_path / filename
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


# --- 未ロード時の挙動 ---


def test_missing_weights_file_is_not_ready_and_returns_other(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    predictor = NBDeckPredictor(weights_path=missing_path)

    assert predictor.is_ready is False
    assert predictor.predict({"SignatureA": 1}, turn=5) == {"other": 1.0}


def test_default_weights_path_resolves_relative_to_module(tmp_path):
    # weights_path を渡さない場合、同ディレクトリの deck_predictor_nb.json を見に行く。
    # 学習パイプライン完了前の現時点ではファイルが存在しないため、未ロード状態になるはず。
    predictor = NBDeckPredictor()
    module_dir = Path(__file__).resolve().parents[2] / "ptcg_ai" / "opponent_modeling"
    expected_path = module_dir / "deck_predictor_nb.json"

    if expected_path.exists():
        assert predictor.is_ready is True
    else:
        assert predictor.is_ready is False
        assert predictor.predict({}, turn=1) == {"other": 1.0}


# --- 確率が合計1 ---


def test_predict_probabilities_sum_to_one(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    probs = predictor.predict({"SignatureA": 1}, turn=3)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert set(probs.keys()) == set(CLASSES)


# --- 専用カード観測でそのクラスが支配的 ---


def test_predict_dominant_class_matches_signature_card(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))

    probs_lucario = predictor.predict({"SignatureA": 2}, turn=3)
    top_lucario = max(probs_lucario, key=probs_lucario.get)
    assert top_lucario == "mega_lucario_ex"
    assert probs_lucario[top_lucario] > 0.9  # 専用アンカーカードなので強く確信してよい

    probs_alakazam = predictor.predict({"SignatureB": 1}, turn=3)
    top_alakazam = max(probs_alakazam, key=probs_alakazam.get)
    assert top_alakazam == "alakazam"
    assert probs_alakazam[top_alakazam] > 0.9


# --- 全クラス共通カード観測では posterior ≈ prior のまま ---


def test_predict_common_card_keeps_posterior_close_to_prior(tmp_path):
    """全クラスで尤度が完全に同一のカードは、対数尤度が定数シフトとして全クラスに
    均等にかかるだけなので、正規化後は posterior が prior と厳密に一致する。"""
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    probs = predictor.predict({"CommonCard": 3}, turn=5)

    prior_by_class = dict(zip(CLASSES, PRIORS))
    for cls in CLASSES:
        assert math.isclose(probs[cls], prior_by_class[cls], abs_tol=1e-9)


# --- 観測なしなら posterior = prior ---


def test_predict_with_empty_observed_cards_equals_prior(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    probs = predictor.predict({}, turn=1)

    prior_by_class = dict(zip(CLASSES, PRIORS))
    for cls in CLASSES:
        assert math.isclose(probs[cls], prior_by_class[cls], abs_tol=1e-9)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


# --- 語彙外カード無視 ---


def test_predict_ignores_unknown_card_names(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    # "ネストボール" は card_names に無いので無視され、例外にならないこと。
    probs_with_unknown = predictor.predict({"SignatureA": 1, "ネストボール": 4}, turn=2)
    probs_without_unknown = predictor.predict({"SignatureA": 1}, turn=2)

    assert math.isclose(sum(probs_with_unknown.values()), 1.0, abs_tol=1e-9)
    for cls in CLASSES:
        assert math.isclose(probs_with_unknown[cls], probs_without_unknown[cls], abs_tol=1e-9)


# --- 5枚以上は k=4 扱い ---


def test_observed_count_of_five_or_more_uses_k4_likelihood(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))

    probs_four = predictor.predict({"SignatureA": 4}, turn=3)
    probs_five = predictor.predict({"SignatureA": 5}, turn=3)
    probs_hundred = predictor.predict({"SignatureA": 100}, turn=3)

    for cls in CLASSES:
        assert math.isclose(probs_four[cls], probs_five[cls], abs_tol=1e-9)
        assert math.isclose(probs_four[cls], probs_hundred[cls], abs_tol=1e-9)


def test_explain_reports_k_used_capped_at_max_k(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    explanation = predictor.explain({"SignatureA": 7}, turn=3)

    assert explanation["cards"]["SignatureA"]["observed_count"] == 7
    assert explanation["cards"]["SignatureA"]["k_used"] == 4
    assert "ネストボール" not in explanation["cards"]


def test_explain_lists_unknown_cards_as_ignored(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    explanation = predictor.explain({"SignatureA": 1, "ネストボール": 2}, turn=3)

    assert "ネストボール" in explanation["ignored_cards"]
    assert "SignatureA" in explanation["cards"]
    assert math.isclose(sum(explanation["probs"].values()), 1.0, abs_tol=1e-9)


# --- predict_top の並び ---


def test_predict_top_order_and_count(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    top = predictor.predict_top({"SignatureA": 3}, turn=4, n=2)

    assert len(top) == 2
    assert top[0][0] == "mega_lucario_ex"
    # 降順に並んでいること。
    assert top[0][1] >= top[1][1]
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in top)


def test_predict_top_n_larger_than_class_count_returns_all(tmp_path):
    predictor = NBDeckPredictor(weights_path=_write_weights(tmp_path))
    top = predictor.predict_top({}, turn=1, n=10)
    assert len(top) == len(CLASSES)
