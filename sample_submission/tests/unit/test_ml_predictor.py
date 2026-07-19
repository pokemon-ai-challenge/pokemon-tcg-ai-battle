"""ml_predictor.MLDeckPredictor のユニットテスト。

deck_predictor_weights.json は学習パイプライン(kaggle_replays/deck_predictor/)の
成果物であり、このリポジトリにはまだ配置されていない。そのため各テストは
tmp_path に合成した小さな重みJSONを書き出し、それを weights_path として渡す形で検証する。
"""

from pathlib import Path
import json
import math
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor


# 合成語彙: __turn__ + 3枚のカード名。3クラス(2アーキタイプ + other)。
FEATURE_NAMES = ["__turn__", "メガルカリオex", "フーディン", "ドラパルトex"]
CLASSES = ["mega_lucario_ex", "alakazam", "other"]


def _write_weights(tmp_path: Path, *, coef=None, intercept=None, classes=None, feature_names=None) -> Path:
    payload = {
        "meta": {"trained_at": "test", "n_episodes": 1, "n_samples": 1, "feature_type": "test", "version": 1},
        "classes": classes if classes is not None else CLASSES,
        "feature_names": feature_names if feature_names is not None else FEATURE_NAMES,
        "coef": coef if coef is not None else [
            # __turn__, メガルカリオex, フーディン, ドラパルトex
            [0.0, 5.0, 0.0, 0.0],   # mega_lucario_ex: strongly driven by メガルカリオex
            [0.0, 0.0, 5.0, 0.0],   # alakazam: strongly driven by フーディン
            [0.0, 0.0, 0.0, 0.0],   # other: baseline
        ],
        "intercept": intercept if intercept is not None else [0.0, 0.0, 0.0],
    }
    weights_path = tmp_path / "deck_predictor_weights.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


def test_predict_probabilities_sum_to_one(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    probs = predictor.predict({"メガルカリオex": 1}, turn=3)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert set(probs.keys()) == set(CLASSES)


def test_predict_argmax_matches_observed_signature_card(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))

    probs_lucario = predictor.predict({"メガルカリオex": 2}, turn=3)
    assert max(probs_lucario, key=probs_lucario.get) == "mega_lucario_ex"

    probs_alakazam = predictor.predict({"フーディン": 1}, turn=3)
    assert max(probs_alakazam, key=probs_alakazam.get) == "alakazam"


def test_predict_ignores_unknown_card_names(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    # "ネストボール" は feature_names に無いので無視され、例外にならないこと。
    probs = predictor.predict({"メガルカリオex": 1, "ネストボール": 4}, turn=2)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_predict_with_empty_observed_cards(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    probs = predictor.predict({}, turn=1)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert set(probs.keys()) == set(CLASSES)


def test_missing_weights_file_is_not_ready_and_returns_other(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    predictor = MLDeckPredictor(weights_path=missing_path)

    assert predictor.is_ready is False
    assert predictor.predict({"メガルカリオex": 1}, turn=5) == {"other": 1.0}


def test_default_weights_path_resolves_relative_to_module(tmp_path):
    # weights_path を渡さない場合、同ディレクトリの deck_predictor_weights.json を見に行く。
    # 学習パイプライン完了前の現時点ではファイルが存在しないため、未ロード状態になるはず。
    predictor = MLDeckPredictor()
    module_dir = Path(__file__).resolve().parents[2] / "ptcg_ai" / "opponent_modeling"
    expected_path = module_dir / "deck_predictor_weights.json"

    if expected_path.exists():
        assert predictor.is_ready is True
    else:
        assert predictor.is_ready is False
        assert predictor.predict({}, turn=1) == {"other": 1.0}


def test_predict_top_order_and_count(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    top = predictor.predict_top({"メガルカリオex": 3}, turn=4, n=2)

    assert len(top) == 2
    assert top[0][0] == "mega_lucario_ex"
    # 降順に並んでいること。
    assert top[0][1] >= top[1][1]
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in top)


def test_predict_top_n_larger_than_class_count_returns_all(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    top = predictor.predict_top({}, turn=1, n=10)
    assert len(top) == len(CLASSES)


# --- キャリブレーション(温度スケーリング) ---


def _write_weights_with_calibration(tmp_path: Path, buckets: list[dict]) -> Path:
    payload = {
        "meta": {
            "trained_at": "test",
            "n_episodes": 1,
            "n_samples": 1,
            "feature_type": "test",
            "version": 1,
            "calibration": {
                "method": "evidence_count_bucketed_temperature_scaling",
                "fitted_at": "test",
                "fitted_on": "test",
                "buckets": buckets,
            },
        },
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "coef": [
            # __turn__, メガルカリオex, フーディン, ドラパルトex
            [0.0, 5.0, 0.0, 0.0],   # mega_lucario_ex: strongly driven by メガルカリオex
            [0.0, 0.0, 5.0, 0.0],   # alakazam: strongly driven by フーディン
            [0.0, 0.0, 0.0, 0.0],   # other: baseline
        ],
        "intercept": [0.0, 0.0, 0.0],
    }
    weights_path = tmp_path / "deck_predictor_weights.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


_CALIBRATION_BUCKETS = [
    {"min_evidence": 0, "max_evidence": 0, "temperature": 1.0, "n_valid_samples": 1, "logloss_before": 0.0,
     "logloss_after": 0.0, "top1_acc": 1.0},
    {"min_evidence": 1, "max_evidence": 2, "temperature": 4.0, "n_valid_samples": 1, "logloss_before": 0.0,
     "logloss_after": 0.0, "top1_acc": 1.0},
    {"min_evidence": 3, "max_evidence": None, "temperature": 1.0, "n_valid_samples": 1, "logloss_before": 0.0,
     "logloss_after": 0.0, "top1_acc": 1.0},
]


def test_missing_calibration_meta_behaves_as_before(tmp_path):
    """meta.calibration の無い(旧)重みJSONは今まで通り動く(回帰確認)。"""
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    assert predictor._calibration_buckets == []
    assert predictor._temperature_for(0) == 1.0
    assert predictor._temperature_for(5) == 1.0

    probs_uncalibrated = predictor.predict({"メガルカリオex": 1}, turn=3)
    z = predictor._raw_logits({"メガルカリオex": 1}, turn=3)
    from ptcg_ai.opponent_modeling.ml_predictor import _softmax

    expected = dict(zip(CLASSES, _softmax(z)))
    for cls in CLASSES:
        assert math.isclose(probs_uncalibrated[cls], expected[cls], abs_tol=1e-9)


def test_calibration_evidence_zero_matches_uncalibrated(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights_with_calibration(tmp_path, _CALIBRATION_BUCKETS))
    observed = {}  # evidence_count == 0 -> T=1.0
    assert predictor.evidence_count(observed) == 0
    assert predictor._temperature_for(0) == 1.0

    probs = predictor.predict(observed, turn=1)
    z = predictor._raw_logits(observed, turn=1)
    from ptcg_ai.opponent_modeling.ml_predictor import _softmax

    expected = dict(zip(CLASSES, _softmax(z)))
    for cls in CLASSES:
        assert math.isclose(probs[cls], expected[cls], abs_tol=1e-9)


def test_calibration_softens_probabilities_but_preserves_argmax(tmp_path):
    """evidence_count が 1〜2 のバケットは T=4 で確率が和らぐ(top1確率が下がる)が、
    argmax(top1クラス)は T=1 のときと変わらない。"""
    weights_calibrated = _write_weights_with_calibration(tmp_path, _CALIBRATION_BUCKETS)
    predictor = MLDeckPredictor(weights_path=weights_calibrated)

    # 別ディレクトリに温度なし版(T=1相当)を書き出して比較。
    uncalibrated_dir = tmp_path / "uncalibrated"
    uncalibrated_dir.mkdir()
    predictor_uncalibrated = MLDeckPredictor(weights_path=_write_weights(uncalibrated_dir))

    observed = {"メガルカリオex": 2}  # evidence_count == 1 -> T=4
    assert predictor.evidence_count(observed) == 1
    assert predictor._temperature_for(1) == 4.0

    probs_t4 = predictor.predict(observed, turn=3)
    probs_t1 = predictor_uncalibrated.predict(observed, turn=3)

    top_t4 = max(probs_t4, key=probs_t4.get)
    top_t1 = max(probs_t1, key=probs_t1.get)
    assert top_t4 == top_t1 == "mega_lucario_ex"  # argmax は不変
    assert probs_t4[top_t4] < probs_t1[top_t1]  # だが確率の「強さ」は和らいでいる


def test_calibration_evidence_three_or_more_reverts_to_t1(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights_with_calibration(tmp_path, _CALIBRATION_BUCKETS))
    observed = {"メガルカリオex": 1, "フーディン": 1, "ドラパルトex": 1}  # evidence_count == 3 -> T=1
    assert predictor.evidence_count(observed) == 3
    assert predictor._temperature_for(3) == 1.0


def test_evidence_count_counts_unique_observed_card_names(tmp_path):
    predictor = MLDeckPredictor(weights_path=_write_weights(tmp_path))
    # 語彙にあるカードのみカウント、枚数0は数えない、__turn__ は含めない。
    assert predictor.evidence_count({}) == 0
    assert predictor.evidence_count({"メガルカリオex": 1}) == 1
    assert predictor.evidence_count({"メガルカリオex": 3}) == 1  # 枚数の合計ではなくユニーク種類数
    assert predictor.evidence_count({"メガルカリオex": 2, "フーディン": 1}) == 2
    assert predictor.evidence_count({"メガルカリオex": 0}) == 0  # 枚数0は数えない
    assert predictor.evidence_count({"ネストボール": 4}) == 0  # 語彙に無いカードは数えない
    assert predictor.evidence_count({"メガルカリオex": 1, "ネストボール": 4}) == 1
