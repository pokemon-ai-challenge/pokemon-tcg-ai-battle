"""prediction_summary.summarize_prediction() のユニットテスト。

- 前半: duck-typing の各分岐(is_ready/predict_top/evidence_count/explain の有無)を、
  実際の予測器を使わない最小限のフェイク予測器で検証する。
- 後半: 実際の HybridDeckPredictor / MLDeckPredictor(合成重みJSON、test_hybrid_predictor.py と
  同じスタイル)を使い、evidence_count が explain() 経由で補完される経路まで含めて検証する。
"""

from pathlib import Path
import json
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.opponent_modeling.prediction_summary import (
    UNCERTAIN_THRESHOLD_DEFAULT,
    summarize_prediction,
)
from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor
from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor


# --- フェイク予測器(duck-typing の各分岐を単独で検証するための最小実装) ---


class _NotReadyPredictor:
    is_ready = False

    def predict_top(self, observed_cards, turn, n=3):  # pragma: no cover -- 呼ばれないはず
        raise AssertionError("unready の場合 predict_top は呼ばれてはいけない")


class _NoIsReadyAttrPredictor:
    """is_ready 属性自体を持たない壊れた予測器。getattr のデフォルト False に倒れることを確認する。"""

    def predict_top(self, observed_cards, turn, n=3):  # pragma: no cover
        raise AssertionError("is_ready が無い場合も unready 扱いになるはず")


class _SimplePredictor:
    """evidence_count も explain も持たない最小の予測器(MLDeckPredictor 相当の一部機能のみ)。"""

    is_ready = True

    def __init__(self, ranked):
        self._ranked = ranked

    def predict_top(self, observed_cards, turn, n=3):
        return self._ranked[:n]


class _WithEvidenceCountPredictor(_SimplePredictor):
    """evidence_count(observed_cards) を直接持つ予測器(MLDeckPredictor 相当)。"""

    def __init__(self, ranked, evidence_count_value):
        super().__init__(ranked)
        self._evidence_count_value = evidence_count_value

    def evidence_count(self, observed_cards):
        return self._evidence_count_value


class _WithExplainPredictor(_SimplePredictor):
    """explain() は持つが evidence_count() は持たない予測器(HybridDeckPredictor 相当)。"""

    def __init__(self, ranked, explanation):
        super().__init__(ranked)
        self._explanation = explanation

    def explain(self, observed_cards, turn):
        return self._explanation


class _BrokenExplainPredictor(_SimplePredictor):
    """explain() が例外を投げる予測器。根拠表示が壊れても確率計算自体は止めない、を検証する。"""

    def explain(self, observed_cards, turn):
        raise RuntimeError("boom")


# --- status 判定 ---


def test_unready_predictor_returns_unready_status():
    result = summarize_prediction(_NotReadyPredictor(), {}, turn=1)
    assert result["status"] == "unready"
    assert result["top"] == []
    assert result["top1_probability"] == 0.0
    assert result["evidence_count"] is None
    assert result["explanation"] is None
    assert result["uncertain_threshold"] == UNCERTAIN_THRESHOLD_DEFAULT


def test_missing_is_ready_attribute_treated_as_unready():
    result = summarize_prediction(_NoIsReadyAttrPredictor(), {}, turn=1)
    assert result["status"] == "unready"


def test_top1_below_threshold_is_uncertain():
    predictor = _SimplePredictor([("alakazam", 0.4), ("mega_lucario_ex", 0.35), ("other", 0.25)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["status"] == "uncertain"
    assert result["top1_probability"] == 0.4
    assert result["top"][0] == {"deck_type": "alakazam", "probability": 0.4}


def test_top1_at_or_above_threshold_is_confident():
    predictor = _SimplePredictor([("alakazam", 0.6), ("other", 0.4)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["status"] == "confident"


def test_top1_just_below_threshold_is_uncertain():
    predictor = _SimplePredictor([("alakazam", 0.599), ("other", 0.401)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["status"] == "uncertain"


def test_custom_uncertain_threshold_is_respected_and_echoed():
    predictor = _SimplePredictor([("alakazam", 0.7), ("other", 0.3)])
    result = summarize_prediction(predictor, {}, turn=1, uncertain_threshold=0.8)
    assert result["status"] == "uncertain"
    assert result["uncertain_threshold"] == 0.8


def test_top_n_is_forwarded_to_predict_top():
    predictor = _SimplePredictor(
        [("a", 0.5), ("b", 0.3), ("c", 0.15), ("d", 0.05)]
    )
    result = summarize_prediction(predictor, {}, turn=1, top_n=2)
    assert len(result["top"]) == 2
    assert [row["deck_type"] for row in result["top"]] == ["a", "b"]


# --- evidence_count の解決順序 ---


def test_evidence_count_uses_direct_method_when_available():
    predictor = _WithEvidenceCountPredictor([("alakazam", 0.9)], evidence_count_value=5)
    result = summarize_prediction(predictor, {"Ultra Ball": 1}, turn=3)
    assert result["evidence_count"] == 5


def test_evidence_count_falls_back_to_explanation_field():
    predictor = _WithExplainPredictor(
        [("alakazam", 0.9)],
        explanation={"mode": "hybrid", "weight_nb": 0.3, "evidence_count": 2, "nb_explanation": None},
    )
    result = summarize_prediction(predictor, {"Ultra Ball": 1}, turn=3)
    assert result["evidence_count"] == 2
    assert result["explanation"]["evidence_count"] == 2


def test_evidence_count_is_none_when_no_source_available():
    predictor = _SimplePredictor([("alakazam", 0.9)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["evidence_count"] is None


# --- explanation の伝播・フォールト耐性 ---


def test_explanation_none_when_predictor_has_no_explain():
    predictor = _SimplePredictor([("alakazam", 0.9)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["explanation"] is None


def test_explanation_propagated_verbatim_when_present():
    payload = {"mode": "nb_only", "weight_nb": 1.0, "evidence_count": None, "nb_explanation": {"cards": {}}}
    predictor = _WithExplainPredictor([("alakazam", 0.9)], explanation=payload)
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["explanation"] == payload


def test_broken_explain_does_not_crash_summary():
    predictor = _BrokenExplainPredictor([("alakazam", 0.9)])
    result = summarize_prediction(predictor, {}, turn=1)
    assert result["status"] == "confident"
    assert result["explanation"] is None
    assert result["evidence_count"] is None


# --- 実際の HybridDeckPredictor / MLDeckPredictor との統合 ---

FEATURE_NAMES = ["__turn__", "SigA", "SigB", "SigC"]
CLASSES = ["mega_lucario_ex", "alakazam", "other"]
_COMMON_PROBS = [0.5, 0.4, 0.3, 0.2]


def _write_lr_weights(tmp_path: Path) -> Path:
    payload = {
        "meta": {"trained_at": "test", "n_episodes": 1, "n_samples": 1, "feature_type": "test", "version": 1},
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "coef": [
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "intercept": [0.0, 0.0, 0.0],
    }
    weights_path = tmp_path / "deck_predictor_weights.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


def _write_nb_weights(tmp_path: Path) -> Path:
    likelihoods = {
        "SigA": {
            "mega_lucario_ex": [0.9, 0.85, 0.8, 0.75],
            "alakazam": [0.02, 0.01, 0.005, 0.001],
            "other": [0.02, 0.01, 0.005, 0.001],
        },
        "SigB": {
            "mega_lucario_ex": [0.02, 0.01, 0.005, 0.001],
            "alakazam": [0.9, 0.85, 0.8, 0.75],
            "other": [0.02, 0.01, 0.005, 0.001],
        },
        "SigC": {cls: list(_COMMON_PROBS) for cls in CLASSES},
    }
    payload = {
        "classes": CLASSES,
        "priors": [1.0 / len(CLASSES)] * len(CLASSES),
        "card_names": ["SigA", "SigB", "SigC"],
        "likelihoods": likelihoods,
        "meta": {
            "trained_at": "test",
            "n_decks_per_class": {c: 10 for c in CLASSES},
            "alpha": 1.0,
            "max_k": 4,
            "prior_window": {"recent_days": 14, "top_rank": 200, "n_window_decks": 10},
            "version": 1,
        },
    }
    weights_path = tmp_path / "deck_predictor_nb.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


def _write_hybrid_config(tmp_path: Path, buckets: list[dict]) -> Path:
    payload = {"buckets": buckets, "meta": {"fitted_at": "test"}}
    config_path = tmp_path / "deck_predictor_hybrid.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return config_path


def test_hybrid_predictor_evidence_count_flows_through_explain(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}])
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    result = summarize_prediction(hybrid, {"SigA": 1}, turn=3)
    # HybridDeckPredictor には evidence_count() メソッドが無いので、explain() の
    # "evidence_count" フィールド経由で解決されているはず。
    assert result["evidence_count"] == hybrid._lr.evidence_count({"SigA": 1})
    assert result["explanation"]["mode"] == "hybrid"
    assert result["explanation"]["nb_explanation"] is not None


def test_hybrid_predictor_no_evidence_uncertain_status(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}])
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    result = summarize_prediction(hybrid, {}, turn=1)
    # 語彙にある観測カードが無いので、prior 相当(3クラス均等に近い)= 閾値未満で uncertain のはず。
    assert result["status"] == "uncertain"
    assert result["top1_probability"] < UNCERTAIN_THRESHOLD_DEFAULT


def test_ml_predictor_alone_has_no_explanation_but_has_evidence_count(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    lr = MLDeckPredictor(weights_path=lr_path)

    result = summarize_prediction(lr, {"SigA": 1}, turn=3)
    assert result["explanation"] is None  # LR単体には explain() を実装しない(仕様C-2)
    assert result["evidence_count"] == 1


def test_unready_hybrid_predictor_returns_unready(tmp_path):
    missing_lr = tmp_path / "missing_lr.json"
    missing_nb = tmp_path / "missing_nb.json"
    hybrid = HybridDeckPredictor(lr_weights_path=missing_lr, nb_weights_path=missing_nb)

    result = summarize_prediction(hybrid, {"SigA": 1}, turn=3)
    assert result["status"] == "unready"
    assert result["top"] == []
