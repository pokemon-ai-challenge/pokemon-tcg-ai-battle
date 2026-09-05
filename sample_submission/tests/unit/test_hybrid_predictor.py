"""hybrid_predictor.HybridDeckPredictor のユニットテスト。

deck_predictor_hybrid.json は学習パイプライン(kaggle_replays/deck_predictor/fit_hybrid.py)の
成果物であり、このリポジトリにはまだ配置されていない。そのため各テストは tmp_path に
合成した小さな LR / NB / hybrid 重みJSONを書き出し、それを weights_path として渡す形で検証する
(test_ml_predictor.py / test_nb_predictor.py と同じスタイル)。
"""

from pathlib import Path
import json
import math
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor
from ptcg_ai.opponent_modeling.ml_predictor import MLDeckPredictor
from ptcg_ai.opponent_modeling.nb_predictor import NBDeckPredictor


# 合成語彙: __turn__ + 3枚のカード名。3クラス(2アーキタイプ + other)。
# LR/NB 両方から同じ観測カードで意味のある(一致した)予測が出るよう、シグネチャカードの
# 役割(SigA -> mega_lucario_ex 寄り、SigB -> alakazam 寄り、SigC -> 汎用)を揃えている。
FEATURE_NAMES = ["__turn__", "SigA", "SigB", "SigC"]
CLASSES = ["mega_lucario_ex", "alakazam", "other"]

_COMMON_PROBS = [0.5, 0.4, 0.3, 0.2]


def _write_lr_weights(tmp_path: Path, *, classes=None, filename="deck_predictor_weights.json") -> Path:
    payload = {
        "meta": {"trained_at": "test", "n_episodes": 1, "n_samples": 1, "feature_type": "test", "version": 1},
        "classes": classes if classes is not None else CLASSES,
        "feature_names": FEATURE_NAMES,
        "coef": [
            # __turn__, SigA, SigB, SigC
            [0.0, 5.0, 0.0, 0.0],  # mega_lucario_ex: strongly driven by SigA
            [0.0, 0.0, 5.0, 0.0],  # alakazam: strongly driven by SigB
            [0.0, 0.0, 0.0, 0.0],  # other: baseline
        ],
        "intercept": [0.0, 0.0, 0.0],
    }
    weights_path = tmp_path / filename
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


def _write_nb_weights(tmp_path: Path, *, classes=None, filename="deck_predictor_nb.json") -> Path:
    classes = classes if classes is not None else CLASSES
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
        "SigC": {
            "mega_lucario_ex": list(_COMMON_PROBS),
            "alakazam": list(_COMMON_PROBS),
            "other": list(_COMMON_PROBS),
        },
    }
    # classes が既定と異なる(クラス不一致テスト用)場合は、likelihoods もそのクラス名で埋め直す。
    likelihoods = {
        card: {cls: list(_COMMON_PROBS) if cls not in per_class else per_class[cls] for cls in classes}
        for card, per_class in likelihoods.items()
    }
    payload = {
        "classes": classes,
        "priors": [1.0 / len(classes)] * len(classes),
        "card_names": ["SigA", "SigB", "SigC"],
        "likelihoods": likelihoods,
        "meta": {
            "trained_at": "test",
            "n_decks_per_class": {c: 10 for c in classes},
            "alpha": 1.0,
            "max_k": 4,
            "prior_window": {"recent_days": 14, "top_rank": 200, "n_window_decks": 10},
            "version": 1,
        },
    }
    weights_path = tmp_path / filename
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return weights_path


def _write_hybrid_config(tmp_path: Path, buckets: list[dict], filename="deck_predictor_hybrid.json") -> Path:
    payload = {"buckets": buckets, "meta": {"fitted_at": "test"}}
    config_path = tmp_path / filename
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return config_path


_ALL_WEIGHT_BUCKET = [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.0}]


# --- w=0 で LR と完全一致 ---


def test_weight_zero_matches_lr_exactly(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.0}]
    )

    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)
    lr_only = MLDeckPredictor(weights_path=lr_path)

    assert hybrid.mode == "hybrid"
    observed = {"SigA": 1}
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_lr = lr_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_lr[cls], abs_tol=1e-9)


# --- w=1 で NB と完全一致 ---


def test_weight_one_matches_nb_exactly(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 1.0}]
    )

    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)
    nb_only = NBDeckPredictor(weights_path=nb_path)

    assert hybrid.mode == "hybrid"
    observed = {"SigB": 2}
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_nb = nb_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_nb[cls], abs_tol=1e-9)


# --- 中間 w で確率合計1 ---


def test_intermediate_weight_probabilities_sum_to_one(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}]
    )

    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)
    probs = hybrid.predict({"SigA": 1}, turn=3)

    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert set(probs.keys()) == set(CLASSES)

    # 中間ブレンドは LR単体・NB単体どちらとも(一般には)一致しない値になっているはず
    # (SigA は LR・NB 双方で mega_lucario_ex 寄りの信号だが、ブレンド確率自体は両者の中間になる)。
    lr_only = MLDeckPredictor(weights_path=lr_path)
    nb_only = NBDeckPredictor(weights_path=nb_path)
    probs_lr = lr_only.predict({"SigA": 1}, turn=3)
    probs_nb = nb_only.predict({"SigA": 1}, turn=3)
    assert not math.isclose(probs["mega_lucario_ex"], probs_lr["mega_lucario_ex"], abs_tol=1e-6)
    assert not math.isclose(probs["mega_lucario_ex"], probs_nb["mega_lucario_ex"], abs_tol=1e-6)


# --- ブレンド設定なし -> LR と一致 ---


def test_missing_hybrid_config_matches_lr(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    missing_hybrid_path = tmp_path / "does_not_exist_hybrid.json"

    hybrid = HybridDeckPredictor(
        lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=missing_hybrid_path
    )
    lr_only = MLDeckPredictor(weights_path=lr_path)

    assert hybrid.mode == "hybrid"  # LR/NBとも読み込めているのでモード自体はhybridだが、常にw=0扱い
    observed = {"SigA": 1, "SigB": 1}
    probs_hybrid = hybrid.predict(observed, turn=2)
    probs_lr = lr_only.predict(observed, turn=2)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_lr[cls], abs_tol=1e-9)


# --- NB重みなし -> LRフォールバック ---


def test_missing_nb_weights_falls_back_to_lr(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    missing_nb_path = tmp_path / "does_not_exist_nb.json"
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 1.0}]
    )

    hybrid = HybridDeckPredictor(
        lr_weights_path=lr_path, nb_weights_path=missing_nb_path, hybrid_config_path=hybrid_path
    )
    lr_only = MLDeckPredictor(weights_path=lr_path)

    assert hybrid.mode == "lr_only"
    assert hybrid.is_ready is True
    observed = {"SigA": 1}
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_lr = lr_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_lr[cls], abs_tol=1e-9)


# --- LR重みなし -> NB単体 ---


def test_missing_lr_weights_falls_back_to_nb(tmp_path):
    missing_lr_path = tmp_path / "does_not_exist_lr.json"
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.0}]
    )

    hybrid = HybridDeckPredictor(
        lr_weights_path=missing_lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path
    )
    nb_only = NBDeckPredictor(weights_path=nb_path)

    assert hybrid.mode == "nb_only"
    assert hybrid.is_ready is True
    observed = {"SigB": 1}
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_nb = nb_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_nb[cls], abs_tol=1e-9)


# --- 両方なし -> {"other": 1.0} ---


def test_missing_both_weights_returns_other(tmp_path):
    missing_lr_path = tmp_path / "does_not_exist_lr.json"
    missing_nb_path = tmp_path / "does_not_exist_nb.json"

    hybrid = HybridDeckPredictor(lr_weights_path=missing_lr_path, nb_weights_path=missing_nb_path)

    assert hybrid.mode == "unready"
    assert hybrid.is_ready is False
    assert hybrid.predict({"SigA": 1}, turn=1) == {"other": 1.0}


# --- バケット外 evidence は w=0 ---


def test_evidence_outside_bucket_range_defaults_to_zero_weight(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    # evidence_count==0 のみ w=1.0(NB)、それ以外はバケットが無いので w=0(LR)になるはず。
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": 0, "weight_nb": 1.0}])

    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)
    lr_only = MLDeckPredictor(weights_path=lr_path)

    observed = {"SigA": 1}  # evidence_count == 1 -> バケット [0,0] の範囲外
    assert hybrid._lr.evidence_count(observed) == 1
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_lr = lr_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_lr[cls], abs_tol=1e-9)


# --- クラス不一致検知 -> LRフォールバック(属性で検知可能) ---


def test_classes_mismatch_falls_back_to_lr_only(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    # NB側だけ余分なクラスを持たせて意図的に不一致を作る。
    nb_path = _write_nb_weights(tmp_path, classes=["mega_lucario_ex", "alakazam", "other", "extra_class"])
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 1.0}]
    )

    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)
    lr_only = MLDeckPredictor(weights_path=lr_path)

    assert hybrid.classes_mismatch is True
    assert hybrid.mode == "lr_only"
    observed = {"SigA": 1}
    probs_hybrid = hybrid.predict(observed, turn=3)
    probs_lr = lr_only.predict(observed, turn=3)
    for cls in CLASSES:
        assert math.isclose(probs_hybrid[cls], probs_lr[cls], abs_tol=1e-9)


# --- predict_top の並び ---


def test_predict_top_order_and_count(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(
        tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}]
    )
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    top = hybrid.predict_top({"SigA": 3}, turn=4, n=2)
    assert len(top) == 2
    assert top[0][0] == "mega_lucario_ex"
    assert top[0][1] >= top[1][1]
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in top)


def test_predict_top_n_larger_than_class_count_returns_all(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path)
    top = hybrid.predict_top({}, turn=1, n=10)
    assert len(top) == len(CLASSES)


# --- 既定パス解決(weights_path/hybrid_config_path省略) ---


# --- rough_prediction によるオーバーライド ---


def test_rough_prediction_none_matches_base_predict(tmp_path):
    """rough_prediction を渡さない(既定)場合は従来と完全に同じ分布を返す。"""
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}])
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    observed = {"SigA": 1}
    base = hybrid.predict(observed, turn=3)
    explicit_none = hybrid.predict(observed, turn=3, rough_prediction=None)
    assert base == explicit_none


def test_rough_prediction_known_class_does_not_change_output(tmp_path):
    """rough が学習済みクラス(CLASSES に含まれる)を confident で返しても出力は変わらない。"""
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}])
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    observed = {"SigA": 1}
    base = hybrid.predict(observed, turn=3)
    overridden = hybrid.predict(
        observed,
        turn=3,
        rough_prediction={"status": "confident", "deck_type": "alakazam", "match_rate": 0.99},
    )
    assert overridden == base


def test_rough_prediction_not_confident_does_not_change_output(tmp_path):
    """rough の status が confident でなければ何もしない。"""
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path)

    observed = {"SigA": 1}
    base = hybrid.predict(observed, turn=3)
    for status in ("insufficient_evidence", "ambiguous", "no_candidate"):
        overridden = hybrid.predict(
            observed,
            turn=3,
            rough_prediction={"status": status, "deck_type": "new_archetype", "match_rate": 0.9},
        )
        assert overridden == base


def test_rough_prediction_new_class_gets_capped_probability_mass(tmp_path):
    """rough が未学習の新アーキタイプを confident で返した場合、そのクラスに
    min(match_rate, 0.9) の確率質量が割り当てられ、残りは既存分布の比率を保って縮小する。
    """
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid_path = _write_hybrid_config(tmp_path, [{"min_evidence": 0, "max_evidence": None, "weight_nb": 0.5}])
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path, hybrid_config_path=hybrid_path)

    observed = {"SigA": 1}
    base = hybrid.predict(observed, turn=3)
    overridden = hybrid.predict(
        observed,
        turn=3,
        rough_prediction={"status": "confident", "deck_type": "lopunny_megafroslass", "match_rate": 0.95},
    )

    assert math.isclose(sum(overridden.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(overridden["lopunny_megafroslass"], 0.9, abs_tol=1e-9)  # match_rate 0.95 は 0.9 に頭打ち
    # 既存クラス同士の相対比率は base のときと変わらない(一律スケールされただけ)。
    for cls in CLASSES:
        assert math.isclose(overridden[cls] / base[cls], 0.1, rel_tol=1e-6)


def test_rough_prediction_new_class_uses_raw_match_rate_when_below_cap(tmp_path):
    lr_path = _write_lr_weights(tmp_path)
    nb_path = _write_nb_weights(tmp_path)
    hybrid = HybridDeckPredictor(lr_weights_path=lr_path, nb_weights_path=nb_path)

    overridden = hybrid.predict(
        {"SigA": 1},
        turn=3,
        rough_prediction={"status": "confident", "deck_type": "yadoking_2", "match_rate": 0.62},
    )
    assert math.isclose(overridden["yadoking_2"], 0.62, abs_tol=1e-9)
    assert math.isclose(sum(overridden.values()), 1.0, abs_tol=1e-9)


def test_rough_prediction_new_class_when_unready_still_surfaces(tmp_path):
    """LR/NB 両方未ロード(unready)でも rough が confident なら新クラスの確率を返す。"""
    missing_lr_path = tmp_path / "does_not_exist_lr.json"
    missing_nb_path = tmp_path / "does_not_exist_nb.json"
    hybrid = HybridDeckPredictor(lr_weights_path=missing_lr_path, nb_weights_path=missing_nb_path)

    overridden = hybrid.predict(
        {"SigA": 1},
        turn=3,
        rough_prediction={"status": "confident", "deck_type": "lopunny_megafroslass", "match_rate": 0.9},
    )
    assert math.isclose(overridden["lopunny_megafroslass"], 0.9, abs_tol=1e-9)
    assert math.isclose(sum(overridden.values()), 1.0, abs_tol=1e-9)


def test_default_paths_resolve_relative_to_module():
    hybrid = HybridDeckPredictor()
    module_dir = Path(__file__).resolve().parents[2] / "ptcg_ai" / "opponent_modeling"
    expected_lr = module_dir / "deck_predictor_weights.json"
    expected_nb = module_dir / "deck_predictor_nb.json"

    if expected_lr.exists() and expected_nb.exists():
        # クラスリストが一致すれば hybrid、不一致なら lr_only にフォールバックしているはず。
        assert hybrid.mode in ("hybrid", "lr_only")
        assert hybrid.is_ready is True
    elif expected_lr.exists():
        assert hybrid.mode == "lr_only"
    elif expected_nb.exists():
        assert hybrid.mode == "nb_only"
    else:
        assert hybrid.mode == "unready"
        assert hybrid.predict({}, turn=1) == {"other": 1.0}
