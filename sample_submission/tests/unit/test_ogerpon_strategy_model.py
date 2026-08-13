"""ogerpon_strategy_model のユニットテスト。design.md §8、Phase3 item2(pure Python parity・
fallbackテスト)。学習(PyTorch)は対象外、手組みの重みJSONで純Python推論の正しさだけを見る。
"""

import json
import math
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))


@pytest.fixture
def mod():
    from ptcg_ai.learning import ogerpon_strategy_model as m
    return m


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _single_model_payload():
    """continuous_dim=2, option_dim=1, slot_count=2, embedding_dim=1 の最小モデル。"""
    return {
        "card_embedding": {"dim": 1, "card_id_max": 1, "table": [[0.0], [1.0]]},
        "standardization": {
            "continuous_mean": [0.0, 0.0], "continuous_std": [1.0, 1.0],
            "option_mean": [0.0], "option_std": [1.0],
        },
        "shared_layers": [
            {"weight": [[1.0, 1.0, 1.0, 1.0, 1.0]], "bias": [0.0]},
        ],
        "heads": {
            "win": {"weight": [[1.0]], "bias": [0.0]},
            "loop_complete": {"weight": [[0.0]], "bias": [0.0]},
            "opponent_ko_count": {"weight": [[1.0]], "bias": [2.0]},
            "signed_terminal_turns": {"weight": [[-1.0]], "bias": [0.0]},
        },
    }


def _write_weights(tmp_path, models, extra_meta=None):
    payload = {
        "schema_version": 1,
        "model_type": "ogerpon_option_q_mlp_ensemble",
        "feature_contract": {
            "continuous_feature_count": 2, "option_feature_count": 1,
            "slot_count": 2, "embedding_dim": 1, "bulu_card_ids": [920],
        },
        "models": models,
        "meta": extra_meta or {},
    }
    path = tmp_path / "ogerpon_strategy_weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- 未配置・壊れたJSON: フォールバック -------------------------------------------

def test_missing_weights_file_is_not_ready(mod, tmp_path):
    model = mod.OgerponStrategyModel(tmp_path / "does_not_exist.json")
    assert model.is_ready is False
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "EX_TEMPO")
    assert pred["win"] == 0.5
    assert pred["win_std"] == 0.0


def test_broken_json_falls_back_to_not_ready(mod, tmp_path):
    path = tmp_path / "ogerpon_strategy_weights.json"
    path.write_text("{not valid json", encoding="utf-8")
    model = mod.OgerponStrategyModel(path)
    assert model.is_ready is False


def test_dimension_mismatch_returns_neutral(mod, tmp_path):
    path = _write_weights(tmp_path, [_single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    assert model.is_ready is True
    # continuous_featuresの次元が契約(2)と不一致。
    pred = model.predict([1.0, 2.0, 3.0], [1, 0], [3.0], "EX_TEMPO")
    assert pred == {**{"win": 0.5, "loop_complete": 0.5, "opponent_ko_count": 0.0,
                       "signed_terminal_turns": 0.0}, "win_std": 0.0}


# --- 手組みの重みでフォワードパスの正しさを検証 -----------------------------------

def test_forward_pass_matches_hand_computed_values(mod, tmp_path):
    """continuous=[1,2], option=[3], slots=[1,0](embedding [1.0],[0.0]) を手計算と突き合わせる。

    shared = sum(continuous ++ option ++ embeddings) = 1+2+3+1+0 = 7
    win = sigmoid(7*1+0) = sigmoid(7)
    loop_complete = sigmoid(7*0+0) = sigmoid(0) = 0.5
    opponent_ko_count = 7*1+2 = 9
    signed_terminal_turns = 7*-1+0 = -7
    """
    path = _write_weights(tmp_path, [_single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "SINGLE_PRIZE_ROTATION")
    assert pred["win"] == pytest.approx(_sigmoid(7.0), abs=1e-9)
    assert pred["loop_complete"] == pytest.approx(0.5, abs=1e-9)
    assert pred["opponent_ko_count"] == pytest.approx(9.0, abs=1e-9)
    assert pred["signed_terminal_turns"] == pytest.approx(-7.0, abs=1e-9)
    assert pred["win_std"] == pytest.approx(0.0, abs=1e-9)  # 1モデルのみ


def test_loop_complete_forced_neutral_for_ex_tempo(mod, tmp_path):
    """EX_TEMPOはloop_completeが意味を持たないため、モデル出力に関わらず0.5に固定する。"""
    path = _write_weights(tmp_path, [_single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "EX_TEMPO")
    assert pred["loop_complete"] == 0.5


def test_missing_slot_ids_are_padded_with_unknown(mod, tmp_path):
    """slot_card_idsがslot_countより短くても壊れない(不足分は0=不明で埋める)。"""
    path = _write_weights(tmp_path, [_single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1], [3.0], "EX_TEMPO")  # 1個しか渡さない(本来2個)
    # slot1が0埋めされるので、embeddingは [1.0]+[0.0] = test_forward_passと同じ結果になる
    assert pred["opponent_ko_count"] == pytest.approx(9.0, abs=1e-9)


def test_shared_trunk_applies_relu_to_last_layer(mod, tmp_path):
    """design.md §8.1: 共有trunkは全層(最終層含む)にReLUがかかる
    (``nn.Sequential(Linear, ReLU, Linear, ReLU)``)。continuous+option+embeddingの
    合計を負にして、最終shared層の出力が0にクリップされることを確認する
    (ReLUを最終層で省略する旧policy_model.py流の実装だと、ここが負のまま漏れて
    PyTorch側と食い違う=parity崩壊のバグになる)。
    """
    payload = _single_model_payload()
    payload["shared_layers"] = [{"weight": [[1.0, 1.0, 1.0, 1.0, 1.0]], "bias": [-100.0]}]
    payload["heads"]["opponent_ko_count"] = {"weight": [[1.0]], "bias": [5.0]}
    path = _write_weights(tmp_path, [payload])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "EX_TEMPO")
    # shared前活性化 = 1+2+3+1+0-100 = -93 -> ReLU適用で0 -> ko_count = 0*1+5 = 5
    assert pred["opponent_ko_count"] == pytest.approx(5.0, abs=1e-9)


# --- ensemble平均・不確実性 -----------------------------------------------------

def test_ensemble_averages_predictions(mod, tmp_path):
    p1 = _single_model_payload()
    p2 = _single_model_payload()
    p2["heads"]["win"] = {"weight": [[1.0]], "bias": [-14.0]}  # 別モデル: z = 7-14 = -7
    path = _write_weights(tmp_path, [p1, p2])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "SINGLE_PRIZE_ROTATION")
    expected_mean = (_sigmoid(7.0) + _sigmoid(-7.0)) / 2.0
    assert pred["win"] == pytest.approx(expected_mean, abs=1e-9)
    assert pred["win_std"] > 0.05  # 2モデルが大きく食い違うので不確実性が出る


def test_identical_ensemble_members_have_zero_std(mod, tmp_path):
    path = _write_weights(tmp_path, [_single_model_payload(), _single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    pred = model.predict([1.0, 2.0], [1, 0], [3.0], "EX_TEMPO")
    assert pred["win_std"] == pytest.approx(0.0, abs=1e-9)


# --- predict_pair -----------------------------------------------------------------

def test_predict_pair_returns_both_options(mod, tmp_path):
    path = _write_weights(tmp_path, [_single_model_payload()])
    model = mod.OgerponStrategyModel(path)
    result = model.predict_pair([1.0, 2.0], [1, 0],
                                {"EX_TEMPO": [3.0], "SINGLE_PRIZE_ROTATION": [3.0]})
    assert set(result.keys()) == {"EX_TEMPO", "SINGLE_PRIZE_ROTATION"}
    assert result["EX_TEMPO"]["loop_complete"] == 0.5
