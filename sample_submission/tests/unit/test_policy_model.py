"""ptcg_ai.learning.policy_model.PolicyModel のユニットテスト。

ゴールデンテスト: 独立実装(numpy)で計算した選択肢スコアを収めた
``tests/fixtures/policy_model_predictions.json``(生成手順は同ファイル生成スクリプト参照)を
読み込み、``PolicyModel.score_options`` が同じ入力に対して 1e-6 以内で一致することを
確認する。これがフォワードパス(標準化・重み行列の向き・状態/選択肢特徴の連結順)の実装が
学習側の重みJSONスキーマと一致していることの担保になる(value_model.py の golden test と
同じ狙い)。fixture が未生成の場合はスキップする。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "policy_model_predictions.json"
_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"
_WEIGHTS = SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "policy_weights.json"

_TOL = 1e-6


@pytest.fixture(scope="module")
def policy_model():
    """デフォルト重み(policy_weights.json)をロードした PolicyModel。cg が無ければスキップ。"""
    try:
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"policy_model / cg engine unavailable: {exc}")
    model = PolicyModel()
    if not model.is_ready:
        pytest.skip(f"policy_weights.json not found at {_WEIGHTS}")
    return model


@pytest.fixture(scope="module")
def golden_predictions() -> list[dict]:
    if not _FIXTURE.exists():
        pytest.skip(f"policy_model_predictions.json not found at {_FIXTURE}")
    with _FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_scores_match(policy_model, golden_predictions):
    """独立実装(numpy)のスコアと 1e-6 以内で一致する(全選択肢)。"""
    from cg.api import to_observation_class

    assert len(golden_predictions) > 0
    mismatches = []
    for row in golden_predictions:
        obs = to_observation_class({**row["observation"], "logs": []})
        got = policy_model.score_options(obs)
        expected = row["expected_scores"]
        if len(got) != len(expected) or any(abs(g - e) > _TOL for g, e in zip(got, expected)):
            mismatches.append({"episode_id": row.get("episode_id"), "got": got, "expected": expected})
    assert not mismatches, (
        f"{len(mismatches)}/{len(golden_predictions)} 件が許容誤差 {_TOL} を超過: "
        + json.dumps(mismatches[:3], ensure_ascii=False)
    )


def test_missing_weights_is_safe():
    """重みファイルが存在しないパスを渡すと is_ready=False になり、例外を出さない。"""
    from cg.api import to_observation_class
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel(
        weights_path=SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "__no_such_weights__.json"
    )
    assert model.is_ready is False

    obs = to_observation_class({"logs": [], "select": None, "current": None})
    assert model.score_options(obs) == []
    assert model.select_option(obs) is None


def test_missing_weights_select_option_falls_back_to_zero(encoder_observations):
    """未ロードでも select.option があれば index 0 を返す(常に有効な選択を返す安全側)。"""
    from cg.api import to_observation_class
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel(
        weights_path=SAMPLE_SUBMISSION_ROOT / "ptcg_ai" / "learning" / "__no_such_weights__.json"
    )
    obs = to_observation_class({**encoder_observations["mid_game"], "logs": []})
    assert model.select_option(obs) == 0


def test_select_option_returns_valid_index(policy_model, encoder_observations):
    """実データ(mid_game / early_active_none)で選択肢の範囲内のインデックスを返すこと。"""
    from cg.api import to_observation_class

    for key in ("mid_game", "early_active_none"):
        obs = to_observation_class({**encoder_observations[key], "logs": []})
        idx = policy_model.select_option(obs)
        assert idx is not None
        assert 0 <= idx < len(obs.select.option)


def test_select_option_none_when_no_select(policy_model):
    """select が None の場合は None を返す(例外にならない)。"""
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    assert policy_model.select_option(obs) is None
    assert policy_model.score_options(obs) == []


def test_use_board_set_defaults_to_false(policy_model):
    """meta.use_board_set が無い(旧)重みでは False のまま(T1相当、後方互換)。"""
    assert policy_model._use_board_set is False


def test_board_pooled_mean_max_sum_order():
    """T2(design-transformer-representation-2026-08-08.md §5.4段階1): _board_pooled が
    mean→max→sum の順で正しく連結されること。既知の小さい埋め込みテーブルで手計算と照合する
    (train.py の PolicyScorerBoardSet._board_pooled と同じ順序であること自体は、
    train.py 側の自己検証・cross-check スモークテストで別途確認済み)。
    """
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel.__new__(PolicyModel)  # __init__ を経由せず内部状態だけ手動セット
    model._card_embedding_table = [
        [0.0, 0.0],  # index 0: 未知/範囲外の予約枠
        [1.0, 2.0],  # index 1
        [3.0, 0.0],  # index 2
    ]
    model._card_id_max = 2

    pooled = model._board_pooled([1, 2, 999])  # 999 は範囲外 -> index 0 にフォールバック
    # vecs = [[1,2], [3,0], [0,0]]
    # mean = [4/3, 2/3]  max = [3, 2]  sum = [4, 2]
    expected = [4 / 3, 2 / 3, 3.0, 2.0, 4.0, 2.0]
    assert pooled == pytest.approx(expected)


def test_forward_uses_board_pooled_when_provided():
    """_forward に board_pooled を渡すと、渡さない場合とスコアが変わること
    (board_pooled が実際に計算へ反映されていることの確認。数値そのものではなく
    "使われているか" を確認する軽量な回帰テスト)。
    """
    from ptcg_ai.learning.policy_model import PolicyModel

    model = PolicyModel.__new__(PolicyModel)
    model._card_embedding_table = [[0.0], [1.0]]
    model._card_id_max = 1
    model._state_mean = [0.0]
    model._state_std = [1.0]
    model._option_mean = [0.0]
    model._option_std = [1.0]
    # in_dim = state(1) + option(1) + card_embed(1) [+ board_pooled(3) if given]
    # 最終層1つだけ: 全入力を単純に加算するだけの重み。
    model._layers = [([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [0.0])]

    score_without_board = model._forward([1.0], [1.0], 1, board_pooled=None)
    score_with_board = model._forward([1.0], [1.0], 1, board_pooled=[1.0, 1.0, 1.0])
    assert score_without_board != score_with_board
    assert score_with_board == pytest.approx(score_without_board + 3.0)
