"""token_shard.py の単体テスト(T0)。

合成データでの往復(write -> read -> decision_slice)を確認する。
cgエンジン・自己対戦は使わない(board_tokens.BOARD_TOKEN_NUMERIC_DIM の参照のみ)。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
for _p in (str(_HERE), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def ts():
    try:
        import token_shard as _ts
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"token_shard unavailable: {exc}")
    return _ts


@pytest.fixture(scope="module")
def board_dim():
    from ptcg_ai.learning.board_tokens import BOARD_TOKEN_NUMERIC_DIM
    return BOARD_TOKEN_NUMERIC_DIM


def _fake_decisions(board_dim: int) -> list[dict]:
    """2決定点、盤面トークン数が異なる合成データ(選択肢数・盤面トークン数を可変にする)。"""
    d0 = {
        "legacy_state_feat": [0.1] * 166,
        "legacy_option_feats": [[0.2] * 65, [0.3] * 65],
        "option_card_ids": [343, 0],
        "board_numeric_feats": [[1.0] * board_dim, [2.0] * board_dim, [3.0] * board_dim],
        "board_card_ids": [343, 756, 756],
        "board_zone_ids": [0, 1, 1],
        "option_target_indices": [0, -1],
        "teacher_logits": [1.5, -0.5],
        "chosen_idx": 0,
        "logprob": -0.1,
    }
    d1 = {
        "legacy_state_feat": [0.4] * 166,
        "legacy_option_feats": [[0.5] * 65],
        "option_card_ids": [678],
        "board_numeric_feats": [[9.0] * board_dim],
        "board_card_ids": [678],
        "board_zone_ids": [2],
        "option_target_indices": [0],
        "teacher_logits": [2.0],
        "chosen_idx": 0,
        "logprob": 0.0,
    }
    return [d0, d1]


def test_round_trip_shapes_and_values(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 1.0, "opp": 2}]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        meta_in = {"run_id": "t0test", "generation": 0, "teacher_model_sha256": "deadbeef"}
        ts.write_token_shard(path, decisions, trajs, meta_in)

        arrays, meta = ts.read_token_shard(path)

        assert meta["token_shard_format"] == ts.TOKEN_SHARD_FORMAT
        assert meta["n_decisions"] == 2
        assert meta["run_id"] == "t0test"

        # legacy(既存形式相当)がそのまま同梱されている(dual-write)。
        assert arrays["legacy_state_features"].shape == (2, 166)
        assert arrays["legacy_option_features"].shape == (3, 65)  # 2+1選択肢
        assert list(arrays["option_card_ids"]) == [343, 0, 678]
        assert list(arrays["counts"]) == [2, 1]

        # 新規: 盤面トークン
        assert arrays["board_token_numeric_features"].shape == (4, board_dim)  # 3+1トークン
        assert list(arrays["board_token_card_ids"]) == [343, 756, 756, 678]
        assert list(arrays["board_token_zone_ids"]) == [0, 1, 1, 2]
        assert list(arrays["board_counts"]) == [3, 1]

        # ポインタ・教師ロジット
        assert list(arrays["option_target_token_indices"]) == [0, -1, 0]
        assert np.allclose(arrays["teacher_logits"], [1.5, -0.5, 2.0])

        # 試合単位
        assert list(arrays["lengths"]) == [2]
        assert list(arrays["rewards"]) == [1.0]
        assert list(arrays["opp"]) == [2]


def test_decision_slice_recovers_per_decision_data(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)

        d0 = ts.decision_slice(arrays, 0)
        assert d0["board_numeric_feats"].shape == (3, board_dim)
        assert list(d0["board_card_ids"]) == [343, 756, 756]
        assert list(d0["option_card_ids"]) == [343, 0]
        assert list(d0["option_target_indices"]) == [0, -1]
        assert d0["chosen_idx"] == 0

        d1 = ts.decision_slice(arrays, 1)
        assert d1["board_numeric_feats"].shape == (1, board_dim)
        assert list(d1["board_card_ids"]) == [678]
        assert list(d1["option_target_indices"]) == [0]


def test_teacher_logits_length_mismatch_raises(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    decisions[0]["teacher_logits"] = [1.5]  # option_card_ids は長さ2のまま -> 不整合
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        with pytest.raises(ValueError):
            ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})


def test_empty_decisions_raises(ts):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        with pytest.raises(SystemExit):
            ts.write_token_shard(path, [], [], {"run_id": "x", "generation": 0})


def test_wrong_format_raises_on_read(ts, board_dim):
    """旧shard(common.SHARD_FORMAT)を新readerに渡すとエラーになる(取り違え防止)。"""
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        # メタを書き換えて古い形式を偽装
        arrays, meta = ts.read_token_shard(path)
        meta["token_shard_format"] = "ptcg-rl-shard/2"
        import json
        np.savez_compressed(path, meta=np.array(json.dumps(meta, ensure_ascii=False)),
                            **{k: v for k, v in arrays.items()})
        with pytest.raises(ValueError):
            ts.read_token_shard(path)


def test_vocabulary_version_mismatch_is_visible_in_meta(ts, board_dim):
    """vocabulary_hash/versionをmetaに積んでおけば、shardとvocabularyの不一致を
    呼び出し側(token_batch.py)が検出できる(token_shard.py自体はmetaの中身を
    検査しない。呼び出し側の責務)。ここではmetaに正しく積まれることだけ確認する。"""
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs,
                             {"run_id": "x", "generation": 0,
                              "vocabulary_hash": "abc123", "vocabulary_version": 1})
        _, meta = ts.read_token_shard(path)
        assert meta["vocabulary_hash"] == "abc123"
        assert meta["vocabulary_version"] == 1


def _fake_decisions_with_debug(board_dim: int) -> list[dict]:
    decisions = _fake_decisions(board_dim)
    decisions[0]["board_serials"] = [1, 2, 3]  # board_card_ids=[343,756,756]と対応
    decisions[0]["debug_target_serial"] = [1, None]
    decisions[0]["debug_target_card_id"] = [343, None]
    decisions[0]["debug_resolver"] = ["area_index", "no_target_no_fields"]
    decisions[1]["board_serials"] = [10]  # board_card_ids=[678]と対応
    decisions[1]["debug_target_serial"] = [10]
    decisions[1]["debug_target_card_id"] = [678]
    decisions[1]["debug_resolver"] = ["active_implicit"]
    return decisions


def test_debug_pointer_arrays_round_trip(ts, board_dim):
    decisions = _fake_decisions_with_debug(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, meta = ts.read_token_shard(path)

        assert list(arrays["board_token_serial"]) == [1, 2, 3, 10]
        assert list(arrays["debug_target_serial"]) == [1, -1, 10]
        assert list(arrays["debug_target_card_id"]) == [343, -1, 678]
        names = meta["resolver_kind_names"]
        resolved = [names[i] for i in arrays["debug_resolver"]]
        assert resolved == ["area_index", "no_target_no_fields", "active_implicit"]

        d0 = ts.decision_slice(arrays, 0)
        assert list(d0["debug_target_serial"]) == [1, -1]


def test_no_debug_arrays_when_not_provided(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        assert "debug_resolver" not in arrays


def test_legacy_global_features_optional_round_trip(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    decisions[0]["legacy_global_feat"] = [1.0] * 34
    decisions[1]["legacy_global_feat"] = [2.0] * 34
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, meta = ts.read_token_shard(path)
        assert arrays["legacy_global_features"].shape == (2, 34)
        assert meta["legacy_global_dim"] == 34
        d0 = ts.decision_slice(arrays, 0)
        assert list(d0["legacy_global_feat"]) == [1.0] * 34


def test_no_legacy_global_features_when_not_provided(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, meta = ts.read_token_shard(path)
        assert "legacy_global_features" not in arrays
        assert meta["legacy_global_dim"] is None


def test_iter_decisions_matches_decision_slice(ts, board_dim):
    decisions = _fake_decisions(board_dim)
    trajs = [{"steps": decisions, "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        collected = list(ts.iter_decisions(arrays))
        assert len(collected) == 2
        assert list(collected[0]["board_card_ids"]) == [343, 756, 756]
        assert list(collected[1]["board_card_ids"]) == [678]


# ---------------------------------------------------------------------------
# concatenate_shards(D2.1: shard互換性検査の拡張)
# ---------------------------------------------------------------------------

def _full_meta(board_dim: int, **overrides) -> dict:
    """collect_tokens.py が実際に書き出すmetaと同じキー構成の合成meta。"""
    meta = {
        "run_id": "x", "generation": 0,
        "teacher_weights_path": "teacher.json", "teacher_model_sha256": "abc123",
        "extended_features_profile": "fuudin_v4",
        "vocabulary_version": 1, "vocabulary_hash": "vochash",
        "card_vocab_size": 1269, "global_feature_manifest_hash": "manihash",
        "games_requested": 1, "temperature": 1.0,
        "action_selection_mode": "softmax_temperature",
        "legacy_state_dim": 389, "legacy_global_dim": 145,
        "legacy_option_dim": 65, "board_token_numeric_dim": board_dim,
    }
    meta.update(overrides)
    return meta


def _write_with_global(ts, path, decisions, trajs, meta, board_dim):
    for i, d in enumerate(decisions):
        d.setdefault("legacy_global_feat", [0.0] * 145)
    ts.write_token_shard(path, decisions, trajs, meta)


def test_concatenate_shards_merges_two_compatible_shards(ts, board_dim):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i in range(2):
            decisions = _fake_decisions(board_dim)
            trajs = [{"steps": decisions, "reward": float(i), "opp": 0}]
            p = tmp / f"s{i}.npz"
            _write_with_global(ts, p, decisions, trajs, _full_meta(board_dim), board_dim)
            paths.append(p)
        arrays, meta = ts.concatenate_shards(paths)
        assert meta["n_decisions"] == 4
        assert meta["n_trajectories"] == 2
        assert meta["n_shards_merged"] == 2
        assert list(arrays["counts"]) == [2, 1, 2, 1]


# token_shard_formatの不一致は write_token_shard 自体が常に現行版を書き、
# read_token_shard がそもそも旧形式を読ませない(test_wrong_format_raises_on_read で別途確認済み)
# ため、ここでは token_shard_format 以外の「metaのラベルとして持つ値」の不一致を確認する。
@pytest.mark.parametrize("key,bad_value", [
    ("extended_features_profile", "fuudin_v2"),
    ("teacher_model_sha256", "different_sha"),
    ("vocabulary_version", 2),
    ("vocabulary_hash", "different_vochash"),
    ("card_vocab_size", 999),
    ("global_feature_manifest_hash", "different_manihash"),
    ("temperature", 0.5),
    ("action_selection_mode", "argmax"),
])
def test_concatenate_shards_rejects_single_field_mismatch(ts, board_dim, key, bad_value):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        decisions0 = _fake_decisions(board_dim)
        trajs0 = [{"steps": decisions0, "reward": 1.0, "opp": 0}]
        p0 = tmp / "s0.npz"
        _write_with_global(ts, p0, decisions0, trajs0, _full_meta(board_dim), board_dim)

        decisions1 = _fake_decisions(board_dim)
        trajs1 = [{"steps": decisions1, "reward": 0.0, "opp": 0}]
        p1 = tmp / "s1.npz"
        _write_with_global(ts, p1, decisions1, trajs1, _full_meta(board_dim, **{key: bad_value}), board_dim)

        with pytest.raises(ts.ShardMetaMismatchError):
            ts.concatenate_shards([p0, p1])


def test_concatenate_shards_rejects_dimension_mismatch(ts, board_dim):
    """legacy_state_dim等は実際の配列shapeから自動計算される(metaのラベルを
    上書きしても効かない、write_token_shardの意図的な仕様)。実際に次元の違う
    データ(166次元 vs 389次元相当)を混ぜようとした場合の拒否を確認する。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        decisions0 = _fake_decisions(board_dim)  # legacy_state_feat長=166
        trajs0 = [{"steps": decisions0, "reward": 1.0, "opp": 0}]
        p0 = tmp / "s0.npz"
        meta0 = _full_meta(board_dim, extended_features_profile=None)
        del meta0["legacy_global_dim"]  # base166相当(global特徴なし)
        ts.write_token_shard(p0, decisions0, trajs0, meta0)

        decisions1 = _fake_decisions(board_dim)
        for d in decisions1:
            d["legacy_state_feat"] = d["legacy_state_feat"] + [0.0] * 223  # 389次元相当
            d["legacy_global_feat"] = [0.0] * 145
        trajs1 = [{"steps": decisions1, "reward": 0.0, "opp": 0}]
        p1 = tmp / "s1.npz"
        meta1 = _full_meta(board_dim)
        ts.write_token_shard(p1, decisions1, trajs1, meta1)

        with pytest.raises(ts.ShardMetaMismatchError):
            ts.concatenate_shards([p0, p1])


def test_concatenate_shards_rejects_missing_field(ts, board_dim):
    """新しいキー(例: card_vocab_size)を持たない古い形式のshardを混ぜようとすると拒否される。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        decisions0 = _fake_decisions(board_dim)
        trajs0 = [{"steps": decisions0, "reward": 1.0, "opp": 0}]
        p0 = tmp / "s0.npz"
        _write_with_global(ts, p0, decisions0, trajs0, _full_meta(board_dim), board_dim)

        decisions1 = _fake_decisions(board_dim)
        trajs1 = [{"steps": decisions1, "reward": 0.0, "opp": 0}]
        p1 = tmp / "s1.npz"
        meta1 = _full_meta(board_dim)
        del meta1["card_vocab_size"]
        _write_with_global(ts, p1, decisions1, trajs1, meta1, board_dim)

        with pytest.raises(ts.ShardMetaMismatchError):
            ts.concatenate_shards([p0, p1])
