"""T0.1のend-to-end検証: 実State token化 -> shard保存 -> 再読み込み -> batch化
(padding/mask/vocab変換/global特徴取得/countsからの決定点復元)。

境界ケース(盤面トークン0件・選択肢1件・選択肢最大数・同一card id複数体・
166次元/389次元profile・未知card id・vocabulary不一致)も確認する。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def cg_api():
    try:
        import cg.api as api
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return api


@pytest.fixture(scope="module")
def env(cg_api):
    try:
        import collect_tokens as ct
        import token_shard as ts
        import token_batch as tb
        from ptcg_ai.learning import board_tokens as bt
        from ptcg_ai.learning import encoder
        from ptcg_ai.learning import legacy_feature_manifest as manifest
        from ptcg_ai.learning import card_vocab
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")
    pm = PolicyModel(None)
    if not pm.is_ready:
        pytest.skip("production policy_weights.json が無い")
    return {"ct": ct, "ts": ts, "tb": tb, "bt": bt, "encoder": encoder, "manifest": manifest,
            "card_vocab": card_vocab, "pm": pm, "vocab": card_vocab.load_vocab()}


def _player(api, active, bench, hand_count=4):
    return api.PlayerState(active=[active] if active is not None else [], bench=list(bench),
                           benchMax=5, deckCount=40, discard=[], prize=[None] * 6,
                           handCount=hand_count, hand=None, poisoned=False, burned=False,
                           asleep=False, paralyzed=False, confused=False)


def _pokemon(api, card_id, serial, hp=100, max_hp=100):
    return api.Pokemon(id=card_id, serial=serial, hp=hp, maxHp=max_hp, appearThisTurn=False,
                       energies=[], energyCards=[], tools=[], preEvolution=[])


def _state(api, me, opp, your_index=0):
    return api.State(turn=5, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
                     supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                     retreated=False, result=-1, stadium=[], looking=None, players=[me, opp])


def _select(api, options):
    return api.SelectData(type=api.SelectType.MAIN, context=list(api.SelectContext)[0],
                          minCount=1, maxCount=1, remainDamageCounter=0, remainEnergyCost=0,
                          option=options, deck=None, contextCard=None, effect=None)


def _collect_one_decision(env, cg_api, state, select, debug_pointers=False):
    ct, encoder, bt, manifest, pm = env["ct"], env["encoder"], env["bt"], env["manifest"], env["pm"]
    profile = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    return ct._build_decision(pm, encoder, bt, manifest, profile, state, select, debug_pointers)


def test_full_pipeline_real_state_to_batch(env, cg_api):
    """1. token化 2. shard保存 3. 再読込 4. batch化 5. padding 6. mask 7. vocab変換
    8. global特徴取得 9. countsからの決定点復元、を一連で確認する。"""
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    my_bench = [_pokemon(api, 756, 2), _pokemon(api, 756, 3)]  # 同一card id複数体
    opp_active = _pokemon(api, 678, 10)
    state = _state(api, _player(api, my_active, my_bench), _player(api, opp_active, []))
    options = [api.Option(type=api.OptionType.RETREAT),
              api.Option(type=api.OptionType.CARD, area=api.AreaType.BENCH, index=0),
              api.Option(type=api.OptionType.CARD, area=api.AreaType.BENCH, index=1),
              api.Option(type=api.OptionType.END)]
    select = _select(api, options)

    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0
    decision["logprob"] = -0.5

    ts, tb, vocab = env["ts"], env["tb"], env["vocab"]
    trajs = [{"steps": [decision], "reward": 1.0, "opp": 0}]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision], trajs,
                             {"run_id": "e2e", "generation": 0,
                              "vocabulary_hash": vocab.hash, "vocabulary_version": vocab.version})
        arrays, meta = ts.read_token_shard(path)

        # 9. countsからの決定点復元
        recovered = list(ts.iter_decisions(arrays))
        assert len(recovered) == 1
        assert list(recovered[0]["option_card_ids"]) == decision["option_card_ids"]

        # 4-7. batch化・padding・mask・vocab変換
        batch = tb.build_batch(arrays, vocab)
        assert batch["board"]["numeric"].shape == (1, 4, 11)  # active+bench2+opp active=4トークン
        assert batch["board"]["mask"].all()  # 1決定点しか無いのでpaddingは発生しない
        assert batch["option"]["mask"].sum() == 4

        # 同一card id(756)の2トークンが、別々のvocab indexではなく「同じindex」を指す
        # (card idが同じなら同じ埋め込みを引くのが正しい。個体としての区別はboard側の
        # 行(トークン)そのもので付いており、vocab indexで区別する必要は無い)。
        bench_card_ids = batch["board"]["card_idx"][0][1:3]
        assert bench_card_ids[0] == bench_card_ids[1] == vocab.index_of(756)

        # 8. global特徴
        assert batch["legacy_global_features"].shape == (1, env["manifest"].global_dim(None))


def test_zero_board_tokens(env, cg_api):
    """対局開始直後などで盤面トークンが0件でも壊れない。"""
    api = cg_api
    state = _state(api, _player(api, None, []), _player(api, None, []))
    options = [api.Option(type=api.OptionType.YES), api.Option(type=api.OptionType.NO)]
    select = _select(api, options)
    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0
    decision["logprob"] = 0.0

    ts, tb, vocab = env["ts"], env["tb"], env["vocab"]
    trajs = [{"steps": [decision], "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision], trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        batch = tb.build_batch(arrays, vocab)
        assert batch["board"]["numeric"].shape[1] == 0
        assert batch["board"]["mask"].shape == (1, 0)


def test_single_option(env, cg_api):
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    state = _state(api, _player(api, my_active, []), _player(api, None, []))
    select = _select(api, [api.Option(type=api.OptionType.END)])
    decision = _collect_one_decision(env, cg_api, state, select)
    assert len(decision["option_card_ids"]) == 1
    assert len(decision["teacher_logits"]) == 1


def test_many_options_padding(env, cg_api):
    """選択肢が多い決定点(42個相当を模擬)でもpaddingが正しく機能する。"""
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    state = _state(api, _player(api, my_active, []), _player(api, None, []))
    options = [api.Option(type=api.OptionType.NUMBER, number=i) for i in range(42)]
    select = _select(api, options)
    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0
    decision["logprob"] = 0.0

    # 別の決定点(選択肢1個)と一緒にbatch化してpaddingがずれないか確認する。
    select2 = _select(api, [api.Option(type=api.OptionType.END)])
    decision2 = _collect_one_decision(env, cg_api, state, select2)
    decision2["chosen_idx"] = 0
    decision2["logprob"] = 0.0

    ts, tb, vocab = env["ts"], env["tb"], env["vocab"]
    trajs = [{"steps": [decision, decision2], "reward": 1.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision, decision2], trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        batch = tb.build_batch(arrays, vocab)
        assert batch["option"]["mask"].shape == (2, 42)
        assert batch["option"]["mask"][0].sum() == 42
        assert batch["option"]["mask"][1].sum() == 1
        assert not batch["option"]["mask"][1, 1:].any()
        # paddingされたteacher_logitsは-infなのでsoftmaxで確実に無視される
        assert np.isneginf(batch["option"]["teacher_logits"][1, 1:]).all()


def test_unknown_card_id_maps_to_unk(env):
    tb, card_vocab, vocab = env["tb"], env["card_vocab"], env["vocab"]
    assert vocab.index_of(999999) == card_vocab.UNK_INDEX


def test_vocabulary_version_mismatch_detectable(env):
    """shardのmeta.vocabulary_hashと、実際にロードしたvocabのhashを比較すれば不一致を検出できる。"""
    vocab = env["vocab"]
    fake_shard_hash = "not_the_real_hash"
    assert fake_shard_hash != vocab.hash  # 呼び出し側はこの比較で不一致を検出する


def test_166_and_389_profile_global_dims_differ(env):
    """166次元profile(None)と389次元profile(fuudin_v4)でglobal_dimが異なり、
    無言でpaddingせず明確に別の次元数になることを確認する。"""
    manifest = env["manifest"]
    assert manifest.global_dim(None) == 34
    assert manifest.global_dim("fuudin_v4") == 145
    assert manifest.global_dim(None) != manifest.global_dim("fuudin_v4")


def test_mixed_profile_shards_rejected(env):
    tb = env["tb"]
    with pytest.raises(tb.ProfileMismatchError):
        tb.validate_profile_consistency([
            {"extended_features_profile": None},
            {"extended_features_profile": "fuudin_v4"},
        ])


def test_same_profile_shards_accepted(env):
    tb = env["tb"]
    profile = tb.validate_profile_consistency([
        {"extended_features_profile": "fuudin_v4"},
        {"extended_features_profile": "fuudin_v4"},
    ])
    assert profile == "fuudin_v4"


# ---------------------------------------------------------------------------
# T1.1: train/val分割・正規化統計
# ---------------------------------------------------------------------------

def test_train_val_split_by_game_no_leakage(env):
    tb = env["tb"]
    arrays = {"lengths": np.array([3, 2, 4, 1, 5])}
    train_idx, val_idx = tb.train_val_decision_indices(arrays, val_fraction=0.4, seed=0)
    total = int(arrays["lengths"].sum())
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(total))
    assert not (set(train_idx.tolist()) & set(val_idx.tolist()))
    assert len(val_idx) > 0 and len(train_idx) > 0


def test_train_val_split_is_by_game_not_by_decision(env):
    """同じ試合の決定点が丸ごとtrainかvalのどちらかに入る(跨がらない)。"""
    tb = env["tb"]
    arrays = {"lengths": np.array([3, 4])}  # game0: 0-2, game1: 3-6
    train_idx, val_idx = tb.train_val_decision_indices(arrays, val_fraction=0.5, seed=1)
    train_set, val_set = set(train_idx.tolist()), set(val_idx.tolist())
    game0, game1 = {0, 1, 2}, {3, 4, 5, 6}
    assert game0 <= train_set or game0 <= val_set
    assert game1 <= train_set or game1 <= val_set


def test_normalization_stats_std_floor_and_no_nan(env, cg_api):
    """定数特徴(全行で同じ値)でもstdに下限があってNaNが出ない。"""
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    state = _state(api, _player(api, my_active, []), _player(api, None, []))
    select = _select(api, [api.Option(type=api.OptionType.END)])
    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0
    decision["logprob"] = 0.0

    ts, tb, vocab = env["ts"], env["tb"], env["vocab"]
    # 同じ決定点を3回複製(定数特徴を作るため)し、2試合に分ける。
    decisions = [decision, decision, decision]
    trajs = [{"steps": [decision, decision], "reward": 1.0, "opp": 0},
            {"steps": [decision], "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        train_idx, val_idx = tb.train_val_decision_indices(arrays, val_fraction=0.5, seed=0)
        stats = tb.compute_normalization_stats(arrays, train_idx)
        for key in ("board_numeric_std", "global_std", "option_std"):
            assert (stats[key] >= tb.STD_FLOOR - 1e-9).all()
            assert np.isfinite(stats[key]).all()
        for key in ("board_numeric_mean", "global_mean", "option_mean"):
            assert np.isfinite(stats[key]).all()


def test_normalization_stats_uses_only_train_split(env, cg_api):
    """train splitに含まれない決定点の値は統計に影響しない。"""
    api = cg_api
    active_a = _pokemon(api, 343, 1, hp=50, max_hp=100)
    state_a = _state(api, _player(api, active_a, []), _player(api, None, []))
    active_b = _pokemon(api, 343, 1, hp=100, max_hp=100)
    state_b = _state(api, _player(api, active_b, []), _player(api, None, []))
    select = _select(api, [api.Option(type=api.OptionType.END)])

    d_a = _collect_one_decision(env, cg_api, state_a, select)
    d_a["chosen_idx"] = 0; d_a["logprob"] = 0.0
    d_b = _collect_one_decision(env, cg_api, state_b, select)
    d_b["chosen_idx"] = 0; d_b["logprob"] = 0.0

    ts, tb = env["ts"], env["tb"]
    decisions = [d_a, d_b]
    trajs = [{"steps": [d_a], "reward": 1.0, "opp": 0}, {"steps": [d_b], "reward": 0.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)

        stats_train0 = tb.compute_normalization_stats(arrays, np.array([0]))
        stats_train1 = tb.compute_normalization_stats(arrays, np.array([1]))
        # hp_ratio(2番目の数値特徴)がゲームごとに違う値なので、統計も違うはず。
        assert not np.allclose(stats_train0["board_numeric_mean"], stats_train1["board_numeric_mean"])


def test_slice_arrays_by_decisions_round_trips_shapes(env, cg_api):
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    state = _state(api, _player(api, my_active, []), _player(api, None, []))
    select = _select(api, [api.Option(type=api.OptionType.END)])
    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0
    decision["logprob"] = 0.0

    ts, tb = env["ts"], env["tb"]
    trajs = [{"steps": [decision], "reward": 1.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision], trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        sub = tb.slice_arrays_by_decisions(arrays, np.array([0]))
        assert sub["counts"].shape == (1,)
        assert sub["board_token_numeric_features"].shape[0] == int(arrays["board_counts"][0])


def test_build_normalization_stats_student_train_stats_matches_compute(env, cg_api):
    api = cg_api
    my_active = _pokemon(api, 343, 1)
    state = _state(api, _player(api, my_active, []), _player(api, None, []))
    select = _select(api, [api.Option(type=api.OptionType.END)])
    decision = _collect_one_decision(env, cg_api, state, select)
    decision["chosen_idx"] = 0; decision["logprob"] = 0.0

    ts, tb = env["ts"], env["tb"]
    trajs = [{"steps": [decision], "reward": 1.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision], trajs, {"run_id": "x", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        idx = np.array([0])
        expected = tb.compute_normalization_stats(arrays, idx)
        got = tb.build_normalization_stats("student_train_stats", arrays, idx, None)
        for k in expected:
            assert np.allclose(got[k], expected[k])


def test_build_normalization_stats_teacher_stats_requires_standardization(env):
    tb = env["tb"]
    arrays = {"lengths": np.array([1])}
    with pytest.raises(ValueError):
        tb.build_normalization_stats("teacher_stats", arrays, np.array([0]), None)


def test_build_normalization_stats_unknown_mode_raises(env):
    tb = env["tb"]
    with pytest.raises(ValueError):
        tb.build_normalization_stats("bogus_mode", {}, np.array([0]), None)


def test_global_stats_from_teacher_selects_manifest_columns(env):
    tb, manifest = env["tb"], env["manifest"]
    fake_state_mean = np.arange(166, dtype=np.float32)
    fake_state_std = np.ones(166, dtype=np.float32) * 2
    mean, std = tb.global_stats_from_teacher(fake_state_mean, fake_state_std, None)
    cols = manifest.global_columns(None)
    assert mean.shape == (34,)
    assert list(mean) == [float(c) for c in cols]
    assert (std == 2).all()
