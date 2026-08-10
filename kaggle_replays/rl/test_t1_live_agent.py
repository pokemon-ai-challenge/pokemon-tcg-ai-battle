"""T1ライブ推論ラッパー(t1_live_agent.py)のparity test(T1単独対戦評価 item2)。

offline収集経路(collect_tokens._build_decision)とライブ経路
(t1_live_agent.build_single_decision_arrays)が、**同じ決定点について**
同じtoken/mask/T1 logits/argmax行動を出すことを実際のcg対局で確認する。
illegal action・checkpoint不一致・空合法手のケースも検査する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
        / "stage100k_teacher_t1_seed1" / "best.pt")


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


@pytest.fixture(scope="module")
def env(torch):
    if not (_TEACHER.exists() and _DECK.exists() and _CKPT.exists()):
        pytest.skip(f"教師/デッキ/checkpointが無い: {_TEACHER} / {_DECK} / {_CKPT}")
    import t1_live_agent as la
    import collect_tokens as ct
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning import board_tokens as board_tokens_mod
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    from run_league import read_deck_csv_file

    pm = PolicyModel(str(_TEACHER))
    if not pm.is_ready:
        pytest.skip("v40教師を読み込めない")
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    model, vocab, ckpt_profile, payload = la.load_t1_for_inference(_CKPT, device="cpu")
    assert ckpt_profile == profile_name, "checkpointとv40のprofileが食い違う"

    return {"la": la, "ct": ct, "pm": pm, "profile": profile_name, "model": model,
           "vocab": vocab, "board_tokens_mod": board_tokens_mod, "encoder": encoder,
           "manifest": manifest, "deck": read_deck_csv_file(str(_DECK))}


@pytest.fixture(scope="module")
def recorded_decisions(env):
    """実際にcgエンジンで数試合対局し、自分側の単一選択の(cur, select)を数件記録する
    (教師のargmaxで両者を動かす。学習用収集ではなくparity検証用の軽量ループ)。"""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    pm = env["pm"]
    deck = env["deck"]
    recorded = []
    for game_seed in range(3):
        obs_dict, start_data = battle_start(deck, deck)
        if start_data.errorType != 0:
            continue
        n = 0
        try:
            while n < 200 and len(recorded) < 6:
                obs = to_observation_class(obs_dict)
                cur = obs.current
                if cur is None or cur.result != -1:
                    break
                select = obs.select
                if select is not None and select.option:
                    scores = pm.score_options_from_state(cur, select)
                    if cur.yourIndex == 0 and select.maxCount == 1:
                        recorded.append((cur, select))
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                else:
                    action = []
                obs_dict = battle_select(action)
                n += 1
        finally:
            battle_finish()
        if len(recorded) >= 6:
            break
    if not recorded:
        pytest.skip("単一選択の決定点を録れなかった")
    return recorded


def test_offline_and_live_tokens_match(env, recorded_decisions):
    """同じ(cur, select)について、offline経路(collect_tokens._build_decision)と
    ライブ経路(build_single_decision_arrays)が同じboard/optionトークンを作ることを確認する。"""
    ct, la = env["ct"], env["la"]
    pm, profile = env["pm"], env["profile"]
    for cur, select in recorded_decisions:
        offline = ct._build_decision(pm, env["encoder"], env["board_tokens_mod"], env["manifest"],
                                     profile, cur, select, debug_pointers=False)
        live = la.build_single_decision_arrays(pm, cur, select, profile)

        assert np.allclose(offline["legacy_option_feats"], live["legacy_option_features"])
        assert list(offline["option_card_ids"]) == list(live["option_card_ids"])
        assert list(offline["board_card_ids"]) == list(live["board_token_card_ids"])
        assert list(offline["board_zone_ids"]) == list(live["board_token_zone_ids"])
        assert len(offline["board_numeric_feats"]) == len(live["board_token_numeric_features"])
        if len(offline["board_numeric_feats"]):
            assert np.allclose(offline["board_numeric_feats"], live["board_token_numeric_features"])
        assert list(offline["option_target_indices"]) == list(live["option_target_token_indices"])


def test_offline_and_live_t1_scores_and_argmax_match(env, recorded_decisions):
    """同じ決定点について、offline収集経路のtoken(shard書き出し→読み込み→build_batch)と
    ライブ経路(build_single_decision_arrays→build_batch)を通したT1のスコア・argmaxが一致する。"""
    import tempfile
    import token_shard as ts
    import token_batch as tb

    ct, la = env["ct"], env["la"]
    pm, profile, model, vocab = env["pm"], env["profile"], env["model"], env["vocab"]

    for cur, select in recorded_decisions:
        offline = ct._build_decision(pm, env["encoder"], env["board_tokens_mod"], env["manifest"],
                                     profile, cur, select, debug_pointers=False)
        offline["chosen_idx"] = 0
        offline["logprob"] = 0.0
        traj = [{"steps": [offline], "reward": 1.0, "opp": 0}]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.npz"
            ts.write_token_shard(path, [offline], traj,
                                 {"run_id": "parity", "generation": 0})
            arrays, _ = ts.read_token_shard(path)
        offline_batch = tb.build_batch(arrays, vocab)
        offline_scores = la._to_tensors(offline_batch, "cpu")
        model.eval()
        import torch
        with torch.no_grad():
            offline_out = model(**offline_scores)[0]
        n_opt = len(select.option)
        offline_scores_np = offline_out[:n_opt].cpu().numpy()

        live_scores_np = la.t1_scores(model, vocab, pm, cur, select, profile, device="cpu")

        assert np.allclose(offline_scores_np, live_scores_np, atol=1e-5)
        assert int(np.argmax(offline_scores_np)) == int(np.argmax(live_scores_np))
        # argmaxは常に合法手(paddingを含まない)の範囲内
        assert 0 <= int(np.argmax(live_scores_np)) < n_opt


def test_illegal_action_never_selected(env):
    """paddingを含むバッチでも、argmaxが常にmask=Trueの範囲内(合法手)を指すことを確認する
    (T1.forwardがpaddingを-infでmaskしているため)。"""
    import torch
    import token_batch as tb

    model, vocab = env["model"], env["vocab"]
    arrays = {
        "board_counts": np.array([1, 1], dtype=np.int32),
        "counts": np.array([2, 5], dtype=np.int32),  # 決定点0は2択、決定点1は5択
        "board_token_numeric_features": np.zeros((2, 11), dtype=np.float32),
        "board_token_card_ids": np.array([1, 1], dtype=np.int32),
        "board_token_zone_ids": np.array([0, 0], dtype=np.int32),
        "legacy_option_features": np.zeros((7, 65), dtype=np.float32),
        "option_card_ids": np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        "teacher_logits": np.zeros(7, dtype=np.float32),
        "option_target_token_indices": np.full(7, -1, dtype=np.int32),
        "chosen": np.array([0, 0], dtype=np.int32),
        "legacy_global_features": np.zeros((2, 145), dtype=np.float32),
    }
    import t1_live_agent as la
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, "cpu")
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)
    argmax_idx = scores.argmax(dim=1)
    assert int(argmax_idx[0]) < 2, "決定点0(合法手2件)のargmaxがpaddingを指した"
    assert int(argmax_idx[1]) < 5, "決定点1(合法手5件)のargmaxがpaddingを指した"


def test_checkpoint_mismatch_raises(env):
    """現在のvocabularyと食い違うcheckpointをload_t1_for_inference相当の検査に通すと
    CheckpointMismatchErrorになる(既存load_checkpointの契約をそのまま使っていることの確認)。"""
    import token_policy_t1 as t1

    with pytest.raises(t1.CheckpointMismatchError):
        t1.load_checkpoint(
            _CKPT,
            token_schema_version="ptcg-rl-token-shard/999-wrong",
            vocabulary_version=env["vocab"].version, vocabulary_hash=env["vocab"].hash,
            card_vocab_size=env["vocab"].size, feature_profile=env["profile"],
            global_feature_manifest_hash=env["manifest"].manifest_hash(env["profile"]),
            map_location="cpu")


def test_empty_legal_moves_raises(env):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_start

    la = env["la"]
    obs_dict, start_data = battle_start(env["deck"], env["deck"])
    assert start_data.errorType == 0
    try:
        obs = to_observation_class(obs_dict)
        cur = obs.current
        select = obs.select

        class _EmptySelect:
            option = []
            minCount = 0
            maxCount = 0

        with pytest.raises(la.EmptyLegalMovesError):
            la.build_single_decision_arrays(env["pm"], cur, _EmptySelect(), env["profile"])
    finally:
        battle_finish()
