"""T1のNumPy推論(sample_submission/ptcg_ai/learning/t1_numpy_policy.py)と
torch版(T1OptionPolicy)の出力が一致することを確認する(提出用NumPy推論の正しさの検証)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
        / "mixed_teacher_t1_seed1" / "best.pt")
_NPZ = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "t1_weights" / "t1_seed1.npz"
_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


@pytest.fixture(scope="module")
def env(torch):
    if not (_CKPT.exists() and _NPZ.exists() and _TEACHER.exists() and _DECK.exists()):
        pytest.skip("checkpoint/npz/教師/デッキが無い")
    import t1_live_agent as la
    import token_batch as tb
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning import t1_numpy_policy as np_policy
    from run_league import read_deck_csv_file

    model, vocab, profile_name, _ = la.load_t1_for_inference(_CKPT, device="cpu")
    pm = PolicyModel(str(_TEACHER))
    numpy_policy = np_policy.load_t1_numpy_policy(str(_NPZ))
    deck = read_deck_csv_file(str(_DECK))
    return {"la": la, "tb": tb, "model": model, "vocab": vocab, "profile": profile_name,
           "pm": pm, "numpy_policy": numpy_policy, "deck": deck}


def _collect_decisions(env, n_games=3, seed0=321):
    import collect_tokens as ct
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import board_tokens as board_tokens_mod
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning import legacy_feature_manifest as manifest

    pm, deck, profile = env["pm"], env["deck"], env["profile"]
    decisions = []
    for g in range(n_games):
        obs_dict, start_data = battle_start(deck, deck)
        if start_data.errorType != 0:
            continue
        n = 0
        try:
            while n < 200 and len(decisions) < 30:
                obs = to_observation_class(obs_dict)
                cur = obs.current
                if cur is None or cur.result != -1:
                    break
                select = obs.select
                if select is not None and select.option:
                    if select.maxCount == 1:
                        d = ct._build_decision(pm, encoder, board_tokens_mod, manifest, profile,
                                               cur, select, False)
                        decisions.append(d)
                    scores = pm.score_options_from_state(cur, select)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                else:
                    action = []
                obs_dict = battle_select(action)
                n += 1
        finally:
            battle_finish()
        if len(decisions) >= 30:
            break
    return decisions


def test_numpy_matches_torch_forward(env, torch):
    """実際のcg対局から集めた決定点について、torch版とnumpy版のスコアが一致することを確認する。"""
    decisions = _collect_decisions(env)
    assert decisions, "決定点を収集できなかった"

    # 単一決定点ずつtorch/numpy両方でforwardして比較する(バッチ化はla.build_single_decision_arrays
    # 依存の都合上パスが別なので、token_batch.build_batchを両方に共通で使う)。
    tb, vocab = env["tb"], env["vocab"]
    max_diff = 0.0
    for d in decisions:
        arrays = {
            "board_counts": np.array([len(d["board_card_ids"])], dtype=np.int32),
            "counts": np.array([len(d["option_card_ids"])], dtype=np.int32),
            "board_token_numeric_features": (np.asarray(d["board_numeric_feats"], dtype=np.float32)
                                             if d["board_numeric_feats"] else np.zeros((0, 11), dtype=np.float32)),
            "board_token_card_ids": np.asarray(d["board_card_ids"], dtype=np.int32),
            "board_token_zone_ids": np.asarray(d["board_zone_ids"], dtype=np.int32),
            "legacy_option_features": np.asarray(d["legacy_option_feats"], dtype=np.float32),
            "option_card_ids": np.asarray(d["option_card_ids"], dtype=np.int32),
            "teacher_logits": np.zeros(len(d["option_card_ids"]), dtype=np.float32),
            "option_target_token_indices": np.asarray(d["option_target_indices"], dtype=np.int32),
            "chosen": np.zeros(1, dtype=np.int32),
            "legacy_global_features": np.asarray(d["legacy_global_feat"], dtype=np.float32).reshape(1, -1),
        }
        batch = tb.build_batch(arrays, vocab)
        n_opt = len(d["option_card_ids"])

        inputs = env["la"]._to_tensors(batch, "cpu")
        env["model"].eval()
        with torch.no_grad():
            torch_scores = env["model"](**inputs)[0, :n_opt].numpy()

        numpy_scores = env["numpy_policy"].forward(
            batch["board"]["numeric"], batch["board"]["card_idx"], batch["board"]["zone_ids"],
            batch["board"]["owner_ids"], batch["board"]["mask"], batch["legacy_global_features"],
            batch["option"]["legacy_option_features"], batch["option"]["card_idx"],
            batch["option"]["mask"])[0, :n_opt]

        diff = np.abs(torch_scores - numpy_scores).max()
        max_diff = max(max_diff, diff)
        assert np.allclose(torch_scores, numpy_scores, atol=1e-3), (
            f"torch/numpyのスコアが一致しない(max diff={diff})")
        assert int(np.argmax(torch_scores)) == int(np.argmax(numpy_scores))

    print(f"\n[T1 numpy parity] {len(decisions)}決定点、最大絶対誤差={max_diff:.2e}")
