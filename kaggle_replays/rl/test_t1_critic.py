"""オーロンゲcritic+GAE PPO pilot実装前のテスト(9項目)。"""

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

import gae  # noqa: E402


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


# 1. 非終端rewardが0、終端だけ±1になっている
def test_trajectory_rewards_only_terminal_nonzero():
    r_win = gae.trajectory_rewards(5, 1.0)
    r_lose = gae.trajectory_rewards(5, -1.0)
    assert list(r_win[:-1]) == [0.0, 0.0, 0.0, 0.0]
    assert r_win[-1] == 1.0
    assert list(r_lose[:-1]) == [0.0, 0.0, 0.0, 0.0]
    assert r_lose[-1] == -1.0
    r_draw = gae.trajectory_rewards(3, 0.0)
    assert list(r_draw) == [0.0, 0.0, 0.0]


# 2. GAEがtrajectory境界を越えない
def test_gae_does_not_leak_across_trajectory_boundary():
    values_a1 = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    rewards_a = gae.trajectory_rewards(3, 1.0)
    adv_b, ret_b = gae.compute_gae_for_trajectory(
        gae.trajectory_rewards(2, -1.0), np.array([0.5, 0.6], dtype=np.float32))

    # trajectory Aの値を変えても、Bを独立に呼んだ結果は変化しない
    # (2つのtrajectoryを1本の配列に結合して計算する実装だったら、境界をまたいだ
    # bootstrapが混入してこの独立性が壊れる)。
    adv_b2, ret_b2 = gae.compute_gae_for_trajectory(
        gae.trajectory_rewards(2, -1.0), np.array([0.5, 0.6], dtype=np.float32))
    assert np.allclose(adv_b, adv_b2)
    assert np.allclose(ret_b, ret_b2)

    values_a2 = np.array([9.0, 9.0, 9.0], dtype=np.float32)  # 全く違う値にしてもBは無関係
    _ = gae.compute_gae_for_trajectory(rewards_a, values_a2)
    adv_b3, ret_b3 = gae.compute_gae_for_trajectory(
        gae.trajectory_rewards(2, -1.0), np.array([0.5, 0.6], dtype=np.float32))
    assert np.allclose(adv_b, adv_b3)
    assert np.allclose(ret_b, ret_b3)


# 3. terminalでは次状態のvalueをbootstrapしない
def test_terminal_step_does_not_bootstrap_next_value():
    # 最終ステップのadvantageはbootstrap無しなら reward[-1] - value[-1] のみになるはず。
    rewards = gae.trajectory_rewards(3, 1.0)
    values = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    adv, ret = gae.compute_gae_for_trajectory(rewards, values, gamma=1.0, lam=0.95)
    assert abs(adv[-1] - (1.0 - 0.9)) < 1e-6
    assert abs(ret[-1] - 1.0) < 1e-6  # return = advantage + value = 0.1 + 0.9 = 1.0


# 4. synthetic trajectoryでGAEとreturnが手計算と一致する
def test_gae_matches_hand_computed_values():
    rewards = np.array([0, 0, 1], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    adv, ret = gae.compute_gae_for_trajectory(rewards, values, gamma=1.0, lam=0.95)
    # 手計算(gamma=1, lambda=0.95):
    #   delta_2 = 1 + 0 - 0.9 = 0.1              -> adv[2] = 0.1
    #   delta_1 = 0 + 0.9 - 0.2 = 0.7             -> adv[1] = 0.7 + 0.95*0.1 = 0.795
    #   delta_0 = 0 + 0.2 - 0.1 = 0.1             -> adv[0] = 0.1 + 0.95*0.795 = 0.85525
    expected_adv = np.array([0.85525, 0.795, 0.1], dtype=np.float32)
    expected_ret = expected_adv + values
    assert np.allclose(adv, expected_adv, atol=1e-5)
    assert np.allclose(ret, expected_ret, atol=1e-5)


# 5. legal-action maskが維持される(criticはoptionに一切依存しない)
def test_value_does_not_depend_on_option_order_or_choice(torch):
    import token_policy_t1 as t1

    model = t1.T1OptionPolicy(global_dim=10, option_dim=5, board_vocab_size=20, num_zones=9)
    model.eval()
    B, n_board, n_opt = 2, 3, 4
    board_numeric = torch.randn(B, n_board, 11)
    board_card_idx = torch.randint(0, 20, (B, n_board))
    board_zone_ids = torch.randint(0, 4, (B, n_board))
    board_owner_ids = torch.randint(0, 2, (B, n_board))
    board_mask = torch.ones(B, n_board, dtype=torch.bool)
    global_features = torch.randn(B, 10)

    with torch.no_grad():
        v1 = model.value(board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
                         global_features)
        v2 = model.value(board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
                         global_features)
    # 同じ状態なら常に同じvalue(option引数が無いので並び順・選択の影響を受けようがない)
    assert torch.allclose(v1, v2)
    assert v1.shape == (B,)

    # forward()にoptionを渡してもvalueの計算には一切関与しない(シグネチャ上optionを
    # 受け取らないことがそのまま保証になっているが、念のためmodel.value()のシグネチャに
    # option系引数が存在しないことも確認する)。
    import inspect
    sig = inspect.signature(model.value)
    assert "option_features" not in sig.parameters
    assert "option_card_idx" not in sig.parameters
    assert "option_mask" not in sig.parameters


# 6. old log-probとold valueが更新中に変化しない
def test_old_logp_and_old_value_unchanged_during_update(torch):
    import train_ppo_t1 as tp
    import train_ppo_t1_critic as tpc
    import t1_rollout as tr
    from ptcg_ai.learning.policy_model import PolicyModel

    ckpt = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_grimmsnarl" / "train"
           / "rule_teacher_seed0" / "best.pt")
    if not ckpt.exists():
        pytest.skip("オーロンゲcheckpointが無い")
    import t1_live_agent as la
    model, vocab, profile, _ = la.load_t1_for_inference(ckpt, device="cpu")
    teacher = PolicyModel(str(_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models"
                              / "model_v40.json"))
    from run_league import read_deck_csv_file
    deck = read_deck_csv_file(str(_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                                  / "marnie_grimmsnarl_ex" / "01.csv"))
    trajs, _ = tr.collect_rollout(model, vocab, teacher,
                                  [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
                                  deck, 4, seed0=999, profile_name=profile, device="cpu")
    batch = tpc.build_critic_batch(trajs, model, vocab, device="cpu")
    old_logps_before = batch["old_logps"].clone()
    old_values_before = batch["old_values"].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    tpc.ppo_update_with_critic(model, optimizer, vocab, batch, device="cpu",
                               clip_eps=0.1, entropy_coef=0.01, grad_clip=0.5, value_loss_coef=0.5,
                               update_epochs=2, target_kl=0.015)

    assert torch.allclose(old_logps_before, batch["old_logps"])
    assert torch.allclose(old_values_before, batch["old_values"])


# 7. dropoutを含むtrain/eval modeの不整合がない
def test_critic_update_forward_is_deterministic_matching_rollout_mode(torch):
    import train_ppo_t1 as tp
    import train_ppo_t1_critic as tpc
    import t1_rollout as tr
    from ptcg_ai.learning.policy_model import PolicyModel

    ckpt = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_grimmsnarl" / "train"
           / "rule_teacher_seed0" / "best.pt")
    if not ckpt.exists():
        pytest.skip("オーロンゲcheckpointが無い")
    import t1_live_agent as la
    model, vocab, profile, _ = la.load_t1_for_inference(ckpt, device="cpu")
    teacher = PolicyModel(str(_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models"
                              / "model_v40.json"))
    from run_league import read_deck_csv_file
    deck = read_deck_csv_file(str(_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                                  / "marnie_grimmsnarl_ex" / "01.csv"))
    trajs, _ = tr.collect_rollout(model, vocab, teacher,
                                  [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
                                  deck, 6, seed0=888, profile_name=profile, device="cpu")
    batch = tpc.build_critic_batch(trajs, model, vocab, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    stats = tpc.ppo_update_with_critic(model, optimizer, vocab, batch, device="cpu",
                                       clip_eps=0.1, entropy_coef=0.01, grad_clip=0.5,
                                       value_loss_coef=0.5, update_epochs=1, target_kl=None)
    assert abs(stats[0]["approx_kl"]) < 1e-3, (
        f"更新前のforwardなのにapprox_klが大きい({stats[0]['approx_kl']:.4f})。"
        "dropoutが有効なまま比較している可能性がある。")


# 8. 既存checkpointを壊さず読み込める
def test_load_checkpoint_without_value_head_initializes_value_head(torch):
    import token_policy_t1 as t1
    import tempfile

    model = t1.T1OptionPolicy(global_dim=10, option_dim=5, board_vocab_size=20, num_zones=9)
    # value headを持たない「旧checkpoint」を模擬するため、state_dictから明示的に除く。
    state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("value_fc")}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.pt"
        payload = {
            "model_state_dict": state_dict,
            "model_config": {"global_dim": 10, "option_dim": 5, "board_vocab_size": 20, "num_zones": 9},
            "token_schema_version": "x", "vocabulary_version": 1, "vocabulary_hash": "h",
            "feature_profile": "p", "global_feature_manifest_hash": "m",
            "normalization_mode": "student_train_stats",
            "normalization_stats": {"board_numeric_mean": [0.0] * 11, "board_numeric_std": [1.0] * 11,
                                    "global_mean": [0.0] * 10, "global_std": [1.0] * 10,
                                    "option_mean": [0.0] * 5, "option_std": [1.0] * 5},
            "use_board_card_id": True, "card_vocab_size": 20,
        }
        torch.save(payload, path)
        loaded, _ = t1.load_checkpoint(path, token_schema_version="x", vocabulary_version=1,
                                       vocabulary_hash="h", card_vocab_size=20, feature_profile="p",
                                       global_feature_manifest_hash="m")
    assert hasattr(loaded, "value_fc1") and hasattr(loaded, "value_fc2")
    # 新規初期化されたvalue headでforwardが破綻しない(NaN/Infなし)
    v = loaded.value(torch.randn(1, 2, 11), torch.randint(0, 20, (1, 2)), torch.randint(0, 4, (1, 2)),
                     torch.randint(0, 2, (1, 2)), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 10))
    assert torch.isfinite(v).all()


# 9. 保存・再ロード後にpolicy/value出力が一致する
def test_save_and_reload_checkpoint_matches_policy_and_value_output(torch):
    import token_policy_t1 as t1
    import tempfile

    model = t1.T1OptionPolicy(global_dim=10, option_dim=5, board_vocab_size=20, num_zones=9)
    model.eval()
    board_numeric = torch.randn(1, 2, 11)
    board_card_idx = torch.randint(0, 20, (1, 2))
    board_zone_ids = torch.randint(0, 4, (1, 2))
    board_owner_ids = torch.randint(0, 2, (1, 2))
    board_mask = torch.ones(1, 2, dtype=torch.bool)
    global_features = torch.randn(1, 10)
    option_features = torch.randn(1, 3, 5)
    option_card_idx = torch.randint(0, 20, (1, 3))
    option_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        score_before = model(board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
                             global_features, option_features, option_card_idx, option_mask)
        value_before = model.value(board_numeric, board_card_idx, board_zone_ids, board_owner_ids,
                                   board_mask, global_features)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        t1.save_checkpoint(model, path, {"global_dim": 10, "option_dim": 5, "board_vocab_size": 20,
                                        "num_zones": 9},
                           token_schema_version="x", vocabulary_version=1, vocabulary_hash="h",
                           feature_profile="p", global_feature_manifest_hash="m",
                           normalization_mode="student_train_stats",
                           normalization_stats={"board_numeric_mean": [0.0] * 11, "board_numeric_std": [1.0] * 11,
                                               "global_mean": [0.0] * 10, "global_std": [1.0] * 10,
                                               "option_mean": [0.0] * 5, "option_std": [1.0] * 5},
                           card_vocab_size=20)
        reloaded, _ = t1.load_checkpoint(path, token_schema_version="x", vocabulary_version=1,
                                         vocabulary_hash="h", card_vocab_size=20, feature_profile="p",
                                         global_feature_manifest_hash="m")
    reloaded.eval()
    with torch.no_grad():
        score_after = reloaded(board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
                              global_features, option_features, option_card_idx, option_mask)
        value_after = reloaded.value(board_numeric, board_card_idx, board_zone_ids, board_owner_ids,
                                     board_mask, global_features)
    assert torch.allclose(score_before, score_after, atol=1e-6)
    assert torch.allclose(value_before, value_after, atol=1e-6)
