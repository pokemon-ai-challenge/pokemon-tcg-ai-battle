"""T1-PPO pilotのpreflight test(rollout・old_logp一致・mask・resume・advantage符号等)。
長時間の学習は行わない。
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
    if not (_CKPT.exists() and _TEACHER.exists() and _DECK.exists()):
        pytest.skip("checkpoint/教師/デッキが無い")
    import t1_live_agent as la
    import t1_rollout as tr
    import train_ppo_t1 as tp
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    model, vocab, profile_name, _ = la.load_t1_for_inference(_CKPT, device="cpu")
    t1_state_pm = PolicyModel(str(_TEACHER))
    assert t1_state_pm.is_ready
    deck = read_deck_csv_file(str(_DECK))
    return {"la": la, "tr": tr, "tp": tp, "model": model, "vocab": vocab,
           "profile": profile_name, "pm": t1_state_pm, "deck": deck}


def _selfplay_specs():
    return [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}]


def test_old_logp_matches_frozen_policy(env, torch):
    """rolloutで保存したold_logpが、収集に使ったのと同じ(凍結)モデルへ同じ入力を
    再度forwardして得られる対数確率と一致することを確認する。"""
    tr = env["tr"]
    trajs, counts = tr.collect_rollout(env["model"], env["vocab"], env["pm"], _selfplay_specs(),
                                       env["deck"], n_games=4, seed0=11, profile_name=env["profile"],
                                       device="cpu")
    assert trajs
    for traj in trajs:
        for d in traj["decisions"]:
            import token_batch as tb
            batch = tb.build_batch(env["tr"].concat_single_decision_arrays([d["arrays"]]), env["vocab"])
            inputs = env["la"]._to_tensors(batch, "cpu")
            with torch.no_grad():
                scores = env["model"](**inputs)[0]
            n_opt = int(d["arrays"]["counts"][0])
            logp = torch.log_softmax(scores[:n_opt], dim=0)[d["chosen"]].item()
            assert abs(logp - d["old_logp"]) < 1e-4


def test_action_always_within_legal_mask(env):
    tr = env["tr"]
    trajs, _ = tr.collect_rollout(env["model"], env["vocab"], env["pm"], _selfplay_specs(),
                                  env["deck"], n_games=4, seed0=22, profile_name=env["profile"],
                                  device="cpu")
    for traj in trajs:
        for d in traj["decisions"]:
            n_opt = int(d["arrays"]["counts"][0])
            assert 0 <= d["chosen"] < n_opt


def test_ppo_update_keeps_legal_argmax_in_range(env, torch):
    tr, tp = env["tr"], env["tp"]
    model, vocab = env["model"], env["vocab"]
    trajs, _ = tr.collect_rollout(model, vocab, env["pm"], _selfplay_specs(), env["deck"],
                                  n_games=6, seed0=33, profile_name=env["profile"], device="cpu")
    decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
    assert len(decisions) > 1
    adv = tp.compute_advantages(returns)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps, adv["advantages"],
                 device="cpu", clip_eps=0.1, entropy_coef=0.01, grad_clip=0.5, update_epochs=2,
                 target_kl=0.015)
    # 更新後も、既存のt1_select_indexが常に合法手範囲内を返すことを確認(別途t1_live_agent側で
    # 既にmasked_fill(-inf)で保証されているが、更新後の重みでも成立することを直接確認する)。
    for traj in trajs[:3]:
        for d in traj["decisions"][:3]:
            import token_batch as tb
            batch = tb.build_batch(tr.concat_single_decision_arrays([d["arrays"]]), vocab)
            inputs = env["la"]._to_tensors(batch, "cpu")
            model.eval()
            with torch.no_grad():
                scores = model(**inputs)[0]
            n_opt = int(d["arrays"]["counts"][0])
            argmax_idx = int(torch.argmax(scores).item())
            assert argmax_idx < n_opt


def test_advantage_shape_dtype_device(env):
    tp = env["tp"]
    adv = tp.compute_advantages([1.0, -1.0, 1.0, -1.0, 1.0])
    a = adv["advantages"]
    assert a.dtype == np.float32
    assert a.shape == (5,)
    assert np.isfinite(a).all()


def test_advantage_sign_matches_outcome(env):
    """勝利(+1)のtrajectoryは正のadvantage、敗北(-1)は負のadvantageになることを確認する。"""
    tp = env["tp"]
    returns = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    adv = tp.compute_advantages(returns)["advantages"]
    assert (adv[:3] > 0).all()
    assert (adv[3:] < 0).all()


def test_advantage_no_nan_when_std_near_zero(env):
    """全員勝ち(標準偏差0)でもNaNにならない(std floorで保護)。"""
    tp = env["tp"]
    adv = tp.compute_advantages([1.0, 1.0, 1.0, 1.0])
    assert np.isfinite(adv["advantages"]).all()


def test_no_nan_inf_or_empty_trajectory_in_rollout(env):
    tr = env["tr"]
    trajs, counts = tr.collect_rollout(env["model"], env["vocab"], env["pm"], _selfplay_specs(),
                                       env["deck"], n_games=6, seed0=44, profile_name=env["profile"],
                                       device="cpu")
    assert trajs
    for traj in trajs:
        assert traj["decisions"], "空のtrajectoryが混入した"
        assert traj["reward"] in (1.0, -1.0)
        for d in traj["decisions"]:
            assert np.isfinite(d["old_logp"])


def test_ppo_checkpoint_save_and_resume(env, torch):
    import tempfile
    tp = env["tp"]
    model = env["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ppo_ckpt.pt"
        tp.save_ppo_checkpoint(model, optimizer, step=500, path=path, extra={"note": "smoke"})
        before = {k: v.clone() for k, v in model.state_dict().items()}
        # わざとパラメータを壊してからresumeし、復元されることを確認する
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        payload = tp.load_ppo_checkpoint(path, model, optimizer, device="cpu")
        assert payload["step"] == 500
        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)


def test_freeze_copy_is_independent(env, torch):
    tp = env["tp"]
    model = env["model"]
    snap = tp.freeze_copy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    changed = any(not torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), snap.parameters()))
    assert changed, "凍結コピーが元モデルの更新から独立していない"
    with torch.no_grad():
        for p in model.parameters():
            p.add_(-0.01)


def test_ppo_update_forward_is_deterministic_matching_rollout_mode(env, torch):
    """[回帰テスト] rollout収集時(model.eval())とPPO更新のforward(かつてはmodel.train()で
    dropoutが有効だった)が一致しない不具合の再現・固定化。

    未更新のモデルでold_logpを収集した直後、ppo_updateの最初のepochで計算される
    approx_kl(=このforwardとold_logpの差)がほぼ0であるべき(まだ実際のパラメータ
    更新は起きていない時点の値のため)。dropoutが有効なまま(train()モード)forwardして
    いた旧実装では、重みが同じでもdropoutノイズでapprox_klが0.015(target_kl)を
    大きく超え、結果としてoptimizer.step()が一度も呼ばれない、という不具合があった。"""
    tr, tp = env["tr"], env["tp"]
    model, vocab = env["model"], env["vocab"]
    trajs, _ = tr.collect_rollout(model, vocab, env["pm"], _selfplay_specs(), env["deck"],
                                  n_games=6, seed0=77, profile_name=env["profile"], device="cpu")
    decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
    assert len(decisions) > 1
    adv = tp.compute_advantages(returns)["advantages"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    stats = tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps, adv,
                          device="cpu", clip_eps=0.1, entropy_coef=0.01, grad_clip=0.5,
                          update_epochs=1, target_kl=None)
    approx_kl_epoch0 = stats[0]["approx_kl"]
    print(f"\n[dropout regression] approx_kl(epoch0, 更新前のforward) = {approx_kl_epoch0:.6f}")
    assert abs(approx_kl_epoch0) < 1e-3, (
        f"更新前のforwardなのにapprox_klが大きい({approx_kl_epoch0:.4f})。"
        "dropoutがtrain()で有効なままold_logpと比較している可能性がある。")


def test_kl_from_initial_is_zero_for_identical_model(env, torch):
    tp = env["tp"]
    model = env["model"]
    trajs, _ = env["tr"].collect_rollout(model, env["vocab"], env["pm"], _selfplay_specs(),
                                         env["deck"], n_games=3, seed0=55,
                                         profile_name=env["profile"], device="cpu")
    probe = tp.build_probe_set(trajs, max_probe=32)
    kl = tp.compute_kl_from_initial(model, model, env["vocab"], probe, device="cpu")
    assert abs(kl) < 1e-5


def test_win_probability_increases_after_ppo_update_toward_winning_actions(env, torch):
    """符号関係のテスト: 勝利trajectoryの選択行動の平均確率が更新後に上がり、
    敗北trajectoryの選択行動の平均確率が下がる傾向になることを確認する。

    単一の決定点で比較すると、合法手が1つしかない(確率が常に1.0で動きようが無い)
    決定点や、既に確率がほぼ飽和している決定点を偶然選んでしまい、符号が
    検出できないことがある(実際に最初のバージョンでこれが起きた)。
    そのため n_opt>1 の決定点に絞り、複数件の平均で比較する。"""
    tr, tp = env["tr"], env["tp"]
    model, vocab = env["model"], env["vocab"]
    trajs, _ = tr.collect_rollout(model, vocab, env["pm"], _selfplay_specs(), env["deck"],
                                  n_games=10, seed0=66, profile_name=env["profile"], device="cpu")
    decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
    assert len(set(returns)) > 1, "勝敗が偏り符号テストができない(再試行が必要)"

    def _logp_of(model, arrays, action):
        import token_batch as tb
        batch = tb.build_batch(tr.concat_single_decision_arrays([arrays]), vocab)
        inputs = env["la"]._to_tensors(batch, "cpu")
        model.eval()
        with torch.no_grad():
            scores = model(**inputs)[0]
        n_opt = int(arrays["counts"][0])
        return torch.log_softmax(scores[:n_opt], dim=0)[action].item()

    win_indices = [i for i, r in enumerate(returns) if r > 0 and int(decisions[i]["counts"][0]) > 1][:20]
    lose_indices = [i for i, r in enumerate(returns) if r < 0 and int(decisions[i]["counts"][0]) > 1][:20]
    assert win_indices and lose_indices, "n_opt>1の勝敗サンプルが十分に無い(再試行が必要)"

    before_win = np.mean([_logp_of(model, decisions[i], actions[i]) for i in win_indices])
    before_lose = np.mean([_logp_of(model, decisions[i], actions[i]) for i in lose_indices])

    adv = tp.compute_advantages(returns)["advantages"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 符号確認のためsmoke testだけ大きめのlr
    for _ in range(5):
        tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps, adv, device="cpu",
                     clip_eps=0.2, entropy_coef=0.0, grad_clip=1.0, update_epochs=1)

    after_win = np.mean([_logp_of(model, decisions[i], actions[i]) for i in win_indices])
    after_lose = np.mean([_logp_of(model, decisions[i], actions[i]) for i in lose_indices])
    print(f"\n[sign check] win logp(avg) {before_win:.4f}->{after_win:.4f}  "
         f"lose logp(avg) {before_lose:.4f}->{after_lose:.4f}")
    assert after_win > before_win, "勝利行動の平均確率が上がらなかった"
    assert after_lose < before_lose, "敗北行動の平均確率が下がらなかった"
