"""T1(Transformer)自身のon-policy rolloutによるPPO学習。

**既存``cg``/``data``/既存MLP推論/既存PPO(``kaggle_replays/rl/distributed/learner.py``等)は
一切変更しない。** 蒸留(``train_distill_t1.py``)とは別の、並存する新規trainer。

**方式: critic-free PPO-Clip(ユーザー確認済み、2026-08-09)。**
- value関数(critic)を持たない。GAEは計算しない。
- advantageは各trajectoryの終局returnをrollout batch内で平均・標準偏差により
  正規化した値(Monte Carlo return-based advantage)。
- 割引は行わない(終局報酬をそのtrajectoryの全決定点へそのまま伝播する。
  ゲーム内に途中報酬が無いため、割引率は実質``gamma=1.0``相当。実装上
  割引計算自体をしていないため、``gamma``という設定値は存在しない)。
- ``value_loss``・``explained_variance``・``GAE``・``GAE lambda``は本実装には
  存在しないためN/Aとして扱う(criticを追加すれば計算できるが、今回のpilotでは
  意図的に追加しない)。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import t1_live_agent as la  # noqa: E402
import t1_rollout as tr  # noqa: E402
import token_batch as tb  # noqa: E402


def trajectories_to_flat_decisions(trajectories: list[dict]):
    """rolloutの試合リストを、PPO更新用にフラットな(decision, action, old_logp, return)へ展開する。
    returnは「その試合の終局報酬(+1/-1)」をその試合の全決定点にそのまま割り当てる
    (episodic・非割引。ゲーム内に途中報酬が無いため)。paddingや無効stepは元々
    ``decisions``に含まれない(``t1_rollout``がT1自身の実決定点しか記録しないため、
    正規化統計に混ざらない)。"""
    decisions, actions, old_logps, returns = [], [], [], []
    for traj in trajectories:
        for d in traj["decisions"]:
            decisions.append(d["arrays"])
            actions.append(d["chosen"])
            old_logps.append(d["old_logp"])
            returns.append(traj["reward"])
    return decisions, actions, old_logps, returns


def compute_advantages(returns: list[float]) -> dict:
    """バッチ平均・標準偏差で正規化(価値関数なしのシンプルなbaseline)。
    正規化前後の統計を両方返す(検査・ログ用)。"""
    r = np.asarray(returns, dtype=np.float32)
    mean, std = float(r.mean()), float(r.std())
    safe_std = max(std, 1e-6)
    adv = (r - mean) / safe_std
    return {"advantages": adv, "return_mean": mean, "return_std": std,
           "advantage_mean_before_norm": mean, "advantage_std_before_norm": std,
           "advantage_mean_after_norm": float(adv.mean()), "advantage_std_after_norm": float(adv.std())}


def build_probe_set(trajectories: list[dict], max_probe: int = 256) -> list[dict]:
    """初期方策からのKL計測用に、決定点の固定サンプルを1回だけ切り出す
    (以後この同じprobe setを使い続ける。比較対象の局面集合を変えないため)。"""
    decisions = [d["arrays"] for traj in trajectories for d in traj["decisions"]]
    if len(decisions) > max_probe:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(decisions), size=max_probe, replace=False)
        decisions = [decisions[i] for i in idx]
    return decisions


def compute_kl_from_initial(current_model, initial_model, vocab, probe_decisions: list[dict],
                            device: str = "cpu") -> float:
    """固定probe setに対する、初期方策からの平均KL(KL(initial || current))。"""
    if not probe_decisions:
        return float("nan")
    arrays = tr.concat_single_decision_arrays(probe_decisions)
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    mask = inputs["option_mask"]
    current_model.eval(); initial_model.eval()
    with torch.no_grad():
        s_cur = current_model(**inputs)
        s_init = initial_model(**inputs)
        logp_cur = F.log_softmax(s_cur, dim=1)
        logp_init = F.log_softmax(s_init, dim=1)
        p_init = logp_init.exp()
        contrib = torch.where(mask, p_init * (logp_init - logp_cur), torch.zeros_like(logp_cur))
        kl = contrib.sum(dim=1).mean()
    return float(kl.item())


def ppo_update(model, optimizer, vocab, decisions: list[dict], actions: list[int],
               old_logps: list[float], advantages: np.ndarray, device: str = "cpu",
               clip_eps: float = 0.2, entropy_coef: float = 0.01, grad_clip: float = 1.0,
               update_epochs: int = 1, target_kl: float | None = None) -> list[dict]:
    """複数epoch(同一バッチを使い回す、ミニバッチ分割はしない)。target_klを超えたら
    そのepochで打ち切る(``approx_kl``はSchulmanのk3推定量、低分散)。
    epochごとの統計をリストで返す。"""
    arrays = tr.concat_single_decision_arrays(decisions)
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    mask = inputs["option_mask"]
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    old_logp_t = torch.tensor(old_logps, dtype=torch.float32, device=device)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    stats_list = []
    for epoch in range(update_epochs):
        # [D2.1後PPO pilotで発見・修正: dropout不整合バグ]
        # T1のTransformer層にはdropout(0.05)があり、model.train()で有効になる。
        # rollout収集時のold_logpはmodel.eval()(dropout無し)で計算しているため、
        # ここでもtrain()にするとold_logp/new_logpの差にdropoutノイズが混入し、
        # 実際のパラメータ更新が無くてもapprox_klが大きく見える(最大0.40の乖離を
        # 実測、target_kl=0.015を常に超えてoptimizer.step()が一度も実行されない
        # 状態になっていた)。eval()のまま(dropout無し)勾配計算する
        # (.eval()でも.backward()・パラメータ更新は正常に行える。T1にbatchnormは無い)。
        model.eval()
        scores = model(**inputs)
        logp_all = F.log_softmax(scores, dim=1)
        new_logp = logp_all.gather(1, actions_t).squeeze(1)

        logratio = new_logp - old_logp_t
        ratio = torch.exp(logratio)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        probs = logp_all.exp()
        logp_masked = torch.where(mask, logp_all, torch.zeros_like(logp_all))
        probs_masked = torch.where(mask, probs, torch.zeros_like(probs))
        entropy = -(probs_masked * logp_masked).sum(dim=1).mean()

        loss = policy_loss - entropy_coef * entropy

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        stop_early = target_kl is not None and float(approx_kl.item()) > target_kl
        if not stop_early:
            optimizer.step()

        stats_list.append({
            "epoch": epoch, "loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()), "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()), "ratio_mean": float(ratio.mean().item()),
            "ratio_max": float(ratio.max().item()), "grad_norm": float(grad_norm),
            "batch_size": len(decisions), "stopped_early_target_kl": stop_early,
        })
        if stop_early:
            break
    return stats_list


def save_ppo_checkpoint(model, optimizer, step: int, path, extra: dict | None = None) -> None:
    import random as _random
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "rng_state": {"python": _random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state()},
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ppo_checkpoint(path, model, optimizer, device: str = "cpu") -> dict:
    import random as _random
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = payload["rng_state"]
    _random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    return payload


def freeze_copy(model) -> "torch.nn.Module":
    """現在方策の凍結コピーを作る(勾配を切った独立インスタンス、rollout中は更新されない)。"""
    snap = copy.deepcopy(model)
    for p in snap.parameters():
        p.requires_grad_(False)
    snap.eval()
    return snap
