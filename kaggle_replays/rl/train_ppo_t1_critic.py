"""critic+GAE付きPPO更新(D2.1後PPO pilot、オーロンゲcritic実験)。

critic無しの``train_ppo_t1.py``は変更しない、新規並存モジュール。dropout不整合の
既知バグ(model.train()でforwardするとold_logpとの比較にdropoutノイズが混入する)を
踏まえ、ここでも更新forwardは常に``model.eval()``で行う。
"""

from __future__ import annotations

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
import gae  # noqa: E402


def build_critic_batch(trajectories: list[dict], model, vocab, gamma: float = 1.0, lam: float = 0.95,
                       device: str = "cpu") -> dict:
    """rolloutの試合リストから、GAE計算込みのPPO更新バッチを作る。

    ``old_value``はrollout収集時点の凍結方策(引数の``model``、まだ更新していない状態)
    で計算する(``old_logp``と同じタイミング)。GAEは試合(trajectory)ごとに個別に
    ``gae.compute_gae_for_trajectory``を呼ぶことで、試合境界をまたいだbootstrapを防ぐ。
    """
    decisions, actions, old_logps, adv_list, ret_list, old_values_list = [], [], [], [], [], []
    model.eval()
    for traj in trajectories:
        traj_decisions = traj["decisions"]
        n = len(traj_decisions)
        if n == 0:
            continue
        arrays_list = [d["arrays"] for d in traj_decisions]
        arrays = tr.concat_single_decision_arrays(arrays_list)
        batch = tb.build_batch(arrays, vocab)
        inputs = la._to_tensors(batch, device)
        with torch.no_grad():
            values = model.value(inputs["board_numeric"], inputs["board_card_idx"],
                                 inputs["board_zone_ids"], inputs["board_owner_ids"],
                                 inputs["board_mask"], inputs["global_features"]).cpu().numpy()
        rewards = gae.trajectory_rewards(n, traj["reward"])
        advantages, returns = gae.compute_gae_for_trajectory(rewards, values, gamma=gamma, lam=lam)

        decisions.extend(arrays_list)
        actions.extend(d["chosen"] for d in traj_decisions)
        old_logps.extend(d["old_logp"] for d in traj_decisions)
        adv_list.extend(advantages.tolist())
        ret_list.extend(returns.tolist())
        old_values_list.extend(values.tolist())

    if not decisions:
        return {"decisions": [], "actions": [], "old_logps": torch.zeros(0), "old_values": torch.zeros(0),
               "returns": torch.zeros(0), "advantages": torch.zeros(0),
               "advantages_raw_mean": float("nan"), "advantages_raw_std": float("nan"),
               "returns_mean": float("nan"), "returns_std": float("nan")}

    adv_arr = np.asarray(adv_list, dtype=np.float32)
    adv_mean, adv_std = float(adv_arr.mean()), float(max(adv_arr.std(), 1e-6))
    adv_norm = (adv_arr - adv_mean) / adv_std

    return {
        "decisions": decisions, "actions": actions,
        "old_logps": torch.tensor(old_logps, dtype=torch.float32, device=device),
        "old_values": torch.tensor(old_values_list, dtype=torch.float32, device=device),
        "returns": torch.tensor(ret_list, dtype=torch.float32, device=device),
        "advantages": torch.tensor(adv_norm, dtype=torch.float32, device=device),
        "advantages_raw_mean": adv_mean, "advantages_raw_std": adv_std,
        "returns_mean": float(np.mean(ret_list)), "returns_std": float(np.std(ret_list)),
    }


def ppo_update_with_critic(model, optimizer, vocab, batch: dict, device: str = "cpu",
                           clip_eps: float = 0.1, entropy_coef: float = 0.01, grad_clip: float = 0.5,
                           value_loss_coef: float = 0.5, update_epochs: int = 2,
                           target_kl: float | None = 0.015) -> list[dict]:
    decisions = batch["decisions"]
    if not decisions:
        return [{"loss": float("nan"), "policy_loss": float("nan"), "value_loss": float("nan"),
                "entropy": float("nan"), "approx_kl": float("nan"), "clip_fraction": float("nan"),
                "grad_norm": float("nan"), "explained_variance": float("nan"),
                "value_pred_mean": float("nan"), "value_pred_std": float("nan"),
                "batch_size": 0, "stopped_early_target_kl": False}]

    arrays = tr.concat_single_decision_arrays(decisions)
    batched = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batched, device)
    mask = inputs["option_mask"]
    actions_t = torch.tensor(batch["actions"], dtype=torch.long, device=device).unsqueeze(1)
    old_logp_t = batch["old_logps"]
    old_value_t = batch["old_values"]
    returns_t = batch["returns"]
    adv_t = batch["advantages"]

    stats_list = []
    for epoch in range(update_epochs):
        # [dropout不整合バグの再発防止] rollout収集時(old_logp/old_value計算時)は
        # model.eval()なので、更新のforwardもeval()で行う(train()でdropoutを
        # 有効にすると、重み不変でもold_logp/old_valueとの差にノイズが混入する
        # ——critic無しPPO pilotで一度実際に踏んだバグと同じ原因)。
        model.eval()
        scores = model(**inputs)
        logp_all = F.log_softmax(scores, dim=1)
        new_logp = logp_all.gather(1, actions_t).squeeze(1)
        new_value = model.value(inputs["board_numeric"], inputs["board_card_idx"],
                                inputs["board_zone_ids"], inputs["board_owner_ids"],
                                inputs["board_mask"], inputs["global_features"])

        logratio = new_logp - old_logp_t
        ratio = torch.exp(logratio)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        v_loss = gae.value_loss_clipped(new_value, old_value_t, returns_t, clip_eps)

        probs = logp_all.exp()
        logp_masked = torch.where(mask, logp_all, torch.zeros_like(logp_all))
        probs_masked = torch.where(mask, probs, torch.zeros_like(probs))
        entropy = -(probs_masked * logp_masked).sum(dim=1).mean()

        loss = policy_loss + value_loss_coef * v_loss - entropy_coef * entropy

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            ev = gae.explained_variance(new_value.detach().cpu().numpy(), returns_t.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        stop_early = target_kl is not None and float(approx_kl.item()) > target_kl
        if not stop_early:
            optimizer.step()

        stats_list.append({
            "epoch": epoch, "loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
            "value_loss": float(v_loss.item()), "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()), "clip_fraction": float(clip_fraction.item()),
            "ratio_mean": float(ratio.mean().item()), "grad_norm": float(grad_norm),
            "explained_variance": ev, "value_pred_mean": float(new_value.mean().item()),
            "value_pred_std": float(new_value.std().item()),
            "batch_size": len(decisions), "stopped_early_target_kl": stop_early,
        })
        if stop_early:
            break
    return stats_list
