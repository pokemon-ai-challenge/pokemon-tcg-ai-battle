"""GAE(Generalized Advantage Estimation)とcriticの価値損失(D2.1後PPO pilot、
オーロンゲcritic実験)。critic無しの``train_ppo_t1.py``は変更しない、新規並存モジュール。

trajectory(試合)境界をまたいでbootstrapしない: 呼び出し側が試合ごとに
``compute_gae_for_trajectory``を個別に呼ぶことで保証する(複数試合を1本の配列に
結合してから計算する設計にはしていない)。
"""

from __future__ import annotations

import numpy as np
import torch


def trajectory_rewards(n_decisions: int, outcome: float) -> np.ndarray:
    """非終端decision=0、最終decisionだけoutcome(勝利+1/敗北-1/引き分け0)、という
    reward配列を作る(``n_decisions``はそのtrajectoryの決定点数)。"""
    r = np.zeros(n_decisions, dtype=np.float32)
    if n_decisions > 0:
        r[-1] = outcome
    return r


def compute_gae_for_trajectory(rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0,
                               lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """1trajectory(試合)ぶんのGAE advantageとreturnを計算する。

    ``rewards``/``values``は同じ長さ(T,)。``values``はrollout収集時点(凍結方策)での
    V(s_t)(old_value)。最終ステップ(t=T-1)では次状態のvalueをbootstrapしない
    (ゲーム終了後の状態は評価対象に無いため、next_value=0・next_nonterminal=0として扱う)。
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
            next_nonterminal = 0.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def value_loss_clipped(new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
                       clip_eps: float) -> torch.Tensor:
    """PPO2式のclipped value loss。"""
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_eps, clip_eps)
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var[return - value] / Var[return]。criticがreturnの分散をどれだけ説明できているか
    (1に近いほど良い、0は「平均を返すのと同じ」、負はそれより悪い)。"""
    var_returns = float(np.var(returns))
    if var_returns < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_returns)
