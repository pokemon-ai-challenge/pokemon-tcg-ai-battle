"""ポテンシャルベースの報酬シェーピング(Ng, Harada & Russell, 1999)。

設計書(``test_plan/ptcg_rl_design.md`` §3.2):

```
Phi(s)    = 相手の残りサイド - 自分の残りサイド      # 有利なほど大きい
F(s, s')  = gamma * Phi(s') - Phi(s)
```

この形を選ぶ理由は、決定 t=0..T-1 について

```
Σ_t gamma^t * F(s_t, s_{t+1}) = gamma^T * Phi(s_T) - Phi(s_0)
```

という望遠鏡和(telescoping sum)がどんな gamma・どんな途中経過でも厳密に成り立つこと。
つまり「サイドを取れ」という中間シグナルの総和は、試合を通じて足し合わせても
終端の Phi と初期の Phi の差だけに畳み込まれ、途中の寄り道では変わらない。
これが Ng et al. の「最適方策を歪めない」ことの直感的な根拠であり、
``kaggle_replays/tests/test_rl_gradient.py`` で数値的に検証する。
"""

from __future__ import annotations


def potential(own_n_prize: float, opp_n_prize: float) -> float:
    """Phi(s) = 相手の残りサイド - 自分の残りサイド。"""
    return float(opp_n_prize) - float(own_n_prize)


def shaping_reward(
    own_n_prize_t: float,
    opp_n_prize_t: float,
    own_n_prize_t1: float,
    opp_n_prize_t1: float,
    gamma: float,
) -> float:
    """F(s_t, s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t)。"""
    phi_t = potential(own_n_prize_t, opp_n_prize_t)
    phi_t1 = potential(own_n_prize_t1, opp_n_prize_t1)
    return gamma * phi_t1 - phi_t
