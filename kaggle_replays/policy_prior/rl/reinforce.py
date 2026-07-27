"""REINFORCE + バッチ平均ベースライン + ポテンシャルベースのシェーピングによる勾配計算。

設計書(``test_plan/ptcg_rl_design.md`` §3):

```
Delta_w = alpha * Sum_t (R + F_t - b) * grad_w log pi(a_t)
```

- ``R``: 試合結果(勝ち+1/負け-1、その decision を下した席から見た値)
- ``F_t = gamma * Phi(s_{t+1}) - Phi(s_t)``(``rl.shaping``)
- ``b``: バッチ平均ベースライン。**このバッチに含まれる全 decision にわたる R の平均**を使う
  (decision 数で重み付けした平均になる。episode 単位の単純平均ではない。学習済みの
  ``value_model.py`` は使わない — 設計書 §3.1: 「実行経路から参照されておらず検証もされて
  いないため依存しない」)。

学習率 ``alpha`` はここでは掛けない(``apply_update`` に渡す)。
"""

from __future__ import annotations

from .policy import LinearPolicy, log_prob_gradient, softmax
from .shaping import shaping_reward


def _decision_potential_pairs(decisions: list[dict], terminal: dict) -> list[tuple[tuple, tuple]]:
    """各 decision について (s_t, s_{t+1}) の (own_n_prize, opp_n_prize) ペアを作る。

    最後の decision の s_{t+1} は試合終端時点のその席から見た残りサイド枚数
    (``terminal``)を使う。
    """
    n = len(decisions)
    pairs = []
    for t in range(n):
        s_t = (decisions[t]["own_n_prize"], decisions[t]["opp_n_prize"])
        if t + 1 < n:
            nxt = decisions[t + 1]
            s_t1 = (nxt["own_n_prize"], nxt["opp_n_prize"])
        else:
            s_t1 = (terminal["own_n_prize"], terminal["opp_n_prize"])
        pairs.append((s_t, s_t1))
    return pairs


def compute_batch_gradient(
    policy: LinearPolicy, episodes: list[dict], temperature: float, gamma: float
) -> tuple[dict[str, float], dict]:
    """バッチ全体の Σ_t (advantage_t) * grad_w log pi(a_t) と診断情報を返す。

    戻り値の grad は ``feature_name -> 値`` のスパースでない dict
    (``policy.feature_names`` の全キーを持つ。未使用特徴は 0.0)。
    """
    flat: list[tuple[int, float, dict]] = []  # (result, f_t, decision)
    for ep in episodes:
        pairs = _decision_potential_pairs(ep["decisions"], ep["terminal"])
        for dec, (s_t, s_t1) in zip(ep["decisions"], pairs):
            f_t = shaping_reward(s_t[0], s_t[1], s_t1[0], s_t1[1], gamma)
            flat.append((ep["result"], f_t, dec))

    n = len(flat)
    grad_accum: dict[str, float] = {name: 0.0 for name in policy.feature_names}
    if n == 0:
        return grad_accum, {
            "n_decisions": 0,
            "n_episodes": len(episodes),
            "baseline": 0.0,
            "mean_shaping": 0.0,
            "mean_advantage": 0.0,
        }

    baseline = sum(r for r, _, _ in flat) / n

    sum_f = 0.0
    sum_advantage = 0.0
    for r, f_t, dec in flat:
        feats = dec["features"]
        scores = policy.score_many(feats)
        probs = softmax(scores, temperature)
        g = log_prob_gradient(feats, probs, dec["chosen_index"], temperature)
        advantage = r + f_t - baseline
        sum_f += f_t
        sum_advantage += advantage
        for name, value in g.items():
            grad_accum[name] += advantage * value

    diagnostics = {
        "n_decisions": n,
        "n_episodes": len(episodes),
        "baseline": baseline,
        "mean_shaping": sum_f / n,
        "mean_advantage": sum_advantage / n,
    }
    return grad_accum, diagnostics


def apply_update(policy: LinearPolicy, grad: dict[str, float], lr: float) -> LinearPolicy:
    """勾配を適用した新しい ``LinearPolicy`` を返す(元の ``policy`` は変更しない)。

    ``intercept`` は更新しない: 全選択肢に同じ定数を足すだけなので softmax の下では
    ``softmax(score + c) == softmax(score)`` となり、方策勾配に対して不変
    (``score_i = intercept + w・x_i`` の intercept 部分は decision 内で全選択肢に共通)。
    """
    new_weights = list(policy.weights)
    index = policy._index
    for name, g in grad.items():
        i = index.get(name)
        if i is not None:
            new_weights[i] += lr * g
    return LinearPolicy(
        feature_names=policy.feature_names,
        weights=new_weights,
        intercept=policy.intercept,
        frequent_card_ids=policy.frequent_card_ids,
        card_attributes=policy.card_attributes,
        meta=dict(policy.meta),
    )
