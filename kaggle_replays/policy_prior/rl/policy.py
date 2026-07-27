"""確率的方策(softmax + 温度)と、その対数尤度勾配。

設計書(``test_plan/ptcg_rl_design.md`` §2)の定式化:

```
score_i = w・x_i          # x_i は選択肢 i の特徴ベクトル(ptcg_ai.learning.policy_features)
pi(i)   = softmax(score_i / T)
```

推論時(提出コード)は常に argmax のまま。ここでの温度付きサンプリングは学習時のみ使う。

重みJSONのスキーマは ``sample_submission/ptcg_ai/learning/policy_model.py`` の契約と
完全に同じにする(``schema_version`` / ``model`` / ``feature_names`` / ``weights`` /
``intercept`` / ``frequent_card_ids`` / ``card_attributes`` / ``meta``)。RLはこの契約の
weights 配列だけを更新し、intercept・feature_names・card_attributes 等は変更しない。

## 勾配の式とT(温度)について【判断メモ】

設計書は ``∇_w log pi(a) = x_a - Σ_i pi_i・x_i`` とだけ書いている。これは T=1 の特別形。
一般の T では

```
log pi(a) = score_a / T - logsumexp(score / T)
∇_w log pi(a) = (1/T) * (x_a - Σ_i pi_i・x_i)
```

となる(score_i = w・x_i なので ∇_w score_i = x_i、logsumexp の勾配は期待値)。
数値微分と一致させるにはこの 1/T が必要なため、本モジュールは 1/T を含めて実装する
(``tests/test_rl_gradient.py`` で検証)。T=1 のときは設計書の式とそのまま一致する。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LinearPolicy:
    """線形 pointwise スコアラー。``policy_model.PolicyModel`` と同じ重みJSON契約を読み書きする。"""

    def __init__(
        self,
        feature_names: list[str],
        weights: list[float],
        intercept: float = 0.0,
        frequent_card_ids: list[int] | None = None,
        card_attributes: dict[str, dict[str, float]] | None = None,
        meta: dict | None = None,
    ) -> None:
        if len(feature_names) != len(weights):
            raise ValueError("feature_names と weights の長さが一致しない")
        self.feature_names: list[str] = list(feature_names)
        self.weights: list[float] = list(weights)
        self.intercept: float = float(intercept)
        self.frequent_card_ids: list[int] = list(frequent_card_ids or [])
        self.card_attributes: dict[str, dict[str, float]] = dict(card_attributes or {})
        self.meta: dict = dict(meta or {})
        self._index: dict[str, int] = {name: i for i, name in enumerate(self.feature_names)}

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(w) for w in payload["weights"]],
            intercept=float(payload.get("intercept", 0.0)),
            frequent_card_ids=[int(c) for c in payload.get("frequent_card_ids", [])],
            card_attributes=dict(payload.get("card_attributes", {}) or {}),
            meta=dict(payload.get("meta", {}) or {}),
        )

    def to_json_payload(self) -> dict:
        return {
            "schema_version": 1,
            "model": "linear_pointwise",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "intercept": self.intercept,
            "frequent_card_ids": self.frequent_card_ids,
            "card_attributes": self.card_attributes,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_payload(), ensure_ascii=False), encoding="utf-8")

    def clone(self) -> "LinearPolicy":
        return LinearPolicy(
            feature_names=self.feature_names,
            weights=list(self.weights),
            intercept=self.intercept,
            frequent_card_ids=self.frequent_card_ids,
            card_attributes=self.card_attributes,
            meta=dict(self.meta),
        )

    def score(self, features: dict[str, float]) -> float:
        """``policy_model.PolicyModel.score`` と同じ計算(語彙に無いキーは無視)。"""
        s = self.intercept
        weights = self.weights
        index = self._index
        for name, value in features.items():
            i = index.get(name)
            if i is not None:
                s += weights[i] * value
        return s

    def score_many(self, feature_dicts: list[dict[str, float]]) -> list[float]:
        return [self.score(f) for f in feature_dicts]

    def filter_known(self, features: dict[str, float]) -> dict[str, float]:
        """語彙(feature_names)に存在するキーだけを残す。

        自己対戦ログを軽量化する(未知の特徴はどのみち score に寄与しない = 勾配にも
        寄与しないので、記録時点で落としてよい)。
        """
        index = self._index
        return {k: v for k, v in features.items() if k in index}


def softmax(scores: list[float], temperature: float) -> list[float]:
    """``softmax(scores / temperature)``。T が極めて小さい場合は argmax の one-hot に収束する。"""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    if total <= 0:
        # 数値的に全部 0 になった場合(理論上ほぼ起きない)は一様分布にフォールバックする
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def sample_index(probs: list[float], rng) -> int:
    """``rng.random()`` を使って ``probs`` からインデックスを1つサンプリングする。"""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1  # 浮動小数点誤差で合計が1未満になった場合の保険


def argmax_index(scores: list[float]) -> int:
    best_i = 0
    best = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best:
            best = scores[i]
            best_i = i
    return best_i


def log_prob(scores: list[float], chosen_index: int, temperature: float) -> float:
    """``log pi(chosen_index)`` を数値的に安定な形(logsumexp)で計算する。"""
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    log_sum_exp = m + math.log(sum(math.exp(v - m) for v in scaled))
    return scaled[chosen_index] - log_sum_exp


def log_prob_gradient(
    feature_dicts: list[dict[str, float]],
    probs: list[float],
    chosen_index: int,
    temperature: float,
) -> dict[str, float]:
    """``∇_w log pi(a) = (1/T) * (x_a - Σ_i pi_i・x_i)`` をスパース dict で返す。

    ``probs`` は呼び出し側で ``softmax(score_many(feature_dicts), temperature)`` を
    使って計算済みのものを渡す(スコア計算を重複させないため)。
    """
    grad: dict[str, float] = {}
    for p, feats in zip(probs, feature_dicts):
        if p == 0.0:
            continue
        for name, value in feats.items():
            grad[name] = grad.get(name, 0.0) - p * value
    for name, value in feature_dicts[chosen_index].items():
        grad[name] = grad.get(name, 0.0) + value
    if temperature != 1.0:
        inv_t = 1.0 / temperature
        for name in grad:
            grad[name] *= inv_t
    return grad
