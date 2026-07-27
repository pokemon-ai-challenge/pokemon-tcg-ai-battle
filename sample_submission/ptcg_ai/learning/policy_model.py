"""Policy Prior(pointwise 線形スコアリング)の推論モジュール。

``kaggle_replays/policy_prior/`` の学習パイプライン(オフライン・このモジュールのスコープ外)が
Behavior Cloning データから線形モデルを学習し、``policy_weights.json`` に出力する。
本モジュールはその重みJSONを読み込み、試合中は純Python(numpy 非依存、リスト演算のみ)の
線形結合だけでスコアを計算する。

設計は ``opponent_modeling/ml_predictor.py`` (:class:`MLDeckPredictor`) /
``learning/value_model.py`` (:class:`ValueModel`) と揃える:

- 重みファイルが存在しなくても例外にせず「未ロード状態」とし、``is_ready`` で判定できる。
  未ロード時は ``score_options()`` が全選択肢に中立スコア(0.0)を返し、``select()`` は
  ``None`` を返す(呼び出し元 ``action_selection/selector.py`` はこれを見て通常の
  ``router.route()`` にフォールバックする。ターンを絶対に止めない)。
- 外部ライブラリに依存しない。特徴抽出は ``ptcg_ai.learning.policy_features`` に一本化。

重みJSONのスキーマ(学習パイプラインと共有する契約。このファイル側の都合で変えない):

```json
{
  "schema_version": 1,
  "model": "linear_pointwise",
  "feature_names": ["..."],
  "weights": [0.0],
  "intercept": 0.0,
  "frequent_card_ids": [1, 2],
  "card_attributes": {"7": {"hp": 60.0, "stage": 0.0}},
  "meta": {"train_rows": 0, "test_accuracy": 0.0}
}
```

``card_attributes`` を重みJSONに埋め込むのは、提出時に ``data/EN_Card_Data.csv`` が
同梱されない可能性があるため(``policy_features.py`` は CSV に直接依存しない)。
"""

from __future__ import annotations

import json
from pathlib import Path

from ptcg_ai.learning import policy_features

_DEFAULT_WEIGHTS_FILENAME = "policy_weights.json"


class PolicyModel:
    """学習済み線形 pointwise スコアラーの重みを読み込み、選択肢をスコアリングする。"""

    def __init__(self, weights_path: str | Path | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)

        self._feature_names: list[str] | None = None
        self._weights: list[float] | None = None
        self._intercept: float = 0.0
        # feature_names -> そのインデックス。特徴 dict -> スコア加算を O(特徴数) にする。
        self._feature_index: dict[str, int] | None = None
        self._frequent_card_ids: list[int] = []
        self._card_attributes: dict[str, dict[str, float]] = {}
        # own/opp active card_id × option_type 交互作用に使うN。重みJSONに無ければ
        # policy_features.py の既定値を使う(古い重みJSONとの後方互換。この交互作用
        # 自体が存在しない古いJSONでは、この値がどうであれ一致する特徴名が無いので
        # 実害は無い)。
        self._active_card_top_n: int = policy_features.DEFAULT_ACTIVE_CARD_TOP_N
        self._meta: dict = {}

        self._load()

    @property
    def is_ready(self) -> bool:
        """重みJSONの読み込みに成功していれば True。"""
        return self._weights is not None

    @property
    def frequent_card_ids(self) -> list[int]:
        return list(self._frequent_card_ids)

    @property
    def card_attributes(self) -> dict[str, dict[str, float]]:
        return self._card_attributes

    def _load(self) -> None:
        if not self._weights_path.exists():
            return  # 未ロード状態のまま(重みは学習パイプライン完了後に配置される)。

        try:
            with self._weights_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)

            feature_names = list(payload["feature_names"])
            weights = [float(w) for w in payload["weights"]]
            if len(feature_names) != len(weights):
                raise ValueError("feature_names と weights の長さが一致しない")

            self._feature_names = feature_names
            self._weights = weights
            self._intercept = float(payload.get("intercept", 0.0))
            self._feature_index = {name: i for i, name in enumerate(feature_names)}
            self._frequent_card_ids = [int(c) for c in payload.get("frequent_card_ids", [])]
            self._card_attributes = dict(payload.get("card_attributes", {}) or {})
            self._active_card_top_n = int(
                payload.get("active_card_top_n", policy_features.DEFAULT_ACTIVE_CARD_TOP_N)
            )
            self._meta = dict(payload.get("meta", {}) or {})
        except Exception:  # noqa: BLE001 -- 壊れた重みファイルで試合を止めない
            self._feature_names = None
            self._weights = None
            self._feature_index = None

    def score(self, features: dict[str, float]) -> float:
        """特徴 dict(疎)から線形スコア ``intercept + sum(weight * value)`` を返す。

        ``feature_names`` に無いキーは無視する(学習データに現れなかった特徴)。
        未ロード時は 0.0(中立)を返す。
        """
        if not self.is_ready:
            return 0.0
        score = self._intercept
        weights = self._weights
        index = self._feature_index
        for name, value in features.items():
            i = index.get(name)
            if i is not None:
                score += weights[i] * value
        return score

    def score_options(self, state: dict, actions: list[dict]) -> list[float]:
        """各選択肢(解決済み Semantic Action)のスコアを返す。

        未ロード時は全選択肢に中立スコア(0.0)を返す(呼び出し元は ``is_ready`` で
        分岐せず、この関数の戻り値だけで判断してよい設計)。
        """
        if not self.is_ready:
            return [0.0] * len(actions)

        return [
            self.score(
                policy_features.extract_features(
                    state, action, self._card_attributes, self._frequent_card_ids,
                    self._active_card_top_n,
                )
            )
            for action in actions
        ]

    def select(self, state: dict, actions: list[dict]) -> int | None:
        """スコア最大の選択肢のインデックスを返す。

        未ロード時・選択肢が空の場合は ``None``(呼び出し元は通常の router へフォールバック)。
        """
        if not self.is_ready or not actions:
            return None
        scores = self.score_options(state, actions)
        best_index = 0
        best_score = scores[0]
        for i in range(1, len(scores)):
            if scores[i] > best_score:
                best_score = scores[i]
                best_index = i
        return best_index
