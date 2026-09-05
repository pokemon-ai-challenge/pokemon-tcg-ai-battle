"""相手デッキ予測器の ML(学習済み重み)版ランタイム推論モジュール。

``kaggle_replays/deck_predictor/`` の学習パイプライン(オフライン・このモジュールのスコープ外)が
リプレイから多クラスロジスティック回帰(softmax回帰)を学習し、``deck_predictor_weights.json``
に出力する。本モジュールはその重みJSONを読み込み、試合中は純Python(外部ライブラリ非依存)の
行列積 + softmax だけで確率分布を計算する。

重みJSONのスキーマは
``sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-plan.md`` の
「deck_predictor_weights.json」節で定義されている契約であり、学習パイプライン側と共有する
インターフェースなので、このファイル側だけの都合で変更しないこと。

重みファイルは学習パイプライン完了後に配置される想定で、現時点ではまだ存在しない。
存在しない場合は例外にせず「未ロード状態」とし、``is_ready`` で判定できるようにする
(未ロード時の ``predict()`` は ``{"other": 1.0}`` を返す)。

## キャリブレーション(温度スケーリング)

観測エビデンス(見えているカードの種類数)が少ない序盤ほど、線形分類器の生ロジットは
過信(overconfidence)しやすい ―― 実際には根拠が薄いのに確率が90%以上に張り付く、といった
現象が起きる。これを緩和するため、``kaggle_replays/deck_predictor/calibrate.py``
(学習パイプライン側、このモジュールのスコープ外)が validation データ上で
「エビデンス数のバケットごとの温度 T」を最適化し、``meta.calibration.buckets`` として
重みJSONに追記する。本モジュールはそれを読み込み、softmax 直前のロジット ``z`` を
``z / T`` に置き換えるだけ(``T`` はエビデンス数に応じて動的に選ぶ)。

- 同一サンプル内の全ロジットを同じ正の T で割るのは単調変換なので、argmax(top-1予測)は
  T を変えても変わらない。変わるのは確率の「強さ」(reliability)だけ。
- ``meta.calibration`` が無い(古い)重みJSONを読んだ場合は常に T=1.0(無補正)となり、
  キャリブレーション導入前と完全に同じ動作になる(後方互換)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DEFAULT_WEIGHTS_FILENAME = "deck_predictor_weights.json"
_TURN_FEATURE_NAME = "__turn__"


class MLDeckPredictor:
    """学習済み softmax 回帰の重みを読み込み、観測カードから相手デッキ確率分布を推論する。"""

    def __init__(self, weights_path: str | Path | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)

        self._classes: list[str] | None = None
        self._feature_names: list[str] | None = None
        self._coef: list[list[float]] | None = None
        self._intercept: list[float] | None = None
        # feature_names -> そのインデックス。特徴ベクトル構築を O(n) にするための逆引き。
        self._feature_index: dict[str, int] | None = None
        # meta.calibration.buckets (無ければ空リスト = 無補正)。
        self._calibration_buckets: list[dict] = []

        self._load()

    @property
    def is_ready(self) -> bool:
        """重みJSONの読み込みに成功していれば True。"""
        return self._classes is not None

    def _load(self) -> None:
        if not self._weights_path.exists():
            return  # 未ロード状態のまま(重みは学習パイプライン完了後に配置される)。

        with self._weights_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._classes = list(payload["classes"])
        self._feature_names = list(payload["feature_names"])
        self._coef = [list(row) for row in payload["coef"]]
        self._intercept = list(payload["intercept"])
        self._feature_index = {name: i for i, name in enumerate(self._feature_names)}
        self._calibration_buckets = list(payload.get("meta", {}).get("calibration", {}).get("buckets", []))

    def predict(self, observed_cards: dict[str, int], turn: int) -> dict[str, float]:
        """観測済みカード枚数とターン数から、全クラスの確率分布(合計1.0)を返す。

        キャリブレーション(meta.calibration.buckets)が設定されている場合、observed_cards の
        エビデンス数に応じた温度 T で生ロジットを割ってから softmax する。T は正の単調な
        スケーリングなので argmax(top-1予測)には影響しない。
        """
        if not self.is_ready:
            return {"other": 1.0}

        z = self._raw_logits(observed_cards, turn)
        temperature = self._temperature_for(self.evidence_count(observed_cards))
        z = [v / temperature for v in z]
        probs = _softmax(z)
        return dict(zip(self._classes, probs))

    def _raw_logits(self, observed_cards: dict[str, int], turn: int) -> list[float]:
        """キャリブレーション適用前の生ロジット z = intercept + coef @ x を返す。

        学習パイプライン側(calibrate.py)もこのメソッドを再利用し、ロジット計算ロジックの
        二重実装によるズレを防ぐ。
        """
        x = self._build_feature_vector(observed_cards, turn)
        return [
            intercept + sum(c * xi for c, xi in zip(coef_row, x))
            for coef_row, intercept in zip(self._coef, self._intercept)
        ]

    def evidence_count(self, observed_cards: dict[str, int]) -> int:
        """観測エビデンス数 = 語彙(__turn__ を除く feature_names)のうち、枚数 > 0 で
        観測されているユニークなカード名の個数(枚数の合計ではない)。

        fit時(calibrate.py)とinfer時で完全に同一の定義を使う必要があるため、両者から
        このメソッドを再利用する。
        """
        return sum(
            1
            for name in self._feature_names
            if name != _TURN_FEATURE_NAME and observed_cards.get(name, 0) > 0
        )

    def _temperature_for(self, evidence_count: int) -> float:
        """evidence_count が属するバケットの温度 T を返す。該当バケットが無ければ 1.0(無補正)。"""
        for bucket in self._calibration_buckets:
            min_evidence = bucket["min_evidence"]
            max_evidence = bucket.get("max_evidence")
            if evidence_count < min_evidence:
                continue
            if max_evidence is not None and evidence_count > max_evidence:
                continue
            return float(bucket["temperature"])
        return 1.0

    def predict_top(
        self, observed_cards: dict[str, int], turn: int, n: int = 3
    ) -> list[tuple[str, float]]:
        """確率降順で上位 n 件を [(クラス名, 確率), ...] で返す。"""
        probs = self.predict(observed_cards, turn)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        return ranked[:n]

    def _build_feature_vector(self, observed_cards: dict[str, int], turn: int) -> list[float]:
        x = [0.0] * len(self._feature_names)
        turn_index = self._feature_index.get(_TURN_FEATURE_NAME)
        if turn_index is not None:
            x[turn_index] = float(turn)

        for name, count in observed_cards.items():
            index = self._feature_index.get(name)
            if index is None:
                continue  # 語彙にないカード名(学習データに現れなかった)は無視する。
            x[index] = float(count)
        return x


def _softmax(z: list[float]) -> list[float]:
    """オーバーフロー対策で max(z) を引いてから exp する softmax。"""
    z_max = max(z)
    exps = [math.exp(v - z_max) for v in z]
    total = sum(exps)
    return [v / total for v in exps]
