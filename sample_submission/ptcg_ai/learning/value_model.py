"""勝率予測器(バリューネットワーク)の推論モジュール(Step1, イシュー #72)。

``kaggle_replays/value_net/`` の学習パイプライン(オフライン・このモジュールのスコープ外)が
Kaggle 上位リプレイから MLP を学習し、標準化パラメータ・各層の重み・ターン帯別の温度較正を
``value_weights.json`` へエクスポートする。本モジュールはその重みJSONを読み込み、試合中は
純Python(numpy/torch 非依存、``math`` のみ)のフォワードパス + 較正だけで勝率を計算する。

設計は ``opponent_modeling/ml_predictor.py`` (:class:`MLDeckPredictor`)と揃える:
- 重みファイルが存在しなくても例外にせず「未ロード状態」とし、``is_ready`` で判定できる。
  未ロード時の推論は安全なフォールバック 0.5(五分)を返す。
- 外部ライブラリに依存しない。行列積・活性化・sigmoid は素の Python で書く。

重みJSONのスキーマ(学習パイプラインと共有する契約。このファイル側の都合で変えない):

```
{
  "feature_names": [...166個, encoder.FEATURE_NAMES と完全一致],
  "standardization": {"mean": [166], "std": [166]},
  "layers": [
    {"W": [[...]], "b": [...], "activation": "relu"},    # W shape [out][in]
    {"W": [[...]], "b": [...], "activation": "relu"},
    {"W": [[...]], "b": [...], "activation": "sigmoid"}   # 最終層 out=1
  ],
  "meta": {"calibration": {"axis": "turn_band", "buckets": [...]}, ...}
}
```

## フォワードパスと較正

``x' = (x - mean) / std`` で標準化し、各層 ``z[k] = b[k] + sum_i(W[k][i] * x'[i])`` を
activation に通して次層へ。最終層(sigmoid)出力が ``p_raw``。
較正は ``logit = ln(p_raw / (1 - p_raw))`` を取り、そのサンプルのターン数から turn_band を
決めてバケットの温度 T を引き、``p_calibrated = sigmoid(logit / T)`` を最終出力とする。
較正の実装は学習パイプライン ``kaggle_replays/value_net/train.py`` の
``pure_python_forward`` / ``turn_band_of`` / sample_predictions 生成経路と一致させてある。

- 温度 T は正の単調変換なので、勝ち/負けの大小関係(argmax相当)は変えず、確率の「強さ」
  (reliability)だけを補正する。
- ``meta.calibration`` が無い(古い)重みJSONを読んだ場合は常に T=1.0(無補正)となる。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from cg.api import Observation

from ptcg_ai.learning import encoder

_DEFAULT_WEIGHTS_FILENAME = "value_weights.json"

# 未ロード時・盤面未確定時の安全なフォールバック(五分)。
_FALLBACK_PROB = 0.5

# p_raw を log に通す前のクランプ幅(train.py と同一)。
_EPS = 1e-12


class ValueModel:
    """学習済み MLP の重みを読み込み、Observation から勝率を推論する。"""

    def __init__(self, weights_path: str | Path | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)

        self._feature_names: list[str] | None = None
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        # 各層は (W[out][in], b[out], activation) のタプル。
        self._layers: list[tuple[list[list[float]], list[float], str]] | None = None
        # meta.calibration.buckets(無ければ空リスト = 無補正)。
        self._calibration_buckets: list[dict] = []

        self._load()

    @property
    def is_ready(self) -> bool:
        """重みJSONの読み込みに成功していれば True。"""
        return self._layers is not None

    def _load(self) -> None:
        if not self._weights_path.exists():
            return  # 未ロード状態のまま(重みは学習パイプライン完了後に配置される)。

        with self._weights_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._feature_names = list(payload["feature_names"])
        std = payload["standardization"]
        self._mean = [float(v) for v in std["mean"]]
        self._std = [float(v) for v in std["std"]]
        self._layers = [
            ([[float(w) for w in row] for row in layer["W"]],
             [float(v) for v in layer["b"]],
             str(layer["activation"]))
            for layer in payload["layers"]
        ]
        self._calibration_buckets = list(
            payload.get("meta", {}).get("calibration", {}).get("buckets", [])
        )

    def predict_win_prob(self, obs: Observation) -> float:
        """Observation(現在盤面)から自分視点の勝率(0..1)を返す。

        未ロード時・``obs.current`` が None(初回デッキ選択など)の場合は 0.5 を返す
        (例外は出さない)。ターン帯は ``obs.current.turn`` から決める。
        """
        if not self.is_ready:
            return _FALLBACK_PROB
        state = obs.current
        if state is None:
            return _FALLBACK_PROB
        features = encoder.encode_state(obs)
        return self._predict_from_features(features, state.turn)

    def predict_win_prob_from_state(self, state) -> float:
        """State(cg.api.State)から直接、自分視点の勝率を返す。``obs.current`` が直接手に入る
        呼び出し元(battle_review_viewer 等)向け。ロジックは ``predict_win_prob`` と同一で、
        Observation でラップする代わりに ``encoder.encode_state_from_state`` を使うだけ。

        未ロード時・``state`` が None の場合は 0.5 を返す(例外は出さない)。
        """
        if not self.is_ready or state is None:
            return _FALLBACK_PROB
        features = encoder.encode_state_from_state(state)
        return self._predict_from_features(features, state.turn)

    def predict_win_prob_from_dict(self, obs_dict: dict) -> float:
        """学習/検証用ヘルパー: obs_dict から勝率を返す(``encode_obs_dict`` 経由)。

        ``obs_dict["current"]`` が None/欠損の場合は 0.5 を返す。
        """
        if not self.is_ready:
            return _FALLBACK_PROB
        current = obs_dict.get("current")
        if current is None:
            return _FALLBACK_PROB
        features = encoder.encode_obs_dict(obs_dict)
        return self._predict_from_features(features, current.get("turn"))

    def predict_win_prob_from_features(self, features: list[float], turn) -> float:
        """既にエンコード済みの特徴ベクトル(``encoder`` 出力)とターン数から勝率を返す。

        オフライン評価(``features.npz`` の X をそのまま流す)用のヘルパー。dict/Observation
        経由の ``predict_*`` と同一のフォワードパス + 較正を通す。未ロード時は 0.5。
        """
        if not self.is_ready:
            return _FALLBACK_PROB
        return self._predict_from_features(list(features), turn)

    # ------------------------------------------------------------------
    # 内部: フォワードパス + 較正
    # ------------------------------------------------------------------
    def _predict_from_features(self, features: list[float], turn) -> float:
        p_raw = self._forward(features)
        return self._calibrate(p_raw, turn)

    def _forward(self, features: list[float]) -> float:
        """標準化 → 各層フォワード → 最終層(sigmoid)の p_raw を返す。"""
        mean = self._mean
        std = self._std
        h = [
            (features[i] - mean[i]) / std[i] if std[i] else 0.0
            for i in range(len(features))
        ]
        for W, b, activation in self._layers:
            z = [
                b[k] + sum(W[k][i] * h[i] for i in range(len(h)))
                for k in range(len(W))
            ]
            if activation == "relu":
                h = [v if v > 0.0 else 0.0 for v in z]
            elif activation == "sigmoid":
                h = [_sigmoid(v) for v in z]
            else:
                raise ValueError(f"未知の activation: {activation}")
        return h[0]

    def _calibrate(self, p_raw: float, turn) -> float:
        """turn 帯の温度 T で ``sigmoid(logit(p_raw) / T)`` を返す。"""
        temperature = self._temperature_for(turn)
        p = min(max(p_raw, _EPS), 1.0 - _EPS)
        logit = math.log(p / (1.0 - p))
        return _sigmoid(logit / temperature)

    def _temperature_for(self, turn) -> float:
        """ターン数が属するバケットの温度 T を返す。該当が無ければ 1.0(無補正)。

        バケット定義(min_turn/max_turn)を用いつつ、境界の判定は学習側 ``turn_band_of``
        と同じ「上限しきい値」意味論にする: turn <= max_turn の最初の数値バケットに割り当て、
        turn が全数値バケットの下限(min_turn=1)未満(= 0)でも先頭の "1-2" 帯に入る。
        """
        band = _turn_band_of(turn)
        for bucket in self._calibration_buckets:
            if bucket.get("band") == band:
                return float(bucket["temperature"])
        return 1.0


def _turn_band_of(turn) -> str:
    """ターン数 → 帯名。学習側 ``kaggle_replays/value_net/train.py`` の ``turn_band_of`` 踏襲。"""
    if turn is None:
        return "unknown"
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "unknown"
    if t < 0:
        return "unknown"
    if t <= 2:
        return "1-2"
    if t <= 5:
        return "3-5"
    if t <= 10:
        return "6-10"
    return "11+"


def _sigmoid(z: float) -> float:
    """オーバーフロー耐性のある sigmoid(train.py と同一)。"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
