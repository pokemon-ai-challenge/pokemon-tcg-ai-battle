"""オーガポン戦略Q-critic(design.md §8)の純Python推論。

学習(``kaggle_replays/rl/train_ogerpon_strategy.py``、PyTorch)がexportした
``ogerpon_strategy_weights.json`` を読み込み、対戦中は ``policy_model.py`` と同じ設計
(``math`` のみ、numpy/torch非依存)でensembleのフォワードパスを計算する。

## 入力

``ogerpon_strategy_encoder.encode_strategy_pair()`` が返す
``continuous_features`` / ``slot_card_ids`` / ``option_features``(Option別)をそのまま使う。
モデル内部で continuous+option を標準化し、``slot_card_ids`` は学習済み埋め込みテーブルで
引いた12スロット分のベクトルを連結する(``policy_model.py`` のcard_embeddingと同じ設計)。

## 出力

各headは以下の意味を持つ(design.md §8.2):
- ``win``: そのOptionを選んだ場合の較正後勝率(0..1)。
- ``loop_complete``: ``SINGLE_PRIZE_ROTATION`` を選んだ場合にループ(気絶までACTIVE継続
  →次アタッカー接続)を完遂できる確率(0..1)。``EX_TEMPO`` では意味を持たないため常に0.5
  (中立)を返す。
- ``opponent_ko_count`` / ``signed_terminal_turns``: 生の回帰値(較正しない)。

## フォールバック

重み未配置・読み込み失敗・特徴量の次元不一致・推論時の例外は、``win=0.5``
(オプション間で差が無い=中立)を返す。design.md §11.2の判断規則はこの中立値のとき
自然に ``EX_TEMPO`` を選ぶため(delta=0 は strict_override_threshold を超えない)、
呼び出し側で特別な分岐を書かなくてもフォールバックが成立する(design.md §15.1 item10)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DEFAULT_WEIGHTS_FILENAME = "ogerpon_strategy_weights.json"
_UNKNOWN_SLOT_EMBEDDING_INDEX = 0
_HEADS = ("win", "loop_complete", "opponent_ko_count", "signed_terminal_turns")
_NEUTRAL = {"win": 0.5, "loop_complete": 0.5, "opponent_ko_count": 0.0, "signed_terminal_turns": 0.0}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _shared_trunk_forward(layers, x):
    """共有trunk(``nn.Sequential(Linear, ReLU, Linear, ReLU)``)の(W[out][in], b[out])を
    順に適用する。design.md §8.1のとおり、trunkは全層にReLUがかかる(最終層のReLUを
    省略する ``policy_model.py`` の単出力ネットとは構造が違うので注意)。head層は別枠で
    活性化なしのLinearとして計算する。
    """
    h = list(x)
    for w, b in layers:
        z = [b[k] + sum(w[k][i] * h[i] for i in range(len(h))) for k in range(len(w))]
        h = [v if v > 0.0 else 0.0 for v in z]
    return h


class _SingleModel:
    """ensembleの1メンバー。1つの重みJSON内の ``models[i]`` に対応する。"""

    def __init__(self, payload: dict, continuous_dim: int, option_dim: int, slot_count: int):
        emb = payload["card_embedding"]
        self.embedding_dim = int(emb["dim"])
        self.card_id_max = int(emb["card_id_max"])
        self.embedding_table = [[float(v) for v in row] for row in emb["table"]]

        std = payload["standardization"]
        self.continuous_mean = [float(v) for v in std["continuous_mean"]]
        self.continuous_std = [float(v) for v in std["continuous_std"]]
        self.option_mean = [float(v) for v in std["option_mean"]]
        self.option_std = [float(v) for v in std["option_std"]]
        if len(self.continuous_mean) != continuous_dim or len(self.option_mean) != option_dim:
            raise ValueError("standardization dimension mismatch")

        self.shared_layers = [
            ([[float(w) for w in row] for row in layer["weight"]], [float(v) for v in layer["bias"]])
            for layer in payload["shared_layers"]
        ]
        self.head_layers = {}
        for head in _HEADS:
            layer = payload["heads"][head]
            self.head_layers[head] = (
                [[float(w) for w in row] for row in layer["weight"]],
                [float(v) for v in layer["bias"]],
            )
        calib = payload.get("calibration") or {}
        self.temperature = {
            "win": float(calib.get("win_temperature", 1.0)),
            "loop_complete": float(calib.get("loop_complete_temperature", 1.0)),
        }
        self.slot_count = slot_count

    def _embedding(self, card_id: int) -> list[float]:
        if card_id is None or card_id <= 0 or card_id > self.card_id_max:
            card_id = _UNKNOWN_SLOT_EMBEDDING_INDEX
        try:
            return list(self.embedding_table[card_id])
        except IndexError:
            return [0.0] * self.embedding_dim

    def forward(self, continuous_features, slot_card_ids, option_features) -> dict[str, float]:
        h = [
            (continuous_features[i] - self.continuous_mean[i]) / self.continuous_std[i]
            if self.continuous_std[i] else 0.0
            for i in range(len(continuous_features))
        ]
        h += [
            (option_features[i] - self.option_mean[i]) / self.option_std[i]
            if self.option_std[i] else 0.0
            for i in range(len(option_features))
        ]
        slots = list(slot_card_ids) + [0] * max(0, self.slot_count - len(slot_card_ids))
        for card_id in slots[: self.slot_count]:
            h += self._embedding(card_id)

        shared = _shared_trunk_forward(self.shared_layers, h)
        out = {}
        for head in _HEADS:
            w, b = self.head_layers[head]
            z = b[0] + sum(w[0][i] * shared[i] for i in range(len(shared)))
            if head in ("win", "loop_complete"):
                z = _sigmoid(z / max(1e-6, self.temperature[head]))
            out[head] = z
        return out


class OgerponStrategyModel:
    """学習済みensembleを読み込み、Option別の較正済み予測を返す。"""

    def __init__(self, weights_path: str | Path | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)
        self._models: list[_SingleModel] = []
        self._continuous_dim = 0
        self._option_dim = 0
        self._slot_count = 12
        self._load()

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    def _load(self) -> None:
        if not self._weights_path.exists():
            return
        try:
            with self._weights_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            contract = payload["feature_contract"]
            self._continuous_dim = int(contract["continuous_feature_count"])
            self._option_dim = int(contract["option_feature_count"])
            self._slot_count = int(contract.get("slot_count", 12))
            models = []
            for entry in payload["models"]:
                models.append(_SingleModel(entry, self._continuous_dim, self._option_dim, self._slot_count))
            self._models = models
        except Exception:  # noqa: BLE001
            self._models = []

    def predict(self, continuous_features, slot_card_ids, option_features, option_name: str) -> dict:
        """1Optionぶんのensemble平均・標準偏差を返す。

        Returns:
            dict: ``{"win": float, "win_std": float, "loop_complete": float,
            "opponent_ko_count": float, "signed_terminal_turns": float}``。
            未ロード・次元不一致・例外時は中立値(win=0.5等、win_std=0.0)。
        """
        if (not self.is_ready
                or len(continuous_features) != self._continuous_dim
                or len(option_features) != self._option_dim):
            return {**_NEUTRAL, "win_std": 0.0}
        try:
            per_model = [m.forward(continuous_features, slot_card_ids, option_features)
                        for m in self._models]
        except Exception:  # noqa: BLE001
            return {**_NEUTRAL, "win_std": 0.0}

        out = {}
        for head in _HEADS:
            values = [p[head] for p in per_model]
            out[head] = sum(values) / len(values)
        wins = [p["win"] for p in per_model]
        mean_win = out["win"]
        out["win_std"] = math.sqrt(sum((w - mean_win) ** 2 for w in wins) / len(wins))
        if option_name != "SINGLE_PRIZE_ROTATION":
            out["loop_complete"] = 0.5
        return out

    def predict_pair(self, continuous_features, slot_card_ids, option_features_by_name: dict) -> dict:
        """EX_TEMPO/SINGLE_PRIZE_ROTATION両方の予測を返す。design.md §11.2の入力形。"""
        return {
            name: self.predict(continuous_features, slot_card_ids, feats, name)
            for name, feats in option_features_by_name.items()
        }
