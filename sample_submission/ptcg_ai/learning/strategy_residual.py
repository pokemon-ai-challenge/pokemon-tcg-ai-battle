"""戦略残差NN(②③レイヤーの最小実装, self-play RL で学習する対象)。

設計: docs/plans/strategy-residual-nn-experiment.md。凍結した模倣 PolicyModel の logits の上に、
option ごとの小さな bias を足す "残差"。最終行動 = argmax(logits_imit + alpha * bias)。

Step1(スケルトン): 重み未ロード(既定)なら全 option に bias 0 を返す = 完全に無効(本番挙動不変)。
self-play RL が JSON 重みを学習して設定する。重みは option 特徴(encoder.encode_options_from_state)
に対する線形: bias_i = scale * (b + W . option_features_i)。将来 MLP/状態特徴/belief に拡張可能だが、
まずは線形の最小形で配線と評価を通す。

JSON 形式(学習後):
  {"W": [float, ...], "b": float, "scale": float}  # len(W) == encoder.OPTION_FEATURE_COUNT が目安
"""

from __future__ import annotations

import json
import os

from cg.api import Observation, SelectData
from ptcg_ai.learning import encoder


class StrategyResidual:
    """option ごとの bias を返す小さな学習残差。未ロード時は zeros(=無効)。"""

    def __init__(self, weights_path: str | None = None):
        self.is_ready = False
        self._W: list[float] = []
        self._b: float = 0.0
        self._scale: float = 1.0
        self._mean: list[float] = []  # option特徴の標準化(学習時と同一)。空なら標準化しない。
        self._std: list[float] = []
        resolved = self._resolve(weights_path)
        if resolved:
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._W = [float(x) for x in data["W"]]
                self._b = float(data.get("b", 0.0))
                self._scale = float(data.get("scale", 1.0))
                self._mean = [float(x) for x in data.get("option_mean", [])]
                self._std = [float(x) for x in data.get("option_std", [])]
                self.is_ready = bool(self._W)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                self.is_ready = False

    @staticmethod
    def _resolve(weights_path: str | None) -> str | None:
        """weights_path をそのまま、無ければ本モジュール(ptcg_ai/learning)基準で解決する
        (config が bare filename を渡しても cwd に依らず読める。移植性)。"""
        if not weights_path:
            return None
        if os.path.isfile(weights_path):
            return weights_path
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, weights_path)
        return cand if os.path.isfile(cand) else None

    def _standardize(self, row: list[float]) -> list[float]:
        if not self._mean or not self._std:
            return row
        out = []
        for i in range(len(row)):
            if i < len(self._mean) and i < len(self._std) and self._std[i]:
                out.append((row[i] - self._mean[i]) / self._std[i])
            else:
                out.append(0.0)
        return out

    def bias_options(self, obs: Observation, select: SelectData) -> list[float]:
        """各 option への bias(長さ = len(select.option))。未ロード/失敗時は zeros。
        option特徴は学習時と同じ mean/std で標準化してから W と内積する。"""
        n = len(select.option) if (select is not None and select.option) else 0
        if not self.is_ready or obs is None or obs.current is None or n == 0:
            return [0.0] * n
        try:
            rows = encoder.encode_options_from_state(obs.current, select)
        except Exception:  # noqa: BLE001 - 残差の失敗が意思決定を止めてはならない
            return [0.0] * n
        if len(rows) != n:
            return [0.0] * n
        W = self._W
        out: list[float] = []
        for row in rows:
            z = self._standardize(row)
            m = min(len(W), len(z))
            s = self._b + sum(W[i] * z[i] for i in range(m))
            out.append(self._scale * s)
        return out
