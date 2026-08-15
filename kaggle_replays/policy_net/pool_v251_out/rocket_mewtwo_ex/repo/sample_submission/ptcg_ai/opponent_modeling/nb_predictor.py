"""相手デッキ予測器の生成的ベイズ(Naive Bayes)版ランタイム推論モジュール。

``kaggle_replays/deck_predictor/train_nb.py``(オフライン・このモジュールのスコープ外)が
ラベル付きの60枚デッキリストから「P(カード採用 | アーキタイプ)」を直接、頻度推定で学習し、
``deck_predictor_nb.json`` に出力する。本モジュールはその重みJSONを読み込み、試合中は
純Python(``math`` / ``json`` / ``pathlib`` のみ、外部ライブラリ非依存)でベイズ更新の
posterior を計算する。

``ml_predictor.MLDeckPredictor``(ロジスティック回帰版)と同じ推論 API
(``predict`` / ``predict_top`` / ``is_ready``)を持ち、ドロップインで差し替え可能。
設計上の背景・決定事項は
``sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md``
の「フェーズB」節を参照。

## 推論式

観測済みカード枚数 ``observed_cards``(カード名 -> 枚数)から、全クラスの事後確率を計算する:

    log P(c | 観測) = log prior(c) + sum_{観測カード n, 枚数 k>0} log P(deck が n を k 枚以上採用 | c)

- ``prior(c)`` は学習時に「直近N日 x 相手ランク上位R位」ウィンドウのラベル分布から求めた
  クラス事前分布(``deck_predictor_nb.json`` の ``priors``)。
- ``P(deck が n を k 枚以上採用 | c)`` は学習パイプライン側がラベル付きデッキから頻度推定した
  尤度テーブル(``likelihoods[n][c]`` = ``[P(>=1枚), P(>=2枚), P(>=3枚), P(>=4枚)]``)。
  観測枚数が5枚以上のときは4枚以上の値(インデックス3)を使う(``min(k, 4)``)。
- LR版と違い、線形結合ではなく素朴ベイズ(観測カードごとの尤度を独立と仮定して単純に足し合わせる)
  なので、汎用カード(全クラスで尤度がほぼ同じ)を1枚見ても posterior はほぼ prior のまま動かない。
  一方、専用カード(そのクラス以外では尤度がほぼ0)は1枚見ただけで他クラスの対数尤度が
  大きく沈み、確信してよい根拠が明確になる(``explain()`` で寄与を個別に確認できる)。
- 語彙(学習データに一度も現れなかったカード名)にない観測カードは無視する(全クラス等しく
  無視されるので、posterior には寄与しない)。
- オーバーフロー対策として、正規化は log-sum-exp(``max(log_posterior)`` を引いてから ``exp``)で行う。

重みファイルはまだ配置されていない可能性がある。存在しない場合は例外にせず「未ロード状態」とし、
``is_ready`` で判定できるようにする(未ロード時の ``predict()`` は ``{"other": 1.0}`` を返す)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DEFAULT_WEIGHTS_FILENAME = "deck_predictor_nb.json"


class NBDeckPredictor:
    """学習済みナイーブベイズの重み(prior + 尤度テーブル)を読み込み、観測カードから
    相手デッキ確率分布を推論する。"""

    def __init__(self, weights_path: str | Path | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)

        self._classes: list[str] | None = None
        self._log_priors: list[float] | None = None
        self._card_names: set[str] | None = None
        # likelihoods[card_name][class_index] = [P(>=1枚), P(>=2枚), P(>=3枚), P(>=4枚)]
        self._likelihoods: dict[str, list[list[float]]] | None = None
        self._max_k: int = 4

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
        priors = list(payload["priors"])
        self._log_priors = [math.log(max(p, 1e-300)) for p in priors]
        self._card_names = set(payload["card_names"])

        # 推論時のインデックスアクセスを速くするため、class名 -> class index の順で
        # likelihoods[name] を [クラス0の[p1..p4], クラス1の[p1..p4], ...] というリストに変換する。
        raw_likelihoods: dict[str, dict[str, list[float]]] = payload["likelihoods"]
        self._likelihoods = {
            name: [list(per_class[c]) for c in self._classes]
            for name, per_class in raw_likelihoods.items()
        }
        self._max_k = int(payload.get("meta", {}).get("max_k", 4))

    def predict(self, observed_cards: dict[str, int], turn: int) -> dict[str, float]:
        """観測済みカード枚数から、全クラスの確率分布(合計1.0)を返す。

        ``turn`` はランタイム(MLDeckPredictor)との API 互換のために受け取るが、NB 版は
        ターン特徴を使わないため無視する。
        """
        if not self.is_ready:
            return {"other": 1.0}

        log_posterior = self._log_posterior(observed_cards)
        probs = _log_normalize(log_posterior)
        return dict(zip(self._classes, probs))

    def _log_posterior(self, observed_cards: dict[str, int]) -> list[float]:
        """正規化前の log posterior(各クラス)を返す。log_prior + 観測カードごとの log 尤度の和。"""
        log_posterior = list(self._log_priors)
        for name, count in observed_cards.items():
            if count <= 0:
                continue
            per_class_probs = self._likelihoods.get(name)
            if per_class_probs is None:
                continue  # 語彙にないカード名(学習データに現れなかった)は無視する。
            k_index = min(count, self._max_k) - 1
            for i, probs in enumerate(per_class_probs):
                log_posterior[i] += math.log(max(probs[k_index], 1e-300))
        return log_posterior

    def predict_top(
        self, observed_cards: dict[str, int], turn: int, n: int = 3
    ) -> list[tuple[str, float]]:
        """確率降順で上位 n 件を [(クラス名, 確率), ...] で返す。"""
        probs = self.predict(observed_cards, turn)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        return ranked[:n]

    def explain(self, observed_cards: dict[str, int], turn: int) -> dict:
        """予測の根拠(観測カードごとのクラス別 log 尤度寄与)を返す。ビュアーでの根拠表示用。

        戻り値スキーマ:
          {
            "classes": [クラス名, ...],
            "log_priors": {クラス名: log prior(c)},
            "cards": {
              観測カード名(語彙にあるもののみ): {
                "observed_count": 生の観測枚数,
                "k_used": min(observed_count, max_k),
                "log_likelihood": {クラス名: log P(>=k_used枚 | c)},
              }, ...
            },
            "ignored_cards": [語彙に無かった観測カード名, ...],
            "log_posterior": {クラス名: 正規化前の log posterior},
            "probs": {クラス名: 正規化後の確率(predict() と同じ)},
          }
        """
        if not self.is_ready:
            return {
                "classes": ["other"],
                "log_priors": {},
                "cards": {},
                "ignored_cards": list(observed_cards.keys()),
                "log_posterior": {},
                "probs": {"other": 1.0},
            }

        cards: dict[str, dict] = {}
        ignored_cards: list[str] = []
        for name, count in observed_cards.items():
            if count <= 0:
                continue
            per_class_probs = self._likelihoods.get(name)
            if per_class_probs is None:
                ignored_cards.append(name)
                continue
            k_index = min(count, self._max_k) - 1
            cards[name] = {
                "observed_count": count,
                "k_used": k_index + 1,
                "log_likelihood": {
                    cls: math.log(max(probs[k_index], 1e-300))
                    for cls, probs in zip(self._classes, per_class_probs)
                },
            }

        log_posterior = self._log_posterior(observed_cards)
        probs = _log_normalize(log_posterior)

        return {
            "classes": list(self._classes),
            "log_priors": dict(zip(self._classes, self._log_priors)),
            "cards": cards,
            "ignored_cards": ignored_cards,
            "log_posterior": dict(zip(self._classes, log_posterior)),
            "probs": dict(zip(self._classes, probs)),
        }


def _log_normalize(log_values: list[float]) -> list[float]:
    """log-sum-exp でオーバーフロー対策しつつ、log 値のリストを合計1.0の確率分布に正規化する。"""
    m = max(log_values)
    exps = [math.exp(v - m) for v in log_values]
    total = sum(exps)
    return [v / total for v in exps]
