"""相手デッキ予測器の LR × NB ハイブリッド版ランタイム推論モジュール。

背景・設計方針は
``sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md``
を参照。要点: ``kaggle_replays/deck_predictor/compare_nb.py`` の比較で、evidence(観測エビデンス数)が
極端に少ない場面(0枚)では NB(``NBDeckPredictor``、生成的ベイズ)が LR(``MLDeckPredictor``、
softmax回帰、温度スケーリング補正済み)より log loss が優れる一方、evidence が多い場面(7枚以上)
では NB の「観測カードごとの尤度が独立」という仮定が崩れて大幅に劣化することが分かった。
evidence 1〜6 枚はほぼ同等。

そこで本モジュールは、evidence 数のバケットごとに LR と NB の予測分布を **log-space の
幾何ブレンド**(重み付き対数線形プール)で混ぜる:

    log p_blend(c) = (1 - w) * log p_LR(c) + w * log p_NB(c)   (w はバケットごとの NB 側の重み)

を計算したあと、log-sum-exp で正規化して確率分布(合計1.0)に戻す。w=0 は LR 単体、w=1 は NB
単体と完全に一致する。バケット境界・重み w は
``kaggle_replays/deck_predictor/fit_hybrid.py``(オフライン・このモジュールのスコープ外)が
validation データ上で evidence バケットごとに最適化し、``deck_predictor_hybrid.json`` に
出力する。evidence 数の定義は ``MLDeckPredictor.evidence_count()``(LR の feature_names 語彙基準)
をそのまま使う(フィット側・推論側で定義がズレないようにするため)。

## ``deck_predictor_hybrid.json`` のスキーマ(本モジュールが定義する側の契約)

    {
      "buckets": [
        {"min_evidence": 0, "max_evidence": 0, "weight_nb": 1.0},
        {"min_evidence": 1, "max_evidence": 1, "weight_nb": 0.3},
        {"min_evidence": 2, "max_evidence": 3, "weight_nb": 0.1},
        {"min_evidence": 4, "max_evidence": null, "weight_nb": 0.0}
      ],
      "meta": { ... フィット日時・サンプル数・バケット別 before/after log loss 等 ... }
    }

- ``buckets`` は ``min_evidence`` 昇順である必要はない(検索は線形走査で最初に一致した
  バケットを採用する)が、フィット側は昇順・網羅的に出力する。
- ``max_evidence`` が ``null`` の場合は上限なし(その ``min_evidence`` 以上すべて)。
- 該当するバケットが無い evidence 数は ``weight_nb=0.0``(LR 単体と同じ)として扱う。
- ``meta`` はランタイムからは参照しない(記録用)。

## フォールバック規則

- ハイブリッド設定JSON(``deck_predictor_hybrid.json``)が無い場合: 全 evidence 数で ``w=0``
  (LR 単体と完全に同一の動作)。
- LR は読み込めたが NB の重み(``deck_predictor_nb.json``)が無い/未ロードの場合: NB 抜きの
  LR 単体にフォールバックする(ハイブリッド設定に何が書いてあっても無視)。
- LR の重みが無い場合: NB 単体にフォールバックする。
- 両方無い場合: ``{"other": 1.0}``(``MLDeckPredictor`` / ``NBDeckPredictor`` の未ロード時と同じ)。
- LR・NB 両方ロードできたが、学習されたクラスリストが一致しない場合(本来は同一のはずだが、
  学習パイプラインの取り違え等の事故を検知するための安全策): ブレンドせず LR 単体に
  フォールバックする。**この場合、共通部分だけで動かす、というような曖昧な処理はしない**
  (どのクラスが欠けているか分からないまま確率を計算すると誤った確信度になりうるため)。
  このフォールバックが起きたかどうかは ``classes_mismatch`` 属性で検知できる。

現在どのモードで動作しているかは ``mode`` 属性(``"hybrid"`` / ``"lr_only"`` / ``"nb_only"`` /
``"unready"``)で判定できる。

## rough_predictor による未学習クラスのオーバーライド(``predict(..., rough_prediction=...)``)

LR(``ml_predictor.py``)・NB(``nb_predictor.py``)は ``kaggle_replays/deck_predictor`` が
学習した固定クラスリスト(現状21クラス + ``"other"``)しか出力できない。新しいアーキタイプを
``rough_predictor.json`` に追加しても、**再学習しない限り**この2つは新クラスを一切返せない
(ML再学習はこのモジュールのスコープ外)。

そこで ``predict()`` はキーワード専用引数 ``rough_prediction``(``rough_predictor.predict()`` の
戻り値、または ``None``)を受け取れるようにしてある。既定は ``None`` で、**渡さない限り従来の
動作と完全に同一**(挙動を変えたくない既存呼び出し元に影響しない)。

``rough_prediction`` を渡した場合の合成規則:

1. ``rough_prediction["status"] != "confident"`` なら何もしない(LR/NB側の分布をそのまま返す)。
2. ``rough_prediction["deck_type"]`` が LR/NB の学習済みクラス(``self._classes``)に**既に
   含まれる**場合も何もしない(例: ``yadoking`` は21クラスに含まれるため、rough が
   ``yadoking`` を確信していても LR/NB 側の確率分布をそのまま使う。学習済みクラスの
   確信度は LR/NB のキャリブレーション済み値の方が信頼できるため、rough で上書きしない)。
3. 上記いずれでもない、つまり **rough が学習済みクラスに無い新アーキタイプを confident で
   返した場合だけ**、そのクラスに確率質量 ``min(rough_prediction["match_rate"],
   _ROUGH_OVERRIDE_MAX_PROB)`` を割り当て、残りの確率を既存分布の比率を保ったまま
   縮小して合計1.0に正規化する(``_ROUGH_OVERRIDE_MAX_PROB`` で頭打ちにするのは、
   rough 側のスコアリングは LR/NB ほど較正されていないため、断定しすぎない側に倒すため。
   `feedback_conservative_confidence` の方針と同じ)。

この設計により、学習済み21クラスの出力は ``rough_prediction`` を渡しても渡さなくても
一切変化しない(条件2で必ず素通しする)。影響が出るのは「rough が新アーキタイプを
確信した」ときの、そのクラス相当の確率質量のみ。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .ml_predictor import MLDeckPredictor
from .nb_predictor import NBDeckPredictor

_DEFAULT_HYBRID_FILENAME = "deck_predictor_hybrid.json"

# rough_predictor 由来のオーバーライドで新クラスに割り当てる確率の上限。
# rough 側のスコアリングは LR/NB ほど較正されていないため、1.0に張り付かせず
# 慎重側に倒す(確信度は生の見積もりより強気にしない、という既存方針と同じ)。
_ROUGH_OVERRIDE_MAX_PROB = 0.9


class HybridDeckPredictor:
    """LR(``MLDeckPredictor``)と NB(``NBDeckPredictor``)を evidence バケット別の重みで
    log-space ブレンドし、観測カードから相手デッキ確率分布を推論する。"""

    def __init__(
        self,
        lr_weights_path: str | Path | None = None,
        nb_weights_path: str | Path | None = None,
        hybrid_config_path: str | Path | None = None,
    ):
        self._lr = MLDeckPredictor(weights_path=lr_weights_path)
        self._nb = NBDeckPredictor(weights_path=nb_weights_path)

        if hybrid_config_path is None:
            hybrid_config_path = Path(__file__).parent / _DEFAULT_HYBRID_FILENAME
        self._hybrid_config_path = Path(hybrid_config_path)

        # meta.buckets(無ければ空リスト = 全 evidence 数で w=0、LR単体と同一動作)。
        self._buckets: list[dict] = []
        self._load_hybrid_config()

        # クラスリストの一致検証、および現在の動作モードの確定。
        self._classes: list[str] = []
        self.classes_mismatch: bool = False
        self.mode: str = self._resolve_mode()

    @property
    def is_ready(self) -> bool:
        """LR・NB のどちらか一方でも読み込めていれば True。両方未ロードのときのみ False。"""
        return self.mode != "unready"

    def _load_hybrid_config(self) -> None:
        if not self._hybrid_config_path.exists():
            return  # 未配置(w=0固定 = LR単体と同一動作)。
        with self._hybrid_config_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._buckets = list(payload.get("buckets", []))

    def _resolve_mode(self) -> str:
        lr_ready = self._lr.is_ready
        nb_ready = self._nb.is_ready

        if not lr_ready and not nb_ready:
            self._classes = []
            return "unready"

        if lr_ready and not nb_ready:
            self._classes = list(self._lr._classes)
            return "lr_only"

        if nb_ready and not lr_ready:
            self._classes = list(self._nb._classes)
            return "nb_only"

        # 両方ロード済み: クラスリストが一致するか検証する。
        lr_classes = list(self._lr._classes)
        nb_classes = list(self._nb._classes)
        if set(lr_classes) != set(nb_classes):
            self.classes_mismatch = True
            self._classes = lr_classes
            return "lr_only"

        self._classes = lr_classes
        return "hybrid"

    def predict(
        self,
        observed_cards: dict[str, int],
        turn: int,
        *,
        rough_prediction: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """観測済みカード枚数とターン数から、全クラスの確率分布(合計1.0)を返す。

        ``mode`` に応じて LR単体 / NB単体 / 両者のブレンドのいずれかを行う(詳細はモジュール
        docstring のフォールバック規則を参照)。

        ``rough_prediction``(``rough_predictor.predict()`` の戻り値、省略時 ``None``)を渡すと、
        rough が LR/NB の学習済みクラスに無い新アーキタイプを confident で返している場合だけ、
        そのクラスの確率質量を最終分布に反映する(詳細はモジュール docstring
        「rough_predictor による未学習クラスのオーバーライド」節を参照)。``None``(既定)の
        ときは従来と完全に同じ分布を返す。
        """
        probs = self._predict_base(observed_cards, turn)
        return _apply_rough_override(probs, rough_prediction, self._classes)

    def _predict_base(self, observed_cards: dict[str, int], turn: int) -> dict[str, float]:
        """LR/NB(のブレンド)だけから確率分布を計算する(rough によるオーバーライド前)。"""
        if self.mode == "unready":
            return {"other": 1.0}
        if self.mode == "lr_only":
            return self._lr.predict(observed_cards, turn)
        if self.mode == "nb_only":
            return self._nb.predict(observed_cards, turn)

        # mode == "hybrid"
        weight_nb = self._weight_for(self._lr.evidence_count(observed_cards))
        if weight_nb <= 0.0:
            return self._lr.predict(observed_cards, turn)
        if weight_nb >= 1.0:
            return self._nb.predict(observed_cards, turn)

        lr_probs = self._lr.predict(observed_cards, turn)
        nb_probs = self._nb.predict(observed_cards, turn)
        log_blend = [
            (1.0 - weight_nb) * math.log(max(lr_probs.get(c, 0.0), 1e-12))
            + weight_nb * math.log(max(nb_probs.get(c, 0.0), 1e-12))
            for c in self._classes
        ]
        probs = _log_normalize(log_blend)
        return dict(zip(self._classes, probs))

    def _weight_for(self, evidence_count: int) -> float:
        """evidence_count が属するバケットの NB 側の重み w を返す。該当バケットが無ければ 0.0(LR単体相当)。"""
        for bucket in self._buckets:
            min_evidence = bucket["min_evidence"]
            max_evidence = bucket.get("max_evidence")
            if evidence_count < min_evidence:
                continue
            if max_evidence is not None and evidence_count > max_evidence:
                continue
            return float(bucket["weight_nb"])
        return 0.0

    def predict_top(
        self, observed_cards: dict[str, int], turn: int, n: int = 3
    ) -> list[tuple[str, float]]:
        """確率降順で上位 n 件を [(クラス名, 確率), ...] で返す。"""
        probs = self.predict(observed_cards, turn)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        return ranked[:n]

    def explain(self, observed_cards: dict[str, int], turn: int) -> dict:
        """予測の根拠を返す(ビュアーの根拠表示、および
        ``prediction_summary.summarize_prediction()`` の evidence_count 補完用)。

        NB 側には実装済みの ``NBDeckPredictor.explain()``(観測カード別のクラス別 log 尤度寄与)を、
        ブレンド文脈(このリクエストで実際に使われた weight_nb)付きでパススルーする。LR単体
        (``MLDeckPredictor``)には根拠分解を実装していない(線形係数×温度の寄与分解は誤解を
        招きやすいため。モジュール docstring 参照)ので、NB 側の重みが 0(= このリクエストで
        NB を使っていない)のときは ``nb_explanation`` は ``None`` になる。

        戻り値スキーマ:
            {
              "mode": "hybrid" | "lr_only" | "nb_only" | "unready",
              "weight_nb": このリクエストの evidence_count で実際に使った NB 側の重み w
                           (0.0-1.0)。mode="unready" のときのみ None。
              "evidence_count": MLDeckPredictor.evidence_count() の定義で数えたエビデンス数。
                           LR が未ロード(mode="nb_only"/"unready")のときは None
                           (NB は evidence_count の定義を持たないため)。
              "nb_explanation": NBDeckPredictor.explain() の戻り値、または None
                           (weight_nb <= 0.0 で NB を使っていない場合)。
            }
        """
        if self.mode == "unready":
            return {"mode": self.mode, "weight_nb": None, "evidence_count": None, "nb_explanation": None}

        if self.mode == "nb_only":
            return {
                "mode": self.mode,
                "weight_nb": 1.0,
                "evidence_count": None,
                "nb_explanation": self._nb.explain(observed_cards, turn),
            }

        # mode in ("lr_only", "hybrid"): LR がロード済みなので evidence_count が定義できる。
        evidence_count = self._lr.evidence_count(observed_cards)

        if self.mode == "lr_only":
            return {"mode": self.mode, "weight_nb": 0.0, "evidence_count": evidence_count, "nb_explanation": None}

        # mode == "hybrid"
        weight_nb = self._weight_for(evidence_count)
        nb_explanation = self._nb.explain(observed_cards, turn) if weight_nb > 0.0 else None
        return {
            "mode": self.mode,
            "weight_nb": weight_nb,
            "evidence_count": evidence_count,
            "nb_explanation": nb_explanation,
        }


def _log_normalize(log_values: list[float]) -> list[float]:
    """log-sum-exp でオーバーフロー対策しつつ、log 値のリストを合計1.0の確率分布に正規化する。"""
    m = max(log_values)
    exps = [math.exp(v - m) for v in log_values]
    total = sum(exps)
    return [v / total for v in exps]


def _apply_rough_override(
    probs: dict[str, float],
    rough_prediction: dict[str, Any] | None,
    classes: list[str],
) -> dict[str, float]:
    """rough_predictor が LR/NB の学習済みクラスに無い新アーキタイプを confident で返した
    場合だけ、そのクラスの確率質量を ``probs`` に合成する。合成規則の詳細はモジュール
    docstring「rough_predictor による未学習クラスのオーバーライド」節を参照。

    ``rough_prediction`` が ``None``、confident でない、またはクラスが ``classes`` に
    既に含まれる場合は ``probs`` をそのまま返す(既存クラスの出力に一切影響しない)。
    """
    if not rough_prediction:
        return probs
    if rough_prediction.get("status") != "confident":
        return probs

    deck_type = rough_prediction.get("deck_type")
    if not deck_type or deck_type == "unknown" or deck_type in classes:
        return probs

    try:
        match_rate = float(rough_prediction.get("match_rate", 0.0))
    except (TypeError, ValueError):
        return probs
    if match_rate <= 0.0:
        return probs

    override_prob = min(match_rate, _ROUGH_OVERRIDE_MAX_PROB)
    remaining = max(1.0 - override_prob, 0.0)
    total = sum(probs.values())

    if total <= 0.0:
        # 既存分布が空/全0(通常は起きないが安全側)の場合は新クラスに全質量を割り当てる。
        scaled = {cls: 0.0 for cls in probs}
    else:
        scaled = {cls: (value / total) * remaining for cls, value in probs.items()}
    scaled[deck_type] = scaled.get(deck_type, 0.0) + override_prob
    return scaled
