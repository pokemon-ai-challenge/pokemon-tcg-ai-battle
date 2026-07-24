"""相手デッキ予測器の出力を「未確定判定つきの候補分布」に変換する共通モジュール。

背景・設計方針は
``sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md``
（「出力の意味論を『診断』から『候補分布』へ」節）と
``sample_submission/docs/plans/opponent-deck-predictor/phase-c-d-implementation.md``（フェーズC）
を参照。

## なぜこのモジュールが要るか

evidence（見えているユニークカード種類数）が少ない場面では、予測器の top-1 確率は
「メタで一番多いデッキ」を言っているだけのことが多い（prior が支配的）。これをビュアーが
毎回バラバラに「断定表示」してしまわないよう、**「未確定判定」と「根拠」の判定ロジックを
ランタイムライブラリ側に一元化**し、ビュアー・将来のエージェント本体（``search_begin`` に渡す
相手デッキの決定、MatchupPolicy の適用判断など）・ML 側のいずれからも同じ基準で
``summarize_prediction()`` を呼べば同じ結果になるようにする。

純Python（外部ライブラリ非依存）。提出環境（Kaggle シミュレーション上のエージェント）からも
そのまま呼べる。

## ``summarize_prediction()`` の戻り値スキーマ（ビュアー・エージェント共通の契約）

以下のキーを持つ dict を返す。**このスキーマはランタイムライブラリ側が定義する契約であり、
利用側（ビュアー・エージェント）の都合で変更しないこと。** 新しいキーを追加する分には
後方互換だが、既存キーの意味・型は変えない。

    {
        "status": "confident" | "uncertain" | "unready",
        "top": [{"deck_type": "mega_lucario_ex", "probability": 0.62}, ...],  # 確率降順・top_n件
        "top1_probability": 0.62,          # top が空なら 0.0
        "uncertain_threshold": 0.6,        # 判定に使ったしきい値（呼び出し時の引数をそのまま反映）
        "evidence_count": 3,               # 予測器から取得できた場合のみ int。取れなければ None
        "explanation": {...} | None,       # 予測器が explain() を持つ場合のみ dict。無ければ None
    }

- ``status``:
  - 予測器が未ロード（``is_ready`` が False、または ``is_ready`` 属性自体が無い）なら ``"unready"``。
  - それ以外で ``top1_probability < uncertain_threshold`` なら ``"uncertain"``
    （候補が割れている・prior に留まっている状態。ビュアーは1位を断定表示せず候補列挙にする）。
  - それ以外は ``"confident"``。
- ``top``: ``predictor.predict_top(observed_cards, turn, n=top_n)`` の結果をそのまま
  ``{"deck_type": ..., "probability": ...}`` の dict リストに変換したもの。``status="unready"``
  のときは空リスト。
- ``evidence_count``: 予測器が ``evidence_count(observed_cards)`` メソッドを持つ場合はそれを
  直接呼ぶ（``MLDeckPredictor`` 等）。持たないが ``explain()`` の戻り値に ``"evidence_count"``
  キーがある場合はそれを流用する（``HybridDeckPredictor.explain()`` はこの経路で埋まる）。
  どちらもなければ ``None``。
- ``explanation``: 予測器が ``explain(observed_cards, turn)`` メソッドを持つ場合はその戻り値
  （スキーマは予測器ごとに異なる。``nb_predictor.NBDeckPredictor.explain()`` /
  ``hybrid_predictor.HybridDeckPredictor.explain()`` のdocstring参照）。持たなければ
  ``None``（``ml_predictor.MLDeckPredictor`` は explain 非対応）。

## duck-typing

``predictor`` は ``MLDeckPredictor`` / ``NBDeckPredictor`` / ``HybridDeckPredictor`` の
いずれでも動く。型チェックはせず、``is_ready`` / ``predict_top`` / ``evidence_count`` /
``explain`` の有無を ``getattr`` で確認して呼ぶ。``predict_top`` を持たない・呼び出しが
例外を投げるような壊れた予測器を渡した場合は例外を伝播させる（呼び出し側の try/except に委ねる。
本モジュール自身はエラーもみ消しをしない）。ただし ``explain`` の呼び出しが失敗した場合は
根拠表示が壊れるだけで確率自体には影響しないため、``explanation`` を ``None`` にして
処理を継続する。
"""

from __future__ import annotations

from typing import Any

UNCERTAIN_THRESHOLD_DEFAULT = 0.6  # 判定基準の一元管理。利用側はこの定数を参照する。


def summarize_prediction(
    predictor: Any,
    observed_cards: dict[str, int],
    turn: int,
    *,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD_DEFAULT,
    top_n: int = 3,
) -> dict[str, Any]:
    """予測器の出力を「未確定判定つきの候補分布」dict にまとめる。

    戻り値スキーマはモジュール docstring を参照。``predictor`` は duck-typing で受ける
    （``MLDeckPredictor`` / ``NBDeckPredictor`` / ``HybridDeckPredictor`` のいずれでも動く）。
    """
    if not getattr(predictor, "is_ready", False):
        return {
            "status": "unready",
            "top": [],
            "top1_probability": 0.0,
            "uncertain_threshold": uncertain_threshold,
            "evidence_count": None,
            "explanation": None,
        }

    ranked = predictor.predict_top(observed_cards, turn, n=top_n)
    top = [{"deck_type": name, "probability": probability} for name, probability in ranked]
    top1_probability = top[0]["probability"] if top else 0.0
    status = "confident" if top1_probability >= uncertain_threshold else "uncertain"

    explanation = _safe_explain(predictor, observed_cards, turn)
    evidence_count = _resolve_evidence_count(predictor, observed_cards, explanation)

    return {
        "status": status,
        "top": top,
        "top1_probability": top1_probability,
        "uncertain_threshold": uncertain_threshold,
        "evidence_count": evidence_count,
        "explanation": explanation,
    }


def _safe_explain(predictor: Any, observed_cards: dict[str, int], turn: int) -> dict | None:
    """予測器が ``explain()`` を持つ場合に呼び出す。失敗しても None を返すだけで例外にしない
    （根拠表示は付加情報であり、確率計算自体を止める理由にはならないため）。
    """
    explain_fn = getattr(predictor, "explain", None)
    if not callable(explain_fn):
        return None
    try:
        return explain_fn(observed_cards, turn)
    except Exception:  # noqa: BLE001 -- 根拠表示が壊れても呼び出し側を止めない
        return None


def _resolve_evidence_count(
    predictor: Any, observed_cards: dict[str, int], explanation: dict | None
) -> int | None:
    """evidence_count を取得する。優先順位:

    1. 予測器が ``evidence_count(observed_cards)`` メソッドを持つ場合はそれを直接呼ぶ
       （``MLDeckPredictor`` はこの経路）。
    2. 持たない場合、``explain()`` の戻り値に ``"evidence_count"`` キーがあればそれを流用する
       （``HybridDeckPredictor.explain()`` はこの経路。LR側の ``evidence_count()`` を内部で
       再利用しているので定義がズレない）。
    3. どちらも無ければ None。
    """
    evidence_count_fn = getattr(predictor, "evidence_count", None)
    if callable(evidence_count_fn):
        try:
            return evidence_count_fn(observed_cards)
        except Exception:  # noqa: BLE001 -- evidence_count が壊れても致命的にしない
            return None

    if isinstance(explanation, dict) and "evidence_count" in explanation:
        return explanation["evidence_count"]

    return None
