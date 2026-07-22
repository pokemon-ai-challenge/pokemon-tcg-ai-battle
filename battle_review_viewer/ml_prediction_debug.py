"""OpponentKnowledge の観測から ML デッキ予測器(MLDeckPredictor / NBDeckPredictor /
HybridDeckPredictor のいずれか)を呼び出し、ビュアー向けの確率分布デバッグ情報を組み立てる。

rough_predictor（ルールベース、根拠カード付きの候補判定）とは別の予測経路として、
両者を同じ debug payload の中に並べて見比べられるようにするための橋渡し。

判定ロジック(未確定判定・根拠)そのものは ``ptcg_ai.opponent_modeling.prediction_summary``
（``summarize_prediction()``）に一元化されている。ここではその戻り値をビュアー payload の
形に薄くラップするだけで、判定基準の実体は持たない(将来エージェント側からも同じ基準を
再利用するため。詳細は
``sample_submission/docs/plans/opponent-deck-predictor/phase-c-d-implementation.md`` フェーズC節)。
"""

from __future__ import annotations

from typing import Any

try:
    from ptcg_ai.opponent_modeling.prediction_summary import summarize_prediction  # noqa: E402
except Exception:  # noqa: BLE001 -- 共通モジュールが無い/壊れていてもビュアーは旧挙動で続行する
    summarize_prediction = None


def build_ml_prediction_debug(
    predictor: Any,
    knowledge: Any,
    state: Any,
    top_n: int = 8,
) -> dict[str, Any] | None:
    """MLDeckPredictor 系の出力をビュアー表示用の dict にまとめる。

    predictor が None（インポート失敗）または重み未配置（is_ready=False）でも
    None/最小限の dict を返すだけで、呼び出し側の処理は止めない。

    後方互換のため、既存キー ``is_ready`` / ``turn`` / ``ranked`` は従来と同じ形のまま返す。
    加えて ``summarize_prediction()`` が使えるときは ``status`` / ``uncertain_threshold`` /
    ``evidence_count`` / ``explanation`` を追加する（過去にエクスポート済みの replay JSON には
    これらのキーが無いので、ビュアー側は欠損を許容してフォールバック表示する）。
    """
    if predictor is None:
        return None
    if not getattr(predictor, "is_ready", False):
        return {"is_ready": False}
    if state is None:
        return {"is_ready": True, "turn": None, "ranked": []}

    observed_cards = knowledge.get_prediction_features().get("observed_cards", {}) if knowledge is not None else {}
    turn = state.turn

    if summarize_prediction is None:
        # 共通モジュールが読み込めない場合は旧来どおり predict_top のみで組み立てる。
        try:
            ranked = predictor.predict_top(observed_cards, turn, n=top_n)
        except Exception as exc:  # noqa: BLE001 -- 予測器が落ちてもビュアー/ライブモードは止めない
            return {"is_ready": True, "error": str(exc)}
        return {
            "is_ready": True,
            "turn": turn,
            "ranked": [{"deck_type": name, "probability": probability} for name, probability in ranked],
        }

    try:
        summary = summarize_prediction(predictor, observed_cards, turn, top_n=top_n)
    except Exception as exc:  # noqa: BLE001 -- 予測器が落ちてもビュアー/ライブモードは止めない
        return {"is_ready": True, "error": str(exc)}

    return {
        "is_ready": True,
        "turn": turn,
        "ranked": [{"deck_type": row["deck_type"], "probability": row["probability"]} for row in summary["top"]],
        "status": summary["status"],
        "uncertain_threshold": summary["uncertain_threshold"],
        "evidence_count": summary["evidence_count"],
        "explanation": summary["explanation"],
    }
