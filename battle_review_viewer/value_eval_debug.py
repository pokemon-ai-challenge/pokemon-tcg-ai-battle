"""OpponentKnowledge とは独立に、Step1 で学習した勝率予測器(ValueModel)を使って
「今の盤面の自分視点勝率」をビュアー向けにまとめる。

判定ロジック(フォワードパス・キャリブレーション)の実体は
``sample_submission/ptcg_ai/learning/value_model.ValueModel`` にあり、ここではその
出力をビュアー payload の形に薄くラップするだけ。設計は
``sample_submission/docs/plans/ml-value-network/design.md`` を参照。
"""

from __future__ import annotations

from typing import Any


def build_value_eval_debug(value_model: Any, real_state: Any) -> dict[str, Any] | None:
    """ValueModel の勝率予測をビュアー表示用の dict にまとめる。

    value_model が None(インポート失敗)または重み未配置(is_ready=False)でも
    None/最小限の dict を返すだけで、呼び出し側の処理は止めない。
    """
    if value_model is None:
        return None
    if not getattr(value_model, "is_ready", False):
        return {"is_ready": False}
    if real_state is None:
        return {"is_ready": True, "turn": None, "win_prob": None, "opponent_win_prob": None}

    try:
        win_prob = value_model.predict_win_prob_from_state(real_state)
    except Exception as exc:  # noqa: BLE001 -- 予測器が落ちてもビュアー/ライブモードは止めない
        return {"is_ready": True, "error": str(exc)}

    return {
        "is_ready": True,
        "turn": real_state.turn,
        "win_prob": win_prob,
        "opponent_win_prob": 1.0 - win_prob,
    }
