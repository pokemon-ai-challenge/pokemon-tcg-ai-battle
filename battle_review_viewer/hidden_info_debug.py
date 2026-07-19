"""``ptcg_ai.hidden_information`` (OwnHiddenState / OpponentHiddenState) の推定結果を、
ビュアー向けの確率分布デバッグ情報に変換する。

``ml_prediction_debug.py``（相手デッキ予測器のブリッジ）と同じ役回りで、``opponentKnowledgeDebug``
dict の中に ``"hidden_info"`` キーとして同居させる。予測器/推定レイヤーが None（インポート失敗）や
未ロードでも None／最小限の dict を返すだけで、呼び出し側（``export_replay.py`` /
``live_match.py``）の処理は止めない。

## 責務分担（重要）

``OwnHiddenState.update()`` / ``resolve_deck_search()`` と ``OpponentHiddenState.update()`` の
呼び出しは、この関数自身が担う（呼び出し側は state のインスタンスと ``select`` を渡すだけ）。
呼び出し側（``run_match()`` / ``LiveMatchSession._build_opponent_knowledge_debug_locked()``）は
1フレームにつきこの関数を高々1回しか呼ばないため、ここで更新しても二重更新にはならない
（``knowledge.update_from_logs/update_from_state`` と同じ「player0 の手番のフレームでだけ」という
呼び出し規約に相乗りする）。
"""

from __future__ import annotations

from typing import Any


def build_hidden_info_debug(
    own_state: Any,
    opponent_state: Any,
    predictor: Any,
    knowledge: Any,
    real_state: Any,
    select: Any = None,
    top_n: int = 15,
) -> dict[str, Any] | None:
    """自分側（サイド落ち候補）・相手側（手札候補）の周辺確率を top_n 件ずつまとめて返す。

    ``own_state`` / ``opponent_state`` が両方 None（``hidden_information`` パッケージが
    インポートできない環境）なら None を返す。``real_state`` が None（まだ対局開始前など）の
    ときも None。それ以外の失敗は握りつぶして ``{"error": str(exc)}`` を返す
    （``ml_prediction_debug.py`` と同じフォールバック規則）。
    """
    if own_state is None and opponent_state is None:
        return None
    if real_state is None:
        return None

    try:
        return _build(own_state, opponent_state, predictor, knowledge, real_state, select, top_n)
    except Exception as exc:  # noqa: BLE001 -- 推定レイヤーが落ちてもリプレイ生成/ライブ対戦は止めない
        return {"error": str(exc)}


def _build(
    own_state: Any,
    opponent_state: Any,
    predictor: Any,
    knowledge: Any,
    real_state: Any,
    select: Any,
    top_n: int,
) -> dict[str, Any]:
    from cg.api import all_card_data  # noqa: PLC0415 -- 遅延import(呼び出し側のsys.path設定に依存するため)

    id_to_name = {card.cardId: card.name for card in all_card_data()}

    my_index = real_state.yourIndex
    opponent_index = 1 - my_index

    own_payload = None
    if own_state is not None:
        own_state.update(real_state, select)
        if select is not None and getattr(select, "deck", None) is not None:
            own_state.resolve_deck_search(select)

        my_player = real_state.players[my_index]
        marginals = own_state.marginals()
        candidates = [
            {
                "card_id": card_id,
                "name": id_to_name.get(card_id, str(card_id)),
                "prize_prob": probs["prize"],
                "deck_prob": probs["deck"],
            }
            for card_id, probs in marginals.items()
        ]
        candidates.sort(key=lambda c: c["prize_prob"], reverse=True)
        own_payload = {
            "prize_count": len(my_player.prize),
            # OwnHiddenState の「未確認プール」= 山札∪サイド全体（design.md §4）。deckCount + サイド枚数と
            # 一致することが update() 側の不変条件で検算済みなので、公開フィールドから独自集計する。
            "pool_size": my_player.deckCount + len(my_player.prize),
            "top_prize_candidates": candidates[:top_n],
        }

    opponent_payload = None
    if opponent_state is not None:
        features = knowledge.get_prediction_features() if knowledge is not None else {}
        observed_card_ids = features.get("observed_card_ids", {})
        observed_cards = features.get("observed_cards", {})

        archetype_posterior: dict[str, float] = {}
        if predictor is not None and getattr(predictor, "is_ready", False):
            try:
                archetype_posterior = predictor.predict(observed_cards, real_state.turn)
            except Exception:  # noqa: BLE001 -- 予測器が落ちてもゾーン推定自体は続行する（重み無しとして扱う）
                archetype_posterior = {}

        opponent_state.update(archetype_posterior, observed_card_ids, real_state.players[opponent_index])

        marginals = opponent_state.marginals()
        candidates = [
            {
                "card_id": card_id,
                "name": id_to_name.get(card_id, str(card_id)),
                "hand_prob": probs["hand"],
                "deck_prob": probs["deck"],
                "prize_prob": probs["prize"],
            }
            for card_id, probs in marginals.items()
        ]
        candidates.sort(key=lambda c: c["hand_prob"], reverse=True)
        opponent_payload = {
            "is_ready": bool(getattr(opponent_state, "is_ready", False)),
            "top_hand_candidates": candidates[:top_n],
        }

    return {"own": own_payload, "opponent": opponent_payload}
