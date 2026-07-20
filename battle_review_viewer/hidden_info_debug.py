"""``ptcg_ai.hidden_information`` (OwnHiddenState / OpponentHiddenState) の推定結果を、
ビュアー向けの確率分布デバッグ情報に変換する。

``ml_prediction_debug.py``（相手デッキ予測器のブリッジ）と同じ役回りで、``opponentKnowledgeDebug``
dict の中に ``"hidden_info"`` キーとして同居させる。予測器/推定レイヤーが None（インポート失敗）や
未ロードでも None／最小限の dict を返すだけで、呼び出し側（``export_replay.py`` /
``live_match.py``）の処理は止めない。

## 表示は「確率」に絞っているが、実データは関数で取れる（重要）

ビュアーが出しているのは各ゾーンの「最低1枚ある確率」（``marginals()``）だけ。これは表示上の
割り切りで、推定レイヤー自体は枚数まで持っている。用途に応じて以下を直接呼べば取得できる:
- ``OwnHiddenState.sample()`` / ``OpponentHiddenState.sample()`` … 山札/サイド(/手札)へ枚数を割り当てた
  determinization（超幾何サンプリング）。ISMCTS の ``search_begin`` にはこちらを使う。
- ``zone_math.expected_in_prize()`` / ``expected_in_zone()`` … 各ゾーンの期待枚数（``k·n/M``）。
- ``OwnHiddenState`` の ``_pool`` は「山札∪サイド」の残り枚数そのもの（``card_id -> 枚数``）。
つまりビュアーは確率表示にしているだけで、枚数が失われているわけではない。

## 責務分担（重要）

``OwnHiddenState.update()`` / ``resolve_deck_search()`` と ``OpponentHiddenState.update()`` の
呼び出しは、この関数自身が担う（呼び出し側は state のインスタンスと ``select`` を渡すだけ）。
呼び出し側（``run_match()`` / ``LiveMatchSession._build_opponent_knowledge_debug_locked()``）は
1フレームにつきこの関数を高々1回しか呼ばないため、ここで更新しても二重更新にはならない
（``knowledge.update_from_logs/update_from_state`` と同じ「player0 の手番のフレームでだけ」という
呼び出し規約に相乗りする）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_hidden_info_debug(
    own_state: Any,
    opponent_state: Any,
    predictor: Any,
    knowledge: Any,
    real_state: Any,
    select: Any = None,
    top_n: int = 15,
    visual_current: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """自分側（サイド落ち候補）・相手側（手札候補）の周辺確率を top_n 件ずつまとめて返す。

    ``own_state`` / ``opponent_state`` が両方 None（``hidden_information`` パッケージが
    インポートできない環境）なら None を返す。``real_state`` が None（まだ対局開始前など）の
    ときも None。それ以外の失敗は握りつぶして ``{"error": str(exc)}`` を返す
    （``ml_prediction_debug.py`` と同じフォールバック規則）。

    ``visual_current`` は神視点（``visualize_data()`` の ``current``、伏せの中身まで見える完全情報）。
    渡された場合、各候補に「実際にそのゾーンに何枚あったか」（``actual_deck`` /
    ``actual_prize`` / ``actual_hand``）を重ねる。推定確率はキャリブレーション（ECE）と同じ
    「そのゾーンに最低1枚ある確率」だが、それ単体では当たり外れが目視できないため、
    神視点の実枚数を並べて1フレームごとの命中/はずれを追えるようにする（用途=キャリブレーション検証）。
    神視点が無い（``None``）ゾーンは実枚数を ``None`` にする（正解不明。0枚とは区別する）。
    """
    if own_state is None and opponent_state is None:
        return None
    if real_state is None:
        return None

    try:
        return _build(
            own_state, opponent_state, predictor, knowledge, real_state, select, top_n, visual_current
        )
    except Exception as exc:  # noqa: BLE001 -- 推定レイヤーが落ちてもリプレイ生成/ライブ対戦は止めない
        return {"error": str(exc)}


def _zone_counts(visual_current: dict[str, Any] | None, seat: int, zone_name: str) -> Counter[int] | None:
    """神視点 dict から ``seat`` の ``zone_name`` ゾーンの ``card_id -> 枚数`` を数える。

    神視点が無い / そのゾーンがリストで表現されていない場合は ``None``（正解不明）を返す。
    ``None`` と「空 Counter（そのゾーンは空）」は区別する: 前者は「見えていない」、後者は「実際に0枚」。
    """
    if visual_current is None:
        return None
    players = visual_current.get("players") or []
    if not (0 <= seat < len(players)):
        return None
    zone = players[seat].get(zone_name)
    if not isinstance(zone, list):
        return None
    counts: Counter[int] = Counter()
    for card in zone:
        if isinstance(card, dict) and card.get("id") is not None:
            counts[card["id"]] += 1
    return counts


def _actual(counts: Counter[int] | None, card_id: int) -> int | None:
    """``counts`` が神視点あり（Counter）なら実枚数（0含む）、無ければ ``None``（正解不明）。"""
    return None if counts is None else counts.get(card_id, 0)


def _build(
    own_state: Any,
    opponent_state: Any,
    predictor: Any,
    knowledge: Any,
    real_state: Any,
    select: Any,
    top_n: int,
    visual_current: dict[str, Any] | None = None,
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
        own_deck_truth = _zone_counts(visual_current, my_index, "deck")
        own_prize_truth = _zone_counts(visual_current, my_index, "prize")
        marginals = own_state.marginals()
        candidates = [
            {
                "card_id": card_id,
                "name": id_to_name.get(card_id, str(card_id)),
                "prize_prob": probs["prize"],
                "deck_prob": probs["deck"],
                # 神視点の実枚数（正解不明なら None）。推定確率の当たり外れを1フレームで突き合わせる用。
                "actual_deck": _actual(own_deck_truth, card_id),
                "actual_prize": _actual(own_prize_truth, card_id),
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

        opp_hand_truth = _zone_counts(visual_current, opponent_index, "hand")
        opp_deck_truth = _zone_counts(visual_current, opponent_index, "deck")
        opp_prize_truth = _zone_counts(visual_current, opponent_index, "prize")
        marginals = opponent_state.marginals()
        candidates = [
            {
                "card_id": card_id,
                "name": id_to_name.get(card_id, str(card_id)),
                "hand_prob": probs["hand"],
                "deck_prob": probs["deck"],
                "prize_prob": probs["prize"],
                # 神視点の実枚数（正解不明なら None）。相手は代表リスト由来の card_id なので、
                # 実デッキに無いカードは全ゾーン0枚になる（＝「予測したが相手は持っていない」も可視化される）。
                "actual_hand": _actual(opp_hand_truth, card_id),
                "actual_deck": _actual(opp_deck_truth, card_id),
                "actual_prize": _actual(opp_prize_truth, card_id),
            }
            for card_id, probs in marginals.items()
        ]
        candidates.sort(key=lambda c: c["hand_prob"], reverse=True)
        opponent_payload = {
            "is_ready": bool(getattr(opponent_state, "is_ready", False)),
            "top_hand_candidates": candidates[:top_n],
        }

    return {"own": own_payload, "opponent": opponent_payload}
