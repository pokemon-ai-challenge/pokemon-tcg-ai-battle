"""``OwnHiddenState`` / ``OpponentHiddenState`` のサンプル結果を、``cg.api.search_begin()`` の
キーワード引数にそのまま展開できる ``dict`` へ変換する薄いアダプタ。

実装プラン Phase 4 の記載どおり、``ptcg_ai/search/``（MCTS本体）はまだ存在しない空パッケージ
（``sample_submission/ptcg_ai/search/__init__.py`` のみ）のため、本モジュールは ``search_begin()``
を実際には呼ばない。返す ``dict`` の各リストの**長さ**が ``search_begin()`` の docstring に書かれた
検証条件（``cg/api.py`` 参照。以下に転記）を満たす形に整えるところまでを行う。

    - ``your_deck``: ``select.deck is not None``（自分の山札サーチ中）なら無視される。それ以外は
      ``len(your_deck) >= deckCount`` が必要。
    - ``your_prize``: ``len(your_prize) >= len(prize)`` が必要。
    - ``opponent_deck`` / ``opponent_prize`` / ``opponent_hand``: 同様に相手側の
      ``deckCount`` / ``len(prize)`` / ``handCount`` 以上が必要。
    - ``opponent_active``: 相手のバトル場が伏せ（``active[0] is None``）のときだけ非空リストが必要
      （中身は Pokémon カードIDである必要がある）。伏せでなければ ``search_begin()`` 側が渡した値を
      無視して ``[]`` 扱いにするので、こちらから空にしておいても実害はない。

## ``None``（不明カード）の扱い

``OpponentHiddenState.sample()`` は、代表リストの総枚数がゾーン合計に満たない場合の不足分を
``None``（不明カード）で埋めて返す（``opponent_hidden_state.py`` モジュールdocstring参照）。
``search_begin()`` は ``list[int]`` を要求する（``ctypes`` 配列化するため ``None`` は渡せない）ため、
本アダプタが「そのまま展開できる」ことを保証する責務として、``None`` を汎用カード1枚（基本エネルギー）
にフォールバック変換する。中身の妥当性（本当にそのカードを持っているか）は保証しない
（design.md 記載どおり、探索結果の妥当性検証自体は ``ptcg_ai/search/`` 実装時の別プランの範囲）。

## ``opponent_active`` の簡易実装

相手のバトル場が伏せのときだけ必要。``OpponentHiddenState`` はゾーン内訳（山札/手札/サイド）しか
モデル化しておらず、「場に出ている伏せポケモンの正体」はそもそも別の4つ目のゾーンであり
``marginals()``/``sample()`` の対象外（design.md にも明記が無い、本フェーズで初めて必要になった
接点）。そのため本アダプタ側で、最有力アーキタイプ（``OpponentHiddenState`` が内部に持つ
事後分布・代表リスト）の中から Basic ポケモンかつ採用枚数が最大のカードを1枚選ぶ、という
簡易ロジックを実装プラン記載どおり組み込む。``OpponentHiddenState`` はこの用途の公開APIを
持たない（Phase 1〜3 の既存ファイルは変更しない制約があるため追加もしない）ので、内部属性
（``_smoothed_normalized_weights()`` / ``_archetype_pool``）を直接参照する。カプセル化の観点では
望ましくないが、疎結合を優先して新しい公開APIを既存モジュールに追加するより、この参照だけに
留める方が変更範囲を最小化できると判断した（実装報告に明記する逸脱点）。
"""

from __future__ import annotations

import random

from cg.api import CardType, Observation, all_card_data

from ptcg_ai.shared.card_cache import get_card

from .opponent_hidden_state import OpponentHiddenState
from .own_hidden_state import OwnHiddenState

_fallback_energy_card_id: int | None = None
_fallback_basic_pokemon_card_id: int | None = None


def _fallback_energy_id() -> int:
    """``None``（不明カード）を埋めるための、汎用的な基本エネルギーのcard_id。

    ``all_card_data()`` は ``card_cache`` 経由でキャッシュ済みの前提が無いため、ここでも
    プロセス内で1回だけ走査してキャッシュする（``card_cache.py`` と同じ思想）。
    """
    global _fallback_energy_card_id
    if _fallback_energy_card_id is None:
        _fallback_energy_card_id = next(
            (card.cardId for card in all_card_data() if card.cardType == CardType.BASIC_ENERGY),
            0,
        )
    return _fallback_energy_card_id


def _fallback_basic_pokemon_id() -> int:
    """``opponent_active`` の推測が全くできない場合の最終フォールバック（何らかのBasicポケモン1体）。"""
    global _fallback_basic_pokemon_card_id
    if _fallback_basic_pokemon_card_id is None:
        _fallback_basic_pokemon_card_id = next(
            (card.cardId for card in all_card_data() if card.basic),
            0,
        )
    return _fallback_basic_pokemon_card_id


def _most_likely_archetype(opponent_state: OpponentHiddenState) -> str | None:
    """``OpponentHiddenState`` が保持するスムージング済み正規化事後分布から最有力アーキタイプを選ぶ。"""
    weights = opponent_state._smoothed_normalized_weights()
    if not weights:
        return None
    return max(weights.items(), key=lambda item: item[1])[0]


def _guess_opponent_active(opponent_state: OpponentHiddenState) -> int | None:
    """相手のバトル場が伏せのときの「正体」を、最有力アーキタイプの代表リストから簡易推測する。

    最有力アーキタイプの代表リスト（``card_counts`` の中央値枚数）のうち、Basicポケモン
    （``CardData.basic``）で絞り込み、採用枚数が最大のものを1体選ぶ。推測できなければ ``None``
    （呼び出し側が ``_fallback_basic_pokemon_id()`` にフォールバックする）。
    """
    archetype = _most_likely_archetype(opponent_state)
    if archetype is None:
        return None
    pool = opponent_state._archetype_pool.get(archetype)
    if not pool:
        return None

    best_card_id: int | None = None
    best_count = -1
    for card_id, count in pool.items():
        try:
            card_data = get_card(card_id)
        except KeyError:
            continue
        if not card_data.basic:
            continue
        if count > best_count:
            best_count = count
            best_card_id = card_id
    return best_card_id


def to_search_begin_kwargs(
    own_state: OwnHiddenState,
    opponent_state: OpponentHiddenState,
    obs: Observation,
    rng: random.Random | None = None,
) -> dict:
    """``search_begin()`` にそのまま展開できるキーワード引数の ``dict`` を組み立てる。

    ``own_state.sample()`` / ``opponent_state.sample()`` を1回ずつ呼ぶ（呼び出しごとに違う
    サンプルになる。決定化のタイミングは呼び出し側の責務）。``obs.current`` が ``None``
    （デッキ選択ターン）の場合は組み立てようがないため ``ValueError`` を送出する
    （呼び出し側は通常ターンでのみ呼ぶ想定）。
    """
    state = obs.current
    if state is None:
        raise ValueError("obs.current が None(デッキ選択ターン)のため search_begin 用の引数を組み立てられない")

    your_index = state.yourIndex
    opponent_index = 1 - your_index

    your_deck, your_prize = own_state.sample(rng)
    opp_deck_raw, opp_hand_raw, opp_prize_raw = opponent_state.sample(rng)

    fallback_energy = _fallback_energy_id()
    opponent_deck = [card_id if card_id is not None else fallback_energy for card_id in opp_deck_raw]
    opponent_hand = [card_id if card_id is not None else fallback_energy for card_id in opp_hand_raw]
    opponent_prize = [card_id if card_id is not None else fallback_energy for card_id in opp_prize_raw]

    opponent_active: list[int] = []
    opponent_player = state.players[opponent_index]
    if opponent_player.active and opponent_player.active[0] is None:
        guessed = _guess_opponent_active(opponent_state)
        opponent_active = [guessed if guessed is not None else _fallback_basic_pokemon_id()]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }
