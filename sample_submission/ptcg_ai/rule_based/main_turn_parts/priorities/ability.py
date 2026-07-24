"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: ability（特性の使用）

特性の使いどころを提案する。特性は Attack とは別物（cg.api の CardData.skills に対応）なので、
knowledge.profile_registry.get_pokemon_profile(card_id) が返す PokemonProfile の
has_ability / ability_category / ability_priority を参照する。
"""

import os

from cg.api import Observation, OptionType

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry

# ドロー系特性（ノココッチのにげあしドロー等）を使ってよい条件。
# 方針: フーディンの主砲ハンドパワーは「手札1枚 = 20ダメージ」で、手札そのものが火力の弾。
# 相手は進化して将来HPが上がる（低HPのたね→高HP ex）ため、今の盤面HPから必要手札を予測するのは
# 難しい。よって基本は手札枚数で止めず引き続けて手札を溜める（＝温存しない）。
# 唯一の歯止めは「自分がデッキアウトしないこと」。にげあしドローは3枚引いてから自身を山札に戻すため
# 正味は概ね山札-2枚/回。山札が _DRAW_ABILITY_MIN_DECK 以下になったらドロー特性を止める。
# 既定フロア=14: 対戦開始時の山札は約47枚（60-手札7-サイド6）なので、14まで引けば手札は最大約33枚まで
# 溜められ、ワンパンに要る17枚の「弾」は十分確保できる一方、デッキアウトまでの緩衝を厚く残せる。
# （手札上限 _DRAW_ABILITY_MAX_HAND は既定で実質無効。手札を絞りたいデッキだけ env で設定する。）
#
# 注意（測定の限界）: この“基本引き続ける”方針は、フーディン同士のミラー自己対戦では検証済み cap6 版に
# 対し 27%(floor10)〜35%(cap17) と負ける。ただしミラーは相手も低HPフーディンで「大きい手札で高HP exを
# ワンパンする」恩恵が全く出ないため、この戦略を構造的に過小評価する。高HP ex デッキ相手での価値は
# このブランチのベンチ（self/mirrorのみ）では測れず、meta-deck ベンチのあるブランチでの検証が必要。
_DRAW_ABILITY_MIN_DECK = int(os.environ.get("PTCG_DRAW_ABILITY_MIN_DECK", "14"))
_DRAW_ABILITY_MAX_HAND = int(os.environ.get("PTCG_DRAW_ABILITY_MAX_HAND", "999"))


def propose(obs: Observation) -> ActionProposal | None:
    """特性使用の行動を1つ提案する。該当行動が無ければ None。"""
    player = obs.current.players[obs.current.yourIndex]
    hand_count = player.handCount or 0
    deck_count = player.deckCount if player.deckCount is not None else 0

    best_index: int | None = None
    best_priority = 0.0
    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ABILITY:
            continue
        # ABILITY の Option は cardId を持たない（cg/api.py: area/index のみ）ため、
        # resolve_card_id で場のポケモンの card_id を引く。
        card_id = common.resolve_card_id(option, obs.current)
        if card_id is None:
            continue
        profile = profile_registry.get_pokemon_profile(card_id)
        if profile is None or not profile.has_ability:
            continue
        # ドロー系特性は基本引き続ける。歯止めは山札のデッキアウト回避のみ（＋手札上限は既定無効）。
        if profile.ability_category == "draw" and (
            deck_count <= _DRAW_ABILITY_MIN_DECK or hand_count >= _DRAW_ABILITY_MAX_HAND
        ):
            continue
        if profile.ability_priority > best_priority:
            best_priority = profile.ability_priority
            best_index = i

    if best_index is None:
        return None
    return ActionProposal(category="ability", select=[best_index], score=best_priority, reason="use ability")
