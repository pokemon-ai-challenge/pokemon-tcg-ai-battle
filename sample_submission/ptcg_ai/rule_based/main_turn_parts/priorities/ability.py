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
# 【方針転換（2026-07-25、ユーザー指示）: ノココッチの特性は毎ターン欠かさず発動する】
# にげあしドローは「3枚引く→（同じ効果解決の中で）ノコッチ＋ノココッチ＋付いてるカードを山札に戻す」
# 効果で、山札が薄くても撃てば必ず2枚以上に補充される（実測: 山札1枚で撃つと deck 1→2）。
# ユーザー方針: 「ベンチ2枠をノコッチ用に確保し（deck_plan.RESERVED_BENCH_*／priorities/board.py）、
# 毎ターン1回ノココッチの特性を使えばデッキアウトしない」。よって山札フロアは既定0（＝止めない）。
#
# 注意（ミラー自己対戦での実測）: フロア0（毎ターン発動）＋ベンチ2枠確保でも、ミラー自己対戦では
# デッキアウト率が高め（約82%、いずれもRESULTログreason=2）。原因はループ維持率が~45%止まりで
# （ノコッチ／ノココッチの手札供給が毎ターンは保証されず穴が空く）、稼働不十分なまま毎ターン撃つと
# 山札の安定域(2枚)に速く到達して穴が致命傷になるため。ただしミラーは両者同時消耗で、非対称
# マッチアップでの真価は測れない。保守的に戻したい場合は env `PTCG_DRAW_ABILITY_MIN_DECK=14`。
# （手札上限 _DRAW_ABILITY_MAX_HAND は既定で実質無効。手札を絞りたいデッキだけ env で設定する。）
_DRAW_ABILITY_MIN_DECK = int(os.environ.get("PTCG_DRAW_ABILITY_MIN_DECK", "0"))
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
