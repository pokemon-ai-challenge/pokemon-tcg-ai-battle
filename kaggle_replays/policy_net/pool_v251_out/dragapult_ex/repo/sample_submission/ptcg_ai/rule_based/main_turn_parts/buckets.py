"""クラスタ③ メイン行動の意思決定／担当B

MAIN の選択肢（Option）を行動カテゴリ（draw/board/ability/energy/retreat/attack/end）に
仕分ける。分類のみを行い、どれを選ぶべきかの判断はしない。

Option.type だけで分類できるもの（ABILITY/ATTACH/RETREAT/ATTACK/END）はそのまま仕分けられるが、
PLAY（グッズ/サポート/どうぐ/スタジアム/進化）は種類の判別に card_id が要るため、
knowledge.card_cache でカード種別を、knowledge.profile_registry の
EffectCategory（"draw"/"search" など）で draw/board のどちらに入れるかを判定する。
"""

from cg.api import CardType, Option, OptionType, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import card_cache, profile_registry

CATEGORIES = ("draw", "board", "ability", "energy", "retreat", "attack", "end")

_DRAW_LIKE_EFFECT_CATEGORIES = ("draw", "search")


def bucketize(options: list[Option], state: State) -> dict[str, list[Option]]:
    """Option.type / card_id を見て CATEGORIES ごとの Option リストに仕分ける。

    対応の目安:
        draw    -> PLAY のうち EffectCategory が "draw"/"search" のサポーター・グッズ
        board   -> PLAY のうちそれ以外（展開/進化/スタジアム/どうぐ系）、EVOLVE
        ability -> ABILITY
        energy  -> ATTACH
        retreat -> RETREAT
        attack  -> ATTACK
        end     -> END
    """
    buckets: dict[str, list[Option]] = {category: [] for category in CATEGORIES}
    for option in options:
        buckets[classify(option, state)].append(option)
    return buckets


def classify(option: Option, state: State) -> str:
    """1つの Option がどの CATEGORIES に属するかを判定する。

    obs.select.option の元の並び順（インデックス）を維持したまま分類したい呼び出し元
    （priorities/*.py）は、bucketize ではなくこちらを enumerate と組み合わせて使う。
    """
    if option.type == OptionType.ABILITY:
        return "ability"
    if option.type == OptionType.ATTACH:
        return "energy"
    if option.type == OptionType.RETREAT:
        return "retreat"
    if option.type == OptionType.ATTACK:
        return "attack"
    if option.type == OptionType.END:
        return "end"
    if option.type == OptionType.PLAY:
        return _classify_play(option, state)
    # EVOLVE・DISCARD など、上記に当てはまらないものは展開系として board 扱いにする。
    return "board"


def _classify_play(option: Option, state: State) -> str:
    # PLAY は option.cardId を持たない（cg/api.py: PLAY は index=手札内インデックスのみ）ため、
    # 必ず resolve_card_id で手札から実カードIDを引く。
    card_id = common.resolve_card_id(option, state)
    if card_id is None:
        return "board"

    card = card_cache.get_card(card_id)
    if card.cardType == CardType.ITEM:
        profile = profile_registry.get_item_profile(card_id)
    elif card.cardType == CardType.SUPPORTER:
        profile = profile_registry.get_supporter_profile(card_id)
    else:
        return "board"

    if profile is not None and profile.category in _DRAW_LIKE_EFFECT_CATEGORIES:
        return "draw"
    return "board"
