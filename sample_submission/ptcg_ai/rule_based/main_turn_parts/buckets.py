"""クラスタ③ メイン行動の意思決定／担当B

MAIN の選択肢（Option）を行動カテゴリ（draw/board/ability/energy/retreat/attack/end）に
仕分ける。分類のみを行い、どれを選ぶべきかの判断はしない。

Option.type だけで分類できるもの（ABILITY/ATTACH/RETREAT/ATTACK/END）はそのまま仕分けられるが、
PLAY（グッズ/サポート/どうぐ/スタジアム/進化）は種類の判別に card_id が要るため、
knowledge.card_cache でカード種別を、knowledge.profile_registry の
EffectCategory（"draw"/"search" など）で draw/board のどちらに入れるかを判定する。
"""

from cg.api import Option

from ptcg_ai.shared import card_cache, profile_registry

CATEGORIES = ("draw", "board", "ability", "energy", "retreat", "attack", "end")


def bucketize(options: list[Option]) -> dict[str, list[Option]]:
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
    raise NotImplementedError
