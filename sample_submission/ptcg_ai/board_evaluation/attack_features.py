"""クラスタ② 盤面評価／担当B

技の打点・与ダメージの解決に関する共通部品。デッキ固有のカードID/カード名をハードコードは
しないが、弱点・抵抗力の判定には「攻撃側ポケモンのタイプ」(CardData.energyType) が要るため、
shared.card_cache 経由で attacker.id から引く（card_cache はカードIDをキーにした汎用の
生データキャッシュであり、特定デッキのカードを直書きするものではない）。
"""

from cg.api import Attack, EnergyType, Pokemon

from ptcg_ai.shared import card_cache

# 現行ルール準拠の簡易モデル: 弱点は2倍、抵抗力は-30（攻撃の追加効果によるダメージ増減は考慮しない）。
_WEAKNESS_MULTIPLIER = 2
_RESISTANCE_REDUCTION = 30


def resolve_damage(attack: Attack, attacker: Pokemon, defender_weakness: EnergyType | None,
                    defender_resistance: EnergyType | None) -> int:
    """弱点・抵抗力を考慮した実際の与ダメージを計算する。

    弱点/抵抗力は「攻撃側ポケモンのタイプ」(CardData.energyType) と防御側の
    weakness/resistance を比較して判定する。
    """
    damage = attack.damage
    attacker_type = card_cache.get_card(attacker.id).energyType

    if defender_weakness is not None and attacker_type == defender_weakness:
        damage *= _WEAKNESS_MULTIPLIER
    if defender_resistance is not None and attacker_type == defender_resistance:
        damage = max(0, damage - _RESISTANCE_REDUCTION)

    return damage


def can_ko(attack: Attack, attacker: Pokemon, defender: Pokemon,
           defender_weakness: EnergyType | None, defender_resistance: EnergyType | None) -> bool:
    """このワザで相手をきぜつさせられるか（残りHP <= 与ダメージ）を判定する。"""
    damage = resolve_damage(attack, attacker, defender_weakness, defender_resistance)
    return damage >= defender.hp
