"""クラスタ② 盤面評価／担当B

技の打点・与ダメージの解決に関する共通部品。カードIDやカード名は扱わず、
cg.api の Attack / Pokemon を入力に取る純粋な計算関数のみを置く。
"""

from cg.api import Attack, EnergyType, Pokemon


def resolve_damage(attack: Attack, attacker: Pokemon, defender_weakness: EnergyType | None,
                    defender_resistance: EnergyType | None) -> int:
    """弱点・抵抗力を考慮した実際の与ダメージを計算する。"""
    raise NotImplementedError


def can_ko(attack: Attack, attacker: Pokemon, defender: Pokemon,
           defender_weakness: EnergyType | None, defender_resistance: EnergyType | None) -> bool:
    """このワザで相手をきぜつさせられるか（残りHP <= 与ダメージ）を判定する。"""
    raise NotImplementedError
