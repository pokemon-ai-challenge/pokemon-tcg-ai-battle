"""クラスタ② 盤面評価／担当B

技の必要エネルギーの解決。ワザのエネルギーコストに対して、現在付いているエネルギーで
足りているか、あと何が必要かを計算する（無色エネルギーの充当も考慮する）。
"""

from cg.api import Attack, EnergyType


def energy_shortfall(attack: Attack, attached: list[EnergyType]) -> dict[EnergyType, int]:
    """attack.energies に対する attached の不足分を EnergyType ごとに返す（不足なしなら空dict）。"""
    raise NotImplementedError


def is_energy_sufficient(attack: Attack, attached: list[EnergyType]) -> bool:
    """このワザを今すぐ使えるだけのエネルギーが揃っているかを判定する。"""
    raise NotImplementedError
