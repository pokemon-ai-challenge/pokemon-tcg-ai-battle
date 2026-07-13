"""クラスタ② 盤面評価／担当B

技の必要エネルギーの解決。ワザのエネルギーコストに対して、現在付いているエネルギーで
足りているか、あと何が必要かを計算する（無色エネルギーの充当も考慮する）。
"""

from collections import Counter

from cg.api import Attack, EnergyType

# にじいろエネルギー相当: どのタイプの要求にも充当できる「万能」枠。
_UNIVERSAL_WILDCARDS = (EnergyType.RAINBOW,)
# ロケット団のエネルギー相当: PSYCHIC/DARKNESS の要求にのみ充当できる。
_PARTIAL_WILDCARDS: dict[EnergyType, tuple[EnergyType, ...]] = {
    EnergyType.TEAM_ROCKET: (EnergyType.PSYCHIC, EnergyType.DARKNESS),
}


def energy_shortfall(attack: Attack, attached: list[EnergyType]) -> dict[EnergyType, int]:
    """attack.energies に対する attached の不足分を EnergyType ごとに返す（不足なしなら空dict）。

    無色（COLORLESS）の要求はタイプを問わず残りのエネルギーで充当できる。特定タイプの要求は
    同タイプのエネルギーを優先して使い、それでも足りない分は部分万能タイプ
    （TEAM_ROCKET: PSYCHIC/DARKNESSのみ）→ 万能タイプ（RAINBOW: どのタイプにも充当可）の順で補う。
    部分万能タイプを先に使うのは、RAINBOW は他のどの色の不足も埋められる唯一の手段になり得るのに対し
    TEAM_ROCKET は PSYCHIC/DARKNESS 以外の不足を絶対に埋められないため。ここで先に「柔軟性の低い方」
    (TEAM_ROCKET) を使い切ってから「柔軟性の高い方」(RAINBOW) を温存しないと、本来なら別の色の不足を
    埋められたはずの RAINBOW を先に消費してしまい、支払い可能なのに不足ありと誤判定しうる。
    どのタイプでも良い（COLORLESS）要求を最後に回すことで、特定タイプ要求を無駄なワイルドカード
    消費なく満たせるようにしている（色付き要求を先に確定させても無色要求側の充当可能量は
    減らないため、この順序で不足なく計算できる）。
    """
    pool = list(attached)
    colored_needed: Counter[EnergyType] = Counter()
    colorless_needed = 0
    for cost in attack.energies:
        if cost == EnergyType.COLORLESS:
            colorless_needed += 1
        else:
            colored_needed[cost] += 1

    shortfall: dict[EnergyType, int] = {}

    for color, needed in colored_needed.items():
        remaining = needed

        exact_available = pool.count(color)
        use = min(exact_available, remaining)
        for _ in range(use):
            pool.remove(color)
        remaining -= use

        if remaining > 0:
            for wildcard, covers in _PARTIAL_WILDCARDS.items():
                if color not in covers:
                    continue
                while remaining > 0 and wildcard in pool:
                    pool.remove(wildcard)
                    remaining -= 1

        if remaining > 0:
            for wildcard in _UNIVERSAL_WILDCARDS:
                while remaining > 0 and wildcard in pool:
                    pool.remove(wildcard)
                    remaining -= 1

        if remaining > 0:
            shortfall[color] = remaining

    if colorless_needed > 0:
        deficit = colorless_needed - len(pool)
        if deficit > 0:
            shortfall[EnergyType.COLORLESS] = deficit

    return shortfall


def is_energy_sufficient(attack: Attack, attached: list[EnergyType]) -> bool:
    """このワザを今すぐ使えるだけのエネルギーが揃っているかを判定する。"""
    return not energy_shortfall(attack, attached)
