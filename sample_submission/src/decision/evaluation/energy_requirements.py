from cg.api import Attack, EnergyType


def has_payable_attack(
    attacks: list[Attack],
    energies: list[EnergyType],
) -> bool:
    # ダメージ量ではなく、技が実際に使えるかどうかだけを判定する。
    return any(energy_gap_for_attack(attack, energies) == 0 for attack in attacks)


def best_payable_attack_damage(
    attacks: list[Attack],
    energies: list[EnergyType],
) -> int:
    # 現在のエネルギーで実際に使える技のうち、最大打点を返す。
    best_damage = 0
    for attack in attacks:
        if energy_gap_for_attack(attack, energies) == 0:
            best_damage = max(best_damage, attack.damage)
    return best_damage


def energy_gap_for_attack(
    attack: Attack | None,
    energies: list[EnergyType],
) -> int:
    if attack is None:
        return 99

    # 先に色指定を埋め、その後で無色要求を残りエネで埋める。
    remaining_energies = list(energies)
    specific_requirements = [
        energy for energy in attack.energies if energy != EnergyType.COLORLESS
    ]
    colorless_requirements = len(attack.energies) - len(specific_requirements)
    unmatched_specific = 0

    for required_energy in specific_requirements:
        matched_index = find_matching_energy_index(remaining_energies, required_energy)
        if matched_index is None:
            unmatched_specific += 1
            continue
        remaining_energies.pop(matched_index)

    remaining_colorless_gap = max(0, colorless_requirements - len(remaining_energies))
    return unmatched_specific + remaining_colorless_gap


def find_matching_energy_index(
    energies: list[EnergyType],
    required_energy: EnergyType,
) -> int | None:
    # 特殊エネルギーの簡易的な色代用もここで扱う。
    for provided_energy in (
        required_energy,
        EnergyType.TEAM_ROCKET,
        EnergyType.RAINBOW,
    ):
        for index, energy in enumerate(energies):
            if energy == provided_energy and provided_energy == required_energy:
                return index
            if provided_energy == EnergyType.TEAM_ROCKET and energy == EnergyType.TEAM_ROCKET:
                if required_energy in {EnergyType.PSYCHIC, EnergyType.DARKNESS}:
                    return index
            if provided_energy == EnergyType.RAINBOW and energy == EnergyType.RAINBOW:
                return index
    return None
