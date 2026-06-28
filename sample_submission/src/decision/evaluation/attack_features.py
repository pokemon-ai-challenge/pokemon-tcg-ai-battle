from cg.api import Attack, CardData

from src.knowledge.card_cache import load_attack_data


def build_attack_lookup() -> dict[int, Attack]:
    """attackId から技情報を引く辞書を作る。"""
    return {attack.attackId: attack for attack in load_attack_data()}


def resolve_attacks(
    pokemon_card: CardData,
    attack_by_id: dict[int, Attack],
) -> list[Attack]:
    # カードが持つ attackId を、実際の Attack 情報へ展開する。
    attacks: list[Attack] = []
    for attack_id in pokemon_card.attacks:
        attack = attack_by_id.get(attack_id)
        if attack is not None:
            attacks.append(attack)
    return attacks


def choose_main_attack(attacks: list[Attack]) -> Attack | None:
    if not attacks:
        return None
    # 暫定的に、最も高打点で重すぎない技を主力技とみなす。
    return max(attacks, key=lambda attack: (attack.damage, -len(attack.energies)))
