"""クラスタ⑤ カード知識アクセス／担当B

cg.api.all_card_data() / all_attack() の結果をカードID/技IDでキャッシュする。
実行中に何度も呼ぶと重い可能性があるため、プロセス内で一度だけ取得して使い回す。
"""

from cg.api import Attack, CardData, all_attack, all_card_data

_card_cache: dict[int, CardData] | None = None
_attack_cache: dict[int, Attack] | None = None


def get_card(card_id: int) -> CardData:
    """card_id からカードの生データを引く。初回呼び出し時に all_card_data() をキャッシュする。"""
    global _card_cache
    if _card_cache is None:
        _card_cache = {card.cardId: card for card in all_card_data()}
    return _card_cache[card_id]


def get_attack(attack_id: int) -> Attack:
    """attack_id から技の生データを引く。初回呼び出し時に all_attack() をキャッシュする。"""
    global _attack_cache
    if _attack_cache is None:
        _attack_cache = {attack.attackId: attack for attack in all_attack()}
    return _attack_cache[attack_id]


def reset_cache() -> None:
    """テスト用: キャッシュを破棄する（対戦間でカードデータが変わることは無いため、通常は不要）。"""
    global _card_cache, _attack_cache
    _card_cache = None
    _attack_cache = None
