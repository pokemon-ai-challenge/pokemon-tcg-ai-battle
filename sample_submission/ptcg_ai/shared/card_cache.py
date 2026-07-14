"""クラスタ⑤ カード知識アクセス／担当B

cg.api.all_card_data() / all_attack() の結果をカードID/技IDでキャッシュする。
実行中に何度も呼ぶと重い可能性があるため、プロセス内で一度だけ取得して使い回す。
"""

from cg.api import Attack, CardData


def get_card(card_id: int) -> CardData:
    """card_id からカードの生データを引く。初回呼び出し時に all_card_data() をキャッシュする。"""
    raise NotImplementedError


def get_attack(attack_id: int) -> Attack:
    """attack_id から技の生データを引く。初回呼び出し時に all_attack() をキャッシュする。"""
    raise NotImplementedError
