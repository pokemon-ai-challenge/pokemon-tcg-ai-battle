"""クラスタ③ メイン行動の意思決定（共有部品）／担当B

グッズ/サポート/スタジアムの usage_condition（担当Aが書く「今使うべきか」の判定関数）を
評価する。board_evaluation.usage_context で盤面から UsageContext を組み立てる（B の純粋な
盤面→データ変換）のと、対応する ItemProfile/SupporterProfile/StadiumProfile（A のデータ）を
引いて組み合わせるのは、energy_eval.py / pokemon_value.py と同じくここ（main_turn_parts）の役目。
"""

from cg.api import CardType, State

from ptcg_ai.board_evaluation import usage_context
from ptcg_ai.shared import card_cache, profile_registry


def is_usable(card_id: int, state: State, your_index: int) -> bool:
    """指定カードの usage_condition を判定する。プロファイルが無い/条件未設定なら常に使用可。"""
    card = card_cache.get_card(card_id)

    profile = None
    if card.cardType == CardType.ITEM:
        profile = profile_registry.get_item_profile(card_id)
    elif card.cardType == CardType.SUPPORTER:
        profile = profile_registry.get_supporter_profile(card_id)
    elif card.cardType == CardType.STADIUM:
        profile = profile_registry.get_stadium_profile(card_id)

    if profile is None or profile.usage_condition is None:
        return True

    context = usage_context.build_usage_context(state, your_index)
    return profile.usage_condition(context)
