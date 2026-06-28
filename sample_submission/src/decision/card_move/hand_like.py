from cg.api import Observation, SelectContext

from src.decision.card_move import common as card_move_common
from src.decision.card_move.common import (
    analyze_card_move_option,
    resolve_option_entry,
)
from src.decision.card_move.to_hand_eval import choose_to_hand_action
from src.decision.fallback import choose_random_legal_action


def choose_hand_like_action(obs: Observation) -> list[int]:
    """TO_HAND の対象選択を専用評価へ渡す。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during card move decisions.")
    if obs.select.context != SelectContext.TO_HAND:
        raise ValueError(
            "choose_hand_like_action only supports SelectContext.TO_HAND."
        )

    option_count = len(obs.select.option)
    if option_count == 0:
        return [] if obs.select.minCount == 0 else choose_random_legal_action(obs)

    # 全候補を取ることが確定しているなら、そのまま返す。
    if obs.select.minCount == obs.select.maxCount == option_count:
        return list(range(option_count))

    # まずは TO_HAND 専用の評価で「何を取るか」を決める。
    chosen = choose_to_hand_action(
        obs,
        analyze_option=analyze_card_move_option,
        card_data_lookup=card_move_common._card_data_lookup,
        resolve_option_entry=resolve_option_entry,
    )
    if chosen is not None:
        return chosen

    # 未対応の TO_HAND 効果に戻し用評価を流用すると意味が逆転するので、
    # ここでは安全側にランダム合法手へ戻す。
    return choose_random_legal_action(obs)
