"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: TO_BENCH, TO_FIELD, TO_HAND, DISCARD, TO_DECK, TO_DECK_BOTTOM,
TO_PRIZE, NOT_MOVE, LOOK, EFFECT_TARGET, DISCARD_CARD_OR_ATTACHED_CARD

カード/対象をどこかへ移動させる系の選択全般。実際にどのカードを選ぶかの判断は
decision.card_move 配下（クラスタ④）のサブモジュールに context ごとに委譲する。
"""

from cg.api import Observation, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.rule_based.card_move import bench_field, discard, hand_like, hidden_zone, not_move_or_look

_MODULE_BY_CONTEXT = {
    SelectContext.TO_BENCH: bench_field,
    SelectContext.TO_FIELD: bench_field,
    SelectContext.TO_HAND: hand_like,
    SelectContext.TO_DECK: hand_like,
    SelectContext.TO_DECK_BOTTOM: hand_like,
    SelectContext.DISCARD: discard,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD: discard,
    SelectContext.TO_PRIZE: hidden_zone,
    SelectContext.LOOK: hidden_zone,
    SelectContext.NOT_MOVE: not_move_or_look,
    SelectContext.EFFECT_TARGET: not_move_or_look,
}


def handle(obs: Observation) -> list[int]:
    """context に応じて decision.card_move 配下の choose() へ委譲する。"""
    module = _MODULE_BY_CONTEXT.get(obs.select.context)
    if module is None:
        return fallback.safe_choice(obs)
    return module.choose(obs.select, obs.current)
