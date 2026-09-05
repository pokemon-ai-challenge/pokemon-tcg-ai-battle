"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE, REMOVE_DAMAGE_COUNTER, HEAL

ダメージカウンターを乗せる/取り除く、回復する対象のポケモンを選ぶ場面。
「次ターン倒されやすいか」「主力アタッカーらしさ」の評価は
decision.evaluation.board_features を参照する。
"""

from cg.api import Observation, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import board_features
from ptcg_ai.rule_based.card_move import common

_DAMAGE_CONTEXTS = (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY)
_SUPPORT_CONTEXTS = (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER)


def handle(obs: Observation) -> list[int]:
    """ダメージ/回復対象の選択肢から、盤面評価に基づいて対象を選ぶ。"""
    context = obs.select.context
    if context in _DAMAGE_CONTEXTS:
        return _choose_damage_target(obs)
    if context in _SUPPORT_CONTEXTS:
        return _choose_support_target(obs)
    return fallback.safe_choice(obs)


def _choose_damage_target(obs: Observation) -> list[int]:
    """相手の主力アタッカー、または倒しきれそうなポケモンを優先して狙う。"""

    def score(option) -> float:
        pokemon = common.resolve_pokemon(option, obs.current)
        if pokemon is None:
            return float("-inf")
        remaining_hp_ratio = pokemon.hp / pokemon.maxHp if pokemon.maxHp else 1.0
        return board_features.attacker_score(pokemon) + (1.0 - remaining_hp_ratio)

    return common.pick_top(obs.select, score)


def _choose_support_target(obs: Observation) -> list[int]:
    """自分の場でダメージを受けている主力ポケモンを優先して回復/ダメカン除去する。"""

    def score(option) -> float:
        pokemon = common.resolve_pokemon(option, obs.current)
        if pokemon is None:
            return float("-inf")
        damage_taken = max(0, pokemon.maxHp - pokemon.hp)
        if damage_taken == 0:
            return float("-inf")
        return board_features.attacker_score(pokemon) * 10.0 + damage_taken

    return common.pick_top(obs.select, score)
