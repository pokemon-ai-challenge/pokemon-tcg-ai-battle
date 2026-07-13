"""クラスタ③ メイン行動の意思決定（共有部品）／担当B

エネルギー付与に関する評価のうち、priorities/energy.py と handlers/energy_tool_turn.py の
両方から使う共通ロジック。「今ターン、どのポケモンにエネルギーを付けるべきか」の判断材料を
decision.evaluation.energy_requirements を使って組み立てる。
"""

from cg.api import Observation, Pokemon, State

from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.rule_based.main_turn_parts import pokemon_value
from ptcg_ai.shared import card_cache, profile_registry

# 不足エネルギーがちょうど1つならすぐ使えるようになるため、優先度を高くする。
_ONE_SHORT_BONUS = 2.0
_MULTI_SHORT_BONUS = 1.0
# deck_plan.energy_priority に載っているポケモンへの加点（先頭に近いほど高い）。
_ENERGY_PRIORITY_BONUS = 3.0


def best_energy_target(obs: Observation) -> Pokemon | None:
    """自分の場のポケモンの中から、今エネルギーを付けるべき最有力候補を返す。"""
    player = obs.current.players[obs.current.yourIndex]
    candidates = [pokemon for pokemon in player.active if pokemon is not None] + list(player.bench)
    if not candidates:
        return None

    state = obs.current
    return max(candidates, key=lambda pokemon: _energy_need_score(pokemon, state))


def _energy_need_score(pokemon: Pokemon, state: State) -> float:
    card = card_cache.get_card(pokemon.id)
    shortage_bonus = 0.0
    for attack_id in card.attacks:
        attack = card_cache.get_attack(attack_id)
        shortfall = energy_requirements.energy_shortfall(attack, pokemon.energies)
        if not shortfall:
            continue
        needed = sum(shortfall.values())
        bonus = _ONE_SHORT_BONUS if needed == 1 else _MULTI_SHORT_BONUS
        shortage_bonus = max(shortage_bonus, bonus)

    active_value = pokemon_value.active_value(pokemon, state, state.yourIndex)
    return active_value + shortage_bonus + _energy_priority_bonus(pokemon.id)


def _energy_priority_bonus(card_id: int) -> float:
    """deck_plan().energy_priority の並び順に応じた加点（先頭ほど高い）。"""
    priority = profile_registry.get_deck_plan().energy_priority
    if card_id not in priority:
        return 0.0
    return _ENERGY_PRIORITY_BONUS - priority.index(card_id) * 0.01
