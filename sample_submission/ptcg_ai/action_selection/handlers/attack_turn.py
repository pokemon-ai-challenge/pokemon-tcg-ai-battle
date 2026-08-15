"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: ATTACK, DISABLE_ATTACK

使用するワザを選ぶ場面。ダメージ計算・エネルギー充足判定は
decision.evaluation.attack_features / energy_requirements に委譲する。
リーサル（相手を倒しきれるか）判定を最優先することが多い。
"""

from cg.api import Observation, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import attack_features
from ptcg_ai.shared import card_cache

# きぜつを取れるワザは、ダメージ量に関わらず最優先にする（main_turn_parts/priorities/attack.py と同じ方針）。
_KO_BONUS = 1000.0


def handle(obs: Observation) -> list[int]:
    """ATTACK / DISABLE_ATTACK の選択肢からワザを1つ選んで返す。"""
    if obs.select.context == SelectContext.DISABLE_ATTACK:
        return _choose_most_dangerous_attack(obs)
    return _choose_best_attack(obs)


def _choose_best_attack(obs: Observation) -> list[int]:
    """自分が使うワザを、きぜつが取れるかを最優先に、それ以外はダメージ量で選ぶ。"""
    state = obs.current
    your_index = state.yourIndex
    player = state.players[your_index]
    opponent = state.players[1 - your_index]
    if not player.active or player.active[0] is None or not opponent.active or opponent.active[0] is None:
        return fallback.safe_choice(obs)

    attacker = player.active[0]
    defender = opponent.active[0]
    defender_card = card_cache.get_card(defender.id)
    # 盤面依存の可変ダメージ技（カミツオロチexデッキの3ワザ、2026-08-12）の推定に使う、
    # 攻撃側自身の場（先頭=バトル場、以降=ベンチ）。
    attacker_side_pokemon = [attacker] + list(player.bench or [])

    best_index = None
    best_score = float("-inf")
    for i, option in enumerate(obs.select.option):
        if option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        score = float(attack.damage)
        if attack_features.can_ko(
            attack, attacker, defender, defender_card.weakness, defender_card.resistance,
            defender_is_benched=False,
            damage_is_effect=attack_features.damage_is_effect_based(attack),
            attacker_side_pokemon=attacker_side_pokemon,
        ):
            score += _KO_BONUS
        if score > best_score:
            best_score = score
            best_index = i

    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]


def _choose_most_dangerous_attack(obs: Observation) -> list[int]:
    """相手のワザを1つ無効化する効果向け: 最もダメージの大きいワザを対象にする。"""
    best_index = None
    best_damage = -1
    for i, option in enumerate(obs.select.option):
        if option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        if attack.damage > best_damage:
            best_damage = attack.damage
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]
