from cg.api import Observation, SelectContext

from src.decision.attack_turn import choose_attack_action
from src.decision.card_move_turn import choose_card_move_action
from src.decision.count_turn import choose_count_action
from src.decision.damage_target_turn import choose_damage_target_action
from src.decision.effect_choice_turn import choose_effect_choice_action
from src.decision.energy_tool_turn import choose_energy_tool_action
from src.decision.evolution_turn import choose_evolution_action
from src.decision.fallback import choose_random_legal_action
from src.decision.main_turn import choose_main_action
from src.decision.setup_turn import choose_setup_action
from src.decision.special_condition_turn import choose_special_condition_action
from src.decision.switch_turn import choose_switch_action
from src.decision.yes_no_turn import choose_yes_no_action


def choose_action(obs: Observation) -> list[int]:
    """ターン中の選択文脈を見て、担当処理へ振り分ける。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during turn decisions.")

    # まずは現在どの文脈の選択なのかを見る。
    context = obs.select.context

    # メインフェーズ中の行動選択。
    if context == SelectContext.MAIN:
        return choose_main_action(obs)

    # 初期配置でバトル場・ベンチに出すポケモン選択。(TODO:福田)
    if context in {
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
    }:
        return choose_setup_action(obs)

    # 入れ替え・きぜつ後のバトル場復帰。(TODO:福田)
    if context in {
        SelectContext.SWITCH,
        SelectContext.TO_ACTIVE,
    }:
        return choose_switch_action(obs)

    # ベンチ・場・手札・山札・サイドなど、移動先を選ぶ処理。(TODO:長島)
    if context in {
        SelectContext.TO_BENCH,
        SelectContext.TO_FIELD,
        SelectContext.TO_HAND,
        SelectContext.DISCARD,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_PRIZE,
        SelectContext.NOT_MOVE,
        SelectContext.LOOK,
    }:
        return choose_card_move_action(obs)

    # ダメカン・ダメージ・回復など、対象のポケモンを選ぶ処理。(TODO:長島
    if context in {
        SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.DAMAGE,
        SelectContext.REMOVE_DAMAGE_COUNTER,
        SelectContext.HEAL,
        SelectContext.EFFECT_TARGET,
    }:
        return choose_damage_target_action(obs)

    # 進化・退化まわりの対象選択。(TODO:福田)
    if context in {
        SelectContext.EVOLVES_FROM,
        SelectContext.EVOLVES_TO,
        SelectContext.DEVOLVE,
        SelectContext.EVOLVE,
        SelectContext.MORE_DEVOLVE,
    }:
        return choose_evolution_action(obs)

    # エネルギーやどうぐの付け替え・トラッシュ・回収先の選択。(TODO:福田)
    if context in {
        SelectContext.ATTACH_FROM,
        SelectContext.ATTACH_TO,
        SelectContext.DETACH_FROM,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
        SelectContext.SWITCH_ENERGY_CARD,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.SWITCH_ENERGY,
    }:
        return choose_energy_tool_action(obs)

    # 特性や効果の順番、使用不可にするワザなどの選択。(TODO:福田)
    if context in {
        SelectContext.SKILL_ORDER,
        SelectContext.DISABLE_ATTACK,
    }:
        return choose_effect_choice_action(obs)

    # 攻撃選択。
    if context == SelectContext.ATTACK:
        return choose_attack_action(obs)

    # 枚数やダメカン個数など、数値を選ぶ処理。(TODO:長島)
    if context in {
        SelectContext.DRAW_COUNT,
        SelectContext.DAMAGE_COUNTER_COUNT,
        SelectContext.REMOVE_DAMAGE_COUNTER_COUNT,
    }:
        return choose_count_action(obs)

    # Yes / No 系の分岐。(TODO:福田)
    if context in {
        SelectContext.IS_FIRST,
        SelectContext.MULLIGAN,
        SelectContext.ACTIVATE,
        SelectContext.FIRST_EFFECT,
        SelectContext.COIN_HEAD,
    }:
        return choose_yes_no_action(obs)

    # 状態異常の付与先・回復対象の選択。(TODO:長島)
    if context in {
        SelectContext.AFFECT_SPECIAL_CONDITION,
        SelectContext.RECOVER_SPECIAL_CONDITION,
    }:
        return choose_special_condition_action(obs)

    # 未対応の文脈は、ひとまず合法手から選ぶ。
    return choose_random_legal_action(obs)
