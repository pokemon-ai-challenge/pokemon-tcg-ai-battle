"""クラスタ③ メイン行動の意思決定（共有部品）／担当B

エネルギー付与に関する評価のうち、priorities/energy.py と handlers/energy_tool_turn.py の
両方から使う共通ロジック。「今ターン、どのポケモンにエネルギーを付けるべきか」の判断材料を
decision.evaluation.energy_requirements を使って組み立てる。
"""

from cg.api import Observation, Pokemon, State

from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.rule_based.main_turn_parts import pokemon_value, usage_gate
from ptcg_ai.shared import card_cache, profile_registry
from ptcg_ai.shared.profile_types import DeckPlan, EnergyCardContext

# 不足エネルギーがちょうど1つならすぐ使えるようになるため、優先度を高くする。
_ONE_SHORT_BONUS = 2.0
_MULTI_SHORT_BONUS = 1.0
# deck_plan.energy_priority に載っているポケモンへの加点（先頭に近いほど高い）。
_ENERGY_PRIORITY_BONUS = 3.0
# エネルギー周回コンボ（deck_plan.energy_recycle_*）の条件が揃ったときの加点。
# 他のボーナスより十分大きくして、条件が揃った場合は優先してそちらへ付けるようにする。
_ENERGY_RECYCLE_BONUS = 10.0


def best_energy_target(obs: Observation) -> Pokemon | None:
    """自分の場のポケモンの中から、今エネルギーを付けるべき最有力候補を返す。

    ENERGY_REQUIRED_COUNT（攻撃に必要なエネルギー総数の上限）に既に達しているポケモンは、
    周回コンボ対象（energy_recycle_target_id）で無い限り候補から外す。
    """
    player = obs.current.players[obs.current.yourIndex]
    candidates = [pokemon for pokemon in player.active if pokemon is not None] + list(player.bench)
    candidates = [pokemon for pokemon in candidates if not _energy_already_sufficient(pokemon)]
    if not candidates:
        return None

    state = obs.current
    return max(candidates, key=lambda pokemon: energy_target_value(pokemon, state))


def _energy_already_sufficient(pokemon: Pokemon) -> bool:
    """ENERGY_REQUIRED_COUNT に対して、既に必要数以上のエネルギーが付いているか。"""
    required = profile_registry.get_energy_required_count(pokemon.id)
    if required is None:
        return False
    return len(pokemon.energies) >= required


def energy_target_value(pokemon: Pokemon, state: State) -> float:
    """このポケモンに今エネルギーを付ける価値をスコア化する。

    best_energy_target が「自分の場全体から無条件で最良の1体」を選ぶのに対し、こちらは
    単体のスコア計算のみを行う公開関数。Wonder Patch のようにベンチのみが対象になる
    効果など、実際に提示された選択肢が場全体に及ばない場面
    （action_selection/handlers/energy_tool_turn.py._choose_attach_target）でも、
    提示された候補どうしをこの関数で比較して選べるようにするため公開している。
    """
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
    return (
        active_value
        + shortage_bonus
        + _energy_priority_bonus(pokemon.id)
        + _energy_recycle_bonus(pokemon, state)
    )


def _energy_priority_bonus(card_id: int) -> float:
    """deck_plan().energy_priority の並び順に応じた加点（先頭ほど高い）。"""
    priority = profile_registry.get_deck_plan().energy_priority
    if card_id not in priority:
        return 0.0
    return _ENERGY_PRIORITY_BONUS - priority.index(card_id) * 0.01


def _energy_recycle_bonus(pokemon: Pokemon, state: State) -> float:
    """エネルギー周回コンボ（例: ノココッチ×リッチエネルギー）向けの加点。

    deck_plan.energy_recycle_target_id で指定された「周回の受け皿」ポケモンが対象で、
    まだエネルギーが付いておらず、周回させたいエネルギーカード（energy_recycle_card_id）が
    手札にある場合に限り検討する。そのうえで、(a) 自分のバトルポケモンが攻撃に必要な
    エネルギーを既に満たしている、または (b) 主力への代替供給手段
    （energy_recycle_backup_item_id、例: ワンダーパッチ）が今使える状態のどちらかを
    満たすときだけ加点する（主力のエネルギー確保を妨げないようにするため）。
    """
    plan = profile_registry.get_deck_plan()
    if plan.energy_recycle_target_id is None or pokemon.id != plan.energy_recycle_target_id:
        return 0.0
    if pokemon.energies:
        return 0.0

    your_index = state.yourIndex
    player = state.players[your_index]
    hand_ids = [card.id for card in (player.hand or [])]
    if plan.energy_recycle_card_id is None or plan.energy_recycle_card_id not in hand_ids:
        return 0.0

    if _active_energy_ready(state, your_index) or _backup_supply_available(plan, state, your_index):
        return _ENERGY_RECYCLE_BONUS
    return 0.0


def _active_energy_ready(state: State, your_index: int) -> bool:
    """自分のバトルポケモンが、攻撃に必要なエネルギーを既に満たしているか。"""
    player = state.players[your_index]
    active = player.active[0] if player.active else None
    if active is None:
        return False
    required = profile_registry.get_energy_required_count(active.id)
    if required is None:
        return False
    return len(active.energies) >= required


def _backup_supply_available(plan: DeckPlan, state: State, your_index: int) -> bool:
    """主力への代替エネルギー供給手段（energy_recycle_backup_item_id）が今使える状態か。"""
    if plan.energy_recycle_backup_item_id is None:
        return False
    return usage_gate.is_usable(plan.energy_recycle_backup_item_id, state, your_index)


def resolve_energy_card_order(context: EnergyCardContext) -> list[int]:
    """ENERGY_CARD_PRIORITY_RULES を先頭から順に試し、最初に条件を満たしたルールの
    card_id 優先順（先頭ほど優先）を返す。どれにも一致しなければ空リスト。

    priorities/energy.py（MAIN選択でのエネルギー付与）と
    action_selection/handlers/energy_tool_turn.py（ATTACH_TOでのカード選択）の
    両方から使う、「対象が決まった後にどのエネルギーカードを使うか」の共通ロジック。
    """
    for rule in profile_registry.get_energy_card_priority_rules():
        if rule.condition(context):
            return rule.order
    return []
