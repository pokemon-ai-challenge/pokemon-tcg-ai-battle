from functools import lru_cache
from pathlib import Path

from cg.api import Card, CardData, CardType, Observation, SelectContext

from src.decision.card_move import common as card_move_common
from src.decision.card_move.common import (
    CardMoveOptionView,
    analyze_card_move_option,
    count_basic_energy_cards,
)
from src.decision.fallback import choose_random_legal_action
from src.knowledge.deck_profiles import get_attack_effect_profile, get_pokemon_profile


def choose_bench_or_field_action(obs: Observation) -> list[int]:
    """自分のポケモンを場に出す候補をスコアで選ぶ。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during card move decisions.")
    if obs.current is None:
        raise ValueError("obs.current must not be None during card move decisions.")

    # option を「誰のカードか / 何のカードか」が分かる形に直す。
    views = [
        analyze_card_move_option(obs, option_index)
        for option_index in range(len(obs.select.option))
    ]
    views = [view for view in views if view is not None]
    if not views:
        return choose_random_legal_action(obs)

    # ここではまず、自分のポケモンを出す判断だけに集中する。
    self_pokemon_views = [
        view
        for view in views
        if view.owner_is_self is True and view.is_pokemon is True and view.card_id is not None
    ]
    if not self_pokemon_views:
        return choose_random_legal_action(obs)

    selected: list[int] = []
    selected_card_ids: dict[int, int] = {}

    # maxCount が複数でも、1 枚ずつ「今追加する価値」が高いものを選ぶ。
    while len(selected) < obs.select.maxCount:
        best_view: CardMoveOptionView | None = None
        best_score = float("-inf")

        for view in self_pokemon_views:
            if view.option_index in selected:
                continue

            score = score_self_deploy_option(
                obs,
                view,
                selected_card_ids,
                self_pokemon_views,
            )
            if score > best_score:
                best_score = score
                best_view = view

        if best_view is None:
            break
        if len(selected) >= obs.select.minCount and best_score <= 0:
            break

        selected.append(best_view.option_index)
        if best_view.card_id is not None:
            selected_card_ids[best_view.card_id] = selected_card_ids.get(best_view.card_id, 0) + 1

    if not selected and obs.select.minCount == 0:
        return []

    # 必須枚数に届いていなければ、残り候補から高い順に補充する。
    if len(selected) < obs.select.minCount:
        remaining = [
            view
            for view in self_pokemon_views
            if view.option_index not in selected
        ]
        remaining.sort(
            key=lambda view: score_self_deploy_option(
                obs,
                view,
                selected_card_ids,
                self_pokemon_views,
            ),
            reverse=True,
        )
        for view in remaining:
            if len(selected) >= obs.select.minCount:
                break
            selected.append(view.option_index)
            if view.card_id is not None:
                selected_card_ids[view.card_id] = selected_card_ids.get(view.card_id, 0) + 1

    # それでも必須枚数に届かないなら、安全のため fallback に戻す。
    if len(selected) < obs.select.minCount:
        return choose_random_legal_action(obs)

    return selected


def score_self_deploy_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    available_views: list[CardMoveOptionView],
) -> float:
    """自分の候補を、役割・枠・相方・将来価値で採点する。"""
    if obs.current is None or view.card_id is None:
        return -1000

    player = obs.current.players[obs.current.yourIndex]
    card_data = card_move_common._card_data_lookup().get(view.card_id)
    if card_data is None or card_data.cardType != CardType.POKEMON:
        return -1000

    profile = get_pokemon_profile(view.card_id)
    active_role_bonus = profile.active_role_bonus if profile is not None else 0
    bench_setup_bonus = profile.bench_setup_bonus if profile is not None else 0

    # 今の盤面に加え、今回すでに選んだ分もベンチ枠を消費しているとみなす。
    selected_count = sum(selected_card_ids.values())
    bench_open_slots = max(player.benchMax - len(player.bench) - selected_count, 0)

    # 同名が今何枚いて、この候補を足すと何枚になるかを見る。
    in_play_count = count_in_play_card_id(player, view.card_id)
    future_count = in_play_count + selected_card_ids.get(view.card_id, 0) + 1
    in_play_card_ids = current_in_play_card_ids(player)
    available_candidate_ids = {
        candidate.card_id
        for candidate in available_views
        if candidate.card_id is not None
    }

    score = -1.0
    # 1. カード単体の役割を土台点にする。
    score += bench_setup_bonus * 2.0
    score += active_role_bonus * 0.45

    # 2. 粗い役割分類を作って、後続補正に使う。
    is_main_attacker = active_role_bonus >= 18
    is_support = bench_setup_bonus >= 5 and active_role_bonus < 10
    is_backup_attacker = 8 <= active_role_bonus < 18
    is_evolution_base = candidate_is_evolution_base(card_data)
    is_wall_candidate = (
        not is_main_attacker
        and not is_support
        and not is_evolution_base
        and card_data.basic
        and card_data.hp > 0
    )

    # 3. ベンチ残数で補正する。
    score += bench_space_score(
        bench_open_slots=bench_open_slots,
        is_main_attacker=is_main_attacker,
        is_evolution_base=is_evolution_base,
        is_support=is_support,
        is_backup_attacker=is_backup_attacker,
    )

    # 4. 進化元は、後続育成につながるほど加点する。
    if is_evolution_base:
        score += 7
        if hand_contains_evolution(card_data, player.hand):
            score += 5
        evolved_count = count_evolved_forms_in_play(card_data, player)
        if evolved_count == 0:
            score += 3
        else:
            score -= 2 * evolved_count

    # 5. 主力不足なら主力候補を押し上げる。
    ready_attackers = count_ready_attackers(player)
    if is_main_attacker:
        if ready_attackers == 0:
            score += 9
        elif ready_attackers >= 2:
            score -= 4
    elif is_backup_attacker and ready_attackers == 0:
        score += 4

    # 6. 個別シナジーを加味する。
    if profile is not None:
        score += partner_synergy_score(
            obs,
            card_data,
            profile=profile,
            future_count=future_count,
            in_play_card_ids=in_play_card_ids,
            available_candidate_ids=available_candidate_ids,
        )
        score += duplicate_penalty_score(
            obs,
            card_data,
            profile=profile,
            future_count=future_count,
            player=player,
        )
        score += discard_draw_support_score(
            obs,
            profile=profile,
        )

    # 7. ベンチ加速が見えているなら、受け皿価値を少し上げる。
    if benched_energy_acceleration_is_online(player):
        if card_data.basic:
            score += 3
        if is_main_attacker or is_evolution_base:
            score += 2

    # 8. アクティブが危険なら、壁候補も少し押し上げる。
    if is_wall_candidate and active_needs_backup_wall(player):
        score += 6

    if profile is None and not is_evolution_base and not is_main_attacker:
        score -= 4

    if obs.select.context == SelectContext.TO_FIELD:
        # TO_FIELD はベンチ限定とは言い切れないので、少しだけ緩める。
        score += 1

    return score


def bench_space_score(
    *,
    bench_open_slots: int,
    is_main_attacker: bool,
    is_evolution_base: bool,
    is_support: bool,
    is_backup_attacker: bool,
) -> float:
    """ベンチ枠が少ないほど、主力・進化元以外を厳しく見る。"""
    if bench_open_slots <= 0:
        return -1000
    if bench_open_slots == 1:
        if is_main_attacker or is_evolution_base:
            return 8
        if is_backup_attacker:
            return 2
        if is_support:
            return -7
        return -12
    if bench_open_slots == 2:
        if is_main_attacker or is_evolution_base:
            return 5
        if is_support:
            return -1
        if is_backup_attacker:
            return 2
        return -4
    if is_support or is_backup_attacker:
        return 2
    return 1


def partner_synergy_score(
    obs: Observation,
    card_data: CardData,
    *,
    profile,
    future_count: int,
    in_play_card_ids: set[int],
    available_candidate_ids: set[int],
) -> float:
    """相方成立を強く見るが、理想枚数を超えたら抑える。"""
    if not profile.partner_card_ids:
        return 0

    score = 0.0
    partner_present = any(partner_id in in_play_card_ids for partner_id in profile.partner_card_ids)
    partner_selectable = any(
        partner_id in available_candidate_ids
        for partner_id in profile.partner_card_ids
    )

    if partner_present:
        score += profile.partner_bonus
    elif partner_selectable:
        score += max(profile.partner_bonus - 3, 0)
    elif profile.partner_required:
        score -= 10
    else:
        score -= 2

    # 理想枚数ぴったりでは減点せず、超えた分だけ減点する。
    if future_count > profile.preferred_bench_count:
        excess = future_count - profile.preferred_bench_count
        score -= 5 * excess

    if future_count > 1 and partner_is_under_pressure(obs, profile.partner_card_ids):
        score += profile.backup_copy_bonus

    if card_data.cardId == 675 and partner_present:
        score += 4

    return score


def duplicate_penalty_score(
    obs: Observation,
    card_data: CardData,
    *,
    profile,
    future_count: int,
    player,
) -> float:
    """価値の薄い重複展開を抑える。"""
    if future_count <= profile.max_useful_copies:
        return 0

    excess = future_count - profile.max_useful_copies
    penalty = -9.0 * excess
    if partner_is_under_pressure(obs, profile.partner_card_ids):
        penalty += profile.backup_copy_bonus
    if active_needs_backup_wall(player) and card_data.basic:
        penalty += 2
    return penalty


def discard_draw_support_score(
    obs: Observation,
    *,
    profile,
) -> float:
    """手札エネをドローと後続資源に変えられる価値。"""
    if obs.current is None or profile.hand_discard_energy_type_for_draw is None:
        return 0

    player = obs.current.players[obs.current.yourIndex]
    hand_count = count_basic_energy_cards(
        player.hand,
        profile.hand_discard_energy_type_for_draw,
    )
    if hand_count <= 0:
        return -2

    score = float(profile.draw_support_bonus)
    discard_count = count_basic_energy_cards(
        player.discard,
        profile.hand_discard_energy_type_for_draw,
    )
    if benched_energy_acceleration_is_online(player):
        score += 3 + min(discard_count, 3) * 2
    elif discard_count > 0:
        score += 1

    if hand_count >= 2:
        score += 2

    if profile.partner_card_ids and partner_is_under_pressure(obs, profile.partner_card_ids):
        score += 2

    return score


def count_in_play_card_id(player, card_id: int) -> int:
    """active + bench に同じ card_id が何枚あるか。"""
    count = 0
    for pokemon in player.active:
        if pokemon is not None and pokemon.id == card_id:
            count += 1
    for pokemon in player.bench:
        if pokemon.id == card_id:
            count += 1
    return count


def current_in_play_card_ids(player) -> set[int]:
    """active + bench に見えている card_id の集合。"""
    card_ids = {pokemon.id for pokemon in player.bench}
    card_ids.update(
        pokemon.id
        for pokemon in player.active
        if pokemon is not None
    )
    return card_ids


def candidate_is_evolution_base(card_data: CardData) -> bool:
    """deck.csv に進化先が入っているたねポケモンかどうか。"""
    return card_data.basic and card_data.name in _deck_evolution_targets()


def hand_contains_evolution(card_data: CardData, hand: list[Card] | None) -> bool:
    """手札に進化先が見えているか。"""
    if hand is None:
        return False
    for card in hand:
        hand_card_data = card_move_common._card_data_lookup().get(card.id)
        if hand_card_data is None:
            continue
        if hand_card_data.evolvesFrom == card_data.name:
            return True
    return False


def count_evolved_forms_in_play(card_data: CardData, player) -> int:
    """その進化元から育つ進化先が、場に何体いるか。"""
    count = 0
    for pokemon in player.bench:
        benched_card_data = card_move_common._card_data_lookup().get(pokemon.id)
        if benched_card_data is not None and benched_card_data.evolvesFrom == card_data.name:
            count += 1
    for pokemon in player.active:
        if pokemon is None:
            continue
        active_card_data = card_move_common._card_data_lookup().get(pokemon.id)
        if active_card_data is not None and active_card_data.evolvesFrom == card_data.name:
            count += 1
    return count


def count_ready_attackers(player) -> int:
    """ざっくり「すでに殴れそうな主力」の数を数える。"""
    count = 0
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            continue
        card_data = card_move_common._card_data_lookup().get(pokemon.id)
        if card_data is None or not card_data.attacks:
            continue
        profile = get_pokemon_profile(pokemon.id)
        if len(pokemon.energies) >= 2 or (profile is not None and profile.active_role_bonus >= 18):
            count += 1
    return count


def benched_energy_acceleration_is_online(player) -> bool:
    """ベンチへエネ加速できる主力が場にいるか。"""
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            continue
        card_data = card_move_common._card_data_lookup().get(pokemon.id)
        if card_data is None:
            continue
        for attack_id in card_data.attacks:
            profile = get_attack_effect_profile(attack_id)
            if profile is not None and profile.accelerates_energy_to_bench > 0:
                return True
    return False


def active_needs_backup_wall(player) -> bool:
    """アクティブが危険域なら、逃がし先・壁候補の価値を上げる。"""
    active = player.active[0] if player.active else None
    if active is None or active.maxHp <= 0:
        return False

    hp_ratio = active.hp / active.maxHp
    if hp_ratio <= 0.45:
        return True
    if hp_ratio <= 0.6 and len(active.energies) >= 2:
        return True
    return False


def partner_is_under_pressure(obs: Observation, partner_card_ids: frozenset[int]) -> bool:
    """相方が落ちそうなら、予備を置く価値を少し戻す。"""
    if obs.current is None or not partner_card_ids:
        return False

    player = obs.current.players[obs.current.yourIndex]
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None or pokemon.id not in partner_card_ids or pokemon.maxHp <= 0:
            continue
        if pokemon.hp / pokemon.maxHp <= 0.55:
            return True
    return False


@lru_cache(maxsize=1)
def _deck_evolution_targets() -> frozenset[str]:
    card_data_by_id = card_move_common._card_data_lookup()
    targets: set[str] = set()
    for card_id in _load_deck_card_ids():
        card_data = card_data_by_id.get(card_id)
        if card_data is not None and card_data.evolvesFrom:
            targets.add(card_data.evolvesFrom)
    return frozenset(targets)


@lru_cache(maxsize=1)
def _load_deck_card_ids() -> frozenset[int]:
    deck_path = Path(__file__).resolve().parents[2] / "deck.csv"
    try:
        card_ids = {
            int(line.strip())
            for line in deck_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError:
        return frozenset()
    return frozenset(card_ids)
