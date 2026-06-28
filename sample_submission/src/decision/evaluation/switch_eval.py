from dataclasses import dataclass

from cg.api import Attack, CardData, Observation, Pokemon

from src.decision.evaluation.attack_features import (
    build_attack_lookup,
    choose_main_attack,
    resolve_attacks,
)
from src.decision.evaluation.board_features import (
    attacker_priority_score,
    is_likely_knocked_out_next_turn,
    resolve_in_play_pokemon,
)
from src.decision.evaluation.energy_requirements import (
    best_payable_attack_damage,
    energy_gap_for_attack,
    has_payable_attack,
)
from src.knowledge.card_cache import load_card_data


@dataclass(frozen=True)
class SwitchTargetEvaluation:
    """前衛として見たときのポケモン単体の評価。"""
    score: int
    can_attack_now: bool
    reaches_main_attack_next_turn: bool
    main_attack_gap: int
    risky: bool


@dataclass(frozen=True)
class SwitchOptionEvaluation:
    """実際の選択肢として返すための評価。"""
    option_index: int
    score: int
    should_offer: bool
    improves_position: bool
    can_attack_now: bool
    reaches_main_attack_next_turn: bool


def choose_best_switch_option(
    obs: Observation,
    option_indices: list[int],
) -> SwitchOptionEvaluation:
    """SWITCH 系の候補から、前に出したいポケモンを 1 つ選ぶ。"""
    if obs.select is None or obs.current is None:
        return SwitchOptionEvaluation(
            option_index=option_indices[0],
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    card_data_by_id = {card.cardId: card for card in load_card_data()}
    attack_by_id = build_attack_lookup()
    # 候補ごとに前衛性能を見て、最も高いものを返す。
    evaluations = [
        evaluate_switch_option(obs, option_index, card_data_by_id, attack_by_id)
        for option_index in option_indices
    ]
    return max(evaluations, key=lambda evaluation: (evaluation.score, -evaluation.option_index))


def choose_best_retreat_option(
    obs: Observation,
    option_indices: list[int],
) -> SwitchOptionEvaluation:
    """retreat 候補の中から、にげる価値が最も高い 1 手を選ぶ。"""
    if obs.select is None or obs.current is None:
        return SwitchOptionEvaluation(
            option_index=option_indices[0],
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    player = obs.current.players[obs.current.yourIndex]
    active = player.active[0] if player.active else None
    if active is None or not player.bench:
        return SwitchOptionEvaluation(
            option_index=option_indices[0],
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    card_data_by_id = {card.cardId: card for card in load_card_data()}
    attack_by_id = build_attack_lookup()
    active_evaluation = evaluate_active_role_candidate(
        obs,
        active,
        card_data_by_id,
        attack_by_id,
    )
    # 現アクティブとの比較で「にげて改善するか」を決める。
    active_card_data = card_data_by_id.get(active.id)
    retreat_cost = active_card_data.retreatCost if active_card_data is not None else 0

    evaluations = [
        evaluate_retreat_option(
            obs,
            option_index,
            active_evaluation,
            retreat_cost,
            card_data_by_id,
            attack_by_id,
        )
        for option_index in option_indices
    ]
    return max(evaluations, key=lambda evaluation: (evaluation.score, -evaluation.option_index))


def evaluate_switch_option(
    obs: Observation,
    option_index: int,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> SwitchOptionEvaluation:
    """選んだ候補をそのまま前衛に置いたときの価値を測る。"""
    if obs.select is None or obs.current is None:
        return SwitchOptionEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    target = resolve_option_target(obs, option_index)
    if target is None:
        return SwitchOptionEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    target_evaluation = evaluate_active_role_candidate(
        obs,
        target,
        card_data_by_id,
        attack_by_id,
    )
    return SwitchOptionEvaluation(
        option_index=option_index,
        score=target_evaluation.score,
        should_offer=True,
        improves_position=True,
        can_attack_now=target_evaluation.can_attack_now,
        reaches_main_attack_next_turn=target_evaluation.reaches_main_attack_next_turn,
    )


def evaluate_retreat_option(
    obs: Observation,
    option_index: int,
    active_evaluation: SwitchTargetEvaluation,
    retreat_cost: int,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> SwitchOptionEvaluation:
    """今の前衛と比較して、にげる価値があるかを評価する。"""
    # 現在のエンジンでは MAIN 中の RETREAT option が逃げ先情報を持たないため、
    # retreat 提案では「にげた後に前へ出したいベンチの最良候補」を予測対象にする。
    target = predict_retreat_target(obs, card_data_by_id, attack_by_id)
    if target is None:
        return SwitchOptionEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            improves_position=False,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    target_evaluation = evaluate_active_role_candidate(
        obs,
        target,
        card_data_by_id,
        attack_by_id,
    )
    # 基本は「逃げ先の前衛性能 - 現アクティブの前衛性能」で比較する。
    score_gap = target_evaluation.score - active_evaluation.score
    score = score_gap - retreat_cost * 10

    if target_evaluation.can_attack_now:
        score += 15
    if target_evaluation.reaches_main_attack_next_turn and active_evaluation.main_attack_gap > 1:
        score += 8
    if active_evaluation.risky and not target_evaluation.risky:
        score += 10
    if active_evaluation.reaches_main_attack_next_turn:
        score -= 12

    improves_position = (
        score_gap > 0
        and (
            target_evaluation.can_attack_now
            or target_evaluation.reaches_main_attack_next_turn
            or (active_evaluation.risky and not target_evaluation.risky)
        )
    )
    # 重い retreat を払っても改善が薄いなら見送る。
    cost_too_heavy = (
        retreat_cost >= 3
        and not target_evaluation.can_attack_now
        and not target_evaluation.reaches_main_attack_next_turn
    )

    return SwitchOptionEvaluation(
        option_index=option_index,
        score=score,
        should_offer=improves_position and not cost_too_heavy and score > 0,
        improves_position=improves_position,
        can_attack_now=target_evaluation.can_attack_now,
        reaches_main_attack_next_turn=target_evaluation.reaches_main_attack_next_turn,
    )


def predict_retreat_target(
    obs: Observation,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> Pokemon | None:
    """MAIN の retreat 提案で使う予測上の逃げ先を返す。"""
    return choose_best_bench_target(obs, card_data_by_id, attack_by_id)


def choose_best_bench_target(
    obs: Observation,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> Pokemon | None:
    """ベンチ全体から、前に出したい候補を単体評価で選ぶ。"""
    if obs.current is None:
        return None

    player = obs.current.players[obs.current.yourIndex]
    if not player.bench:
        return None

    return max(
        player.bench,
        key=lambda pokemon: evaluate_active_role_candidate(
            obs,
            pokemon,
            card_data_by_id,
            attack_by_id,
        ).score,
    )


def resolve_option_target(
    obs: Observation,
    option_index: int,
) -> Pokemon | None:
    """option が指しているポケモンを、area/index から解決する。"""
    if obs.select is None or obs.current is None:
        return None

    player = obs.current.players[obs.current.yourIndex]
    option = obs.select.option[option_index]
    target = resolve_in_play_pokemon(
        player.active,
        player.bench,
        option.area,
        option.index,
    )
    if target is not None:
        return target
    # option によっては inPlayArea / inPlayIndex 側に座標が入る。
    return resolve_in_play_pokemon(
        player.active,
        player.bench,
        option.inPlayArea,
        option.inPlayIndex,
    )


def evaluate_active_role_candidate(
    obs: Observation,
    pokemon: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> SwitchTargetEvaluation:
    """そのポケモンを前衛に置いたときの強さを粗く数値化する。"""
    card_data = card_data_by_id.get(pokemon.id)
    if card_data is None:
        return SwitchTargetEvaluation(
            score=0,
            can_attack_now=False,
            reaches_main_attack_next_turn=False,
            main_attack_gap=99,
            risky=True,
        )

    attacks = resolve_attacks(card_data, attack_by_id)
    energies = list(pokemon.energies)
    main_attack = choose_main_attack(attacks)
    main_attack_gap = energy_gap_for_attack(main_attack, energies)
    can_attack_now = has_payable_attack(attacks, energies)
    reaches_main_attack_next_turn = not can_attack_now and main_attack_gap == 1
    best_damage_now = best_payable_attack_damage(attacks, energies)
    risky = is_likely_knocked_out_next_turn(obs, pokemon, card_data_by_id, attack_by_id)

    score = 0
    # いますぐ攻撃できるか、主力技までどれだけ近いかを最優先で見る。
    if can_attack_now:
        score += 50 + best_damage_now // 10
    elif reaches_main_attack_next_turn:
        score += 25
    elif main_attack_gap == 2:
        score += 10
    elif main_attack_gap >= 3:
        score -= 8

    score += attacker_priority_score(pokemon, card_data_by_id, attack_by_id) // 8
    score += max(0, pokemon.hp) // 20

    # 次ターンに倒されやすい前衛は評価を下げる。
    if risky:
        score -= 18

    return SwitchTargetEvaluation(
        score=score,
        can_attack_now=can_attack_now,
        reaches_main_attack_next_turn=reaches_main_attack_next_turn,
        main_attack_gap=main_attack_gap,
        risky=risky,
    )
