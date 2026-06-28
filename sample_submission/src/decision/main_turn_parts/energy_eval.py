from dataclasses import dataclass

from cg.api import AreaType, Attack, Card, CardData, EnergyType, Observation, Option, Pokemon

from src.decision.evaluation.attack_features import (
    build_attack_lookup,
    choose_main_attack,
    resolve_attacks,
)
from src.decision.evaluation.board_features import (
    is_likely_knocked_out_next_turn,
    is_main_attacker,
)
from src.decision.evaluation.energy_requirements import (
    best_payable_attack_damage,
    energy_gap_for_attack,
    has_payable_attack,
)
from src.knowledge.card_cache import load_card_data


@dataclass(frozen=True)
class AttachEvaluation:
    option_index: int
    # 貼り先どうしを比較するための内部スコア。
    score: int
    # MAIN でエネ貼り候補として出してよいかどうか。
    should_offer: bool
    # この貼りで今すぐアクティブが攻撃可能になるか。
    enables_attack_now: bool
    # この貼りで主力技まで残り 1 エネになるか。
    reaches_main_attack_next_turn: bool


def choose_best_attach_option(
    obs: Observation,
    attach_option_indices: list[int],
) -> AttachEvaluation:
    if obs.select is None or obs.current is None:
        return AttachEvaluation(
            option_index=attach_option_indices[0],
            score=0,
            should_offer=False,
            enables_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    # 各候補を同じ基準で比較できるよう、カード情報を先に引いておく。
    card_data_by_id = {card.cardId: card for card in load_card_data()}
    attack_by_id = build_attack_lookup()
    evaluations: list[AttachEvaluation] = []

    for option_index in attach_option_indices:
        evaluation = evaluate_attach_option(
            obs,
            option_index,
            card_data_by_id,
            attack_by_id,
        )
        evaluations.append(evaluation)

    return max(
        evaluations,
        key=lambda evaluation: (evaluation.score, -evaluation.option_index),
    )


def evaluate_attach_option(
    obs: Observation,
    option_index: int,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> AttachEvaluation:
    if obs.select is None or obs.current is None:
        return AttachEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            enables_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    option = obs.select.option[option_index]
    target = resolve_attach_target(player.active, player.bench, option)
    if target is None:
        return AttachEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            enables_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    target_card_data = card_data_by_id.get(target.id)
    if target_card_data is None:
        return AttachEvaluation(
            option_index=option_index,
            score=0,
            should_offer=False,
            enables_attack_now=False,
            reaches_main_attack_next_turn=False,
        )

    attacks = resolve_attacks(target_card_data, attack_by_id)
    current_energies = list(target.energies)
    attached_energy = resolve_attached_energy_type(player.hand, option, card_data_by_id)
    next_energies = current_energies.copy()
    if attached_energy is not None:
        next_energies.append(attached_energy)

    score = 0
    main_attack = choose_main_attack(attacks)
    before_main_gap = energy_gap_for_attack(main_attack, current_energies)  # 貼る前に主力技まであと何エネ足りないか
    after_main_gap = energy_gap_for_attack(main_attack, next_energies)  # 貼った後に主力技まであと何エネ足りないか
    best_payable_before = best_payable_attack_damage(attacks, current_energies)  # 貼る前に出せる最大打点
    best_payable_after = best_payable_attack_damage(attacks, next_energies)  # 貼った後に出せる最大打点
    can_attack_before = has_payable_attack(attacks, current_energies)  # 貼る前に何らかの技を使えるか
    can_attack_after = has_payable_attack(attacks, next_energies)  # 貼った後に何らかの技を使えるか

    # ここで作る真偽値は、加点減点だけでなく外側の提案判定にも再利用できる。
    enables_attack_now = (
        option.inPlayArea == AreaType.ACTIVE
        and not can_attack_before
        and can_attack_after
    )
    reaches_main_attack_next_turn = (
        main_attack is not None
        and before_main_gap > 1
        and after_main_gap == 1
    )
    improves_attack_plan = (
        after_main_gap < before_main_gap
        or best_payable_after > best_payable_before
    )
    supports_retreat = retreat_energy_progress(option, target, target_card_data.retreatCost)
    risky_target = is_likely_knocked_out_next_turn(
        obs,
        target,
        card_data_by_id,
        attack_by_id,
    )
    low_priority_target = is_low_priority_target(
        obs,
        option,
        target,
        next_energies,
        card_data_by_id,
        attack_by_id,
    )
    already_satisfied_without_upgrade = (
        before_main_gap == 0 and best_payable_after <= best_payable_before
    )

    # この貼りでアクティブがすぐ攻撃できるなら最優先。
    if enables_attack_now:
        score += 30

    # 主力技まで残り 1 エネになる貼りを強く評価する。
    if reaches_main_attack_next_turn:
        score += 20

    # 盤面の主力候補に貼れるなら加点する。
    if is_main_attacker(obs, target, card_data_by_id, attack_by_id):
        score += 15

    # 必要エネに近づく、または実質的に打点が上がる貼りを評価する。
    if improves_attack_plan:
        score += 10

    # アクティブの逃げエネとして意味がある場合も少し加点する。
    if supports_retreat:
        score += 5

    # 次の相手ターンで倒されそうなポケモンへの貼りは下げる。
    if risky_target:
        score -= 20

    # 主力でなく、貼ってもまだ遠いベンチは優先度を下げる。
    if low_priority_target:
        score -= 10

    # すでに必要エネを満たしているポケモンへの上積みは少し抑える。
    if already_satisfied_without_upgrade:
        score -= 5

    return AttachEvaluation(
        option_index=option_index,
        score=score,
        # 当面は score を暫定利用し、将来は独立した判定ロジックへ育てる。
        should_offer=score > 0,
        enables_attack_now=enables_attack_now,
        reaches_main_attack_next_turn=reaches_main_attack_next_turn,
    )


def is_low_priority_target(
    obs: Observation,
    option: Option,
    target: Pokemon,
    next_energies: list[EnergyType],
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> bool:
    if is_main_attacker(obs, target, card_data_by_id, attack_by_id):
        return False

    card_data = card_data_by_id.get(target.id)
    if card_data is None:
        return True

    main_attack = choose_main_attack(resolve_attacks(card_data, attack_by_id))
    after_gap = energy_gap_for_attack(main_attack, next_energies)
    return option.inPlayArea != AreaType.ACTIVE and after_gap > 1


def retreat_energy_progress(
    option: Option,
    target: Pokemon,
    retreat_cost: int,
) -> bool:
    # 今の実装では、アクティブの逃げコスト補助になる貼りだけを見る。
    if option.inPlayArea != AreaType.ACTIVE or retreat_cost <= 0:
        return False
    return len(target.energies) < retreat_cost


def resolve_attach_target(
    active: list[Pokemon | None],
    bench: list[Pokemon],
    option: Option,
) -> Pokemon | None:
    # ATTACH option が指している盤面上のポケモンを復元する。
    if option.inPlayArea == AreaType.ACTIVE:
        if option.inPlayIndex is None or option.inPlayIndex >= len(active):
            return None
        return active[option.inPlayIndex]
    if option.inPlayArea == AreaType.BENCH:
        if option.inPlayIndex is None or option.inPlayIndex >= len(bench):
            return None
        return bench[option.inPlayIndex]
    return None


def resolve_attached_energy_type(
    hand: list[Card] | None,
    option: Option,
    card_data_by_id: dict[int, CardData],
) -> EnergyType | None:
    # どの手札エネルギーを貼る候補かを option から取り出す。
    if option.area != AreaType.HAND or hand is None or option.index is None:
        return None
    if option.index < 0 or option.index >= len(hand):
        return None

    hand_card = hand[option.index]
    card_data = card_data_by_id.get(hand_card.id)
    if card_data is None:
        return None
    return card_data.energyType
