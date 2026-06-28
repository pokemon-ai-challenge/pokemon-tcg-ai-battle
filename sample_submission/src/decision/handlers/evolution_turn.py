from collections import Counter

from cg.api import Attack, CardData, EnergyType, Observation, Pokemon, SelectContext, all_attack, all_card_data

from src.decision.fallback import choose_random_legal_action
from src.decision.evaluation.switch_eval import resolve_option_target


_CARD_DATA = {card.cardId: card for card in all_card_data()}
_CARD_DATA_BY_NAME = {card.name: card for card in all_card_data()}
_ATTACK_BY_ID = {a.attackId: a for a in all_attack()}


def _resolve_attacks(card_data: CardData) -> list[Attack]:
    """CardData が持つ攻撃IDを Attack オブジェクトのリストに変換する。"""
    return [a for aid in card_data.attacks if (a := _ATTACK_BY_ID.get(aid)) is not None]


def _can_pay(required: list[EnergyType], available: list[EnergyType]) -> bool:
    """required を available で賄えるか判定する（RAINBOW / COLORLESS 対応）。"""
    avail = Counter(available)
    rainbow = avail.pop(EnergyType.RAINBOW, 0)

    for e in required:
        if e == EnergyType.COLORLESS:
            continue  # 後でまとめて処理
        if avail.get(e, 0) > 0:
            avail[e] -= 1
        elif rainbow > 0:
            rainbow -= 1
        else:
            return False

    colorless_needed = sum(1 for e in required if e == EnergyType.COLORLESS)
    return sum(avail.values()) + rainbow >= colorless_needed


def _has_payable_attack(card_data: CardData, energies: list[EnergyType]) -> bool:
    """現在のエネルギーで使えるワザが 1 つ以上あれば True を返す。"""
    return any(_can_pay(attack.energies, energies) for attack in _resolve_attacks(card_data))


def choose_evolution_action(obs: Observation) -> list[int]:
    """進化・退化の対象を選ぶ。"""

    if obs.select is None or obs.current is None:
        return choose_random_legal_action(obs)

    context = obs.select.context
    your_index = obs.current.yourIndex

    # 退化系は未実装
    if context in {
        SelectContext.DEVOLVE,
        SelectContext.MORE_DEVOLVE,
    }:
        return choose_random_legal_action(obs)

    lucario_data = _CARD_DATA_BY_NAME.get("Lucario")

    best_option = None
    best_score = float("-inf")

    for option_index, option in enumerate(obs.select.option):

        score = 0

        # ==========================================================
        # TODO:
        # - 特性が強い進化先を優先
        # - 複数の進化ラインに対応
        # - 分岐進化(EVOLVES_TO)に対応
        # ==========================================================

        # ----------------------------------------------------------
        # 進化先を選ぶ
        # ----------------------------------------------------------
        if context == SelectContext.EVOLVES_TO:

            if option.index is None:
                continue

            hand = obs.current.players[your_index].hand
            if hand is None:
                continue

            hand_card = hand[option.index]
            card_data = _CARD_DATA.get(hand_card.id)
            if card_data is None:
                continue

            if card_data.name == "Lucario":
                score += 100

        # ----------------------------------------------------------
        # 進化元を選ぶ
        # ----------------------------------------------------------
        else:

            target = resolve_option_target(obs, option_index)
            if target is None:
                continue

            card_data = _CARD_DATA.get(target.id)
            if card_data is None:
                continue

            # Riolu を最優先
            if card_data.name == "Riolu":
                score += 100

            # ----------------------------------------
            # エネルギーが多いほど優先
            # ----------------------------------------
            score += len(target.energies) * 10

            # ----------------------------------------
            # バトル場なら少し優先
            # ----------------------------------------
            player = obs.current.players[your_index]
            if target is player.active:
                score += 10

            # ----------------------------------------
            # 残りHPが少ないなら減点
            # ----------------------------------------
            if card_data.hp > 0:
                hp_ratio = target.hp / card_data.hp

                if hp_ratio <= 0.3:
                    score -= 30
                elif hp_ratio <= 0.5:
                    score -= 15

            # ----------------------------------------
            # 進化後すぐ攻撃できるなら加点
            # ----------------------------------------
            if (
                card_data.name == "Riolu"
                and lucario_data is not None
                and _has_payable_attack(lucario_data, list(target.energies))
            ):
                score += 40

        if score > best_score:
            best_score = score
            best_option = option_index

    if best_option is None:
        return choose_random_legal_action(obs)

    return [best_option]