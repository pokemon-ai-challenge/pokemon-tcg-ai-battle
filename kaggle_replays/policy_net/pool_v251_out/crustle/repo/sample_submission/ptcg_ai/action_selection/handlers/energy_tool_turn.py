"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: ATTACH_FROM, ATTACH_TO, DETACH_FROM, DISCARD_ENERGY_CARD,
DISCARD_TOOL_CARD, SWITCH_ENERGY_CARD, DISCARD_ENERGY, TO_HAND_ENERGY,
TO_DECK_ENERGY, SWITCH_ENERGY

エネルギー・ポケモンのどうぐの付け外し・入れ替えに関する選択。
どのポケモンに付けるべきかの評価は decision.evaluation.energy_requirements /
decision.main_turn_parts.energy_eval を参照する。
"""

from cg.api import Observation, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import board_features
from ptcg_ai.rule_based.card_move import common, discard
from ptcg_ai.rule_based.main_turn_parts import energy_eval
from ptcg_ai.shared.profile_types import EnergyCardContext

# discard.choose は DISCARD 系 SelectContext（protected_card_ids を避ける判断）を
# まとめて面倒みているため、ここでもそのまま使い回す。
_DISCARD_LIKE_CONTEXTS = (
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_ENERGY,
)

# ATTACH_TO の時点でまだ対象ポケモンが決まっていない場合に使う、実在しないダミーのcard_id。
# ENERGY_CARD_PRIORITY_RULES の「対象がフーディンかどうか」判定を必ず外し、
# 常に既定（フォールバック）ルールを使わせるための番兵値。
_UNKNOWN_TARGET_CARD_ID = -1


def handle(obs: Observation) -> list[int]:
    """ATTACH/DETACH/DISCARD/SWITCH のエネルギー・どうぐ系選択肢を処理する。"""
    context = obs.select.context
    if context in _DISCARD_LIKE_CONTEXTS:
        return discard.choose(obs.select, obs.current)
    if context == SelectContext.ATTACH_FROM:
        return _choose_attach_target(obs)
    if context == SelectContext.ATTACH_TO:
        return _choose_attach_card(obs)
    # DETACH_FROM / SWITCH_ENERGY_CARD / SWITCH_ENERGY / TO_HAND_ENERGY / TO_DECK_ENERGY:
    # 場から手放す/動かす対象は、既定では最も価値の低いポケモンから選ぶ
    # （主力アタッカーのエネルギーを不用意に失わないようにする）。
    return common.pick_top(obs.select, lambda option: -_owner_attacker_score(option, obs.current))


def _choose_attach_target(obs: Observation) -> list[int]:
    """ATTACH_FROM: エネルギー/どうぐを付けるべきポケモンを選ぶ。

    実際のリプレイ確認により、Wonder Patch のような効果では ATTACH_FROM の選択肢が
    ベンチのポケモンだけに絞られる（バトル場は選択肢に出てこない）ことが分かっている。
    energy_eval.best_energy_target は自分の場全体から無条件で最良の1体を選ぶため、
    そのポケモンが今回の選択肢に含まれるとは限らない。ここでは必ず「実際に提示された
    選択肢」の中だけで energy_eval.energy_target_value を比較する
    （提示範囲の制約を守りつつ、フーディンのように役割の大きいポケモンがベンチにいれば
    自然に選ばれる）。
    """
    state = obs.current
    scored = [
        (i, energy_eval.energy_target_value(pokemon, state))
        for i, option in enumerate(obs.select.option)
        if (pokemon := common.resolve_pokemon(option, state)) is not None
    ]
    if not scored:
        return fallback.safe_choice(obs)
    best_index = max(scored, key=lambda item: item[1])[0]
    return [best_index]


def _choose_attach_card(obs: Observation) -> list[int]:
    """ATTACH_TO: 手札/トラッシュのどのエネルギー/どうぐを付けるか選ぶ。

    実際のリプレイ確認により、ATTACH_TO は ATTACH_FROM より先に発生しうる
    （対象ポケモンがまだ決まっていない状態でカードだけ選ぶ）ため、対象を前提にした
    ENERGY_CARD_PRIORITY_RULES の判定はできない。存在しないカードIDをダミーの対象として
    渡すことで、常に「それ以外」の既定ルール（フォールバック優先順位）を使う。
    """
    context = EnergyCardContext(target_card_id=_UNKNOWN_TARGET_CARD_ID, target_energy_count=0)
    order = energy_eval.resolve_energy_card_order(context)

    def score(option) -> float:
        card_id = common.resolve_card_id(option, obs.current)
        if card_id in order:
            return float(len(order) - order.index(card_id))
        return 0.0

    return common.pick_top(obs.select, score)


def _owner_attacker_score(option, state) -> float:
    """option の area/index が指すポケモンの attacker_score を返す（不明なら0.0）。"""
    pokemon = common.resolve_pokemon(option, state)
    if pokemon is None:
        return 0.0
    return board_features.attacker_score(pokemon)
