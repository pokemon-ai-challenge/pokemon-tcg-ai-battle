"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: AFFECT_SPECIAL_CONDITION, RECOVER_SPECIAL_CONDITION

どく・やけど・ねむり・まひ・こんらんなどの状態異常を、誰に付与するか/誰から回復するかを
選ぶ場面（SelectType.SPECIAL_CONDITION）。
"""

from cg.api import Observation, SelectContext, SpecialConditionType

from ptcg_ai.action_selection import fallback

# 相手に付与する際の優先度。にげる/攻撃を封じるほど盤面への影響が大きいものを優先する。
_AFFECT_PRIORITY: dict[SpecialConditionType, int] = {
    SpecialConditionType.PARALYZE: 4,
    SpecialConditionType.SLEEP: 3,
    SpecialConditionType.CONFUSE: 2,
    SpecialConditionType.POISON: 1,
    SpecialConditionType.BURN: 0,
}


def handle(obs: Observation) -> list[int]:
    """AFFECT_SPECIAL_CONDITION / RECOVER_SPECIAL_CONDITION の選択肢を処理する。"""
    if obs.select.context == SelectContext.AFFECT_SPECIAL_CONDITION:
        return _choose_most_disruptive(obs)
    # RECOVER_SPECIAL_CONDITION: 治す状態異常を選べる場合、判断材料が乏しいため
    # 提示された順に必要数だけ選ぶ（提示される状態異常はどれも治す価値がある想定）。
    return fallback.safe_choice(obs)


def _choose_most_disruptive(obs: Observation) -> list[int]:
    best_index = None
    best_priority = -1
    for i, option in enumerate(obs.select.option):
        if option.specialConditionType is None:
            continue
        priority = _AFFECT_PRIORITY.get(option.specialConditionType, 0)
        if priority > best_priority:
            best_priority = priority
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]
