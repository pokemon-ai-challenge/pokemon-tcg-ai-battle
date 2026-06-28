from cg.api import (
    AreaType,
    OptionType,
    SelectContext,
    Observation,
    Option,
    Pokemon,
)

from src.decision.fallback import choose_random_legal_action


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _my_index(obs: Observation) -> int:
    return obs.current.yourIndex


def _my_state(obs: Observation):
    return obs.current.players[_my_index(obs)]


def _active(obs: Observation) -> Pokemon | None:
    """自分のアクティブポケモンを返す（いなければ None）。"""
    active_list = _my_state(obs).active
    return active_list[0] if active_list else None


def _bench(obs: Observation) -> list[Pokemon]:
    return _my_state(obs).bench


def _all_my_pokemon(obs: Observation) -> list[tuple[AreaType, int, Pokemon]]:
    """(area, index, pokemon) のリストを返す。アクティブ→ベンチ順。"""
    result: list[tuple[AreaType, int, Pokemon]] = []
    active = _active(obs)
    if active is not None:
        result.append((AreaType.ACTIVE, 0, active))
    for i, p in enumerate(_bench(obs)):
        result.append((AreaType.BENCH, i, p))
    return result


def _hp_ratio(pokemon: Pokemon) -> float:
    """HP残量の割合（0.0〜1.0）。"""
    if pokemon.maxHp == 0:
        return 0.0
    return pokemon.hp / pokemon.maxHp


def _energy_count(pokemon: Pokemon) -> int:
    return len(pokemon.energies)


def _find_option_index(options: list[Option], predicate) -> int | None:
    """条件に合う最初のオプションのインデックスを返す。"""
    for i, opt in enumerate(options):
        if predicate(opt):
            return i
    return None


def _find_all_option_indices(options: list[Option], predicate) -> list[int]:
    return [i for i, opt in enumerate(options) if predicate(opt)]


# ---------------------------------------------------------------------------
# 各 context のハンドラ
# ---------------------------------------------------------------------------

def _handle_attach_from(obs: Observation) -> list[int] | None:
    """
    ATTACH_FROM: エネルギー/どうぐを「どのポケモンに」付けるか選ぶ。
    優先順位:
      1. アタッカー候補（アクティブ）でエネルギーが少ないポケモン
      2. HPが多く生存可能性が高いポケモン
    """
    options = obs.select.option
    all_pokemon = _all_my_pokemon(obs)

    # CARD オプションを対象とする（ポケモンを選ぶ）
    card_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.CARD)
    if not card_opts:
        return None

    def score(idx: int) -> float:
        opt = options[idx]
        # 対応するポケモンを area/index で特定
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                # アクティブを最優先、次にエネルギー不足、HPが高い順
                active_bonus = 10.0 if area == AreaType.ACTIVE else 0.0
                energy_need = max(0, 3 - _energy_count(poke))  # 3枚を目安
                hp_score = _hp_ratio(poke) * 5.0
                return active_bonus + energy_need * 2.0 + hp_score
        return 0.0

    best = max(card_opts, key=score)
    return [best]


def _handle_attach_to(obs: Observation) -> list[int] | None:
    """
    ATTACH_TO: 手札から「どのカードを」付けるか選ぶ。
    基本的に minCount==maxCount の場合は必ず選ぶ必要があるため、
    できるだけ全体で必要なエネルギーカードを優先する。
    ここではシンプルに最初の選択肢を返す（多くは1択）。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))
    if count == 0:
        return None
    # ENERGY_CARD > TOOL_CARD > CARD の優先度で選ぶ
    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY_CARD)
    if energy_opts:
        return energy_opts[:count]
    tool_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.TOOL_CARD)
    if tool_opts:
        return tool_opts[:count]
    card_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.CARD)
    return card_opts[:count] if card_opts else None


def _handle_detach_from(obs: Observation) -> list[int] | None:
    """
    DETACH_FROM: 「どのポケモンから」外すか選ぶ。
    優先順位: HPが低い（瀕死に近い）ポケモン、ベンチ優先。
    """
    options = obs.select.option
    all_pokemon = _all_my_pokemon(obs)
    card_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.CARD)
    if not card_opts:
        return None

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                # ベンチかつHPが低いものを外す（アクティブは極力温存）
                bench_bonus = 5.0 if area == AreaType.BENCH else 0.0
                low_hp_score = (1.0 - _hp_ratio(poke)) * 3.0
                return bench_bonus + low_hp_score
        return 0.0

    best = max(card_opts, key=score)
    return [best]


def _handle_discard_energy_card(obs: Observation) -> list[int] | None:
    """
    DISCARD_ENERGY_CARD: 付いているエネルギーカードをトラッシュする。
    優先順位: ベンチの瀕死ポケモンから、次にアクティブの余剰エネルギー。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))

    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY_CARD)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                # HPが低いポケモン（どうせ倒される）のエネルギーを先にトラッシュ
                low_hp_score = (1.0 - _hp_ratio(poke)) * 5.0
                # ベンチの余剰エネルギーはトラッシュしやすい
                bench_bonus = 2.0 if area == AreaType.BENCH else 0.0
                return low_hp_score + bench_bonus
        return 0.0

    sorted_opts = sorted(energy_opts, key=score, reverse=True)
    return sorted_opts[:count]


def _handle_discard_tool_card(obs: Observation) -> list[int] | None:
    """
    DISCARD_TOOL_CARD: 付いているどうぐをトラッシュする。
    基本的に minCount 分だけ選ぶ。どのポケモンのどうぐでもよければ
    HPが低いポケモンのどうぐを優先してトラッシュ。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))

    tool_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.TOOL_CARD)
    if not tool_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                return (1.0 - _hp_ratio(poke)) * 5.0
        return 0.0

    sorted_opts = sorted(tool_opts, key=score, reverse=True)
    return sorted_opts[:count]


def _handle_switch_energy_card(obs: Observation) -> list[int] | None:
    """
    SWITCH_ENERGY_CARD: 付いているエネルギーカードを入れ替える。
    1択が多いが、複数ある場合はアクティブに付いているものを優先。
    """
    options = obs.select.option
    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY_CARD)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, _ in all_pokemon:
            if opt.area == area and opt.index == pos:
                return 10.0 if area == AreaType.ACTIVE else 0.0
        return 0.0

    best = max(energy_opts, key=score)
    return [best]


def _handle_discard_card_or_attached_card(obs: Observation) -> list[int] | None:
    """
    DISCARD_CARD_OR_ATTACHED_CARD: 手札カードまたは付いているカードをトラッシュ。
    付いているエネルギー/どうぐより手札の不要カードを先にトラッシュしたい。
    ここでは CARD (手札) → ENERGY_CARD → TOOL_CARD の優先度。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))

    hand_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.CARD)
    if hand_opts:
        return hand_opts[:count]
    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY_CARD)
    if energy_opts:
        return energy_opts[:count]
    tool_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.TOOL_CARD)
    if tool_opts:
        return tool_opts[:count]
    return None


def _handle_discard_energy(obs: Observation) -> list[int] | None:
    """
    DISCARD_ENERGY: エネルギー単位でトラッシュ先を選ぶ。
    remainEnergyCost を参考に必要な数だけ選ぶ。
    選び方: ベンチの瀕死ポケモンから、COLORLESS(汎用)エネルギーを優先してトラッシュ。
    """
    options = obs.select.option
    sel = obs.select
    needed = sel.remainEnergyCost if sel.remainEnergyCost > 0 else sel.minCount
    count = min(needed, sel.maxCount, len(options))

    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        # COLORLESS(汎用)エネルギーを優先トラッシュ
        from cg.api import EnergyType
        colorless_bonus = 2.0 if (opt.count is not None and opt.count == 1) else 0.0
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                low_hp_score = (1.0 - _hp_ratio(poke)) * 5.0
                bench_bonus = 3.0 if area == AreaType.BENCH else 0.0
                return low_hp_score + bench_bonus + colorless_bonus
        return colorless_bonus

    sorted_opts = sorted(energy_opts, key=score, reverse=True)
    return sorted_opts[:count]


def _handle_to_hand_energy(obs: Observation) -> list[int] | None:
    """
    TO_HAND_ENERGY: エネルギーを手札に戻す。
    次ターン再利用できるので、アクティブの余剰エネルギーを優先して回収。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))

    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                # 手札に戻す価値: アクティブが瀕死なら回収して再配置
                active_bonus = 3.0 if area == AreaType.ACTIVE else 0.0
                low_hp_score = (1.0 - _hp_ratio(poke)) * 5.0
                return active_bonus + low_hp_score
        return 0.0

    sorted_opts = sorted(energy_opts, key=score, reverse=True)
    return sorted_opts[:count]


def _handle_to_deck_energy(obs: Observation) -> list[int] | None:
    """
    TO_DECK_ENERGY: エネルギーをデッキに戻す。
    手札に戻す場合と似た判断。ベンチの余剰エネルギーを戻す。
    """
    options = obs.select.option
    sel = obs.select
    count = min(sel.maxCount, len(options))

    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, poke in all_pokemon:
            if opt.area == area and opt.index == pos:
                bench_bonus = 3.0 if area == AreaType.BENCH else 0.0
                low_hp_score = (1.0 - _hp_ratio(poke)) * 5.0
                return bench_bonus + low_hp_score
        return 0.0

    sorted_opts = sorted(energy_opts, key=score, reverse=True)
    return sorted_opts[:count]


def _handle_switch_energy(obs: Observation) -> list[int] | None:
    """
    SWITCH_ENERGY: エネルギーの差し替え先を選ぶ。
    アクティブポケモンにあるエネルギーを優先して入れ替え対象にする。
    """
    options = obs.select.option
    energy_opts = _find_all_option_indices(options, lambda o: o.type == OptionType.ENERGY)
    if not energy_opts:
        return None

    all_pokemon = _all_my_pokemon(obs)

    def score(idx: int) -> float:
        opt = options[idx]
        for area, pos, _ in all_pokemon:
            if opt.area == area and opt.index == pos:
                return 10.0 if area == AreaType.ACTIVE else 0.0
        return 0.0

    best = max(energy_opts, key=score)
    return [best]


# ---------------------------------------------------------------------------
# メインエントリ
# ---------------------------------------------------------------------------

def choose_energy_tool_action(obs: Observation) -> list[int]:
    """エネルギーやどうぐの付け替え・回収・トラッシュ先を選ぶ。

    担当 context:
        ATTACH_FROM, ATTACH_TO,
        DETACH_FROM,
        DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD, SWITCH_ENERGY_CARD,
        DISCARD_CARD_OR_ATTACHED_CARD,
        DISCARD_ENERGY,
        TO_HAND_ENERGY, TO_DECK_ENERGY, SWITCH_ENERGY
    """
    context = obs.select.context

    handler_map = {
        SelectContext.ATTACH_FROM:                _handle_attach_from,
        SelectContext.ATTACH_TO:                  _handle_attach_to,
        SelectContext.DETACH_FROM:                _handle_detach_from,
        SelectContext.DISCARD_ENERGY_CARD:        _handle_discard_energy_card,
        SelectContext.DISCARD_TOOL_CARD:          _handle_discard_tool_card,
        SelectContext.SWITCH_ENERGY_CARD:         _handle_switch_energy_card,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD: _handle_discard_card_or_attached_card,
        SelectContext.DISCARD_ENERGY:             _handle_discard_energy,
        SelectContext.TO_HAND_ENERGY:             _handle_to_hand_energy,
        SelectContext.TO_DECK_ENERGY:             _handle_to_deck_energy,
        SelectContext.SWITCH_ENERGY:              _handle_switch_energy,
    }

    handler = handler_map.get(context)
    if handler is not None:
        result = handler(obs)
        if result is not None:
            return result

    # どのハンドラも処理できなかった場合のフォールバック
    return choose_random_legal_action(obs)