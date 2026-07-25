"""サポートの使用条件・対象選択。

1182 ボスの指令だけは「攻撃直前に判断する」複雑な決定木を持つため、真偽値だけでなく
対象ポケモンそのものを返す専用関数 `boss_orders_target` を用意する。
"""

from __future__ import annotations

from cg.api import CardType, Pokemon, SelectData, State

from ptcg_ai.custom_agent import attack, board_context, constants
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import card_cache


def boss_orders_target(state: State) -> Pokemon | None:
    """ボスの指令の使用判定（仕様の決定木）。使うべきなら交代先の相手ベンチポケモンを返す。

    1. 自分の最大ダメージ < 相手アクティブHP かつ >= 相手ベンチいずれかのHP
       → 使用（複数候補時はエネルギー数優先、次にサイド枚数優先）。
    2. 自分の最大ダメージが相手の場の全ポケモンのHPを上回る場合:
       - 相手アクティブ・ベンチ両方EX → ベンチのほうがエネルギー数が多ければ使用。
       - 相手アクティブ・ベンチ双方EXでない → 上記と同じ比較（エネルギー数が多い方を使用）。
       - 相手アクティブのみEX（ベンチにEXなし）→ 使用しない（仕様に明記が無いため保守的に見送る）。
       - 相手アクティブはEXでない、ベンチにEXがいる → 使用（ベンチのEXを狩る）。
       - 相手アクティブはEXでない、ベンチにもEXがいない → 使用しない。
    3. それ以外は使用しない。
    """
    opp_active = board_context.opponent_active(state)
    if opp_active is None:
        return None
    opp_bench = board_context.opponent_bench(state)
    if not opp_bench:
        return None

    my_damage = attack.max_damage_against(opp_active, state)
    if my_damage is None:
        return None

    can_ko_active = my_damage >= opp_active.hp

    if not can_ko_active:
        candidates = [
            p for p in opp_bench if (dmg := attack.max_damage_against(p, state)) is not None and dmg >= p.hp
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda p: (board_context.energy_count(p), board_context.prize_value(p.id)),
        )

    # 自分の最大ダメージで相手アクティブは倒せる。ベンチ全体も倒しきれる前提かを見る。
    all_ko = all(
        (dmg := attack.max_damage_against(p, state)) is not None and dmg >= p.hp for p in opp_bench
    )
    if not all_ko:
        return None

    active_is_ex = board_context.is_ex(opp_active.id)
    bench_ex = [p for p in opp_bench if board_context.is_ex(p.id)]

    if active_is_ex and bench_ex:
        best_bench_ex = max(bench_ex, key=board_context.energy_count)
        if board_context.energy_count(best_bench_ex) > board_context.energy_count(opp_active):
            return best_bench_ex
        return None
    if not active_is_ex and not bench_ex:
        best_bench = max(opp_bench, key=board_context.energy_count)
        if board_context.energy_count(best_bench) > board_context.energy_count(opp_active):
            return best_bench
        return None
    if not active_is_ex and bench_ex:
        return max(bench_ex, key=board_context.energy_count)
    # active_is_ex and not bench_ex: 仕様に明記が無い組み合わせ。保守的に見送る。
    return None


def _lanas_aid_condition(state: State) -> bool:
    """スイレンのお世話:
    (トラッシュにエネルギーがあり、バトル場のエネルギーが足りず、手札にもエネルギーが無い)
    または (トラッシュにフーディン・ユンゲラー・ケーシィの3体がそろっている) 場合。
    """
    own_active = board_context.own(state).active
    discard = board_context.discard_ids(state)
    hand = board_context.hand_ids(state)
    discard_has_energy = any(cid in discard for cid in constants.ENERGY_CARD_IDS)
    hand_has_energy = any(cid in hand for cid in constants.ENERGY_CARD_IDS)

    energy_shortage = False
    if own_active and own_active[0] is not None:
        pokemon = own_active[0]
        required = constants.ENERGY_REQUIRED_COUNT.get(pokemon.id)
        if required is not None and board_context.energy_count(pokemon) < required:
            energy_shortage = discard_has_energy and not hand_has_energy

    fudin_line_in_discard = all(cid in discard for cid in constants.FUDIN_LINE)
    return energy_shortage or fudin_line_in_discard


def _xerosics_condition(state: State) -> bool:
    """クセロシキのたくらみ: 相手の手札が10枚を超えていたら優先的に使用。"""
    return board_context.opponent_hand_count(state) > 10


def _touko_condition(state: State) -> bool:
    """トウコ: 手札にあれば基本的に使用する（山札を消費するため手札20枚以上なら見送る）。"""
    return board_context.own_hand_count(state) < constants.DECK_DRAW_STOP_HAND_SIZE


def _hikari_condition(state: State) -> bool:
    """ヒカリ: 手札にあれば基本的に使用する（山札を消費するため手札20枚以上なら見送る）。"""
    return board_context.own_hand_count(state) < constants.DECK_DRAW_STOP_HAND_SIZE


_CONDITIONS = {
    constants.BOSS_ORDERS: lambda state: boss_orders_target(state) is not None,
    constants.LANAS_AID: _lanas_aid_condition,
    constants.XEROSICS_MACHINATIONS: _xerosics_condition,
    constants.TOUKO: _touko_condition,
    constants.HIKARI: _hikari_condition,
}


def is_usable(card_id: int, state: State) -> bool:
    condition = _CONDITIONS.get(card_id)
    return condition(state) if condition is not None else True


# --- トウコ／ヒカリのサーチ対象選択（TO_HAND, select.effect.id で判定） ---

_TOUKO_ENERGY_ORDER: tuple[int, ...] = (constants.RICH_ENERGY, constants.TELEPATH_ENERGY, constants.BASIC_PSYCHIC_ENERGY)
# トウコの進化ポケモン優先順位: 進化先が場・手札に無いラインを優先。それ以外はフーライン→ノコライン。
_TOUKO_EVOLUTION_IDS: tuple[int, ...] = (constants.KADABRA, constants.ALAKAZAM, constants.DUDUNSPARCE)
_HIKARI_TARGETS: tuple[int, ...] = (
    constants.CASEY,
    constants.DUNSPARCE,
    constants.SHAYMIN,
    constants.KADABRA,
    constants.ALAKAZAM,
    constants.DUDUNSPARCE,
)


def _missing_successor_bonus(card_id: int, state: State) -> int:
    """card_id がその進化元の「まだ場・手札に居ない後継」であれば高いスコアを返す。"""
    ids_present = board_context.own_ids_present(state)
    for base_id, successor_id in constants.EVOLUTION_SUCCESSOR.items():
        if successor_id == card_id and base_id in ids_present and successor_id not in ids_present:
            return 1000
    return 0


def choose_touko_target(select: SelectData, state: State) -> list[int]:
    """トウコ: エネルギーは常に [リッチ, テレパス, 基本] を優先。ポケモンは進化先不足ラインを優先、
    それ以外は フーライン(742/743) → ノコライン(66) の順。
    """

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id is None:
            return 0.0
        card = card_cache.get_card(card_id)
        if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            if card_id in _TOUKO_ENERGY_ORDER:
                return 100.0 - _TOUKO_ENERGY_ORDER.index(card_id)
            return 0.0
        bonus = _missing_successor_bonus(card_id, state)
        if card_id in _TOUKO_EVOLUTION_IDS:
            return bonus + (50.0 - _TOUKO_EVOLUTION_IDS.index(card_id))
        return bonus

    return common.pick_top(select, score)


def choose_hikari_target(select: SelectData, state: State) -> list[int]:
    """ヒカリ: たね/1進化/2進化それぞれ、基本優先順位（POKEMON_PRIORITY）に沿って選ぶ。"""

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in constants.POKEMON_PRIORITY:
            return float(len(constants.POKEMON_PRIORITY) - constants.POKEMON_PRIORITY.index(card_id))
        return 0.0

    return common.pick_top(select, score)


def choose_lanas_aid_target(select: SelectData, state: State) -> list[int]:
    """スイレンのお世話: エネルギー優先、次いで進化先不足ラインの基本ポケモンを優先。"""

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id is None:
            return 0.0
        if card_id in constants.ENERGY_CARD_IDS:
            return 200.0 - constants.ENERGY_CARD_IDS.index(card_id)
        bonus = _missing_successor_bonus(card_id, state)
        if card_id in constants.POKEMON_PRIORITY:
            return bonus + (100.0 - constants.POKEMON_PRIORITY.index(card_id))
        return bonus

    return common.pick_top(select, score)
