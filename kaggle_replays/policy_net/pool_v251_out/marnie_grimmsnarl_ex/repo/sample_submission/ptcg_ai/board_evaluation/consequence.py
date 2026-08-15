"""選択肢の1手先「結果」を仮実行して読む共有ユーティリティ(Tier3設計書 Stage3a)。

docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md
の実装。``ptcg_ai.search.attack_plan`` の仮実行ロジック(``_evaluate_attack``/
``_resolve_attack_damage``/``_energy_removed`` 等)と同型の計算を、探索(search)から
独立した共有ユーティリティとして提供する。**``attack_plan.py`` 自体は変更しない**
(担当領域の尊重、既にKaggle提出済みの動作への影響を避けるため。意図的な複製であり、
``ml_policy_agent.py`` が ``action_selection`` への依存をゼロに保つために ``_is_valid_action``
を複製しているのと同じ方針)。

Stage3a: ATTACK選択肢のみ。
- :func:`best_effective_attack_damage` — 盤面上でMAIN選択肢のうち ``type == ATTACK``
  であるもの全てを仮実行し、実際に通る直接ダメージ(resistance/lock等エンジン解決込み)
  の最大値を返す(Tier3方針書 §3.2.1 の ``best_effective_attack_damage(state)``)。
- :func:`opp_hp_loss` — 特定の1選択肢(通常はATTACK)を仮実行した際の相手アクティブHP
  減少量(Tier3方針書 §3.1 のグループA特徴)。

Stage3b: 準備行動(ATTACH/EVOLVE/エネルギー除去ITEM等、複数選択を要する行動)。
- :func:`option_consequence` — 任意の1選択肢を仮実行し(``_transaction`` 相当のロジックで
  対象選択等の派生コールバックを解決)、グループA(即時結果)+グループB(将来の行動可能性、
  特に ``delta_best_effective_attack_damage``)をまとめて返す(方針書 §3/§4)。
  「1 option = search_step 1回」を仮定しない(方針書 §4)。複数の解決候補がある場合は
  ``delta_best_effective_attack_damage`` が最大になる解決を代表値として採用する(§4.1)。
  deckに触れる効果(ドロー/サーチ等)も除外しない(attack_planのv0/v1とは用途が異なり、
  本モジュールは「手札にある任意の行動の価値を記録するだけ」のため。方針書 §4)。

非決定性(コイン技): 1回の仮実行結果だけではコイン技の裏表で値がぶれるため、
``LogType.COIN`` が観測された結果は不採用として扱う(``attack_plan._has_coin`` と同じ
考え方、方針書 §2.5)。

静的計算(方針書 §5、ATTACHのエネルギー不足量計算等)は未実装(TODO)。現状は本モジュールの
全機能が仮実行(``search_step``)経由。ATTACKの実効ダメージ判定はカード効果の解決が
本質的に必要なため常に仮実行が必要だが、単純な基本エネルギーATTACHのような決定的な
ケースは将来的に ``ptcg_ai.board_evaluation.energy_requirements`` を使った静的計算に
置き換えてレイテンシを削減できる(方針書 §5の未実装分)。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable

from cg import api as cg_api
from cg.api import CardType, LogType, Observation, OptionType, SelectType

from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.shared import card_cache

# encoder.py の _MAX_SHORTFALL と同じ値(エネ不足数のクリップ上限。技が撃てない/
# 攻撃を持たない場合の既定値)。
_MAX_SHORTFALL = 10.0


class ConsequenceTimeout(Exception):
    """仮実行が時間予算を超えた。"""


@dataclass(frozen=True)
class AttackResult:
    """1つのATTACK選択肢を仮実行した結果。"""

    option_index: int
    direct_damage: int
    coin_seen: bool


def _check(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise ConsequenceTimeout()


def _by_serial(entries: list) -> dict[int, object]:
    return {p.serial: p for p in entries if p is not None}


def _hp_loss(before_entries: list, after_entries: list) -> int:
    """``before_entries``/``after_entries`` はPokemonのリスト(``active`` や ``bench``
    のような``PlayerState``の1フィールド)。存在し続けたポケモンのHP減少量の合計を返す。"""
    after = _by_serial(after_entries)
    return sum(
        max(0, poke.hp - after[poke.serial].hp)
        for poke in before_entries
        if poke is not None and poke.serial in after
    )


def _has_coin(logs) -> bool:
    return any(getattr(log, "type", None) == LogType.COIN for log in (logs or []))


def _begin(obs: Observation, hidden: dict):
    return cg_api.search_begin(
        obs,
        hidden["your_deck"],
        hidden["your_prize"],
        hidden["opponent_deck"],
        hidden["opponent_prize"],
        hidden["opponent_hand"],
        hidden["opponent_active"],
    )


def _release(search_ids: list[int]) -> None:
    for search_id in reversed(search_ids):
        try:
            cg_api.search_release(search_id)
        except Exception:
            pass


def _resolve_attack(node, option_index: int, me: int, deadline: float, search_ids: list[int]) -> AttackResult | None:
    """指定したATTACK選択肢(``option_index``)を仮実行し、相手アクティブへの直接ダメージを読む。"""
    _check(deadline)
    before = node.observation.current
    try:
        child = cg_api.search_step(node.searchId, [option_index])
    except (ValueError, RuntimeError):
        return None
    search_ids.append(child.searchId)
    after = child.observation.current
    if before is None or after is None:
        return None
    opponent = 1 - me
    logs = child.observation.logs or []
    direct = _hp_loss(before.players[opponent].active, after.players[opponent].active)
    return AttackResult(option_index, direct, _has_coin(logs))


def _evaluate_attack_option(
    obs: Observation,
    option_index: int,
    me: int,
    hidden_state_factory: Callable[[], dict | None],
    deadline: float,
) -> AttackResult | None:
    """``obs`` からATTACK選択肢を1つ仮実行する(候補ごとに独立した ``search_begin``)。

    エンジンの ``search_step`` は呼び出すたびにその探索セッションを1手進めるため、
    同一局面から複数の候補手を独立に試すには候補ごとに ``search_begin`` し直す必要がある
    (``ptcg_ai.search.attack_plan._v0`` / ``lethal_simple`` と同じ制約)。
    """
    _check(deadline)
    hidden = hidden_state_factory()
    if hidden is None:
        return None
    root = _begin(obs, hidden)
    search_ids = [root.searchId]
    try:
        return _resolve_attack(root, option_index, me, deadline, search_ids)
    finally:
        _release(search_ids)


def best_effective_attack_damage(
    obs: Observation,
    hidden_state_factory: Callable[[], dict | None],
    deadline: float,
) -> int:
    """``obs`` のMAIN選択肢のうち ``type == ATTACK`` であるもの全てを仮実行し、
    実際に通る直接ダメージ(resistance/lock等エンジン解決込みの実効値)の最大値を返す。

    ATTACK選択肢が無い/隠れ状態が得られない/全て仮実行に失敗/コイン絡み/タイムアウトの
    場合は 0 を返す(Tier3方針書 §3.2.1)。例外は投げない(呼び出し側の意思決定を止めない)。
    """
    select = obs.select
    state = obs.current
    if state is None or select is None:
        return 0

    best = 0
    for i, opt in enumerate(select.option):
        if opt.type != OptionType.ATTACK:
            continue
        try:
            result = _evaluate_attack_option(obs, i, state.yourIndex, hidden_state_factory, deadline)
        except ConsequenceTimeout:
            break
        if result is not None and not result.coin_seen:
            best = max(best, result.direct_damage)
    return best


def opp_hp_loss(
    obs: Observation,
    option_index: int,
    hidden_state_factory: Callable[[], dict | None],
    deadline: float,
) -> int | None:
    """指定した選択肢(通常はATTACK)を仮実行した際の相手アクティブHP減少量を返す。

    仮実行できない/コイン絡み/タイムアウトの場合は None
    (「0ダメージだった」と「判定不能だった」を呼び出し側が区別できるようにするため。
    Tier3方針書 §3.1)。
    """
    select = obs.select
    state = obs.current
    if state is None or select is None:
        return None
    try:
        result = _evaluate_attack_option(obs, option_index, state.yourIndex, hidden_state_factory, deadline)
    except ConsequenceTimeout:
        return None
    if result is None or result.coin_seen:
        return None
    return result.direct_damage


# ---------------------------------------------------------------------------
# Stage3b: 複数選択を要する行動(transaction解決)+ 統合consequence特徴
# ---------------------------------------------------------------------------


# 1回の option_consequence 呼び出しで試す追加選択の組み合わせ総数の既定上限
# (attack_plan.py の max_combinations_per_select と同じ考え方の予算制御)。
_DEFAULT_MAX_COMBINATIONS = 8


@dataclass
class TransactionResult:
    """1つの選択肢を起点に、MAIN復帰(または終局)まで解決した結果。"""

    node: object  # SearchState(解決後)
    search_ids: list[int]
    logs: list


@dataclass(frozen=True)
class OptionConsequence:
    """1つの選択肢を仮実行した際の即時結果(グループA)+将来の行動可能性への影響(グループB)。

    Tier3方針書 §3 の特徴定義に対応する。
    """

    option_index: int
    # グループA: Immediate consequence
    opp_hp_loss: int
    self_hp_gain: int
    opp_energy_removed: bool
    opp_special_energy_removed: bool
    self_energy_added: bool
    cards_drawn: int
    pokemon_evolved: bool
    stadium_changed: bool
    # グループB: Future capability delta(本Tierの主役)
    delta_best_effective_attack_damage: int
    delta_can_ko: bool
    delta_attack_ready: bool
    delta_energy_shortfall: float


def _by_serial_dict(player_state) -> dict[int, object]:
    return _by_serial([*player_state.active, *player_state.bench])


def _hp_gain(before_entries: list, after_entries: list) -> int:
    """``_hp_loss`` の逆(回復量)。存在し続けたポケモンのHP増加量の合計。"""
    after = _by_serial(after_entries)
    return sum(
        max(0, after[poke.serial].hp - poke.hp)
        for poke in before_entries
        if poke is not None and poke.serial in after
    )


def _energy_card_ids(player_state, card_type: CardType | None) -> set[int]:
    ids: set[int] = set()
    for poke in [*player_state.active, *player_state.bench]:
        if poke is None:
            continue
        for card in poke.energyCards:
            if card_type is not None:
                try:
                    if card_cache.get_card(card.id).cardType != card_type:
                        continue
                except KeyError:
                    continue
            ids.add(card.serial)
    return ids


def _energy_removed(before_player, after_player, card_type: CardType | None = None) -> bool:
    """``before_player``に付いていたエネルギー(``card_type``で絞り込み可)のうち、
    ``after_player``で1枚以上外れたものがあるか。"""
    before_ids = _energy_card_ids(before_player, card_type)
    after_ids = _energy_card_ids(after_player, card_type)
    return bool(before_ids - after_ids)


def _energy_added(before_player, after_player) -> bool:
    before_ids = _energy_card_ids(before_player, None)
    after_ids = _energy_card_ids(after_player, None)
    return bool(after_ids - before_ids)


def _active_evolved(before_player, after_player) -> bool:
    """自分のアクティブスロットのカードidが変わったか(進化の簡易検出)。
    伏せ/不在から出現した場合は進化ではないため対象外。"""
    before_active = before_player.active[0] if before_player.active else None
    after_active = after_player.active[0] if after_player.active else None
    if before_active is None or after_active is None:
        return False
    return before_active.id != after_active.id


def _stadium_id(state) -> int:
    stadium = getattr(state, "stadium", None) or []
    return stadium[0].id if stadium else 0


def _pokemon_readiness(pokemon) -> tuple[bool, float]:
    """``(has_ready_attack, min_energy_shortfall)`` を返す(方針書 §5: 静的計算)。

    ``encoder._pokemon_features`` と同じロジック(``energy_requirements.energy_shortfall``)
    だが、仮実行(search_step)を一切使わない。技の実ダメージ判定(resistance/lock込み)とは
    異なり、「エネルギーが足りているか」はカードの静的データ(``card.attacks`` の
    エネルギーコスト)と ``pokemon.energies`` だけで決まる決定的な計算のため。
    """
    if pokemon is None:
        return False, _MAX_SHORTFALL
    try:
        card = card_cache.get_card(pokemon.id)
    except KeyError:
        return False, _MAX_SHORTFALL
    has_ready = False
    min_shortfall = _MAX_SHORTFALL
    for attack_id in card.attacks or []:
        try:
            attack = card_cache.get_attack(attack_id)
        except KeyError:
            continue
        shortfall = energy_requirements.energy_shortfall(attack, pokemon.energies or [])
        shortfall_sum = float(sum(shortfall.values()))
        min_shortfall = min(min_shortfall, shortfall_sum)
        if not shortfall:
            has_ready = True
    return has_ready, min_shortfall


def _transaction_candidates(
    node,
    selection: list[int],
    me: int,
    deadline: float,
    search_ids: list[int],
    logs: list,
    budget: list[int],
) -> list["TransactionResult"]:
    """``selection`` を起点に、同一ターン・同一 ``yourIndex`` でMAINへ復帰する(または終局する)
    まで解決を試す。派生する追加選択がある場合は組み合わせを ``budget`` の範囲内で列挙し、
    見つかった解決を(1つに絞らず)すべて返す(§4.1: 「代表値」を後段でスコアにより選ぶため)。

    attack_plan.py の ``_transaction`` と異なり、deckに触れる効果(ドロー/サーチ等)は
    除外しない(方針書 §4: 本モジュールは「手札にある任意の行動の価値を記録するだけ」)。
    コイン絡みの線は判定不能として除外する(§2.5)。
    """
    _check(deadline)
    try:
        child = cg_api.search_step(node.searchId, selection)
    except (ValueError, RuntimeError):
        return []
    search_ids.append(child.searchId)
    state = child.observation.current
    all_logs = [*logs, *(child.observation.logs or [])]
    if state is None or _has_coin(all_logs):
        return []
    if state.result != -1:
        return [TransactionResult(child, list(search_ids), all_logs)]

    select = child.observation.select
    if state.yourIndex != me or select is None or select.deck is not None:
        return []
    if select.type == SelectType.MAIN:
        return [TransactionResult(child, list(search_ids), all_logs)]

    results: list[TransactionResult] = []
    n_options = len(select.option)
    for n in range(max(0, select.minCount), min(select.maxCount, n_options) + 1):
        for combo in itertools.combinations(range(n_options), n):
            if budget[0] <= 0:
                return results
            budget[0] -= 1
            sub = _transaction_candidates(
                child, list(combo), me, deadline, search_ids, all_logs, budget
            )
            results.extend(sub)
    return results


def _best_effective_attack_damage_from_node(
    node,
    select,
    me: int,
    deadline: float,
    search_ids: list[int],
) -> int:
    """既に開いている search セッション(``node``)を起点に、``select`` のうち
    ``type == ATTACK`` の選択肢を評価し、最大の直接ダメージを返す。

    **新規 ``search_begin`` は行わない。** ``node.searchId`` を直接 ``search_step`` する
    (``lethal_simple._dfs`` が同一親ノードから複数候補を ``search_step`` するのと同じ
    使い方で、これはエンジンAPIとして妥当)。

    重要な注意(実測で確認した不具合の回避): ``search_step`` の戻り値の
    ``Observation.search_begin_input`` は ``None``(``cg.api.SearchState`` のdocstring
    「New observation. search_begin_input is None.」参照)。この仮実行後の
    Observationを :func:`best_effective_attack_damage` のように新規 ``search_begin`` の
    入力として渡すと、実エンジンの内部状態が壊れ、**同一プロセス内の以降の対戦まで
    巻き添えで壊れる**(2026-07-23、league/_diag_consequence_latency.py の実測で確認)。
    このため ``option_consequence`` の「実行後」評価は必ずこちらを使うこと。
    """
    best = 0
    for i, opt in enumerate(select.option):
        if opt.type != OptionType.ATTACK:
            continue
        try:
            result = _resolve_attack(node, i, me, deadline, search_ids)
        except ConsequenceTimeout:
            break
        if result is not None and not result.coin_seen:
            best = max(best, result.direct_damage)
    return best


def _resolve_transaction_candidates(
    obs: Observation,
    first_selection: list[int],
    me: int,
    hidden_state_factory: Callable[[], dict | None],
    deadline: float,
    max_combinations: int,
) -> tuple[list[TransactionResult], list[int]]:
    """``first_selection`` を起点にtransactionを解決する候補を全て集める
    (新規 ``search_begin`` を1回だけ使う。候補ごとの独立性は同一transaction内では不要
    ―― 分岐は同じセッション内の ``search_step`` で表現できるため、Stage3aの
    「候補ごとに独立したsearch_begin」とは異なる)。

    戻り値: (候補リスト, このtransactionで使った全search_id列)。呼び出し側が
    ``_release`` する責任を持つ(候補のobservationを読み終えるまで解放できないため)。
    """
    _check(deadline)
    hidden = hidden_state_factory()
    if hidden is None:
        return [], []
    root = _begin(obs, hidden)
    search_ids = [root.searchId]
    budget = [max_combinations]
    try:
        candidates = _transaction_candidates(root, first_selection, me, deadline, search_ids, [], budget)
    except ConsequenceTimeout:
        candidates = []
    return candidates, search_ids


def option_consequence(
    obs: Observation,
    option_index: int,
    hidden_state_factory: Callable[[], dict | None],
    deadline: float,
    max_combinations: int = _DEFAULT_MAX_COMBINATIONS,
) -> OptionConsequence | None:
    """``option_index`` の選択肢を仮実行し、グループA+グループBの結果を返す。

    複数の追加選択(改造ハンマーの対象選択等)が必要な行動は ``_transaction_candidates``
    で解決する。複数の解決候補がある場合は ``delta_best_effective_attack_damage`` が
    最大になる解決を代表値として採用する(方針書 §4.1: 「上手く使えばどれだけの価値が
    あるか」を学習データとして与える設計判断)。

    解決不能(選択肢が存在しない/派生選択の組み合わせが尽きた/相手ターンに及んだ/
    コイン絡み/タイムアウト)の場合は None を返す(「unknown」。方針書 §4.1)。
    """
    state = obs.current
    select = obs.select
    if state is None or select is None:
        return None
    if not (0 <= option_index < len(select.option)):
        return None

    me = state.yourIndex
    opponent = 1 - me

    before_best = best_effective_attack_damage(obs, hidden_state_factory, deadline)

    candidates, search_ids = _resolve_transaction_candidates(
        obs, [option_index], me, hidden_state_factory, deadline, max_combinations
    )
    try:
        if not candidates:
            return None

        best_candidate = None
        best_delta = None
        best_after_best = None
        for candidate in candidates:
            after_select = candidate.node.observation.select
            if after_select is None:
                after_best = 0
            else:
                try:
                    after_best = _best_effective_attack_damage_from_node(
                        candidate.node, after_select, me, deadline, search_ids
                    )
                except ConsequenceTimeout:
                    after_best = 0
            delta = after_best - before_best
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_candidate = candidate
                best_after_best = after_best

        after_state = best_candidate.node.observation.current
        if after_state is None:
            return None

        before_me, before_opp = state.players[me], state.players[opponent]
        after_me, after_opp = after_state.players[me], after_state.players[opponent]

        before_opp_hp = before_opp.active[0].hp if before_opp.active and before_opp.active[0] else 0
        after_opp_hp = after_opp.active[0].hp if after_opp.active and after_opp.active[0] else 0
        opp_hp_loss_v = _hp_loss(before_opp.active, after_opp.active)
        self_hp_gain_v = _hp_gain(before_me.active, after_me.active)
        opp_energy_removed_v = _energy_removed(before_opp, after_opp)
        opp_special_energy_removed_v = _energy_removed(before_opp, after_opp, CardType.SPECIAL_ENERGY)
        self_energy_added_v = _energy_added(before_me, after_me)
        cards_drawn_v = max(0, (after_me.handCount or 0) - (before_me.handCount or 0))
        pokemon_evolved_v = _active_evolved(before_me, after_me)
        stadium_changed_v = _stadium_id(state) != _stadium_id(after_state)
        # 「今すぐ攻撃したらKOできるか」(before)と「この行動の後の最善攻撃でKOできるか」(after)。
        # after側は行動適用後の相手HP(after_opp_hp)を使う(行動自体が相手にダメージを
        # 与えていた場合にも正しく判定するため、beforeのHPを使い回さない)。
        can_ko_before = before_opp_hp > 0 and before_opp_hp <= before_best
        can_ko_after = after_opp_hp > 0 and after_opp_hp <= best_after_best
        delta_can_ko_v = (not can_ko_before) and can_ko_after

        # delta_attack_ready / delta_energy_shortfall(方針書 §5: 静的計算、仮実行不要)。
        # 自分のアクティブ1体のエネルギー充足状況を before/after で比較する。
        before_self_active = before_me.active[0] if before_me.active else None
        after_self_active = after_me.active[0] if after_me.active else None
        before_ready, before_shortfall = _pokemon_readiness(before_self_active)
        after_ready, after_shortfall = _pokemon_readiness(after_self_active)
        delta_attack_ready_v = (not before_ready) and after_ready
        delta_energy_shortfall_v = before_shortfall - after_shortfall  # 正 = 不足が減った(改善)

        return OptionConsequence(
            option_index=option_index,
            opp_hp_loss=opp_hp_loss_v,
            self_hp_gain=self_hp_gain_v,
            opp_energy_removed=opp_energy_removed_v,
            opp_special_energy_removed=opp_special_energy_removed_v,
            self_energy_added=self_energy_added_v,
            cards_drawn=cards_drawn_v,
            pokemon_evolved=pokemon_evolved_v,
            stadium_changed=stadium_changed_v,
            delta_best_effective_attack_damage=best_delta,
            delta_can_ko=delta_can_ko_v,
            delta_attack_ready=delta_attack_ready_v,
            delta_energy_shortfall=delta_energy_shortfall_v,
        )
    finally:
        _release(search_ids)
