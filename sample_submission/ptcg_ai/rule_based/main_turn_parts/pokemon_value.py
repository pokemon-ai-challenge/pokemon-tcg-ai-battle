"""クラスタ③ メイン行動の意思決定（共有部品）／担当B

ポケモン1体を「バトル場に置く価値」として評価する。B の盤面評価
（board_evaluation: HP・エネルギー本数・次ターン被KO予測）に、A の PokemonProfile
（role_score・bench_value・combo_with）を加味した総合値を返す。

priorities/retreat.py の撤退判断と、action_selection/handlers/switch_turn.py の
交代先選択の両方から使われる、「バトル場に誰を置くか」の評価の唯一の入り口。
board_evaluation 側は今後もカードIDを扱わない純粋な計算のままにしておき、
A のデータを混ぜる層はここに集約する（main_turn_parts/energy_eval.py が
deck_plan.energy_priority を混ぜているのと同じ構造）。
"""

from cg.api import Pokemon, State

from ptcg_ai.board_evaluation import attack_features, board_features, switch_eval
from ptcg_ai.shared import card_cache, profile_registry

# PokemonProfile.role_score（0.0〜1.0想定）の重み。
_ROLE_SCORE_WEIGHT = 3.0
# bench_value（ベンチに温存したい度合い）が高いポケモンほど、アクティブに出すのは避けたい。
_BENCH_VALUE_PENALTY_WEIGHT = 2.0
# combo_with の相棒が自分の場に既にいるなら、今アクティブにする/しておく理由として加点する。
_COMBO_BONUS = 1.5
# KO_REPLACEMENT_PRIORITY での順位に応じた加点の基準値（順位が下がるほど少しずつ減らす）。
_KO_REPLACEMENT_BASE_SCORE = 4.0
# このポケモンに交代すれば「今持っているエネルギーだけで」相手アクティブを即きぜつさせられる場合の
# 加点。switch_target_value の他項（attacker_score・_SAFE_BONUS・_profile_bonus・
# _ko_replacement_bonus）の実運用上の合計はおおむね20未満に収まるため、これを1桁以上
# 上回る値にして「即KOできる候補は他のどんな理由があっても最優先で選ばれる」ことを保証する
# （attack.py の _KO_BONUS=1000 と同じ考え方。retreat.py のトリガーB＝「後退で相手を今ターン中に
# きぜつさせられるか」が、この値のおかげで retreat.py 側から交代先を明示的に指定しなくても
# best_switch_target 経由で自然に成立する）。
_IMMEDIATE_KO_BONUS = 100.0


def active_value(pokemon: Pokemon, state: State, your_index: int) -> float:
    """このポケモンが「今バトル場にいる/これから出す」ことの総合評価。"""
    return board_features.attacker_score(pokemon) + _profile_bonus(pokemon, state, your_index)


def switch_target_value(candidate: Pokemon, state: State, your_index: int) -> float:
    """交代先候補としての総合評価（B の switch_eval + A の PokemonProfile + KO_REPLACEMENT_PRIORITY +
    即KOボーナス）。"""
    return (
        switch_eval.switch_target_score(candidate, state, your_index)
        + _profile_bonus(candidate, state, your_index)
        + _ko_replacement_bonus(candidate)
        + _immediate_ko_bonus(candidate, state, your_index)
    )


def best_switch_target(candidates: list[Pokemon], state: State, your_index: int) -> Pokemon | None:
    """候補の中から switch_target_value が最も高いものを返す。候補が無ければ None。"""
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: switch_target_value(candidate, state, your_index))


def _profile_bonus(pokemon: Pokemon, state: State, your_index: int) -> float:
    profile = profile_registry.get_pokemon_profile(pokemon.id)
    if profile is None:
        return 0.0

    bonus = profile.role_score * _ROLE_SCORE_WEIGHT
    # bench_value: 進化前で温存したい/コンボパーツなど「ベンチに置いておきたい」度合いが高いほど、
    # わざわざアクティブへ出す/留める理由を減点する（倒されるリスクにさらしたくないため）。
    bonus -= profile.bench_value * _BENCH_VALUE_PENALTY_WEIGHT
    if profile.combo_with and _combo_partner_in_play(profile.combo_with, state, your_index):
        bonus += _COMBO_BONUS
    return bonus


def _combo_partner_in_play(combo_with: list[int], state: State, your_index: int) -> bool:
    """combo_with に挙げられた相棒カードが、自分の場（アクティブ/ベンチ）に既にいるかを判定する。"""
    player = state.players[your_index]
    own_card_ids = {pokemon.id for pokemon in player.active if pokemon is not None}
    own_card_ids |= {pokemon.id for pokemon in player.bench}
    return any(card_id in own_card_ids for card_id in combo_with)


def _immediate_ko_bonus(pokemon: Pokemon, state: State, your_index: int) -> float:
    """このポケモンに交代すれば、今持っているエネルギーだけで相手アクティブを即きぜつ
    させられるワザを1つでも持つ場合に `_IMMEDIATE_KO_BONUS` を返す（無ければ0.0）。

    attack.py / retreat.py と共通の `attack_features.can_ko_with_any_available_attack` を使う
    （ロジックの重複を避ける）。手札依存の可変ダメージ推定には、交代の有無に関わらず変わらない
    「自分の」手札枚数（state.players[your_index].handCount）を渡す。
    """
    opponent = state.players[1 - your_index]
    opponent_active = opponent.active[0] if opponent.active else None
    if opponent_active is None:
        return 0.0

    opponent_card = card_cache.get_card(opponent_active.id)
    hand_size = state.players[your_index].handCount

    if attack_features.can_ko_with_any_available_attack(
        pokemon, opponent_active, opponent_card.weakness, opponent_card.resistance, hand_size
    ):
        return _IMMEDIATE_KO_BONUS
    return 0.0


def _ko_replacement_bonus(pokemon: Pokemon) -> float:
    """KO_REPLACEMENT_PRIORITY（きぜつ後の後継優先順位）に応じた加点。"""
    priority = profile_registry.get_ko_replacement_priority()
    rank = _ko_replacement_rank(pokemon.id, len(pokemon.energies), priority)
    if rank is None:
        return 0.0
    return _KO_REPLACEMENT_BASE_SCORE - rank * 0.01


def _ko_replacement_rank(card_id: int, energy_count: int, priority: list[int]) -> int | None:
    """priority 内での card_id の順位（小さいほど優先）を返す。載っていなければ None。

    同じ card_id が複数回登場する場合（例: 同じポケモンでもエネルギー充足度で優先度を
    分けたいケース）は、ENERGY_REQUIRED_COUNT に対してエネルギーがほぼ足りている
    （あと1本以内）候補は先頭寄りの登場位置を、そうでなければ末尾寄りの登場位置を採用する。
    """
    matches = [i for i, cid in enumerate(priority) if cid == card_id]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    required = profile_registry.get_energy_required_count(card_id)
    if required is None:
        return matches[0]

    is_almost_ready = energy_count >= required - 1
    return matches[0] if is_almost_ready else matches[-1]
