"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: retreat（にげる）

「逃げに具体的な必要性がある場合だけ」提案する（2026-07-23 再設計、docs/plans/rule_based/
retreat-margin-unification-2026-07-23.md 参照）。汎用スコア比較ではなく、以下の2トリガーの
いずれかを満たす場合のみ逃げを提案する:

    トリガーA（回避）: 次ターンKOされそう、かつベンチの最良候補は次ターンKOされなさそう
        （＝逃げれば実際に助かる。候補も即死するなら逃げても被害の先送りにしかならない）。
    トリガーB（後退での即きぜつ）: 現在のアクティブでは今ターン相手アクティブを倒せないが、
        ベンチ候補の中に、今持っているエネルギーだけで倒せるものがいる。

どちらにも該当しなければ None を返す（逃げない）。逃げ先の実際の選定は
main_turn_parts.pokemon_value.best_switch_target に委ねる。トリガーBで「KOできる候補が
選ばれる」保証は、pokemon_value.switch_target_value 側に組み込んだ即KOボーナス
（_IMMEDIATE_KO_BONUS、他の全項を確実に上回る値）によって、retreat.py から交代先を
明示的に指定しなくても自然に成立する。

今ターン既に交代済みなら提案しない（State.retreated を確認する）。
"""

from cg.api import Observation, OptionType

from ptcg_ai.board_evaluation import attack_features, board_features, energy_requirements
from ptcg_ai.core.config import load_config
from ptcg_ai.rule_based.main_turn_parts import pokemon_value
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import card_cache

_URGENT_RETREAT_SCORE = 5.0  # トリガーA（回避）
_OFFENSIVE_RETREAT_SCORE = 5.0  # トリガーB（後退での即きぜつ）。どちらも「明確に価値のある逃げ」として同格に扱う。

# トリガーAの確率化（probabilistic_ko）関連。selector.py の _CONFIG_CACHE と同じパターンで、
# retreat.propose() は router.route(obs) 経由で呼ばれ config を引数で受け取らないため、
# 自前で core.config.load_config() を読む。
_CONFIG_CACHE: dict | None = None
_DEFAULT_PROBABILISTIC_KO_THRESHOLD = 0.2


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def propose(obs: Observation) -> ActionProposal | None:
    """にげる行動を1つ提案する。トリガーA/Bのいずれも成立しなければ None。"""
    state = obs.current
    if state.retreated:
        return None

    your_index = state.yourIndex
    player = state.players[your_index]
    if not player.active or player.active[0] is None:
        return None
    active = player.active[0]

    active_card = card_cache.get_card(active.id)
    if not energy_requirements.can_afford_retreat(active_card.retreatCost, active.energies):
        # にげるコストを払えない（合法手として出ていても念のための防御）。
        return None

    bench = list(player.bench)
    if not bench:
        return None

    best_bench_target = pokemon_value.best_switch_target(bench, state, your_index)
    if best_bench_target is None:
        return None

    trigger_a = _trigger_a_avoid_ko(active, best_bench_target, state, your_index)
    trigger_b = not trigger_a and _trigger_b_offensive_switch(active, bench, state, your_index)

    if not trigger_a and not trigger_b:
        return None

    retreat_index = next((i for i, option in enumerate(obs.select.option) if option.type == OptionType.RETREAT), None)
    if retreat_index is None:
        return None

    score = _URGENT_RETREAT_SCORE if trigger_a else _OFFENSIVE_RETREAT_SCORE
    return ActionProposal(category="retreat", select=[retreat_index], score=score, reason="retreat")


def _trigger_a_avoid_ko(active, best_bench_target, state, your_index: int) -> bool:
    """トリガーA: 次ターンKOされそう、かつ逃げれば実際に助かる（候補も即死しない）。

    config の probabilistic_ko.enabled が真の場合は、is_likely_ko_next_turn の
    ブール判定の代わりに board_features.likely_ko_probability_next_turn（ベイズ推定による
    確率）と threshold を比較する。active/候補どちらも必ず同じ判定方式（確率 or ブール）で
    揃える（片方だけ確率化すると「安全」の意味がズレるため）。フラグが偽（既定）なら
    従来のブール判定のまま（非回帰）。
    """
    cfg = (_config() or {}).get("probabilistic_ko") or {}
    if cfg.get("enabled", False):
        threshold = cfg.get("threshold", _DEFAULT_PROBABILISTIC_KO_THRESHOLD)
        active_prob = board_features.likely_ko_probability_next_turn(active, state, your_index)
        if active_prob < threshold:
            return False
        candidate_prob = board_features.likely_ko_probability_next_turn(best_bench_target, state, your_index)
        return candidate_prob < threshold

    if not board_features.is_likely_ko_next_turn(active, state, your_index):
        return False
    return not board_features.is_likely_ko_next_turn(best_bench_target, state, your_index)


def _trigger_b_offensive_switch(active, bench, state, your_index: int) -> bool:
    """トリガーB: 現在のアクティブでは今ターン相手アクティブを倒せないが、
    ベンチ候補の中に、今持っているエネルギーだけで倒せるものがいる。
    """
    if _can_ko_opponent_active(active, state, your_index):
        # 現在のアクティブで既にKOできるなら、attack.py が直接そのワザを選ぶので撤退は不要。
        return False
    return any(_can_ko_opponent_active(candidate, state, your_index) for candidate in bench)


def _can_ko_opponent_active(pokemon, state, your_index: int) -> bool:
    """このポケモンが、今持っているエネルギーだけで相手アクティブを今すぐきぜつさせられるか。

    attack.py / pokemon_value.py と共通の attack_features.can_ko_with_any_available_attack を使う。
    """
    opponent = state.players[1 - your_index]
    opponent_active = opponent.active[0] if opponent.active else None
    if opponent_active is None:
        return False

    opponent_card = card_cache.get_card(opponent_active.id)
    hand_size = state.players[your_index].handCount

    return attack_features.can_ko_with_any_available_attack(
        pokemon, opponent_active, opponent_card.weakness, opponent_card.resistance, hand_size
    )
