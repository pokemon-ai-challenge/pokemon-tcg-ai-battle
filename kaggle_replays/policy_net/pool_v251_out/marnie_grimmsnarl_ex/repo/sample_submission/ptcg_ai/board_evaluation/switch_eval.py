"""クラスタ② 盤面評価／担当B

逃げ先・入れ替え先の評価。バトル場を空けたいとき、どのベンチポケモンに交代するのが
最も良いかを board_features / attack_features のスコアを組み合わせて判定する。
"""

from cg.api import Pokemon, State

from ptcg_ai.board_evaluation import board_features

# 次の相手ターンで倒されにくい（安全な）交代先には加点する。
_SAFE_BONUS = 3.0


def switch_target_score(candidate: Pokemon, state: State, your_index: int) -> float:
    """交代先候補としてのスコアを返す（安全性・主力アタッカーらしさ・エネルギー状況などから算出）。"""
    score = board_features.attacker_score(candidate)
    if not board_features.is_likely_ko_next_turn(candidate, state, your_index):
        score += _SAFE_BONUS
    return score


def best_switch_target(candidates: list[Pokemon], state: State, your_index: int) -> Pokemon | None:
    """候補の中から最もスコアの高い交代先を返す。候補が無ければ None。"""
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: switch_target_score(candidate, state, your_index))
