"""クラスタ② 盤面評価／担当B

「盤面の状態 -> 数値」に変換する共通部品。サイド差・テンポ・主力アタッカーらしさなど、
main_turn_parts/priorities/*.py や handlers/damage_target_turn.py から使われる。
"""

from cg.api import Pokemon, State


def attacker_score(pokemon: Pokemon) -> float:
    """「主力アタッカーらしさ」をスコア化する（HP、付いているエネルギー量などから算出）。"""
    raise NotImplementedError


def prize_diff(state: State, your_index: int) -> int:
    """自分と相手の残りサイド枚数の差を返す（正なら自分が有利）。"""
    raise NotImplementedError


def is_likely_ko_next_turn(pokemon: Pokemon, state: State, your_index: int) -> bool:
    """このポケモンが次の相手ターンで倒されやすいかを判定する。"""
    raise NotImplementedError
