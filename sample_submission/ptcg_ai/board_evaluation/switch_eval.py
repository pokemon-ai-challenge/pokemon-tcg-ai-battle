"""クラスタ② 盤面評価／担当B

逃げ先・入れ替え先の評価。バトル場を空けたいとき、どのベンチポケモンに交代するのが
最も良いかを board_features / attack_features のスコアを組み合わせて判定する。
"""

from cg.api import Pokemon, State


def switch_target_score(candidate: Pokemon, state: State, your_index: int) -> float:
    """交代先候補としてのスコアを返す（安全性・主力アタッカーらしさ・エネルギー状況などから算出）。"""
    raise NotImplementedError


def best_switch_target(candidates: list[Pokemon], state: State, your_index: int) -> Pokemon | None:
    """候補の中から最もスコアの高い交代先を返す。候補が無ければ None。"""
    raise NotImplementedError
