"""クラスタ③ メイン行動の意思決定（共有部品）／担当B

エネルギー付与に関する評価のうち、priorities/energy.py と handlers/energy_tool_turn.py の
両方から使う共通ロジック。「今ターン、どのポケモンにエネルギーを付けるべきか」の判断材料を
decision.evaluation.energy_requirements を使って組み立てる。
"""

from cg.api import Observation, Pokemon


def best_energy_target(obs: Observation) -> Pokemon | None:
    """自分の場のポケモンの中から、今エネルギーを付けるべき最有力候補を返す。"""
    raise NotImplementedError
