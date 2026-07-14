"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE, REMOVE_DAMAGE_COUNTER, HEAL

ダメージカウンターを乗せる/取り除く、回復する対象のポケモンを選ぶ場面。
「次ターン倒されやすいか」「主力アタッカーらしさ」の評価は
decision.evaluation.board_features を参照する。
"""

from cg.api import Observation

from ptcg_ai.board_evaluation import board_features


def handle(obs: Observation) -> list[int]:
    """ダメージ/回復対象の選択肢から、盤面評価に基づいて対象を選ぶ。"""
    raise NotImplementedError
