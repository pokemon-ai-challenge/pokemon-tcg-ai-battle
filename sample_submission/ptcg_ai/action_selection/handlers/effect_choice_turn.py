"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SKILL_ORDER, ACTIVATE, FIRST_EFFECT, COIN_HEAD

カード効果の発動可否・発動順・コイン面選択など、はい/いいえ・順序系の判断。
どの効果を有利に使うかは、カードごとの効果分類（decks.active の *_profiles）を参照する。
"""

from cg.api import Observation


def handle(obs: Observation) -> list[int]:
    """SKILL_ORDER / ACTIVATE / FIRST_EFFECT / COIN_HEAD の選択肢を処理する。"""
    raise NotImplementedError
