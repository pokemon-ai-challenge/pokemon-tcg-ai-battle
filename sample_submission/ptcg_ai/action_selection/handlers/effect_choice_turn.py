"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SKILL_ORDER, ACTIVATE, FIRST_EFFECT, COIN_HEAD

カード効果の発動可否・発動順・コイン面選択など、はい/いいえ・順序系の判断。
どの効果を有利に使うかは、カードごとの効果分類（decks.active の *_profiles）を参照する。
"""

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.action_selection import fallback

# ACTIVATE/FIRST_EFFECT/COIN_HEAD は基本的に自分のカード効果に関する Yes/No で、
# 判断材料が乏しい場合は「発動する/最初に選ぶ/表を選ぶ」を既定にしておく
# （効果を使わない・後回しにする理由が無い限りは発動側が無難なため）。
_DEFAULT_YES_CONTEXTS = (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT, SelectContext.COIN_HEAD)


def handle(obs: Observation) -> list[int]:
    """SKILL_ORDER / ACTIVATE / FIRST_EFFECT / COIN_HEAD の選択肢を処理する。"""
    if obs.select.context in _DEFAULT_YES_CONTEXTS:
        return _choose_yes(obs)
    # SKILL_ORDER: どの順で発動すべきかの判断材料が無いため、提示された順序をそのまま使う。
    return fallback.safe_choice(obs)


def _choose_yes(obs: Observation) -> list[int]:
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.YES:
            return [i]
    return fallback.safe_choice(obs)
