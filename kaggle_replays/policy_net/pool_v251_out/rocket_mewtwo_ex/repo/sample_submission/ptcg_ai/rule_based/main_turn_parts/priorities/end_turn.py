"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: end（ターン終了）

他のどのカテゴリからも有意な提案が得られなかった場合の最終手段として、ターン終了を提案する。
常に選択可能な保険的な提案であり、スコアは他カテゴリより十分低く設定する
（weights.py の基礎点で調整）。
"""

from cg.api import Observation, OptionType

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal

# 他カテゴリの提案が採用されないよう、常に最も低いスコアにしておく（weights.py で最終調整）。
_END_TURN_SCORE = -1000.0


def propose(obs: Observation) -> ActionProposal:
    """ターン終了の提案を返す（常に提案可能）。"""
    end_indices = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.END]
    if end_indices:
        return ActionProposal(category="end", select=[end_indices[0]], score=_END_TURN_SCORE, reason="no better action found")

    # 理論上は常に END が選択肢に含まれる想定だが、無い場合でも合法手を返せるようにしておく。
    count = min(max(obs.select.minCount, 1), len(obs.select.option))
    select = list(range(count))
    return ActionProposal(category="end", select=select, score=_END_TURN_SCORE, reason="fallback: no END option")
