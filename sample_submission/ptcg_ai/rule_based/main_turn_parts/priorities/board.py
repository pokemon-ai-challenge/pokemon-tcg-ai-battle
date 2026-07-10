"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: board（展開/進化/グッズ/スタジアム/どうぐ）

ベンチ展開・進化・場を強化するグッズ/スタジアム/どうぐの使用を提案する。
展開・進化の優先順位は knowledge.profile_registry.get_deck_plan() の
opening_priority / evolution_priority を参照する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry


def propose(obs: Observation) -> ActionProposal | None:
    """展開/進化/グッズ/スタジアム/どうぐ系の行動を1つ提案する。該当行動が無ければ None。"""
    raise NotImplementedError
