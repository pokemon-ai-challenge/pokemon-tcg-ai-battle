"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: draw（ドロー/サーチ系サポーター・グッズ）

手札のリソースが尽きていないか、探したいカード
（knowledge.profile_registry.get_deck_plan().search_priority）がまだ手札/場に無いかを見て、
優先度を判断する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry


def propose(obs: Observation) -> ActionProposal | None:
    """ドロー/サーチ系の行動を1つ提案する。該当行動が無ければ None。"""
    raise NotImplementedError
