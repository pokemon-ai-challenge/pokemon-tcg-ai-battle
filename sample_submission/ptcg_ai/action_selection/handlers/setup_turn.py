"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON, IS_FIRST, MULLIGAN

対戦開始時の初期配置・先攻後攻・マリガン（引き直し）に関する選択。
バトル場/ベンチに出す Basic ポケモンの優先順位は
knowledge.profile_registry.get_deck_plan().opening_priority を参照する。
"""

from cg.api import Observation

from ptcg_ai.shared import profile_registry


def handle(obs: Observation) -> list[int]:
    """SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON / IS_FIRST / MULLIGAN を振り分けて処理する。"""
    raise NotImplementedError
