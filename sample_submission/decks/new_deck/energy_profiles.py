"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使う基本/特殊エネルギーの種類。knowledge.profile_types.EnergyProfile の型に沿って
card_id をキーとした辞書を埋める。category は "basic" / "special" のいずれか。
"""

from ptcg_ai.shared.profile_types import EnergyProfile

# TODO(担当A): 新デッキの60枚確定後に card_id -> EnergyProfile を埋める。
PROFILES: dict[int, EnergyProfile] = {}
