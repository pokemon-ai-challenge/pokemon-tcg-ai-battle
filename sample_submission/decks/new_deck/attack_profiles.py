"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うワザごとの追加効果分類。knowledge.profile_types.AttackProfile の型に沿って
attack_id をキーとした辞書を埋める（ベンチ狙撃、状態異常、ドロー、次ターン攻撃不可など）。
"""

from ptcg_ai.shared.profile_types import AttackProfile

# TODO(担当A): 新デッキの60枚確定後に attack_id -> AttackProfile を埋める。
PROFILES: dict[int, AttackProfile] = {}
