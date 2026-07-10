"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うサポートごとの効果分類。knowledge.profile_types.SupporterProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙から選ぶ。
1ターン1枚制限の判断は handlers 側で State.supporterPlayed を見て行うため、ここでは
効果分類と priority（tie-break用）のみを持つ。
"""

from ptcg_ai.shared.profile_types import SupporterProfile

# TODO(担当A): 新デッキの60枚確定後に card_id -> SupporterProfile を埋める。
PROFILES: dict[int, SupporterProfile] = {}
