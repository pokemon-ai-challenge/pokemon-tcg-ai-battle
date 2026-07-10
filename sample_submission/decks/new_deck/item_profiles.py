"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うグッズごとの効果分類。knowledge.profile_types.ItemProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙
（search/draw/heal/disruption/setup/lock/other）から選ぶ。priority は同カテゴリ内の
tie-break用（例: どのサーチカードを優先するか）。
"""

from ptcg_ai.shared.profile_types import ItemProfile

# TODO(担当A): 新デッキの60枚確定後に card_id -> ItemProfile を埋める。
PROFILES: dict[int, ItemProfile] = {}
