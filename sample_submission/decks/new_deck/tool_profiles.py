"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うポケモンのどうぐごとの効果分類。knowledge.profile_types.ToolProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙から選ぶ。
"""

from ptcg_ai.shared.profile_types import ToolProfile

# TODO(担当A): 新デッキの60枚確定後に card_id -> ToolProfile を埋める。
PROFILES: dict[int, ToolProfile] = {}
