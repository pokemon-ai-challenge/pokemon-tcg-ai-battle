"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うポケモンごとの役割データ。knowledge.profile_types.PokemonProfile の型に沿って
card_id をキーとした辞書を埋める。特性を持つポケモンは has_ability/ability_category/
ability_priority も忘れずに埋める（EffectCategory の語彙は profile_types.py 参照）。
"""

from ptcg_ai.shared.profile_types import PokemonProfile

# TODO(担当A): 新デッキの60枚確定後に card_id -> PokemonProfile を埋める。
PROFILES: dict[int, PokemonProfile] = {}
