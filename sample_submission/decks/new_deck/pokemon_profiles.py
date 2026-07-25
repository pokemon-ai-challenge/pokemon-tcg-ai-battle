
"""ポケモン別プロファイル（⑦、担当A領域）。

役割・主力度・ベンチ価値・特性情報。deck_plan.py の方針を数値・語彙に落とし込んだもの。
"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import PokemonProfile

PROFILES: dict[int, PokemonProfile] = {
    # --- フーディン系列（メインアタッカーライン） ---
    741: PokemonProfile(  # ケーシィ
        role="main_attacker",
        role_score=0.2,
        bench_value=0.3,
        has_ability=False,
    ),
    742: PokemonProfile(  # ユンゲラー
        role="main_attacker",
        role_score=0.5,
        bench_value=0.3,
        has_ability=True,
        ability_category="draw",
        ability_priority=0.6,  # サイコドロー：手札から進化させたとき1回、2ドロー
    ),
    743: PokemonProfile(  # フーディン
        role="main_attacker",
        role_score=1.0,
        bench_value=0.2,
        has_ability=True,
        ability_category="draw",
        ability_priority=0.7,  # サイコドロー：手札から進化させたとき1回、3ドロー
    ),
    # --- ベンチ固定の特性要員（deck_plan.BENCH_ONLY_SUPPORT参照） ---
    65: PokemonProfile(  # ノコッチ
        role="support",
        role_score=0.1,
        bench_value=0.7,
        has_ability=False,
    ),
    66: PokemonProfile(  # ノココッチ
        role="support",
        role_score=0.15,
        bench_value=0.9,
        has_ability=True,
        ability_category="draw",
        ability_priority=0.9,  # にげあしドロー：毎ターン1回、3ドロー後に自身は山札へ
    ),
    343: PokemonProfile(  # シェイミ
        role="support",
        role_score=0.15,
        bench_value=0.8,
        has_ability=True,
        # はなのカーテンは常時効果（ダメージ無効化）で、上記7分類のどれにも
        # きれいに当てはまらないため "other" とする。迷ったら other、と
        # profile-contract に従う。
        ability_category="other",
        ability_priority=0.0,  # 常時効果のため使用タイミングの選択余地はない
    ),
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うポケモンごとの役割データ。knowledge.profile_types.PokemonProfile の型に沿って
card_id をキーとした辞書を埋める。特性を持つポケモンは has_ability/ability_category/
ability_priority も忘れずに埋める（EffectCategory の語彙は profile_types.py 参照）。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> PokemonProfile を埋める。
#PROFILES: dict[int, PokemonProfile] = {}

