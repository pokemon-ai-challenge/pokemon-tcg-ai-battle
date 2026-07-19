"""デッキ切替の唯一の変更箇所／担当B: 雛形、担当A: 差替

現在使用するデッキのモジュールをまとめて re-export する。ptcg_ai.shared.profile_registry は
必ずこのモジュール経由で担当Aのデータ（decks/new_deck/*）を参照する。

デッキを切り替える際は、この import 先を差し替えるだけでよい
（ptcg_ai/action_selection や ptcg_ai/rule_based、ptcg_ai/shared 側の変更は不要）。
"""

from decks.new_deck import (
    attack_profiles,
    deck_plan,
    energy_profiles,
    item_profiles,
    pokemon_profiles,
    stadium_profiles,
    supporter_profiles,
    tool_profiles,
)

__all__ = [
    "deck_plan",
    "pokemon_profiles",
    "attack_profiles",
    "item_profiles",
    "supporter_profiles",
    "tool_profiles",
    "stadium_profiles",
    "energy_profiles",
]
