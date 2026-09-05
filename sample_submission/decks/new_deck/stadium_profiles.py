
"""スタジアム別プロファイル（⑦、担当A領域）。

usage_condition は、deck_plan.py の STADIUM_USAGE_NOTES に書いていた自由記述の使用条件を
実際に判定できる関数に落としたもの（Issue #3 の対応）。
"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import StadiumProfile, UsageContext

_BATTLE_COLOSSEUM_ID = 1264


def _battle_colosseum_condition(ctx: UsageContext) -> bool:
    """バトルコロシアム: 場のスタジアムがバトルコロシアム以外なら使用する。"""
    return ctx.stadium_id != _BATTLE_COLOSSEUM_ID


PROFILES: dict[int, StadiumProfile] = {
    # バトルコロシアム：両者のベンチポケモンへの「ダメカンをのせる」効果（ワザ・特性由来）を
    # 無効化する常時スタジアム。攻撃対象の制限や妨害には当たらないため category="other"。
    1264: StadiumProfile(category="other", priority=0.5, usage_condition=_battle_colosseum_condition),
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うスタジアムごとの効果分類。knowledge.profile_types.StadiumProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙から選ぶ。
1ターン1枚制限の判断は handlers 側で State.stadiumPlayed を見て行うため、ここでは
効果分類と priority（tie-break用）のみを持つ。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> StadiumProfile を埋める。
#PROFILES: dict[int, StadiumProfile] = {}

