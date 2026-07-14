
"""サポート別プロファイル（⑦、担当A領域）。"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import SupporterProfile

PROFILES: dict[int, SupporterProfile] = {
    1182: SupporterProfile(category="disruption", priority=0.9),  # ボスの指令：ベンチ狙撃・詰め
    1184: SupporterProfile(category="search", priority=0.4),  # スイレンのお世話：トラッシュから回収
    1225: SupporterProfile(category="search", priority=0.8),  # トウコ：進化ポケモン+エネ同時サーチ
    1231: SupporterProfile(category="search", priority=0.9),  # ヒカリ：たね/1進化/2進化を1枚ずつサーチ
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うサポートごとの効果分類。knowledge.profile_types.SupporterProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙から選ぶ。
1ターン1枚制限の判断は handlers 側で State.supporterPlayed を見て行うため、ここでは
効果分類と priority（tie-break用）のみを持つ。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> SupporterProfile を埋める。
#PROFILES: dict[int, SupporterProfile] = {}

