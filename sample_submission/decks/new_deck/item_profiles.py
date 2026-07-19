
"""グッズ別プロファイル（⑦、担当A領域）。"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import ItemProfile

PROFILES: dict[int, ItemProfile] = {
    1079: ItemProfile(category="setup", priority=0.9),  # ふしぎなアメ：進化短縮の要
    1081: ItemProfile(category="disruption", priority=0.5),  # 改造ハンマー：相手の特殊エネルギー破壊
    1086: ItemProfile(category="setup", priority=0.8),  # なかよしポフィン：たねポケモン展開
    1097: ItemProfile(category="search", priority=0.5),  # 夜のタンカ：トラッシュからポケモン/基本エネ回収
    1129: ItemProfile(category="other", priority=0.3),  # せいなるはい：トラッシュ整理・山札に戻す
    1146: ItemProfile(category="setup", priority=0.5),  # ワンダーパッチ：トラッシュの基本超エネを再利用
    1152: ItemProfile(category="search", priority=0.7),  # ポケパッド：ポケモンサーチ
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うグッズごとの効果分類。knowledge.profile_types.ItemProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙
（search/draw/heal/disruption/setup/lock/other）から選ぶ。priority は同カテゴリ内の
tie-break用（例: どのサーチカードを優先するか）。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> ItemProfile を埋める。
#PROFILES: dict[int, ItemProfile] = {}

