"""サポート別プロファイル（⑦、担当A領域）。"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import SupporterProfile

PROFILES: dict[int, SupporterProfile] = {
    1182: SupporterProfile(category="disruption", priority=0.9),  # ボスの指令：ベンチ狙撃・詰め
    1184: SupporterProfile(category="search", priority=0.4),  # スイレンのお世話：トラッシュから回収
    1225: SupporterProfile(category="search", priority=0.8),  # トウコ：進化ポケモン+エネ同時サーチ
    1231: SupporterProfile(category="search", priority=0.9),  # ヒカリ：たね/1進化/2進化を1枚ずつサーチ
}
