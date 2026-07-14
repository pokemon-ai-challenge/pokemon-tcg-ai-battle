"""スタジアム別プロファイル（⑦、担当A領域）。"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import StadiumProfile

PROFILES: dict[int, StadiumProfile] = {
    # バトルコロシアム：両者のベンチポケモンへの「ダメカンをのせる」効果（ワザ・特性由来）を
    # 無効化する常時スタジアム。攻撃対象の制限や妨害には当たらないため category="other"。
    1264: StadiumProfile(category="other", priority=0.5),
}
