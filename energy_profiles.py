"""エネルギー別プロファイル（⑦、担当A領域）。"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import EnergyProfile

PROFILES: dict[int, EnergyProfile] = {
    5: EnergyProfile(category="basic"),  # 基本【超】エネルギー
    19: EnergyProfile(category="special"),  # テレパス【超】エネルギー（超1個ぶん）
    13: EnergyProfile(category="special"),  # リッチエネルギー（ACE SPEC、無1個ぶん）
}
