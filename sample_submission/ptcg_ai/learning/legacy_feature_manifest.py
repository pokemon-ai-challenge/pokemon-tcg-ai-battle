"""legacy_state_features(166/389次元)の列manifest(T0.1)。唯一の定義元。

以後、legacy_state_featuresから特定の意味を持つ列を取り出すコードは、生のslice
(``vec[132:166]`` 等)を書かず、必ずこのモジュールの ``global_columns()`` /
``GROUPS`` を経由すること。

分類方針: board_token_replaces(盤面トークン(自分/相手のactive/bench)に置き換わる列。
T1のCLS計算後は不要)/ global_keep(盤面個体表現ではないのでT1でも
legacy_global_featuresとしてそのまま使う)の2種類。**T0.1修正により
「削除」カテゴリは無い**(旧設計had 自分の山札/サイド marginals・自分トラッシュを
削除する予定だったが、これらは盤面個体表現ではないため global_keep に分類し直した)。

各グループの境界は ``encoder.FEATURE_NAMES`` / ``extended_features.Profile.feature_names``
から実際に確認した順序をそのまま反映している(``tests`` で両リストとの整合を検証)。
"""

from __future__ import annotations

from dataclasses import dataclass

USAGE_GLOBAL_KEEP = "global_keep"
USAGE_BOARD_REPLACES = "board_token_replaces"


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    end: int  # exclusive
    profile: str | None  # None = base166(全profile共通の先頭166列)。文字列 = 拡張特徴のプロファイル名
    usage: str

    @property
    def dim(self) -> int:
        return self.end - self.start


# 基本166次元(全profile共通、encoder.FEATURE_NAMES の並びそのまま)。
_BASE_SPEC = [
    ("board_self_active", 11, USAGE_BOARD_REPLACES),
    ("board_self_bench", 55, USAGE_BOARD_REPLACES),
    ("board_opp_active", 11, USAGE_BOARD_REPLACES),
    ("board_opp_bench", 55, USAGE_BOARD_REPLACES),
    ("self_counts", 8, USAGE_GLOBAL_KEEP),
    ("opp_counts", 5, USAGE_GLOBAL_KEEP),
    ("self_special_conditions", 5, USAGE_GLOBAL_KEEP),
    ("opp_special_conditions", 5, USAGE_GLOBAL_KEEP),
    ("game_progress", 7, USAGE_GLOBAL_KEEP),
    ("aggregate", 4, USAGE_GLOBAL_KEEP),
]

# 拡張特徴(extended_features.py)。own_zone="count"(fuudin_v2)と"marginals"(fuudin_v4)で
# 山札/サイドブロックの中身が違うため、プロファイルごとに定義する。
_EXTENDED_SPEC: dict[str, list[tuple[str, int, str]]] = {
    "fuudin_v2": [
        ("self_deck_prize_marginals", 23, USAGE_GLOBAL_KEEP),   # deck_prize_remain(22) + unaccounted(1)
        ("self_discard_breakdown", 22, USAGE_GLOBAL_KEEP),
        ("board_self_active_identity", 7, USAGE_BOARD_REPLACES),
        ("board_self_bench_identity", 7, USAGE_BOARD_REPLACES),
        ("board_opp_active_identity", 49, USAGE_BOARD_REPLACES),
        ("board_opp_bench_identity", 49, USAGE_BOARD_REPLACES),
        ("opp_archetype", 21, USAGE_GLOBAL_KEEP),
    ],
    "fuudin_v4": [
        ("self_deck_prize_marginals", 68, USAGE_GLOBAL_KEEP),   # remain/in_deck_p/in_prize_p(66) + unaccounted(1) + zone_known(1)
        ("self_discard_breakdown", 22, USAGE_GLOBAL_KEEP),
        ("board_self_active_identity", 7, USAGE_BOARD_REPLACES),
        ("board_self_bench_identity", 7, USAGE_BOARD_REPLACES),
        ("board_opp_active_identity", 49, USAGE_BOARD_REPLACES),
        ("board_opp_bench_identity", 49, USAGE_BOARD_REPLACES),
        ("opp_archetype", 21, USAGE_GLOBAL_KEEP),
    ],
}


def _build_groups(profile: str | None) -> list[FeatureGroup]:
    groups: list[FeatureGroup] = []
    offset = 0
    for name, dim, usage in _BASE_SPEC:
        groups.append(FeatureGroup(name, offset, offset + dim, None, usage))
        offset += dim
    if profile is not None:
        if profile not in _EXTENDED_SPEC:
            raise ValueError(f"未知のextended_features_profile: {profile!r}")
        for name, dim, usage in _EXTENDED_SPEC[profile]:
            groups.append(FeatureGroup(name, offset, offset + dim, profile, usage))
            offset += dim
    return groups


def groups(profile: str | None) -> list[FeatureGroup]:
    """``profile``(Noneなら基本166次元のみ)に対応する ``FeatureGroup`` のリストを返す。"""
    return _build_groups(profile)


def total_dim(profile: str | None) -> int:
    gs = groups(profile)
    return gs[-1].end if gs else 0


def global_columns(profile: str | None) -> list[int]:
    """legacy_global_features を作るために抜き出すべき列indexのリスト(元の順序を維持)。"""
    cols: list[int] = []
    for g in groups(profile):
        if g.usage == USAGE_GLOBAL_KEEP:
            cols.extend(range(g.start, g.end))
    return cols


def global_dim(profile: str | None) -> int:
    return len(global_columns(profile))


def extract_global_features(state_features, profile: str | None):
    """``state_features``(1決定点ぶん、または (n, total_dim) のnumpy配列)から
    global特徴だけを取り出す。1次元・2次元どちらでも動く。"""
    cols = global_columns(profile)
    import numpy as np
    arr = np.asarray(state_features)
    if arr.ndim == 1:
        return arr[cols]
    return arr[:, cols]


def manifest_hash(profile: str | None) -> str:
    """このprofileのグループ構成(name/start/end/usage)のhash。checkpointに埋め込み、
    ロード時に現在のmanifest定義とズレていないか検出するために使う。"""
    import hashlib
    import json
    payload = json.dumps([(g.name, g.start, g.end, g.usage) for g in groups(profile)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
