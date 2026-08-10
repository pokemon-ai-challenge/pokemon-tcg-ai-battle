"""legacy_feature_manifest.py の単体テスト(T0.1)。manifestが唯一の定義元として
encoder.FEATURE_NAMES / extended_features.Profile.feature_names と食い違わないことを確認する。
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def manifest():
    try:
        from ptcg_ai.learning import legacy_feature_manifest as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"unavailable: {exc}")
    return m


@pytest.fixture(scope="module")
def encoder():
    try:
        from ptcg_ai.learning import encoder as enc
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return enc


def test_base_total_matches_encoder_base_feature_count(manifest, encoder):
    assert manifest.total_dim(None) == encoder.BASE_FEATURE_COUNT == 166


@pytest.mark.parametrize("profile_name,expected_total", [("fuudin_v2", 389 - 166 + 166), ("fuudin_v4", 389)])
def test_extended_total_matches_profile_count(manifest, profile_name, expected_total):
    from ptcg_ai.learning import extended_features
    prof = extended_features.load_profile(profile_name)
    assert manifest.total_dim(profile_name) == 166 + prof.count


def test_groups_do_not_overlap_and_cover_range(manifest):
    for profile in (None, "fuudin_v2", "fuudin_v4"):
        gs = manifest.groups(profile)
        offset = 0
        for g in gs:
            assert g.start == offset, (profile, g)
            offset = g.end
        assert offset == manifest.total_dim(profile)


def test_no_removed_category_remains(manifest):
    """T0.1修正: 'removed'カテゴリが存在しない(global_keep/board_token_replacesのみ)。"""
    for profile in (None, "fuudin_v2", "fuudin_v4"):
        for g in manifest.groups(profile):
            assert g.usage in (manifest.USAGE_GLOBAL_KEEP, manifest.USAGE_BOARD_REPLACES)


def test_self_deck_prize_marginals_and_discard_are_global_keep(manifest):
    """自分の山札/サイドmarginals・自分トラッシュ内訳は削除せずglobal_keepであること。"""
    for profile in ("fuudin_v2", "fuudin_v4"):
        gs = {g.name: g.usage for g in manifest.groups(profile)}
        assert gs["self_deck_prize_marginals"] == manifest.USAGE_GLOBAL_KEEP
        assert gs["self_discard_breakdown"] == manifest.USAGE_GLOBAL_KEEP
        assert gs["opp_archetype"] == manifest.USAGE_GLOBAL_KEEP


def test_global_dim_base_and_fuudin_v4(manifest):
    assert manifest.global_dim(None) == 34
    assert manifest.global_dim("fuudin_v4") == 145
    assert manifest.global_dim("fuudin_v2") == 34 + 23 + 22 + 21


def test_unknown_profile_raises(manifest):
    with pytest.raises(ValueError):
        manifest.groups("no_such_profile")


def test_extract_global_features_1d_and_2d(manifest):
    import numpy as np
    vec = np.arange(166, dtype=np.float32)
    g = manifest.extract_global_features(vec, None)
    assert g.shape == (34,)
    batch = np.stack([vec, vec + 1000])
    g2 = manifest.extract_global_features(batch, None)
    assert g2.shape == (2, 34)
    assert (g2[1] - g2[0] == 1000).all()
