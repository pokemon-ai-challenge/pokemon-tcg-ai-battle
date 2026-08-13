"""matchup_common / 専門Policy学習基盤の自動テスト(最低限)。"""

import sys
from pathlib import Path

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

import matchup_common as mc  # noqa: E402


def test_frozen_climb_sha256_matches_expected():
    assert mc.sha256_file(mc.FROZEN_CLIMB_WEIGHTS) == mc.FROZEN_CLIMB_SHA256


def test_frozen_plan_a_deck_sha256_matches_expected():
    assert mc.sha256_file(mc.FROZEN_PLAN_A_DECK) == mc.FROZEN_PLAN_A_SHA256


def test_frozen_plan_a_deck_is_legal():
    deck = mc.read_deck(mc.FROZEN_PLAN_A_DECK)
    assert len(deck) == 60
    assert mc.validate_deck(deck) == []


@pytest.mark.parametrize("archetype", mc.TARGET_ARCHETYPES)
def test_all_11_archetypes_resolve_opponent_and_decks(archetype):
    wres = mc.resolve_opponent_weights(archetype)
    assert wres["path"] is not None, wres["error"]
    assert wres["tier"] in ("g2", "plain")
    decks = mc.discover_archetype_decks(archetype)
    usable = [d for d in decks if not d["errors"] and d["dup_of"] is None]
    assert len(usable) > 0, f"no usable deck for {archetype}"


def test_resolve_path_does_not_truncate_subdirectories():
    rel = "kaggle_replays/meta_analysis/archetype_decks/crustle/01.csv"
    resolved = mc.resolve_path(rel)
    assert resolved == (mc.ROOT / rel).resolve()
    assert resolved.name == "01.csv"
    assert "archetype_decks" in str(resolved)
    assert "crustle" in str(resolved)


def test_resolve_path_absolute_passthrough():
    p = mc.FROZEN_CLIMB_WEIGHTS
    assert mc.resolve_path(str(p)) == p


def test_deck_multiset_jaccard_identical_decks_is_one():
    deck = mc.read_deck(mc.FROZEN_PLAN_A_DECK)
    assert mc.deck_multiset_jaccard(deck, list(deck)) == pytest.approx(1.0)


def test_deck_multiset_jaccard_disjoint_decks_is_zero():
    assert mc.deck_multiset_jaccard([1, 2, 3], [4, 5, 6]) == 0.0


def test_discover_archetype_decks_flags_exact_duplicates():
    deck = mc.read_deck(mc.FROZEN_PLAN_A_DECK)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.csv"
        p2 = Path(td) / "b.csv"
        p1.write_text("\n".join(str(c) for c in deck), encoding="utf-8")
        p2.write_text("\n".join(str(c) for c in deck), encoding="utf-8")
        decks = mc.discover_archetype_decks("_unused", explicit=[str(p1), str(p2)])
        assert decks[0]["dup_of"] is None
        assert decks[1]["dup_of"] == str(p1)


def test_basic_energy_unlimited_but_other_cards_capped():
    assert mc.validate_deck([5] * 10 + [741] * 4 + [1] * (60 - 14)) == []
    errs = mc.validate_deck([741] * 5 + [5] * 55)
    assert any("741" in e for e in errs)


def test_ace_spec_capped_at_one():
    # id 13 = リッチエネルギー(ACE SPEC)。
    errs = mc.validate_deck([13, 13] + [5] * 58)
    assert any("13" in e for e in errs)


def test_forbidden_initial_weights_names():
    assert "policy_weights.json" in mc.FORBIDDEN_INITIAL_WEIGHTS_NAMES
    assert "policy_weights_alakazam_rl_vscrustle.json" in mc.FORBIDDEN_INITIAL_WEIGHTS_NAMES


def test_train_matchup_specialist_rejects_forbidden_initial_weights(tmp_path):
    import subprocess

    out_dir = tmp_path / "guard_test"
    proc = subprocess.run(
        [sys.executable, str(_RL_DIR / "train_matchup_specialist.py"),
         "--target-archetype", "crustle",
         "--learner-weights", "sample_submission/ptcg_ai/learning/policy_weights.json",
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(mc.ROOT), timeout=60)
    assert proc.returncode != 0
    assert "forbidden initial weights" in proc.stderr


def test_build_anchor_opp_specs_excludes_target():
    opp_specs, shares, meta = mc.build_anchor_opp_specs("crustle", exclude={"crustle"})
    archs_used = {m["archetype"] for m in meta if m.get("status") == "ok"}
    assert "crustle" not in archs_used
