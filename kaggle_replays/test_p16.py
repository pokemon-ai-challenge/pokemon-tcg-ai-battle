"""Phase16 §34: checkpoint / delta replay / branch / raw state logging の検証。

engine 依存部分は**生成された 260 group の不変条件**として検証する。
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "value_net"))

import _l1l2_core as C  # noqa: E402
import delta_features as DF  # noqa: E402

DATA = str(_HERE / "_p16_m*.jsonl.gz")
needs = pytest.mark.skipif(not glob.glob(DATA), reason="Phase16 データ未生成")


def _rows():
    out = []
    for f in sorted(glob.glob(DATA)):
        out += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return out


# ---------------- Checkpoint ----------------

def test_checkpoint_shas():
    exp = {"Q0-expanded.pt": "85ac1a597ebe328b", "DeltaQ_D1.pt": "6e922976e2dfb636",
           "S3.pt": "58cbd32b752853ec"}
    for n, s in exp.items():
        p = _HERE / "value_net" / "frozen" / n
        assert hashlib.sha256(p.read_bytes()).hexdigest()[:16] == s, n


def test_policy_and_value_shas():
    root = _HERE.parent
    exp = {"sample_submission/ptcg_ai/learning/policy_weights_alakazam_rl_climb.json":
           "395b02486f851be9",
           "sample_submission/ptcg_ai/learning/value_weights.json": "d6d7cd8907f53689"}
    for f, s in exp.items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] == s, f


def test_production_and_phase15_unchanged():
    root = _HERE.parent
    exp = {"sample_submission/configs/abl_5_full.json": "ca6c37af4b1a4149",
           "sample_submission/deck.csv": "8ae7a618b2655669"}
    for f, s in exp.items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] == s, f
    p12 = _HERE / "_teacher_blocks_p12.jsonl.gz"
    assert hashlib.sha256(p12.read_bytes()).hexdigest()[:16] == "d7aec0ca63d1eeb8"


# ---------------- Delta replay / schema ----------------

@needs
def test_delta_width_is_frozen_23():
    for r in _rows():
        assert len(r["delta_q0"]) == DF.N_DELTA
        assert len(r["delta_dq"]) == DF.N_DELTA
        for d in r["cand_deltas"]:
            assert len(d) == DF.N_DELTA


@needs
def test_stored_delta_matches_recomputation_from_after_summaries():
    """保存された delta が before/after から再計算した値と一致するか(§12)。"""
    n = 0
    for r in _rows():
        if r["after_q0"] is None:
            continue
        assert DF.single(r["before"], r["after_q0"]) == pytest.approx(r["delta_q0"], abs=1e-4)
        n += 1
    assert n > 0


@needs
def test_delta_identical_flag_is_consistent_with_distance():
    for r in _rows():
        d = DF.pair_distance(r["delta_q0"], r["delta_dq"])
        assert r["delta_identical"] == (d < 1e-9)
        assert abs(d - r["delta_distance"]) < 1e-4


@needs
def test_delta_seed_recorded():
    assert all(isinstance(r["delta_seed"], int) for r in _rows())


# ---------------- Branch ----------------

@needs
def test_branches_differ_only_in_first_action():
    for r in _rows():
        assert r["q0_action"] != r["s3_action"], r["group_id"]


@needs
def test_paired_branches_have_equal_continuation_counts():
    for r in _rows():
        assert len(r["Q0"]["l1_values"]) == len(r["S3"]["l1_values"]) == 8


@needs
def test_outcomes_valid_and_consistent():
    for r in _rows():
        for tag in ("Q0", "S3"):
            oc = r[tag]["l2_outcomes"]
            assert all(o in (0.0, 0.5, 1.0) for o in oc)
            if oc:
                assert r[tag]["l2_mean"] == pytest.approx(statistics.mean(oc))
            assert all(0.0 <= v <= 1.0 and v == v for v in r[tag]["l1_values"])


@needs
def test_terminal_reached_for_every_branch():
    for r in _rows():
        assert r["Q0"]["terminal_rate"] == 1.0 and r["S3"]["terminal_rate"] == 1.0


# ---------------- Raw state logging / leak ----------------

@needs
def test_raw_before_state_saved_without_opponent_hand():
    for r in _rows():
        e = r["entity_before"]
        assert "hand" in e["self"]
        assert "hand" not in e["opp"], r["group_id"]        # 相手手札はリークしない
        assert e["self"]["active"] is None or "id" in e["self"]["active"]


@needs
def test_before_and_after_summary_widths():
    for r in _rows():
        assert len(r["before"]) == DF.N_BASE
        for k in ("after_q0", "after_delta"):
            assert r[k] is None or len(r[k]) == DF.N_BASE


# ---------------- 統計ヘルパ ----------------

def test_tie_epsilon_unchanged_from_phase13():
    import _l1l2_analyze as A
    assert A.EPS_L1 == 0.05 and A.EPS_L2 == 0.05


def test_winner_uses_frozen_epsilon():
    assert C.winner(0.06, 0.05) == "S3" and C.winner(-0.06, 0.05) == "Q0"
    assert C.winner(0.05, 0.05) == "tie"


@needs
def test_no_duplicate_group_ids():
    ids = [r["group_id"] for r in _rows()]
    assert len(set(ids)) == len(ids)
