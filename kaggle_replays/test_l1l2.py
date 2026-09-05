"""Phase13 §26: L1/L2 のテスト。

engine を起動しない範囲は単体テストで、engine 依存部分は
**実際に生成されたデータの不変条件**として検証する(こちらの方が強い)。
"""
from __future__ import annotations

import glob
import gzip
import json
import random
import statistics
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _l1l2_analyze as A  # noqa: E402
import _l1l2_core as C  # noqa: E402

DATA = str(_HERE / "_l1l2_main_w*.jsonl.gz")


# ---------------- Continuation: terminal outcome の視点 ----------------

@pytest.mark.parametrize("term,me,exp", [(0, 0, 1.0), (1, 0, 0.0), (1, 1, 1.0),
                                         (0, 1, 0.0), (2, 0, 0.5), (None, 0, None)])
def test_outcome_perspective(term, me, exp):
    assert C.outcome_of(term, me) == exp


def test_outcome_is_symmetric_between_players():
    for t in (0, 1):
        assert C.outcome_of(t, 0) + C.outcome_of(t, 1) == 1.0


# ---------------- tie epsilon (§13) ----------------

def test_winner_epsilon_boundaries():
    assert C.winner(0.06, 0.05) == "S3"
    assert C.winner(-0.06, 0.05) == "Q0"
    assert C.winner(0.05, 0.05) == "tie"
    assert C.winner(-0.05, 0.05) == "tie"
    assert C.winner(0.0, 0.05) == "tie"
    assert C.winner(None, 0.05) is None


def test_analyze_uses_frozen_epsilon():
    assert A.EPS_L1 == 0.05 and A.EPS_L2 == 0.05


# ---------------- teacher vote ----------------

def test_teacher_all_blocks_favor_q0():
    t = C.teacher_vote([0.1, 0.1, 0.1, 0.1])
    assert t["support_q0"] == 1.0 and t["cls"] == "stable"
    assert C.teacher_side(t["support_q0"]) == "supports_Q0"


def test_teacher_all_blocks_favor_s3():
    t = C.teacher_vote([-0.1] * 4)
    assert t["support_q0"] == 0.0 and t["cls"] == "stable"
    assert C.teacher_side(t["support_q0"]) == "supports_S3"


def test_teacher_split_is_near_tie():
    t = C.teacher_vote([0.1, 0.1, -0.1, -0.1])
    assert t["support_q0"] == 0.5 and t["cls"] == "unstable"
    assert C.teacher_side(0.5) == "near_tie"


def test_teacher_all_ties():
    t = C.teacher_vote([0.001, -0.001, 0.0, 0.002])
    assert t["cls"] == "always_tie" and t["support_q0"] == 0.5


def test_teacher_confidence_matches_phase12_formula():
    assert C.teacher_vote([0.1, 0.1, 0.1, -0.1])["confidence"] == pytest.approx(0.5)


# ---------------- CRN 機構 ----------------

def test_same_seed_reproduces_determinization_rng():
    a = [random.Random(123).random() for _ in range(3)]
    b = [random.Random(123).random() for _ in range(3)]
    assert a == b
    assert random.Random(124).random() != random.Random(123).random()


# ---------------- Statistics ----------------

def test_bootstrap_is_reproducible():
    v = [0.1, -0.2, 0.3, 0.0, 0.5]
    assert A.boot(v, b=500) == A.boot(v, b=500)


def test_bootstrap_ci_brackets_mean():
    v = [0.1] * 40
    r = A.boot(v, b=500)
    assert r["mean"] == pytest.approx(0.1)
    assert r["ci95"][0] <= 0.1 <= r["ci95"][1]
    assert r["n"] == 40


def test_bootstrap_empty_returns_none():
    assert A.boot([]) is None


def test_prep_uses_seed_mean_as_group_value():
    """group 値は continuation 平均。個々の continuation を独立扱いしない(§17)。"""
    row = {"group_id": "g", "game": 1, "turn": 5, "turn_band": "early",
           "cand_band": "small", "arch": "a", "me_first": True, "n_cands": 3,
           "action_type_q0": 0, "action_type_s3": 1, "q0_action": 0, "s3_action": 1,
           "policy_action": 0, "q0_margin": 0.1, "s3_margin": 0.2,
           "q0_scores": [], "s3_scores": [],
           "Q0": {"l1_mean": 0.4, "l1_values": [0.3, 0.5], "l2_mean": 0.0,
                  "l2_outcomes": [0.0, 0.0], "terminal_rate": 1.0, "steps_mean": 50,
                  "horizon_rate": 1.0, "prize_diff_mean": -1.0},
           "S3": {"l1_mean": 0.6, "l1_values": [0.5, 0.7], "l2_mean": 1.0,
                  "l2_outcomes": [1.0, 1.0], "terminal_rate": 1.0, "steps_mean": 60,
                  "horizon_rate": 1.0, "prize_diff_mean": 1.0}}
    g = A.prep([row])[0]
    assert g["l1_delta"] == pytest.approx(0.2)
    assert g["l2_delta"] == pytest.approx(1.0)
    assert g["w_l1"] == "S3" and g["w_l2"] == "S3"


def test_prep_skips_groups_without_l1():
    row = {"Q0": {"l1_mean": None}, "S3": {"l1_mean": 0.5}}
    assert A.prep([row]) == []


def test_subset_counts_sum_to_group_count():
    rows = []
    for i in range(6):
        rows.append({"group_id": f"g{i}", "game": i, "turn": 3, "turn_band": "early",
                     "cand_band": "small", "arch": "a" if i % 2 else "b",
                     "me_first": True, "n_cands": 3, "action_type_q0": 0,
                     "action_type_s3": 1, "q0_action": 0, "s3_action": 1,
                     "policy_action": 0, "q0_margin": 0.0, "s3_margin": 0.0,
                     "q0_scores": [], "s3_scores": [],
                     "Q0": {"l1_mean": 0.5, "l1_values": [0.5], "l2_mean": 0.0,
                            "l2_outcomes": [0.0], "terminal_rate": 1.0,
                            "steps_mean": 1, "horizon_rate": 1.0, "prize_diff_mean": 0.0},
                     "S3": {"l1_mean": 0.5, "l1_values": [0.5], "l2_mean": 1.0,
                            "l2_outcomes": [1.0], "terminal_rate": 1.0,
                            "steps_mean": 1, "horizon_rate": 1.0, "prize_diff_mean": 0.0}})
    G = A.prep(rows)
    tab = A.subset_table(G, lambda g: g["arch"])
    assert sum(v["groups"] for v in tab.values()) == len(G) == 6


# ---------------- 生成データの不変条件(engine 依存部分の検証) ----------------

def _rows():
    out = []
    for f in sorted(glob.glob(DATA)):
        out += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return out


needs_data = pytest.mark.skipif(not glob.glob(DATA), reason="L1/L2 データ未生成")


@needs_data
def test_data_branches_differ_only_in_first_action():
    for r in _rows():
        assert r["q0_action"] != r["s3_action"], r["group_id"]


@needs_data
def test_data_paired_branches_have_equal_continuation_counts():
    for r in _rows():
        assert len(r["Q0"]["l1_values"]) == len(r["S3"]["l1_values"]), r["group_id"]


@needs_data
def test_data_outcomes_are_valid_and_consistent_with_mean():
    for r in _rows():
        for tag in ("Q0", "S3"):
            oc = r[tag]["l2_outcomes"]
            assert all(o in (0.0, 0.5, 1.0) for o in oc), r["group_id"]
            if oc:
                assert r[tag]["l2_mean"] == pytest.approx(statistics.mean(oc))


@needs_data
def test_data_l1_values_in_unit_range():
    for r in _rows():
        for tag in ("Q0", "S3"):
            assert all(0.0 <= v <= 1.0 for v in r[tag]["l1_values"]), r["group_id"]
            assert all(v == v for v in r[tag]["l1_values"])          # NaN なし


@needs_data
def test_data_teacher_recomputes_from_stored_blocks():
    """保存された block から support/cls を再計算し、記録値と一致するか。"""
    n = 0
    for r in _rows():
        t = r.get("teacher")
        if not t:
            continue
        diffs = [b[0] - b[1] for b in t["blocks"]]
        rec = C.teacher_vote(diffs)
        assert rec["support_q0"] == pytest.approx(t["support_q0"]), r["group_id"]
        assert rec["cls"] == t["cls"], r["group_id"]
        n += 1
    assert n > 0


@needs_data
def test_data_no_duplicate_group_ids():
    ids = [r["group_id"] for r in _rows()]
    assert len(set(ids)) == len(ids)
