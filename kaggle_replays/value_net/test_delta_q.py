"""Phase15 §33: delta 抽出 / multi-determinization / model / dataset のテスト。

engine を起動しない範囲は単体テスト、engine 依存部分は**生成データの不変条件**で検証する。
"""
from __future__ import annotations

import glob
import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import delta_features as DF  # noqa: E402
import train_delta_q as TD  # noqa: E402

DATA = str(_HERE.parent / "_dlt_probes_w*.jsonl.gz")
B0 = [10.0] * DF.N_BASE


# ---------------- Delta schema ----------------

def test_schema_is_frozen_23_dims():
    s = DF.schema()
    assert len(s) == DF.N_DELTA == 23
    assert s[-1]["feature"] == "valid_flag"
    assert [r["idx"] for r in s] == list(range(23))


def test_schema_kinds_are_predeclared():
    kinds = {r["kind"] for r in DF.schema()}
    assert kinds <= {"count", "resolve", "meta"}
    assert len(DF.DETERMINISTIC_DIMS) > 0


# ---------------- Delta 計算 ----------------

def test_zero_change_action_gives_zero_delta():
    d = DF.single(B0, list(B0))
    assert all(abs(x) < 1e-12 for x in d[:DF.N_BASE]) and d[-1] == 1.0


def test_positive_and_negative_delta():
    a = list(B0); a[7] = 7.0        # 手札 -3
    b = list(B0); b[9] = 13.0       # トラッシュ +3
    assert DF.single(B0, a)[7] == pytest.approx(-3.0 / DF.SCALE)
    assert DF.single(B0, b)[9] == pytest.approx(+3.0 / DF.SCALE)


def test_missing_after_yields_zero_and_invalid_flag():
    d = DF.single(B0, None)
    assert d[-1] == 0.0 and all(x == 0.0 for x in d[:DF.N_BASE])


def test_expected_equals_single_when_all_dets_identical():
    a = list(B0); a[7] = 8.0
    assert DF.expected(B0, [a, a, a, a]) == pytest.approx(DF.single(B0, a))


def test_expected_averages_across_determinizations():
    a = list(B0); a[7] = 8.0        # -2
    b = list(B0); b[7] = 14.0       # +4
    assert DF.expected(B0, [a, b])[7] == pytest.approx(1.0 / DF.SCALE)


def test_std_zero_for_deterministic_and_positive_for_stochastic():
    a = list(B0); a[7] = 8.0
    b = list(B0); b[7] = 14.0
    assert DF.std(B0, [a, a])[7] == pytest.approx(0.0)
    assert DF.std(B0, [a, b])[7] > 0


def test_std_needs_two_samples():
    assert DF.std(B0, [list(B0)]) == [0.0] * DF.N_BASE


def test_expected_ignores_missing_samples():
    a = list(B0); a[7] = 8.0
    assert DF.expected(B0, [a, None, a]) == pytest.approx(DF.single(B0, a))


def test_variance_scalar_zero_when_deterministic():
    a = list(B0); a[7] = 8.0
    assert DF.variance_scalar(B0, [a, a, a, a]) == pytest.approx(0.0)
    b = list(B0); b[7] = 14.0
    assert DF.variance_scalar(B0, [a, b]) > 0


def test_deterministic_only_zeroes_resolve_dims():
    a = list(B0)
    a[0] = 0.0        # self_active_hp = resolve
    a[7] = 8.0        # self_hand_count = count
    d = DF.deterministic_only(B0, [a])
    assert d[7] != 0.0
    assert d[0] == 0.0


def test_next_state_differs_from_delta():
    a = list(B0); a[7] = 8.0
    assert DF.next_state(a)[7] == pytest.approx(8.0 / DF.SCALE)
    assert DF.single(B0, a)[7] == pytest.approx(-2.0 / DF.SCALE)


def test_pair_distance_detects_candidate_discrimination():
    a = list(B0); a[7] = 8.0
    b = list(B0); b[9] = 12.0
    assert DF.pair_distance(DF.single(B0, a), DF.single(B0, a)) == 0.0
    assert DF.pair_distance(DF.single(B0, a), DF.single(B0, b)) > 0


# ---------------- shuffle 対照 ----------------

def _grp(C=4):
    g = {"before": list(B0),
         "candidates": [{"option_feat": [0.0] * 65, "action_card_id": 10 + i,
                         "option_type": 1, "policy_score": 0.0, "policy_rank": i,
                         "blocks": [0.5 + 0.01 * i] * 4,
                         "afters": [[10.0 + i] + [10.0] * (DF.N_BASE - 1)] * 4}
                        for i in range(C)],
         "state_feat": [0.1] * 166, "turn": 5, "turn_band": "early", "game": 1}
    import soft_target as ST
    g["_pairs"] = ST.group_pairs(g)
    return g


def test_shuffle_permutes_delta_without_changing_multiset():
    g = _grp()
    TD.featurize([g])
    a, b = g["_dcache"]["single"], g["_dcache"]["single_shuffle"]
    assert a != b
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))   # 値の分布は不変


def test_shuffle_is_a_derangement_for_every_candidate():
    g = _grp(C=5)
    TD.featurize([g])
    a, b = g["_dcache"]["single"], g["_dcache"]["single_shuffle"]
    assert all(a[i] != b[i] for i in range(len(a)))


# ---------------- Model ----------------

@pytest.mark.parametrize("arm", list(TD.ARMS))
def test_model_forward_shape_and_finite(arm):
    g = _grp()
    TD.featurize([g])
    gs = [g, g]
    b = TD.make_batch(gs, np.zeros(166, np.float32), np.ones(166, np.float32), arm)
    m = TD.build_model(arm, 166, 65, 16)
    q = m(b)
    assert q.shape == (2, 4) and torch.isfinite(q).all()


def test_shuffle_arm_has_identical_param_count_to_D1():
    a = sum(p.numel() for p in TD.build_model("D1", 166, 65, 32).parameters())
    b = sum(p.numel() for p in TD.build_model("D1_shuffle", 166, 65, 32).parameters())
    assert a == b


def test_D0_cap_matches_D1_param_count_within_5pct():
    d1 = sum(p.numel() for p in TD.build_model("D1", 166, 65, 32).parameters())
    cap = sum(p.numel() for p in TD.build_model("D0_cap", 166, 65, 32).parameters())
    assert abs(cap - d1) / d1 < 0.05


def test_D0_has_fewer_params_than_D1():
    assert (sum(p.numel() for p in TD.build_model("D0", 166, 65, 32).parameters())
            < sum(p.numel() for p in TD.build_model("D1", 166, 65, 32).parameters()))


def test_masked_candidates_excluded():
    g = _grp()
    TD.featurize([g])
    b = TD.make_batch([g], np.zeros(166, np.float32), np.ones(166, np.float32), "D1")
    b["mask"][:, 3] = False
    assert (TD.build_model("D1", 166, 65, 16)(b)[:, 3] < -1e8).all()


def test_seed_reproducibility():
    torch.manual_seed(3)
    a = TD.build_model("D2", 166, 65, 16)
    torch.manual_seed(3)
    c = TD.build_model("D2", 166, 65, 16)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))


def test_gradients_finite_and_reach_delta_encoder():
    g = _grp()
    TD.featurize([g])
    b = TD.make_batch([g], np.zeros(166, np.float32), np.ones(166, np.float32), "D1")
    m = TD.build_model("D1", 166, 65, 16)
    q = m(b).masked_fill(~b["mask"], 0.0)
    q.sum().backward()
    gr = m.delta_mlp[0].weight.grad
    assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0


def test_save_load_roundtrip(tmp_path):
    m = TD.build_model("D2", 166, 65, 16)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = TD.build_model("D2", 166, 65, 16)
    m2.load_state_dict(torch.load(p, map_location="cpu"))
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))


def test_delta_dim_table_matches_feature_module():
    assert TD.DELTA_DIM["single"] == DF.N_DELTA
    assert TD.DELTA_DIM["expected_std"] == DF.N_DELTA + DF.N_BASE
    assert TD.DELTA_DIM["none"] == 0


# ---------------- 生成データの不変条件 ----------------

def _rows():
    out = []
    for f in sorted(glob.glob(DATA)):
        out += [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
    return out


needs = pytest.mark.skipif(not glob.glob(DATA), reason="Phase15 データ未生成")


@needs
def test_data_every_candidate_has_M_determinizations():
    for r in _rows():
        for c in r["candidates"]:
            assert len(c["afters"]) == r["m_dets"], r["group_id"]


@needs
def test_data_determinization_seeds_unique_per_candidate():
    for r in _rows():
        for c in r["candidates"]:
            assert len(set(c["det_seeds"])) == len(c["det_seeds"]), r["group_id"]


@needs
def test_data_saves_selected_option_and_outcome():
    rows = _rows()
    assert sum(1 for r in rows if r.get("selected_option") is not None) / len(rows) > 0.9
    assert sum(1 for r in rows if r.get("game_outcome") is not None) / len(rows) > 0.9


@needs
def test_data_before_and_afters_have_frozen_width():
    for r in _rows():
        assert len(r["before"]) == DF.N_BASE
        for c in r["candidates"]:
            for a in c["afters"]:
                assert a is None or len(a) == DF.N_BASE


@needs
def test_data_no_duplicate_group_ids():
    ids = [r["group_id"] for r in _rows()]
    assert len(set(ids)) == len(ids)


@needs
def test_data_split_has_no_game_leak():
    import train_soft as TS
    rows = _rows()
    TS.stratified_split(rows)
    per = {}
    for r in rows:
        per.setdefault(r["game"], set()).add(r["_split"])
    assert all(len(v) == 1 for v in per.values())


@needs
def test_data_teacher_blocks_complete():
    for r in _rows():
        for c in r["candidates"]:
            assert len(c["blocks"]) == 4, r["group_id"]


# ---------------- 回帰 ----------------

def test_frozen_anchors_unchanged():
    import hashlib
    exp = {"Q0-expanded.pt": "85ac1a597ebe328b", "S3.pt": "58cbd32b752853ec"}
    for n, s in exp.items():
        assert hashlib.sha256((_HERE / "frozen" / n).read_bytes()).hexdigest()[:16] == s, n
    root = _HERE.parent.parent
    prod = {"sample_submission/configs/abl_5_full.json": "ca6c37af4b1a4149",
            "sample_submission/deck.csv": "8ae7a618b2655669",
            "sample_submission/ptcg_ai/learning/value_weights.json": "d6d7cd8907f53689"}
    for f, s in prod.items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] == s, f


def test_phase12_13_datasets_unchanged():
    import hashlib
    p = _HERE.parent / "_teacher_blocks_p12.jsonl.gz"
    assert hashlib.sha256(p.read_bytes()).hexdigest()[:16] == "d7aec0ca63d1eeb8"
