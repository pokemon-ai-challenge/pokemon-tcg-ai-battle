"""Phase12 §23: soft target / confidence / loss / evaluation の単体テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import soft_eval as SE  # noqa: E402
import soft_target as ST  # noqa: E402
import train_soft as TS  # noqa: E402


def mkgroup(blocks, gid="g0", game=0, n_feat=8, n_opt=4):
    """blocks[c] = そのcandidateの4block score。"""
    return {"group_id": gid, "game": game, "turn": 6, "turn_band": "middle",
            "cand_band": "small", "arch": "a", "stratum": 0,
            "state_feat": [0.0] * n_feat,
            "candidates": [{"option_index": c, "option_feat": [0.0] * n_opt,
                            "action_card_id": -1, "option_type": 0,
                            "policy_score": 0.0, "policy_rank": c, "blocks": b}
                           for c, b in enumerate(blocks)]}


# ---------------- Soft target (§23 Soft target) ----------------

def test_4of4_gives_hard_target():
    p = ST.group_pairs(mkgroup([[0.9] * 4, [0.1] * 4]))[0]
    assert p["support"] == 1.0
    assert p["cls"] == "stable"
    assert ST.arm_target(p, "S2")[0] == 1.0


def test_4of4_reversed_gives_zero():
    p = ST.group_pairs(mkgroup([[0.1] * 4, [0.9] * 4]))[0]
    assert p["support"] == 0.0


def test_3of4_gives_075():
    p = ST.group_pairs(mkgroup([[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]))[0]
    assert p["support"] == 0.75
    assert p["cls"] == "mostly"


def test_1of4_gives_025():
    p = ST.group_pairs(mkgroup([[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]]))[0]
    assert p["support"] == 0.25


def test_2of2_gives_05():
    p = ST.group_pairs(mkgroup([[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]]))[0]
    assert p["support"] == 0.5
    assert p["cls"] == "unstable"


def test_all_tie_votes_give_05():
    p = ST.group_pairs(mkgroup([[0.5] * 4, [0.5] * 4]))[0]
    assert p["support"] == 0.5
    assert p["cls"] == "always_tie"


def test_tie_vote_counts_as_half():
    # 3block で i 勝ち, 1block 同点 -> (1+1+1+0.5)/4
    p = ST.group_pairs(mkgroup([[0.9, 0.9, 0.9, 0.5], [0.1, 0.1, 0.1, 0.5]]))[0]
    assert p["support"] == pytest.approx(0.875)


def test_candidate_order_reversal_flips_target():
    a = ST.group_pairs(mkgroup([[0.9] * 4, [0.2] * 4]))[0]
    b = ST.group_pairs(mkgroup([[0.2] * 4, [0.9] * 4]))[0]
    assert a["support"] + b["support"] == pytest.approx(1.0)
    assert a["mean_margin"] == pytest.approx(-b["mean_margin"])


def test_tie_epsilon_boundary():
    assert ST.vote(0.004) == 0.5
    assert ST.vote(0.006) == 1.0
    assert ST.vote(-0.006) == 0.0


# ---------------- Confidence (§23 Confidence) ----------------

@pytest.mark.parametrize("p,c", [(1.0, 1.0), (0.75, 0.5), (0.5, 0.0),
                                 (0.25, 0.5), (0.0, 1.0)])
def test_confidence_values(p, c):
    assert ST.confidence(p) == pytest.approx(c)


def test_confidence_symmetry():
    for p in (0.0, 0.125, 0.25, 0.375, 0.5):
        assert ST.confidence(p) == pytest.approx(ST.confidence(1.0 - p))


def test_margin_weight_clip():
    assert ST.margin_weight(0.0) == 0.0
    assert ST.margin_weight(0.035) == pytest.approx(0.5)
    assert ST.margin_weight(0.07) == 1.0
    assert ST.margin_weight(0.5) == 1.0
    assert ST.margin_weight(-0.5) == 1.0


def test_margin_bands_use_phase11d_bounds():
    assert ST.margin_band(0.005) == "very_small"
    assert ST.margin_band(0.02) == "small"
    assert ST.margin_band(0.05) == "medium"
    assert ST.margin_band(0.10) == "large"


# ---------------- arm target/weight ----------------

def test_S1_excludes_split_pairs_and_hardens_others():
    split = ST.group_pairs(mkgroup([[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]]))[0]
    assert ST.arm_target(split, "S1")[1] == 0.0
    mostly = ST.group_pairs(mkgroup([[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]))[0]
    assert ST.arm_target(mostly, "S1") == (1.0, 1.0)


def test_S0_uses_single_block_only():
    # block0 だけ i 勝ち。S0 は block0 の hard label を使う
    g = mkgroup([[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]])
    p = ST.group_pairs(g)[0]
    assert ST.arm_target(p, "S0") == (1.0, 1.0)
    assert ST.arm_target(p, "S2")[0] == 0.25


def test_S3_weight_is_confidence():
    p = ST.group_pairs(mkgroup([[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]]))[0]
    assert ST.arm_target(p, "S3") == (0.5, 0.0)          # 2/2 は順位Lossを与えない
    q = ST.group_pairs(mkgroup([[0.9] * 4, [0.1] * 4]))[0]
    assert ST.arm_target(q, "S3") == (1.0, 1.0)


# ---------------- Loss (§23 Loss) ----------------

def _loss(qvals, tgt_pairs, mask=None):
    C = len(qvals)
    q = torch.tensor([qvals], dtype=torch.float32)
    t = torch.zeros(1, C, C)
    w = torch.zeros(1, C, C)
    for (i, j, tv, wv) in tgt_pairs:
        t[0, i, j] = tv
        w[0, i, j] = wv
    m = torch.ones(1, C, dtype=torch.bool) if mask is None else torch.tensor([mask])
    return float(TS.pair_loss(q, t, w, m))


def test_loss_decreases_toward_target():
    good = _loss([3.0, -3.0], [(0, 1, 1.0, 1.0)])
    bad = _loss([-3.0, 3.0], [(0, 1, 1.0, 1.0)])
    assert good < bad


def test_near_tie_penalises_extreme_q_gap():
    """target 0.5 では Q 差が大きいほど Loss が大きい(near-tie を無理に割らない)。"""
    flat = _loss([0.0, 0.0], [(0, 1, 0.5, 1.0)])
    wide = _loss([4.0, -4.0], [(0, 1, 0.5, 1.0)])
    assert flat < wide
    assert flat == pytest.approx(0.6931, abs=1e-3)


def test_stable_pair_learned_more_strongly_than_mostly():
    """同じ Q 差でも target 1.0 の方が 0.75 より正方向の勾配が強い。"""
    q = torch.tensor([[0.0, 0.0]], requires_grad=True)
    t = torch.zeros(1, 2, 2)
    w = torch.zeros(1, 2, 2)
    t[0, 0, 1], w[0, 0, 1] = 1.0, 1.0
    m = torch.ones(1, 2, dtype=torch.bool)
    TS.pair_loss(q, t, w, m).backward()
    g_stable = float(q.grad[0, 0])
    q2 = torch.tensor([[0.0, 0.0]], requires_grad=True)
    t[0, 0, 1] = 0.75
    TS.pair_loss(q2, t, w, m).backward()
    assert abs(g_stable) > abs(float(q2.grad[0, 0]))


def test_zero_weight_pairs_excluded():
    # (0,2) は weight 0 なので、どんな target でも損失に寄与してはならない
    a = _loss([3.0, -3.0, 5.0], [(0, 1, 1.0, 1.0), (0, 2, 1.0, 0.0)])
    b = _loss([3.0, -3.0, 5.0], [(0, 1, 1.0, 1.0)])
    assert a == pytest.approx(b)


def test_padding_candidates_excluded_by_mask():
    with_pad = _loss([3.0, -3.0, 99.0],
                     [(0, 1, 1.0, 1.0), (0, 2, 1.0, 1.0)], mask=[True, True, False])
    plain = _loss([3.0, -3.0], [(0, 1, 1.0, 1.0)])
    assert with_pad == pytest.approx(plain)


def test_loss_no_nan_on_extreme_values():
    for v in (1e3, -1e3, 1e6):
        out = _loss([v, -v], [(0, 1, 1.0, 1.0), (1, 0, 0.5, 0.5)])
        assert out == out and abs(out) < 1e6


def test_loss_zero_total_weight_is_finite_and_differentiable():
    q = torch.tensor([[0.0, 1.0]], requires_grad=True)
    z = torch.zeros(1, 2, 2)
    out = TS.pair_loss(q, z, z, torch.ones(1, 2, dtype=torch.bool))
    out.backward()
    assert float(out) == 0.0 and q.grad is not None


# ---------------- Evaluation (§23 Evaluation) ----------------

def _eg():
    # c0 が全block勝ち(stable, large margin), c1 vs c2 は 2/2(unstable, near tie)
    return mkgroup([[0.90, 0.90, 0.90, 0.90],
                    [0.50, 0.50, 0.502, 0.498],
                    [0.502, 0.498, 0.50, 0.50]])


def test_stable_metric_counts_only_stable_pairs():
    g = _eg()
    m = SE.evaluate([g], lambda x: [3.0, 1.0, 0.0])
    assert m["stable"] == 1.0
    assert m["stable_n"] == 2                     # c0-c1, c0-c2
    assert m["large_margin"] == 1.0


def test_wrong_order_scores_zero_on_stable():
    m = SE.evaluate([_eg()], lambda x: [0.0, 1.0, 3.0])
    assert m["stable"] == 0.0


def test_support_weighted_ignores_zero_confidence_pairs():
    m = SE.evaluate([_eg()], lambda x: [3.0, 1.0, 0.0])
    assert m["support_weighted"] == 1.0           # 2/2 pair は confidence 0 で除外


def test_regret_is_zero_when_top1_matches():
    m = SE.evaluate([_eg()], lambda x: [3.0, 1.0, 0.0])
    assert m["regret"] == pytest.approx(0.0)
    assert m["top1"] == 1.0


def test_regret_positive_when_top1_wrong():
    m = SE.evaluate([_eg()], lambda x: [0.0, 1.0, 3.0])
    assert m["regret"] > 0.3


def test_calibration_bins_report_predictions():
    m = SE.evaluate([_eg()], lambda x: [3.0, 1.0, 0.0])
    assert "1.0" in m["calibration"] and "0.5" in m["calibration"]
    assert m["calibration"]["1.0"]["pred_mean"] > 0.5
    assert m["calibration_error"] is not None


def test_abs_q_diff_by_margin_reported():
    m = SE.evaluate([_eg()], lambda x: [3.0, 1.0, 0.0])
    assert m["abs_q_diff_by_margin"]["large"] is not None
    assert m["abs_q_diff_by_margin"]["very_small"] == pytest.approx(1.0)


def test_bootstrap_group_level_resampling():
    pg = SE.evaluate([_eg(), _eg()], lambda x: [3.0, 1.0, 0.0])["_per_group"]
    pg2 = SE.evaluate([_eg(), _eg()], lambda x: [0.0, 1.0, 3.0])["_per_group"]
    b = TS.paired_bootstrap(pg, pg2, "stable", B=500)
    assert b["mean_diff"] == pytest.approx(1.0)
    assert b["n_groups"] == 2


def test_bootstrap_regret_uses_group_values():
    pg = SE.evaluate([_eg()], lambda x: [3.0, 1.0, 0.0])["_per_group"]
    b = TS.paired_bootstrap(pg, pg, "regret", B=200)
    assert b["mean_diff"] == 0.0


# ---------------- split (§12) ----------------

def test_split_keeps_games_together():
    groups = [mkgroup([[0.9] * 4, [0.1] * 4], gid=f"g{i}", game=i // 3) for i in range(30)]
    TS.stratified_split(groups)
    per_game = {}
    for g in groups:
        per_game.setdefault(g["game"], set()).add(g["_split"])
    assert all(len(v) == 1 for v in per_game.values())


def test_split_covers_three_partitions():
    groups = [mkgroup([[0.9] * 4, [0.1] * 4], gid=f"g{i}", game=i) for i in range(60)]
    TS.stratified_split(groups)
    assert {g["_split"] for g in groups} == {0, 1, 2}
