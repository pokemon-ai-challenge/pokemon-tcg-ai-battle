"""Phase10 §20: Candidate Set Transformer の集合特性・候補間通信テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_q import CandidateSetQNet  # noqa: E402

S, O = 166, 65


def _mk(mode="attn", B=2, C=5, seed=0):
    torch.manual_seed(seed)
    m = CandidateSetQNet(S, O, mode=mode).eval()
    st = torch.randn(B, S)
    zones = [torch.zeros(B, 1, dtype=torch.long)] * 3
    of = torch.randn(B, C, O)
    ci = torch.randint(1, 60, (B, C))
    ot = torch.randint(0, 10, (B, C))
    mk = torch.ones(B, C, dtype=torch.bool)
    return m, (st, zones, of, ci, ot, mk)


@pytest.mark.parametrize("C", [2, 3, 5, 7, 8, 12])
def test_shapes_variable_candidates(C):
    m, a = _mk(C=C)
    q, p = m(*a)
    assert q.shape == (2, C) and p.shape == (2, C)


def test_permutation_equivariance():
    """候補順序を並べ替えたら、出力Qも同じ順序で並べ替わること(集合として扱えている)。"""
    m, (st, z, of, ci, ot, mk) = _mk(B=2, C=6, seed=1)
    q1, _ = m(st, z, of, ci, ot, mk)
    perm = torch.randperm(6)
    q2, _ = m(st, z, of[:, perm], ci[:, perm], ot[:, perm], mk[:, perm])
    inv = torch.argsort(perm)
    assert torch.allclose(q1, q2[:, inv], atol=1e-5)


def test_padding_invariance():
    """padding 候補を足しても実候補の Q が変わらないこと。"""
    m, (st, z, of, ci, ot, mk) = _mk(B=2, C=4, seed=2)
    q1, _ = m(st, z, of, ci, ot, mk)
    pad = 3
    of2 = torch.cat([of, torch.randn(2, pad, O)], 1)
    ci2 = torch.cat([ci, torch.randint(1, 60, (2, pad))], 1)
    ot2 = torch.cat([ot, torch.randint(0, 10, (2, pad))], 1)
    mk2 = torch.cat([mk, torch.zeros(2, pad, dtype=torch.bool)], 1)
    q2, _ = m(st, z, of2, ci2, ot2, mk2)
    assert torch.allclose(q1, q2[:, :4], atol=1e-5)


def test_batch_matches_individual():
    m, (st, z, of, ci, ot, mk) = _mk(B=4, C=5, seed=3)
    qb, _ = m(st, z, of, ci, ot, mk)
    for b in range(4):
        zb = [x[b:b + 1] for x in z]
        qi, _ = m(st[b:b + 1], zb, of[b:b + 1], ci[b:b + 1], ot[b:b + 1], mk[b:b + 1])
        assert torch.allclose(qb[b], qi[0], atol=1e-5)


def test_save_load(tmp_path):
    m, a = _mk(seed=4)
    q1, _ = m(*a)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = CandidateSetQNet(S, O, mode="attn")
    m2.load_state_dict(torch.load(p, weights_only=True))
    m2.eval()
    assert torch.allclose(q1, m2(*a)[0], atol=1e-6)


def test_candidate_communication_attn():
    """他候補を差し替えると対象候補の Q が変わること(= 候補間通信が起きている)。"""
    m, (st, z, of, ci, ot, mk) = _mk(mode="attn", B=1, C=5, seed=5)
    q1, _ = m(st, z, of, ci, ot, mk)
    of2 = of.clone()
    of2[:, 1:] = torch.randn_like(of2[:, 1:])      # 候補0 以外を変更
    q2, _ = m(st, z, of2, ci, ot, mk)
    assert not torch.allclose(q1[:, 0], q2[:, 0], atol=1e-6), "候補0のQが不変=通信なし"


def test_no_communication_in_capacity_and_none():
    """QC / Q0 相当は候補間通信を持たない(他候補を変えても対象候補のQは不変)。"""
    for mode in ("none", "capacity"):
        m, (st, z, of, ci, ot, mk) = _mk(mode=mode, B=1, C=5, seed=6)
        q1, _ = m(st, z, of, ci, ot, mk)
        of2 = of.clone()
        of2[:, 1:] = torch.randn_like(of2[:, 1:])
        q2, _ = m(st, z, of2, ci, ot, mk)
        assert torch.allclose(q1[:, 0], q2[:, 0], atol=1e-6), f"{mode} が通信している"


def test_identity_mode_has_no_communication():
    """identity は self のみ参照 = 候補間通信なし(ブロック追加効果だけを分離)。"""
    m, (st, z, of, ci, ot, mk) = _mk(mode="identity", B=1, C=5, seed=7)
    q1, _ = m(st, z, of, ci, ot, mk)
    of2 = of.clone()
    of2[:, 1:] = torch.randn_like(of2[:, 1:])
    q2, _ = m(st, z, of2, ci, ot, mk)
    assert torch.allclose(q1[:, 0], q2[:, 0], atol=1e-5)


def test_mean_mode_communicates():
    m, (st, z, of, ci, ot, mk) = _mk(mode="mean", B=1, C=5, seed=8)
    q1, _ = m(st, z, of, ci, ot, mk)
    of2 = of.clone()
    of2[:, 1:] = torch.randn_like(of2[:, 1:])
    q2, _ = m(st, z, of2, ci, ot, mk)
    assert not torch.allclose(q1[:, 0], q2[:, 0], atol=1e-6)


def test_padding_not_attended():
    """padding 候補は Attention の Key から除外されること。"""
    m, (st, z, of, ci, ot, mk) = _mk(mode="attn", B=1, C=6, seed=9)
    mk[:, 4:] = False
    m(st, z, of, ci, ot, mk)
    w = m.last_attn[0]              # [C, C]
    assert w[:, 4:].abs().max().item() < 1e-6, "padding へ attention が漏れている"


def test_capacity_matches_attn_params():
    a = sum(p.numel() for p in CandidateSetQNet(S, O, mode="attn").parameters())
    c = sum(p.numel() for p in CandidateSetQNet(S, O, mode="capacity").parameters())
    assert abs(a - c) / a < 0.10, f"容量対照が離れすぎ QT={a} QC={c}"


# --- Phase10 で踏んだバグの回帰テスト(§16)---
# identity(self のみ許可の attn_mask)と key_padding_mask を併用すると、
# padding 候補の行が「唯一許可された self も padding」で全マスクになり NaN が出た。
# 初回測定が pairwise 0.0 になった原因。以後この組み合わせを機械的に検出する。

@pytest.mark.parametrize("mode", ["attn", "identity"])
@pytest.mark.parametrize("n_real,C", [(1, 5), (2, 6), (4, 4), (1, 2), (7, 8)])
def test_no_nan_with_padding(mode, n_real, C):
    m, (st, z, of, ci, ot, mk) = _mk(mode=mode, B=3, C=C, seed=11)
    mk[:] = False
    mk[:, :n_real] = True
    q, p = m(st, z, of, ci, ot, mk)
    real = q[:, :n_real]
    assert torch.isfinite(real).all(), f"{mode}/n_real={n_real} で NaN/inf"
    assert torch.isfinite(p[:, :n_real]).all()
    assert not torch.isnan(q).any() or (q[:, n_real:] < -1e8).all()


@pytest.mark.parametrize("mode", ["attn", "identity", "capacity", "mean", "none"])
def test_all_real_candidates_finite(mode):
    m, a = _mk(mode=mode, B=2, C=6, seed=12)
    q, _ = m(*a)
    assert torch.isfinite(q).all()


def test_single_real_candidate_is_safe():
    """実候補1件のみ(残り全 padding)でも有限値を返す。"""
    for mode in ("attn", "identity"):
        m, (st, z, of, ci, ot, mk) = _mk(mode=mode, B=2, C=5, seed=13)
        mk[:] = False
        mk[:, 0] = True
        q, _ = m(st, z, of, ci, ot, mk)
        assert torch.isfinite(q[:, 0]).all()
