"""Phase8 §18: Card-Action Cross Attention の形状・不変性・感度テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_q import CrossActionQNet  # noqa: E402

S, O = 166, 65


def _mk(B=3, C=5, N=7, seed=0, **kw):
    torch.manual_seed(seed)
    m = CrossActionQNet(S, O, **kw).eval()
    st = torch.randn(B, S)
    hand = torch.randint(1, 60, (B, N))
    of = torch.randn(B, C, O)
    ci = torch.randint(1, 60, (B, C))
    ot = torch.randint(0, 10, (B, C))
    mk = torch.ones(B, C, dtype=torch.bool)
    return m, (st, hand, of, ci, ot, mk)


# --- Shape ---

@pytest.mark.parametrize("B,C,N", [(1, 1, 1), (2, 3, 5), (4, 8, 20), (3, 2, 1)])
def test_shapes(B, C, N):
    m, a = _mk(B, C, N)
    q, p = m(*a)
    assert q.shape == (B, C) and p.shape == (B, C)


def test_empty_hand_is_safe():
    """全 padding の手札でも NaN を出さない。"""
    m, (st, hand, of, ci, ot, mk) = _mk(2, 3, 4)
    hand = torch.zeros_like(hand)          # 全て padding
    q, _ = m(st, hand, of, ci, ot, mk)
    assert torch.isfinite(q).all()


def test_candidate_mask_applied():
    m, (st, hand, of, ci, ot, mk) = _mk(2, 4, 5)
    mk[:, 2:] = False
    q, _ = m(st, hand, of, ci, ot, mk)
    assert (q[:, 2:] < -1e8).all()


# --- 不変性 ---

def test_hand_order_permutation_invariance():
    """手札 token の順序を変えても Q は一致する(集合として扱えているか)。"""
    m, (st, hand, of, ci, ot, mk) = _mk(3, 4, 6, seed=1)
    q1, _ = m(st, hand, of, ci, ot, mk)
    perm = torch.randperm(hand.shape[1])
    q2, _ = m(st, hand[:, perm], of, ci, ot, mk)
    assert torch.allclose(q1, q2, atol=1e-5)


def test_padding_does_not_change_output():
    """末尾に padding を足しても Q は変わらない。"""
    m, (st, hand, of, ci, ot, mk) = _mk(2, 3, 5, seed=2)
    q1, _ = m(st, hand, of, ci, ot, mk)
    hand2 = torch.cat([hand, torch.zeros(hand.shape[0], 4, dtype=hand.dtype)], dim=1)
    q2, _ = m(st, hand2, of, ci, ot, mk)
    assert torch.allclose(q1, q2, atol=1e-5)


def test_batch_matches_individual():
    m, (st, hand, of, ci, ot, mk) = _mk(4, 3, 6, seed=3)
    qb, _ = m(st, hand, of, ci, ot, mk)
    for b in range(4):
        qi, _ = m(st[b:b + 1], hand[b:b + 1], of[b:b + 1], ci[b:b + 1],
                  ot[b:b + 1], mk[b:b + 1])
        assert torch.allclose(qb[b], qi[0], atol=1e-5)


def test_deterministic():
    m, a = _mk(2, 3, 5, seed=4)
    assert torch.allclose(m(*a)[0], m(*a)[0])


def test_save_load_roundtrip(tmp_path):
    m, a = _mk(2, 3, 5, seed=5)
    q1, _ = m(*a)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = CrossActionQNet(S, O)
    m2.load_state_dict(torch.load(p, weights_only=True))
    m2.eval()
    assert torch.allclose(q1, m2(*a)[0], atol=1e-6)


# --- 感度 ---

def test_attention_varies_with_action():
    """候補行動を変えると Attention 分布が変わる(= Action 依存になっている)。

    ここが変わらないなら Cross Attention は退化しており、pooling と同じ。
    """
    m, (st, hand, of, ci, ot, mk) = _mk(1, 4, 8, seed=6)
    m(st, hand, of, ci, ot, mk)
    w = m.cross.last_weights[0]              # [C, N]
    diffs = [(w[i] - w[j]).abs().max().item()
             for i in range(w.shape[0]) for j in range(i + 1, w.shape[0])]
    assert max(diffs) > 1e-4, "全候補が同じ Attention = 退化"


def test_q_changes_with_hand_identity():
    m, (st, hand, of, ci, ot, mk) = _mk(2, 3, 6, seed=7)
    q1, _ = m(st, hand, of, ci, ot, mk)
    hand2 = hand.clone()
    hand2[:, 0] = hand2[:, 0] + 37           # 別のカードIDへ
    q2, _ = m(st, hand2, of, ci, ot, mk)
    assert not torch.allclose(q1, q2, atol=1e-6)


def test_q_changes_with_action_card_id():
    m, (st, hand, of, ci, ot, mk) = _mk(2, 3, 6, seed=8)
    q1, _ = m(st, hand, of, ci, ot, mk)
    ci2 = ci.clone()
    ci2[:, 0] = ci2[:, 0] + 23
    q2, _ = m(st, hand, of, ci2, ot, mk)
    assert not torch.allclose(q1, q2, atol=1e-6)


def test_no_action_card_id_ignores_it():
    m, (st, hand, of, ci, ot, mk) = _mk(2, 3, 6, seed=9, use_action_card=False)
    q1, _ = m(st, hand, of, ci, ot, mk)
    q2, _ = m(st, hand, of, ci + 11, ot, mk)
    assert torch.allclose(q1, q2, atol=1e-6)


def test_capacity_control_has_no_attention():
    m, _ = _mk(2, 3, 5, capacity_only=True)
    assert not hasattr(m, "cross")


def test_capacity_control_params_close_to_cross():
    a = sum(p.numel() for p in CrossActionQNet(S, O).parameters())
    b = sum(p.numel() for p in CrossActionQNet(S, O, capacity_only=True).parameters())
    assert abs(a - b) / a < 0.10, f"容量対照が離れすぎ: QX={a} QC={b}"
