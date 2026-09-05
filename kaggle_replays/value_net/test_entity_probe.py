"""Phase14 §32: feature 抽出 / probe / collision / recall のテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import _collision_audit as CA  # noqa: E402
import _entity_extract as EX  # noqa: E402
import entity_probe as EP  # noqa: E402
import train_probe as TP  # noqa: E402


def pk(cid, hp=100, mx=100, en=(), tools=(), pre=(), new=False):
    return {"id": cid, "hp": hp, "max_hp": mx, "dmg": mx - hp, "new": new,
            "en": list(en), "tools": list(tools), "pre": list(pre)}


def ent(hand=(10, 10, 11), sb=(101, 102), ob=(201,), disc=(50, 51),
        odisc=(60,), opp_hand=None):
    e = {"turn": 5, "me": 0,
         "self": {"active": pk(100, 80, 120, en=(1, 1)), "bench": [pk(c) for c in sb],
                  "discard": list(disc), "deck_count": 30, "hand_count": len(hand),
                  "prize_count": 6, "status": [0] * 5, "hand": list(hand)},
         "opp": {"active": pk(200, 150, 200, en=(0,), tools=(900,)),
                 "bench": [pk(c) for c in ob], "discard": list(odisc),
                 "deck_count": 28, "hand_count": 5, "prize_count": 5,
                 "status": [0] * 5}}
    if opp_hand is not None:
        e["opp"]["hand"] = list(opp_hand)
    return e


# ---------------- Feature extraction ----------------

def test_hand_bag_preserves_multiplicity():
    assert sorted(EP.hand_bag(ent(hand=(10, 10, 11)))) == [10, 10, 11]


def test_hand_bag_drops_out_of_vocab_and_zero():
    assert EP.hand_bag(ent(hand=(0, 99999, 12))) == [12]


def test_board_tokens_active_first_and_slot_encoded():
    t = EP.board_tokens(ent(), "self")
    assert len(t) == 3                                # active + bench2
    assert t[0][1][0] == 1.0 and t[1][1][0] == 0.0    # is_active
    assert t[0][0][0] == 100                          # active card id


def test_board_tokens_carry_evolution_stage_and_tool():
    e = ent()
    e["self"]["active"] = pk(300, pre=(298, 299), tools=(901,))
    t = EP.board_tokens(e, "self")
    assert t[0][0][1] == 901                          # tool id
    assert t[0][1][7] == pytest.approx(1.0)           # stage = len(pre)/2


def test_board_tokens_skip_missing_active():
    e = ent()
    e["self"]["active"] = None
    t = EP.board_tokens(e, "self")
    assert len(t) == 2 and all(x[1][0] == 0.0 for x in t)


def test_unseen_deck_subtracts_seen_cards():
    # ent() の active は card 100。bench に 101 を置き、手札に 10 を 2 枚持たせる。
    deck = [10] * 4 + [100] * 2 + [101] * 2 + [777] * 3
    e = ent(hand=(10, 10), sb=(101,), disc=())
    bag = EP.unseen_deck_bag(e, deck)
    from collections import Counter
    c = Counter(bag)
    assert c[10] == 2          # 4 - 2(手札)
    assert c[100] == 1         # 2 - 1(バトル場)
    assert c[101] == 1         # 2 - 1(ベンチ)
    assert c[777] == 3


def test_unseen_deck_never_negative():
    e = ent(hand=(10,) * 6)
    assert all(x != 10 for x in EP.unseen_deck_bag(e, [10] * 2))


def test_history_tokens_preserve_order_and_recency():
    h = [{"type": i, "sel_type": 0, "card_id": 100 + i, "turn": i} for i in range(10)]
    t = EP.history_tokens(h, n=4)
    assert len(t) == 4
    assert [x[0][0] for x in t] == [106, 107, 108, 109]     # 直近 4 手・時系列順
    assert t[-1][1][2] > t[0][1][2]                        # recency が単調


def test_delta_vec_matches_after_minus_before():
    before = [1.0] * EX.SUMMARY_KEYS.__len__()
    after = list(before)
    after[7] -= 1.0
    d = EP.delta_vec(before, after)
    assert d[7] == pytest.approx(-1.0 / 20.0)
    assert d[-1] == 1.0                                    # 有効フラグ


def test_delta_vec_when_after_missing():
    d = EP.delta_vec([1.0] * len(EX.SUMMARY_KEYS), None)
    assert d[-1] == 0.0 and all(x == 0.0 for x in d[:-1])


def test_summary_keys_length_matches_model():
    assert len(EX.SUMMARY_KEYS) == EP.N_SUMMARY


# ---------------- 情報リーク(§27) ----------------

def test_featurizers_ignore_opponent_hand_entirely():
    """相手手札を entity に混ぜても、どの特徴量も変化してはならない。"""
    a, b = ent(), ent(opp_hand=[1234, 1235, 1236])
    deck = [10] * 4 + [100] * 4
    assert EP.hand_bag(a) == EP.hand_bag(b)
    assert EP.board_tokens(a, "opp") == EP.board_tokens(b, "opp")
    assert EP.discard_bag(a, "opp") == EP.discard_bag(b, "opp")
    assert EP.unseen_deck_bag(a, deck) == EP.unseen_deck_bag(b, deck)


def test_extractor_does_not_emit_opponent_hand_key():
    class P:
        active = []
        bench = []
        discard = []
        deckCount = 10
        handCount = 4
        prize = []
        hand = [type("C", (), {"id": 5})()]
        poisoned = burned = asleep = paralyzed = confused = False
    side = EX._side(P(), own=False)
    assert "hand" not in side
    assert "hand" in EX._side(P(), own=True)


# ---------------- Probe モデル ----------------

def _batch(fam, B=3, C=4, sdim=166, odim=65):
    g = {"state_feat": [0.1] * sdim, "entity": ent(),
         "history": [{"type": 1, "sel_type": 0, "card_id": 10, "turn": 1}],
         "before": [1.0] * EP.N_SUMMARY,
         "candidates": [{"option_feat": [0.0] * odim, "action_card_id": 10 + i,
                         "option_type": 1, "policy_score": 0.0, "policy_rank": i,
                         "blocks": [0.5 + 0.01 * i] * 4,
                         "after": [1.0] * EP.N_SUMMARY} for i in range(C)]}
    import soft_target as ST
    g["_pairs"] = ST.group_pairs(g)
    gs = [dict(g) for _ in range(B)]
    TP.featurize(gs, [10] * 4)
    return TP.make_batch(gs, np.zeros(sdim, np.float32), np.ones(sdim, np.float32),
                         fam, [10] * 4)


@pytest.mark.parametrize("name,fam", list(TP.ARMS.items()))
def test_probe_forward_shape_and_finite(name, fam):
    b = _batch(fam)
    m = EP.ProbeNet(166, 65, families=fam, hidden=16)
    q = m(b)
    assert q.shape == (3, 4)
    assert torch.isfinite(q).all()


def test_probe_masked_candidates_are_excluded():
    b = _batch(())
    b["mask"][:, 3] = False
    q = EP.ProbeNet(166, 65, families=(), hidden=16)(b)
    assert (q[:, 3] < -1e8).all()


def test_probe_same_seed_reproduces_weights():
    torch.manual_seed(7)
    a = EP.ProbeNet(166, 65, families=("board",), hidden=16)
    torch.manual_seed(7)
    c = EP.ProbeNet(166, 65, families=("board",), hidden=16)
    for p, q in zip(a.parameters(), c.parameters()):
        assert torch.equal(p, q)


def test_probe_families_change_parameter_count():
    base = sum(p.numel() for p in EP.ProbeNet(166, 65, (), 16).parameters())
    for fam in ("hand", "board", "history"):
        assert sum(p.numel() for p in
                   EP.ProbeNet(166, 65, (fam,), 16).parameters()) > base


def test_batch_padding_shapes():
    b = _batch(("board", "history", "hand", "opp", "deck", "delta"))
    assert b["board_ids"].shape[-1] == 2 and b["board_num"].shape[-1] == 10
    assert b["hist_ids"].shape[-1] == 1 and b["hist_num"].shape[-1] == 3
    assert b["delta"].shape[-1] == EP.N_SUMMARY + 1
    assert b["board_mask"].sum() > 0


def test_gradients_flow_to_added_family():
    b = _batch(("board",))
    m = EP.ProbeNet(166, 65, families=("board",), hidden=16)
    q = m(b).masked_fill(~b["mask"], 0.0)
    q.sum().backward()
    g = m.enc["board"].emb.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------- Collision ----------------

def test_entity_distance_zero_for_identical():
    assert CA.entity_distance(ent(), ent()) == 0


def test_entity_distance_counts_hand_difference():
    assert CA.entity_distance(ent(hand=(10, 10)), ent(hand=(10, 12))) == 2


def test_board_identity_diff_detects_bench_change():
    assert CA.board_identity_diff(ent(sb=(101, 102)), ent(sb=(101, 103))) == 1
    assert CA.board_identity_diff(ent(), ent()) == 0


def test_hand_identity_diff_symmetric():
    a, b = ent(hand=(1, 2, 3)), ent(hand=(1, 2))
    assert CA.hand_identity_diff(a, b) == CA.hand_identity_diff(b, a) == 1


# ---------------- 回帰: 凍結 SHA ----------------

def test_frozen_checkpoints_unchanged():
    import hashlib
    exp = {"Q0-expanded.pt": "85ac1a597ebe328b", "S3.pt": "58cbd32b752853ec"}
    for n, s in exp.items():
        p = _HERE / "frozen" / n
        assert hashlib.sha256(p.read_bytes()).hexdigest()[:16] == s, n


def test_production_anchors_unchanged():
    import hashlib
    root = _HERE.parent.parent
    exp = {"sample_submission/configs/abl_5_full.json": "ca6c37af4b1a4149",
           "sample_submission/deck.csv": "8ae7a618b2655669",
           "sample_submission/ptcg_ai/learning/value_weights.json": "d6d7cd8907f53689"}
    for f, s in exp.items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] == s, f
