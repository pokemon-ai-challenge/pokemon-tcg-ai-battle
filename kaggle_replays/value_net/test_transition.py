"""Phase17 §50: entity tokenize / relation / Transformer / 対照の妥当性テスト。"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import entity_tokens as ET  # noqa: E402
import train_transition as TT  # noqa: E402
import transition_net as TN  # noqa: E402

DATA = str(_HERE.parent / "_p17_probes_w*.jsonl.gz")
needs = pytest.mark.skipif(not glob.glob(DATA), reason="Phase17 データ未生成")


def pk(cid, hp=100, mx=100, en=(), tools=(), pre=(), new=False):
    return {"id": cid, "hp": hp, "max_hp": mx, "dmg": mx - hp, "new": new,
            "en": list(en), "tools": list(tools), "pre": list(pre)}


def ent(hand=(10, 10, 11), sb=(101, 102), ob=(201,), disc=(50, 50, 51),
        odisc=(60,), opp_hand=None, active=None):
    e = {"turn": 5, "me": 0,
         "self": {"active": active or pk(100, 80, 120, en=(1, 1)),
                  "bench": [pk(c) for c in sb], "discard": list(disc),
                  "deck_count": 30, "hand_count": len(hand), "prize_count": 6,
                  "status": [0] * 5, "hand": list(hand)},
         "opp": {"active": pk(200, 150, 200, en=(0,), tools=(900,)),
                 "bench": [pk(c) for c in ob], "discard": list(odisc),
                 "deck_count": 28, "hand_count": 5, "prize_count": 5,
                 "status": [0] * 5}}
    if opp_hand is not None:
        e["opp"]["hand"] = list(opp_hand)
    return e


# ---------------- Entity extraction / tokenize ----------------

def test_state_token_is_first():
    t = ET.tokenize(ent())
    assert t["type"][0] == ET.TYPE_STATE and t["key"][0] == "STATE"


def test_pokemon_tokens_carry_owner_zone_slot():
    t = ET.tokenize(ent())
    pk_idx = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON]
    assert len(pk_idx) == 3 + 2                      # self a+2bench, opp a+1bench
    i = pk_idx[0]
    assert t["owner"][i] == ET.OWN_SELF and t["zone"][i] == ET.ZONE_ACTIVE
    assert t["slot"][pk_idx[1]] == 1


def test_energy_type_vector_is_encoded():
    a = ET.tokenize(ent(active=pk(100, en=(1, 1))))
    b = ET.tokenize(ent(active=pk(100, en=(1, 2))))
    ia = [i for i, x in enumerate(a["type"]) if x == ET.TYPE_POKEMON][0]
    assert a["num"][ia] != b["num"][ia]              # 同数でも型構成が違えば別表現


def test_tool_and_evolution_encoded():
    t = ET.tokenize(ent(active=pk(300, tools=(901,), pre=(298, 299))))
    i = [k for k, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON][0]
    assert t["num"][i][9] > 0                        # 進化段数
    assert t["num"][i][8] == 1.0                     # tool 数
    assert t["pre"][i] == [298, 299]


def test_hand_tokens_are_compressed_with_count():
    t = ET.tokenize(ent(hand=(10, 10, 11)))
    h = [(t["card_id"][i], t["num"][i][0]) for i, x in enumerate(t["type"])
         if x == ET.TYPE_HAND]
    assert (10, 0.5) in h and (11, 0.25) in h        # count/4


def test_discard_tokens_for_both_players():
    t = ET.tokenize(ent())
    own = {t["owner"][i] for i, x in enumerate(t["type"]) if x == ET.TYPE_DISCARD}
    assert own == {ET.OWN_SELF, ET.OWN_OPP}


def test_missing_active_is_skipped():
    e = ent(); e["self"]["active"] = None
    t = ET.tokenize(e)
    pk_idx = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON]
    assert all(not (t["owner"][i] == ET.OWN_SELF and t["zone"][i] == ET.ZONE_ACTIVE)
               for i in pk_idx)


# ---------------- Hidden leak (§7) ----------------

def test_opponent_hand_never_tokenized():
    a = ET.tokenize(ent())
    b = ET.tokenize(ent(opp_hand=[1234, 1235, 1236]))
    assert a == b


def test_no_opponent_hand_token_type_exists():
    t = ET.tokenize(ent(opp_hand=[1234]))
    for i, ty in enumerate(t["type"]):
        assert not (ty == ET.TYPE_HAND and t["owner"][i] == ET.OWN_OPP)


# ---------------- Relation (§15) ----------------

def test_relation_self_and_state():
    t = ET.tokenize(ent()); R = ET.relation_matrix(t)
    assert R[1][1] == ET.REL_SELF
    assert R[0][3] == ET.REL_STATE and R[3][0] == ET.REL_STATE


def test_relation_opposing_and_same_owner():
    t = ET.tokenize(ent()); R = ET.relation_matrix(t)
    pk_idx = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON]
    s = [i for i in pk_idx if t["owner"][i] == ET.OWN_SELF]
    o = [i for i in pk_idx if t["owner"][i] == ET.OWN_OPP]
    assert R[s[0]][o[0]] == ET.REL_OPPOSING
    assert R[s[0]][s[1]] == ET.REL_ACTIVE_BENCH


def test_relation_hand_board():
    t = ET.tokenize(ent()); R = ET.relation_matrix(t)
    h = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_HAND][0]
    p = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON
         and t["owner"][i] == ET.OWN_SELF][0]
    assert R[h][p] == ET.REL_HAND_BOARD and R[p][h] == ET.REL_HAND_BOARD


def test_relation_evolution_detected():
    e = ent(hand=(298,), active=pk(300, pre=(298, 299)))
    t = ET.tokenize(e); R = ET.relation_matrix(t)
    h = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_HAND][0]
    p = [i for i, x in enumerate(t["type"]) if x == ET.TYPE_POKEMON][0]
    assert R[h][p] == ET.REL_EVOLUTION


def test_relation_matrix_is_square_and_in_range():
    t = ET.tokenize(ent()); R = ET.relation_matrix(t)
    T = len(t["card_id"])
    assert len(R) == T and all(len(r) == T for r in R)
    assert all(0 <= v < ET.N_REL for r in R for v in r)


# ---------------- Transformer ----------------

def _grp(C=3, seed=0):
    g = {"state_feat": [0.1] * 166, "entity": ent(), "before": [10.0] * 22,
         "turn": 5, "turn_band": "early", "game": seed, "arch": "a",
         "candidates": [{"option_feat": [0.0] * 65, "action_card_id": 10 + i,
                         "option_type": 7, "policy_score": 0.0, "policy_rank": i,
                         "blocks": [0.5 + 0.02 * i] * 4,
                         "afters": [[10.0 + i] + [10.0] * 21] * 4,
                         "after_entity": ent(hand=(10 + i, 20 + i))} for i in range(C)]}
    import soft_target as ST
    g["_pairs"] = ST.group_pairs(g)
    return g


@pytest.mark.parametrize("arm", list(TT.ARMS))
def test_forward_shape_and_finite(arm):
    gs = [_grp(), _grp(seed=1)]
    TT.prepare(gs)
    b = TT.make_batch(gs, np.zeros(166, np.float32), np.ones(166, np.float32), arm)
    m = TT.build_model(arm, 166, 65)
    q = m(b)
    assert q.shape == (2, 3) and torch.isfinite(q).all()


def test_before_after_encoder_is_shared():
    m = TN.TransitionQNet(65, mode="transition")
    assert sum(1 for _ in m.enc.parameters()) > 0
    names = [n for n, _ in m.named_parameters() if "enc." in n]
    assert names and not any("enc_after" in n for n in names)


def test_identical_state_gives_zero_latent_delta():
    g = _grp(C=2)
    for c in g["candidates"]:
        c["after_entity"] = g["entity"]              # after == before
    TT.prepare([g])
    b = TT.make_batch([g], np.zeros(166, np.float32), np.ones(166, np.float32), "T4")
    m = TN.TransitionQNet(65, mode="transition")
    m.eval()
    with torch.no_grad():
        zb = m.enc(b["before"])
        flat = {k: v.reshape(2, *v.shape[2:]) for k, v in b["after"].items()}
        za = m.enc(flat)
    assert torch.allclose(zb.expand_as(za), za, atol=1e-5)


def test_deterministic_output_for_identical_input():
    gs = [_grp()]
    TT.prepare(gs)
    b = TT.make_batch(gs, np.zeros(166, np.float32), np.ones(166, np.float32), "T4")
    torch.manual_seed(0)
    m = TN.TransitionQNet(65, mode="transition")
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(b), m(b))


def test_padding_invariance():
    """padding token を増やしても STATE 出力が変わらない。"""
    g = _grp()
    TT.prepare([g])
    b1 = TT.make_batch([g], np.zeros(166, np.float32), np.ones(166, np.float32), "T2")
    m = TN.TransitionQNet(65, mode="current")
    m.eval()
    with torch.no_grad():
        z1 = m.enc(b1["before"])
    T = b1["before"]["card_id"].shape[1]
    pad = {k: (torch.cat([v, torch.zeros_like(v[:, :3])], 1) if v.dim() == 2 else
               (torch.cat([v, torch.zeros_like(v[:, :3, :])], 1) if k == "num" else
                torch.nn.functional.pad(v, (0, 3, 0, 3))))
           for k, v in b1["before"].items()}
    pad["mask"] = torch.cat([b1["before"]["mask"],
                             torch.zeros(1, 3, dtype=torch.bool)], 1)
    with torch.no_grad():
        z2 = m.enc(pad)
    assert torch.allclose(z1, z2, atol=1e-5)


def test_hand_permutation_invariance():
    """手札 token の並び替えで STATE 出力が変わらない(順序なし entity)。"""
    a, b_ = ent(hand=(10, 11, 12)), ent(hand=(12, 11, 10))
    ta, tb = ET.tokenize(a), ET.tokenize(b_)
    pa = TT._pack([ta], [ET.relation_matrix(ta)])
    pb = TT._pack([tb], [ET.relation_matrix(tb)])
    m = TN.TransitionQNet(65, mode="current")
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m.enc(pa), m.enc(pb), atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    m = TN.TransitionQNet(65, mode="transition")
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = TN.TransitionQNet(65, mode="transition")
    m2.load_state_dict(torch.load(p, map_location="cpu"))
    assert all(torch.equal(x, y) for x, y in zip(m.parameters(), m2.parameters()))


def test_gradients_finite():
    gs = [_grp()]
    TT.prepare(gs)
    b = TT.make_batch(gs, np.zeros(166, np.float32), np.ones(166, np.float32), "T4")
    m = TT.build_model("T4", 166, 65)
    q = m(b).masked_fill(~b["mask"], 0.0)
    q.sum().backward()
    g = m.enc.card.weight.grad
    assert g is not None and torch.isfinite(g).all()


# ---------------- 対照の妥当性(§50 Controls) ----------------

def test_after_shuffle_actually_breaks_correspondence():
    g = _grp(C=3)
    TT.prepare([g])
    z = np.zeros(166, np.float32); o = np.ones(166, np.float32)
    b0 = TT.make_batch([g], z, o, "T4")
    b1 = TT.make_batch([g], z, o, "T4_shuffle")
    changed = [k for k in b0["after"]
               if not torch.equal(b0["after"][k], b1["after"][k])]
    assert changed, "after が入れ替わっていない"
    for k in changed:                                  # 値の分布は不変(対応だけ破壊)
        a, b = b0["after"][k], b1["after"][k]
        assert torch.equal(a.flatten().sort().values.to(torch.float64),
                           b.flatten().sort().values.to(torch.float64)), k


def test_shuffle_has_identical_param_count():
    a = sum(p.numel() for p in TT.build_model("T4", 166, 65).parameters())
    b = sum(p.numel() for p in TT.build_model("T4_shuffle", 166, 65).parameters())
    assert a == b


def test_norel_actually_disables_relation_bias():
    m = TT.build_model("T4_norel", 166, 65)
    assert all(blk.att.rel_bias is None for blk in m.enc.blocks)
    m2 = TT.build_model("T4", 166, 65)
    assert all(blk.att.rel_bias is not None for blk in m2.enc.blocks)
    assert sum(p.numel() for p in m2.parameters()) > sum(p.numel() for p in m.parameters())


def test_t2_does_not_use_after_state():
    gs = [_grp()]
    TT.prepare(gs)
    b = TT.make_batch(gs, np.zeros(166, np.float32), np.ones(166, np.float32), "T2")
    assert "after" not in b


def test_collision_flags_match_delta_distance():
    import delta_features as DF
    g = _grp(C=3)
    TT.prepare([g])
    for i in range(3):
        for j in range(3):
            d = DF.pair_distance(g["_delta"][i], g["_delta"][j])
            assert g["_collision"][i][j] == (d < 1e-9)


# ---------------- データ不変条件 ----------------

@needs
def test_data_has_after_entity_for_every_candidate():
    n = 0
    for f in sorted(glob.glob(DATA)):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(l)
            for c in r["candidates"]:
                assert "after_entity" in c
                n += 1
    assert n > 0


@needs
def test_data_after_entity_has_no_opponent_hand():
    for f in sorted(glob.glob(DATA)):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(l)
            assert "hand" not in r["entity"]["opp"]
            for c in r["candidates"]:
                if c["after_entity"]:
                    assert "hand" not in c["after_entity"]["opp"]


# ---------------- Regression ----------------

def test_frozen_anchors_unchanged():
    import hashlib
    exp = {"Q0-expanded.pt": "85ac1a597ebe328b", "S3.pt": "58cbd32b752853ec",
           "DeltaQ_D1.pt": "6e922976e2dfb636"}
    for n, s in exp.items():
        assert hashlib.sha256((_HERE / "frozen" / n).read_bytes()).hexdigest()[:16] == s, n
    root = _HERE.parent.parent
    prod = {"sample_submission/configs/abl_5_full.json": "ca6c37af4b1a4149",
            "sample_submission/deck.csv": "8ae7a618b2655669",
            "sample_submission/ptcg_ai/learning/value_weights.json": "d6d7cd8907f53689"}
    for f, s in prod.items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] == s, f
