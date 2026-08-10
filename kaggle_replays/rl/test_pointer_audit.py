"""pointerがserial単位で正しい個体を指しているかの機械的検証(T1.1)。

card idの一致だけでは、同一card idの別個体を誤って指しても検出できない。ここでは
``board_token_serial[target_index] == debug_target_serial`` を直接assertする
(card idではなくserialで検証する)。同一card idの個体が複数いる合成例を含み、
実際の自己対戦データ(小規模)でも同じ検証を行う。一回限りの手元スクリプトではなく
再実行可能なpytestテストとして残す。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def cg_api():
    try:
        import cg.api as api
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return api


@pytest.fixture(scope="module")
def env(cg_api):
    try:
        import collect_tokens as ct
        import token_shard as ts
        from ptcg_ai.learning import board_tokens as bt
        from ptcg_ai.learning import encoder
        from ptcg_ai.learning import legacy_feature_manifest as manifest
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")
    pm = PolicyModel(None)
    if not pm.is_ready:
        pytest.skip("production policy_weights.json が無い")
    return {"ct": ct, "ts": ts, "bt": bt, "encoder": encoder, "manifest": manifest, "pm": pm}


def _audit(arrays) -> tuple[int, int, int]:
    """(検証した非NO_TARGET選択肢数, serial不一致数, NO_TARGET数) を返す。"""
    board_counts, counts = arrays["board_counts"], arrays["counts"]
    checked = mismatches = no_target = 0
    board_off = opt_off = 0
    for i in range(len(counts)):
        bn, on = int(board_counts[i]), int(counts[i])
        for j in range(on):
            k = opt_off + j
            target = int(arrays["option_target_token_indices"][k])
            dbg_serial = int(arrays["debug_target_serial"][k])
            if target == -1:
                no_target += 1
                continue
            checked += 1
            actual_serial = int(arrays["board_token_serial"][board_off + target])
            if actual_serial != dbg_serial:
                mismatches += 1
        board_off += bn
        opt_off += on
    return checked, mismatches, no_target


def test_pointer_serial_audit_duplicate_card_id_synthetic(env, cg_api):
    """同一card id(756)の個体が複数いる合成局面で、pointerがserial単位で正しいこと。"""
    api = cg_api
    my_active = api.Pokemon(id=343, serial=1, hp=80, maxHp=80, appearThisTurn=False,
                            energies=[], energyCards=[], tools=[], preEvolution=[])
    my_bench = [
        api.Pokemon(id=756, serial=2, hp=300, maxHp=300, appearThisTurn=False,
                   energies=[], energyCards=[], tools=[], preEvolution=[]),
        api.Pokemon(id=756, serial=3, hp=150, maxHp=300, appearThisTurn=False,  # 同一card id、別個体
                   energies=[], energyCards=[], tools=[], preEvolution=[]),
    ]
    me = api.PlayerState(active=[my_active], bench=my_bench, benchMax=5, deckCount=40,
                         discard=[], prize=[None] * 6, handCount=4, hand=None,
                         poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False)
    opp = api.PlayerState(active=[], bench=[], benchMax=5, deckCount=40, discard=[],
                          prize=[None] * 6, handCount=5, hand=None, poisoned=False,
                          burned=False, asleep=False, paralyzed=False, confused=False)
    state = api.State(turn=5, turnActionCount=0, yourIndex=0, firstPlayer=0,
                      supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                      retreated=False, result=-1, stadium=[], looking=None, players=[me, opp])
    options = [
        api.Option(type=api.OptionType.CARD, area=api.AreaType.BENCH, index=0),  # serial=2
        api.Option(type=api.OptionType.CARD, area=api.AreaType.BENCH, index=1),  # serial=3(同card id)
        api.Option(type=api.OptionType.RETREAT),  # serial=1(active)
    ]
    select = api.SelectData(type=api.SelectType.MAIN, context=list(api.SelectContext)[0],
                            minCount=1, maxCount=1, remainDamageCounter=0, remainEnergyCost=0,
                            option=options, deck=None, contextCard=None, effect=None)

    ct, ts, bt, encoder, manifest, pm = (env["ct"], env["ts"], env["bt"], env["encoder"],
                                         env["manifest"], env["pm"])
    profile = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, True)
    decision["chosen_idx"] = 0
    decision["logprob"] = 0.0

    trajs = [{"steps": [decision], "reward": 1.0, "opp": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, [decision], trajs, {"run_id": "audit", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        checked, mismatches, no_target = _audit(arrays)

    assert checked == 3  # BENCH0, BENCH1, RETREAT はすべて解決可能
    assert mismatches == 0
    assert no_target == 0
    # card idだけの一致では区別できない2つのbenchトークン(どちらも756)を、
    # serialで正しく区別できていることを直接確認する。
    idx0 = int(arrays["option_target_token_indices"][0])
    idx1 = int(arrays["option_target_token_indices"][1])
    assert idx0 != idx1
    assert arrays["board_token_card_ids"][idx0] == arrays["board_token_card_ids"][idx1] == 756
    assert arrays["board_token_serial"][idx0] != arrays["board_token_serial"][idx1]


def test_pointer_serial_audit_real_selfplay(env):
    """実際の自己対戦(小規模)でserial単位のpointer監査を行う。再実行可能な検証として
    残す(手元での一回限りの確認で終わらせない)。"""
    ct, ts = env["ct"], env["ts"]
    deck = str((_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
               / "alakazam" / "01.csv").resolve())
    weights = str((_ROOT / "sample_submission" / "ptcg_ai" / "learning"
                  / "policy_weights.json").resolve())
    from run_league import read_deck_csv_file
    deck_l = read_deck_csv_file(deck)

    trajs, wins, valid, errors = ct.parallel_collect_tokens(
        weights, deck_l, deck_l, n_games=4, seed0=42, temperature=1.0, workers=2,
        debug_pointers=True)
    assert valid > 0
    decisions = [step for tr in trajs for step in tr["steps"]]
    assert decisions

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "audit_real", "generation": 0})
        arrays, _ = ts.read_token_shard(path)
        checked, mismatches, no_target = _audit(arrays)

    assert checked > 0
    assert mismatches == 0
    assert no_target >= 0  # 存在確認のみ(0件でも異常ではない)
