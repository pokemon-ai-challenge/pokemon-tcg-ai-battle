"""collect_tokens.py の単体テスト(T0)。

自己対戦(cgエンジンでの対局)は行わず、``_build_decision`` を合成Stateに対して直接呼び、
- 既存MLPの推論結果(teacher_logits)が、通常の推論経路(PolicyModel.score_options_from_state)
  と完全に一致すること(T0完了条件「既存MLPの推論結果を変更しない」)
- legacy_state_feat / legacy_option_feats / option_card_ids が既存encoder関数の出力と一致すること
  (dual-writeの整合性)
を確認する。既定の教師重み(``sample_submission/ptcg_ai/learning/policy_weights.json``、
production既定)を使うため、cgエンジンのカードデータ参照が必要(DLLが無い環境ではスキップ)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def cg_api():
    try:
        import cg.api as api  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return api


@pytest.fixture(scope="module")
def env(cg_api):
    """collect_tokens / PolicyModel / encoder / board_tokens / manifest をまとめて用意する。"""
    try:
        import collect_tokens as ct
        from ptcg_ai.learning import board_tokens as board_tokens_mod
        from ptcg_ai.learning import encoder
        from ptcg_ai.learning import legacy_feature_manifest as manifest
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")
    pm = PolicyModel(None)  # production既定(sample_submission/ptcg_ai/learning/policy_weights.json)
    if not pm.is_ready:
        pytest.skip("production policy_weights.json が無い(is_ready=False)")
    # productionのpolicy_weights.jsonは拡張特徴を使わない(166次元)ことを確認済み(T0のsmokeテスト)。
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    return {"ct": ct, "board_tokens": board_tokens_mod, "encoder": encoder, "manifest": manifest,
            "pm": pm, "profile_name": profile_name}


def _basic_state(api):
    my_active = api.Pokemon(id=343, serial=1, hp=80, maxHp=80, appearThisTurn=False,
                            energies=[0], energyCards=[], tools=[], preEvolution=[])
    my_bench = [
        api.Pokemon(id=756, serial=2, hp=300, maxHp=300, appearThisTurn=False,
                   energies=[], energyCards=[], tools=[], preEvolution=[]),
    ]
    opp_active = api.Pokemon(id=678, serial=10, hp=340, maxHp=340, appearThisTurn=False,
                             energies=[], energyCards=[], tools=[], preEvolution=[])
    me = api.PlayerState(active=[my_active], bench=my_bench, benchMax=5, deckCount=40,
                         discard=[], prize=[None] * 6, handCount=4, hand=None,
                         poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False)
    opp = api.PlayerState(active=[opp_active], bench=[], benchMax=5, deckCount=40,
                          discard=[], prize=[None] * 6, handCount=5, hand=None,
                          poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False)
    return api.State(turn=5, turnActionCount=0, yourIndex=0, firstPlayer=0,
                     supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                     retreated=False, result=-1, stadium=[], looking=None, players=[me, opp])


def _basic_select(api, state):
    options = [
        api.Option(type=api.OptionType.RETREAT),
        api.Option(type=api.OptionType.CARD, area=api.AreaType.BENCH, index=0),
        api.Option(type=api.OptionType.END),
    ]
    return api.SelectData(type=api.SelectType.MAIN, context=list(api.SelectContext)[0],
                          minCount=1, maxCount=1, remainDamageCounter=0, remainEnergyCost=0,
                          option=options, deck=None, contextCard=None, effect=None)


def test_teacher_logits_match_normal_inference_path(env, cg_api):
    """T0完了条件: 既存MLPの推論結果を変更しない。

    _build_decision が出す teacher_logits と、通常の推論経路
    (PolicyModel.score_options_from_state)の出力が完全一致することを確認する。
    """
    state = _basic_state(cg_api)
    select = _basic_select(cg_api, state)

    ct, bt, encoder, manifest, pm, profile = (env["ct"], env["board_tokens"], env["encoder"],
                                              env["manifest"], env["pm"], env["profile_name"])
    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, False)

    reference = pm.score_options_from_state(state, select)
    assert decision["teacher_logits"] == reference


def test_legacy_features_match_existing_encoder_functions(env, cg_api):
    """dual-write整合性: legacy_* が既存encoder関数の出力とそのまま一致する。"""
    state = _basic_state(cg_api)
    select = _basic_select(cg_api, state)
    ct, bt, encoder, manifest, pm, profile = (env["ct"], env["board_tokens"], env["encoder"],
                                              env["manifest"], env["pm"], env["profile_name"])

    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, False)

    assert decision["legacy_state_feat"] == pm.encode_state_features(state)
    assert decision["legacy_option_feats"] == encoder.encode_options_from_state(state, select)
    assert decision["option_card_ids"] == encoder.encode_option_card_ids(state, select)


def test_legacy_global_feat_matches_manifest(env, cg_api):
    """T0.1: legacy_global_featはmanifest.extract_global_featuresの出力と一致する
    (唯一の定義元がmanifestであることの確認)。"""
    state = _basic_state(cg_api)
    select = _basic_select(cg_api, state)
    ct, bt, encoder, manifest, pm, profile = (env["ct"], env["board_tokens"], env["encoder"],
                                              env["manifest"], env["pm"], env["profile_name"])
    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, False)

    expected = manifest.extract_global_features(pm.encode_state_features(state), profile).tolist()
    assert decision["legacy_global_feat"] == expected
    assert len(decision["legacy_global_feat"]) == manifest.global_dim(profile)


def test_option_target_indices_length_matches_options(env, cg_api):
    state = _basic_state(cg_api)
    select = _basic_select(cg_api, state)
    ct, bt, encoder, manifest, pm, profile = (env["ct"], env["board_tokens"], env["encoder"],
                                              env["manifest"], env["pm"], env["profile_name"])

    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, False)
    assert len(decision["option_target_indices"]) == len(select.option)

    # RETREAT(index0) は自分のactive(serial=1)を指す
    assert decision["option_target_indices"][0] != bt.NO_TARGET
    # CARD(index1, bench0) はserial=2(bench)を指す
    assert decision["option_target_indices"][1] != bt.NO_TARGET
    # END(index2) は対象を持たない
    assert decision["option_target_indices"][2] == bt.NO_TARGET


def test_debug_pointers_flag_adds_debug_fields(env, cg_api):
    state = _basic_state(cg_api)
    select = _basic_select(cg_api, state)
    ct, bt, encoder, manifest, pm, profile = (env["ct"], env["board_tokens"], env["encoder"],
                                              env["manifest"], env["pm"], env["profile_name"])

    decision = ct._build_decision(pm, encoder, bt, manifest, profile, state, select, True)
    assert "debug_target_serial" in decision
    assert decision["debug_target_serial"][0] == 1  # RETREAT -> 自分のactive(serial=1)
    assert decision["debug_resolver"][0] == bt.RESOLVER_ACTIVE_IMPLICIT
    assert decision["debug_target_serial"][2] is None  # END -> 対象なし
