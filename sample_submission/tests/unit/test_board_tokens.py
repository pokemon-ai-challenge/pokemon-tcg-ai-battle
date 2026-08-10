"""クラスタ⑨ 検証（学習基盤）／盤面トークン化(Transformer化 T0)

ptcg_ai.learning.board_tokens の単体テスト。実cgエンジンのカードデータ参照が必要
(``_pokemon_features`` 経由で ``card_cache``/``board_features`` を呼ぶため)なので、
DLL がロードできない環境ではスキップされる(test_encoder.py と同じ方針)。

State/PlayerState/Pokemon/Option はリプレイJSONを介さず、cg.api のdataclassを
直接構築する(ベンチの並び順・重複カードid・serialを意図的に制御するため)。
使用するcard id(343/756/678)は既存フィクスチャ
(tests/fixtures/encoder_observations.json)から採った実在カードで、
``card_cache`` が解決できることを確認済み。
"""

from __future__ import annotations

import math

import pytest


@pytest.fixture(scope="module")
def cg_api():
    try:
        import cg.api as api  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return api


@pytest.fixture(scope="module")
def bt():
    try:
        from ptcg_ai.learning import board_tokens as _bt
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"board_tokens / cg engine unavailable: {exc}")
    return _bt


def _pokemon(api, card_id: int, serial: int, hp: int = 100, max_hp: int = 100, energies=None):
    return api.Pokemon(
        id=card_id, serial=serial, hp=hp, maxHp=max_hp, appearThisTurn=False,
        energies=list(energies or []), energyCards=[], tools=[], preEvolution=[],
    )


def _player(api, active, bench, hand_count=4, deck_count=40, prize=None, discard=None):
    return api.PlayerState(
        active=[active] if active is not None else [],
        bench=list(bench), benchMax=5, deckCount=deck_count,
        discard=list(discard or []), prize=list(prize if prize is not None else [None] * 6),
        handCount=hand_count, hand=None,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def _state(api, players, your_index=0, turn=5):
    return api.State(
        turn=turn, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False, retreated=False,
        result=-1, stadium=[], looking=None, players=list(players),
    )


def _basic_state(api):
    """自分: active(343,serial=1) + bench(756,serial=2, 756,serial=3)(同名重複)。
    相手: active(678,serial=10)のみ。"""
    my_active = _pokemon(api, 343, serial=1)
    my_bench = [_pokemon(api, 756, serial=2), _pokemon(api, 756, serial=3)]
    opp_active = _pokemon(api, 678, serial=10)
    me = _player(api, my_active, my_bench)
    opp = _player(api, opp_active, [])
    return _state(api, [me, opp])


# ---------------------------------------------------------------------------
# build_board_tokens
# ---------------------------------------------------------------------------

def test_state_none_returns_empty(bt):
    tokens = bt.build_board_tokens(None)
    assert len(tokens) == 0


def test_token_count_matches_present_pokemon(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    # 自分: active 1 + bench 2 = 3 / 相手: active 1 + bench 0 = 1 -> 合計4
    assert len(tokens) == 4


def test_no_fixed_slot_padding(bt, cg_api):
    """ベンチが0体でもゼロ埋めトークンが生成されない(固定5スロットではない)。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    opp_bench_tokens = [z for z in tokens.zone_ids if z == bt.ZONE_OPP_BENCH]
    assert opp_bench_tokens == []


def test_numeric_feature_dim_matches_pokemon_features(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    assert all(len(f) == bt.BOARD_TOKEN_NUMERIC_DIM for f in tokens.numeric_features)


def test_no_nan_or_inf(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    for row in tokens.numeric_features:
        assert all(isinstance(x, float) for x in row)
        assert all(math.isfinite(x) for x in row)


def test_deterministic_same_input_same_output(bt, cg_api):
    """同じStateを複数回encodeして完全に同じトークン列になること(T0完了条件)。"""
    state = _basic_state(cg_api)
    t1 = bt.build_board_tokens(state)
    t2 = bt.build_board_tokens(state)
    assert t1.numeric_features == t2.numeric_features
    assert t1.card_ids == t2.card_ids
    assert t1.zone_ids == t2.zone_ids
    assert t1.serials == t2.serials


def test_card_ids_preserved_per_token(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    pairs = dict(zip(tokens.serials, tokens.card_ids))
    assert pairs[1] == 343  # self active
    assert pairs[2] == 756  # self bench 0
    assert pairs[3] == 756  # self bench 1(同名重複)
    assert pairs[10] == 678  # opp active


def test_duplicate_card_id_tokens_are_distinct(bt, cg_api):
    """同じcard idのポケモンがベンチに複数いても、別トークンとして区別される。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    bench_serials = [s for s, z in zip(tokens.serials, tokens.zone_ids) if z == bt.ZONE_SELF_BENCH]
    assert sorted(bench_serials) == [2, 3]
    assert len(set(bench_serials)) == 2


# ---------------------------------------------------------------------------
# resolve_option_target_index: 並び順・重複への頑健性
# ---------------------------------------------------------------------------

def test_pointer_survives_bench_reorder(bt, cg_api):
    """ベンチの並び順を変えても、serialから解決したポインタは同じ個体を指す。"""
    state = _basic_state(cg_api)
    tokens_original = bt.build_board_tokens(state)

    # ベンチのPythonリスト順を逆にした別Stateを作る(cgエンジンが返す順序が変わる状況を模擬)。
    me = state.players[0]
    reordered_me = cg_api.PlayerState(
        active=me.active, bench=list(reversed(me.bench)), benchMax=me.benchMax,
        deckCount=me.deckCount, discard=me.discard, prize=me.prize,
        handCount=me.handCount, hand=me.hand,
        poisoned=me.poisoned, burned=me.burned, asleep=me.asleep,
        paralyzed=me.paralyzed, confused=me.confused,
    )
    reordered_state = _state(cg_api, [reordered_me, state.players[1]])
    tokens_reordered = bt.build_board_tokens(reordered_state)

    # 選択肢: ベンチのserial=3(元の並びでは2番目、逆順では1番目)を対象にするCARD選択肢。
    option = cg_api.Option(type=cg_api.OptionType.CARD, area=cg_api.AreaType.BENCH, index=1)
    idx_original = bt.resolve_option_target_index(option, state, tokens_original)
    assert tokens_original.serials[idx_original] == 3

    option_reordered = cg_api.Option(type=cg_api.OptionType.CARD, area=cg_api.AreaType.BENCH, index=0)
    idx_reordered = bt.resolve_option_target_index(option_reordered, reordered_state, tokens_reordered)
    assert tokens_reordered.serials[idx_reordered] == 3  # 並びが変わっても同じ個体(serial=3)を指す


def test_pointer_resolves_uniquely_for_duplicate_card_id(bt, cg_api):
    """同一card idのポケモンが複数いても、area/indexからの解決は一意に定まる。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)

    opt_bench0 = cg_api.Option(type=cg_api.OptionType.CARD, area=cg_api.AreaType.BENCH, index=0)
    opt_bench1 = cg_api.Option(type=cg_api.OptionType.CARD, area=cg_api.AreaType.BENCH, index=1)
    idx0 = bt.resolve_option_target_index(opt_bench0, state, tokens)
    idx1 = bt.resolve_option_target_index(opt_bench1, state, tokens)

    assert idx0 != idx1
    assert tokens.serials[idx0] == 2
    assert tokens.serials[idx1] == 3
    assert tokens.card_ids[idx0] == tokens.card_ids[idx1] == 756  # card idは同じでも別トークン


# ---------------------------------------------------------------------------
# resolve_option_target_index: NO_TARGET / 暗黙対象
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option_type_name", ["NUMBER", "YES", "NO", "END"])
def test_no_target_options(bt, cg_api, option_type_name):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option_type = getattr(cg_api.OptionType, option_type_name)
    option = cg_api.Option(type=option_type)
    assert bt.resolve_option_target_index(option, state, tokens) == bt.NO_TARGET


def test_play_is_no_target_in_t1(bt, cg_api):
    """PLAY(手札のカード)はT1では未対応ゾーンなのでNO_TARGET。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.PLAY, index=0)
    assert bt.resolve_option_target_index(option, state, tokens) == bt.NO_TARGET


@pytest.mark.parametrize("option_type_name", ["ATTACK", "RETREAT"])
def test_attack_and_retreat_resolve_to_own_active(bt, cg_api, option_type_name):
    """ATTACK/RETREATはarea/indexを持たないが、ルール上常に自分のバトルポケモンが対象。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option_type = getattr(cg_api.OptionType, option_type_name)
    option = cg_api.Option(type=option_type, attackId=0 if option_type_name == "ATTACK" else None)
    idx = bt.resolve_option_target_index(option, state, tokens)
    assert idx != bt.NO_TARGET
    assert tokens.serials[idx] == 1  # 自分のactive(serial=1)


def test_skill_resolves_via_serial(bt, cg_api):
    """SKILLはoption.serialを直接持つ場合、area/index無しでも解決できる。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.SKILL, cardId=756, serial=3)
    idx = bt.resolve_option_target_index(option, state, tokens)
    assert idx != bt.NO_TARGET
    assert tokens.serials[idx] == 3


def test_skill_special_condition_marker_returns_no_target(bt, cg_api):
    """SKILLでcardId=0(特殊状態の宣言、cg/api.pyコメント)かつserial未設定ならNO_TARGET。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.SKILL, cardId=0, serial=None)
    assert bt.resolve_option_target_index(option, state, tokens) == bt.NO_TARGET


def test_attach_resolves_via_in_play_pointer(bt, cg_api):
    """ATTACHはinPlayArea/inPlayIndexで装着先ポケモンを解決する。"""
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(
        type=cg_api.OptionType.ATTACH, area=cg_api.AreaType.HAND, index=0,
        inPlayArea=cg_api.AreaType.BENCH, inPlayIndex=1,
    )
    idx = bt.resolve_option_target_index(option, state, tokens)
    assert idx != bt.NO_TARGET
    assert tokens.serials[idx] == 3


def test_option_type_pointer_table_covers_all_enum_members(bt, cg_api):
    """OPTION_TYPE_POINTER_TABLEがOptionTypeの全メンバーを網羅していること。"""
    all_types = set(cg_api.OptionType)
    assert set(bt.OPTION_TYPE_POINTER_TABLE.keys()) == all_types


# ---------------------------------------------------------------------------
# owner_of_zone(T0.1: zone_id -> owner_id の決定的マッピング)
# ---------------------------------------------------------------------------

def test_owner_of_zone_self_and_opp(bt):
    assert bt.owner_of_zone(bt.ZONE_SELF_ACTIVE) == bt.OWNER_SELF
    assert bt.owner_of_zone(bt.ZONE_SELF_BENCH) == bt.OWNER_SELF
    assert bt.owner_of_zone(bt.ZONE_OPP_ACTIVE) == bt.OWNER_OPP
    assert bt.owner_of_zone(bt.ZONE_OPP_BENCH) == bt.OWNER_OPP


def test_owner_of_zone_unknown_raises(bt):
    with pytest.raises(ValueError):
        bt.owner_of_zone(999)


def test_owner_of_zone_covers_all_reserved_zones(bt):
    for zone_id in bt.ZONE_NAMES:
        assert bt.owner_of_zone(zone_id) in (bt.OWNER_SELF, bt.OWNER_OPP)


# ---------------------------------------------------------------------------
# resolve_option_target_index_debug(pointer監査情報)
# ---------------------------------------------------------------------------

def test_debug_resolver_matches_plain_resolver(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.CARD, area=cg_api.AreaType.BENCH, index=1)
    plain = bt.resolve_option_target_index(option, state, tokens)
    debug = bt.resolve_option_target_index_debug(option, state, tokens)
    assert debug.target_index == plain
    assert debug.target_serial == 3
    assert debug.target_card_id == 756
    assert debug.resolver == bt.RESOLVER_AREA_INDEX


def test_debug_resolver_reports_skill_serial(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.SKILL, cardId=756, serial=3)
    debug = bt.resolve_option_target_index_debug(option, state, tokens)
    assert debug.resolver == bt.RESOLVER_SKILL_SERIAL
    assert debug.target_serial == 3
    assert debug.target_card_id == 756


def test_debug_resolver_reports_active_implicit(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.RETREAT)
    debug = bt.resolve_option_target_index_debug(option, state, tokens)
    assert debug.resolver == bt.RESOLVER_ACTIVE_IMPLICIT
    assert debug.target_serial == 1


def test_debug_resolver_reports_no_target_reason(bt, cg_api):
    state = _basic_state(cg_api)
    tokens = bt.build_board_tokens(state)
    option = cg_api.Option(type=cg_api.OptionType.END)
    debug = bt.resolve_option_target_index_debug(option, state, tokens)
    assert debug.target_index == bt.NO_TARGET
    assert debug.resolver == bt.RESOLVER_NO_TARGET_NO_FIELDS

    option_play = cg_api.Option(type=cg_api.OptionType.PLAY, index=0)
    debug_play = bt.resolve_option_target_index_debug(option_play, state, tokens)
    assert debug_play.target_index == bt.NO_TARGET
    # PLAYはarea=Noneのまま解決を試みるため、_resolve_card_id等がNoneになりcase5に落ちる。
