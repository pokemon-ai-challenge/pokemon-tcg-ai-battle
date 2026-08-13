"""ogerpon_strategy_encoder のユニットテスト。design.md §7、Phase3 item1(encoder契約テスト)。"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX = 96
TAPU_BULU = 920
GRASS = 1


@pytest.fixture
def enc():
    try:
        from ptcg_ai.learning import ogerpon_strategy_encoder as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ogerpon_strategy_encoder / cg engine unavailable: {exc}")
    return m


@pytest.fixture
def STRAT():
    from ptcg_ai.ml_policy import ogerpon_strategy as m
    return m


class _Pokemon:
    _n = [8000]

    def __init__(self, card_id, energies=(), hp=None, max_hp=None):
        self.id = card_id
        self.energies = list(energies)
        self.maxHp = max_hp if max_hp is not None else (210 if card_id == OGERPON_EX else 140)
        self.hp = hp if hp is not None else self.maxHp
        _Pokemon._n[0] += 1
        self.serial = _Pokemon._n[0]


class _Player:
    def __init__(self, active=(), bench=(), hand=(), prize_count=6):
        self.active = list(active)
        self.bench = list(bench)
        self.hand = list(hand)
        self.prize = [object()] * prize_count
        self.handCount = len(hand)
        self.deckCount = 40
        self.discard = []
        self.poisoned = self.burned = self.asleep = self.paralyzed = self.confused = False


class _State:
    def __init__(self, players, your_index=0, result=-1, turn=3):
        self.players = players
        self.yourIndex = your_index
        self.result = result
        self.turn = turn
        self.firstPlayer = your_index
        self.supporterPlayed = False
        self.stadiumPlayed = False
        self.energyAttached = False
        self.retreated = False
        self.stadium = []


class _Card:
    def __init__(self, card_id, serial):
        self.id = card_id
        self.serial = serial


class _AttachOption:
    def __init__(self, area, index, in_play_area, in_play_index, card_id=GRASS):
        from cg.api import OptionType
        self.type = int(OptionType.ATTACH)
        self.area = area
        self.index = index
        self.inPlayArea = in_play_area
        self.inPlayIndex = in_play_index
        self.serial = None
        self.cardId = card_id


class _SimpleOption:
    def __init__(self, otype):
        self.type = int(otype)
        self.area = self.index = self.inPlayArea = self.inPlayIndex = None
        self.serial = None
        self.cardId = None


class _Select:
    def __init__(self, stype, options, max_count=1, context=None):
        from cg.api import SelectContext
        self.type = stype
        self.option = list(options)
        self.minCount = 1
        self.maxCount = max_count
        self.context = int(context if context is not None else SelectContext.MAIN)


class _Obs:
    def __init__(self, state, select):
        self.current = state
        self.select = select


def _area_int(name):
    from cg.api import AreaType
    return int(getattr(AreaType, name))


def _main_select(options):
    from cg.api import SelectType
    return _Select(int(SelectType.MAIN), options)


# --- encode_slot_card_ids ---------------------------------------------------------

def test_slot_card_ids_length_and_order(enc):
    active = _Pokemon(OGERPON_EX)
    bench0 = _Pokemon(TAPU_BULU)
    opp_active = _Pokemon(OGERPON_EX)
    state = _State([_Player(active=[active], bench=[bench0]),
                    _Player(active=[opp_active])])
    ids = enc.encode_slot_card_ids(state, 0)
    assert len(ids) == enc.SLOT_COUNT == 12
    assert ids[0] == OGERPON_EX          # 自分 active
    assert ids[1] == TAPU_BULU           # 自分 bench0
    assert ids[2:6] == [0, 0, 0, 0]      # 自分 bench1-4(空)
    assert ids[6] == OGERPON_EX          # 相手 active
    assert ids[7:12] == [0, 0, 0, 0, 0]  # 相手 bench(空)


def test_slot_card_ids_all_empty_is_all_zero(enc):
    state = _State([_Player(), _Player()])
    assert enc.encode_slot_card_ids(state, 0) == [0] * 12


# --- encode_continuous_features -----------------------------------------------------

def test_continuous_features_length_matches_names(enc):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 2)
    state = _State([_Player(active=[active], bench=[bulu], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    feats = enc.encode_continuous_features(state, 0)
    assert len(feats) == len(enc.CONTINUOUS_FEATURE_NAMES)
    assert all(isinstance(x, float) for x in feats)


def test_continuous_features_deterministic(enc):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(active=[active], prize_count=6), _Player(prize_count=6)])
    a = enc.encode_continuous_features(state, 0)
    b = enc.encode_continuous_features(state, 0)
    assert a == b


def test_continuous_features_reflects_bulu_location(enc):
    """ブルルがactiveのときとbenchのときで bulu_location 特徴が変わる。"""
    idx = enc.CONTINUOUS_FEATURE_NAMES.index("bulu_location")

    bulu_bench = _Pokemon(TAPU_BULU, [GRASS] * 2)
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state_bench = _State([_Player(active=[active], bench=[bulu_bench], prize_count=6),
                          _Player(prize_count=6)])
    feats_bench = enc.encode_continuous_features(state_bench, 0)
    assert feats_bench[idx] == 1.0

    bulu_active = _Pokemon(TAPU_BULU, [GRASS] * 2)
    state_active = _State([_Player(active=[bulu_active], prize_count=6), _Player(prize_count=6)])
    feats_active = enc.encode_continuous_features(state_active, 0)
    assert feats_active[idx] == 2.0

    state_none = _State([_Player(prize_count=6), _Player(prize_count=6)])
    feats_none = enc.encode_continuous_features(state_none, 0)
    assert feats_none[idx] == 0.0


# --- encode_option_features / encode_strategy_pair -----------------------------------

def test_option_features_length_matches_names_for_real_candidates(enc, STRAT):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    attach_opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([attach_opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    pair = STRAT.build_candidates(obs, 0, {}, [OGERPON_EX] * 4 + [TAPU_BULU] + [7] * 55,
                                  STRAT.TRIGGER_MAIN_ATTACH, [1])
    assert pair is not None
    ex_c, single_c = pair

    ex_feats = enc.encode_option_features(ex_c, state, 0, sel)
    single_feats = enc.encode_option_features(single_c, state, 0, sel)
    assert len(ex_feats) == len(enc.OPTION_FEATURE_NAMES)
    assert len(single_feats) == len(enc.OPTION_FEATURE_NAMES)

    ex_idx = enc.OPTION_FEATURE_NAMES.index("option_is_EX_TEMPO")
    single_idx = enc.OPTION_FEATURE_NAMES.index("option_is_SINGLE_PRIZE_ROTATION")
    assert ex_feats[ex_idx] == 1.0 and ex_feats[single_idx] == 0.0
    assert single_feats[ex_idx] == 0.0 and single_feats[single_idx] == 1.0

    trigger_idx = enc.OPTION_FEATURE_NAMES.index("trigger_is_main_attach")
    assert ex_feats[trigger_idx] == 1.0
    assert single_feats[trigger_idx] == 1.0

    legal_idx = enc.OPTION_FEATURE_NAMES.index("safety_legal")
    assert single_feats[legal_idx] == 1.0


def test_encode_strategy_pair_shapes(enc, STRAT):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    attach_opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([attach_opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    pair = STRAT.build_candidates(obs, 0, {}, [OGERPON_EX] * 4 + [TAPU_BULU] + [7] * 55,
                                  STRAT.TRIGGER_MAIN_ATTACH, [1])
    assert pair is not None
    result = enc.encode_strategy_pair(obs, 0, pair)
    assert len(result["continuous_features"]) == len(enc.CONTINUOUS_FEATURE_NAMES)
    assert len(result["slot_card_ids"]) == enc.SLOT_COUNT
    assert set(result["option_features"].keys()) == {STRAT.EX_TEMPO, STRAT.SINGLE_PRIZE_ROTATION}
    for feats in result["option_features"].values():
        assert len(feats) == len(enc.OPTION_FEATURE_NAMES)
