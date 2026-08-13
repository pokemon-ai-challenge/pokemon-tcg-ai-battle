"""ogerpon_strategy(Strategy Window Builder)のユニットテスト。design.md Phase2 item1。"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX = 96
TAPU_BULU = 920
GRASS = 1
OGERPON_DECK = [OGERPON_EX] * 4 + [TAPU_BULU] + [7] * 55
ALAKAZAM_DECK = [741, 742, 743] * 4 + [5] * 48


@pytest.fixture
def mod():
    try:
        from ptcg_ai.ml_policy import ogerpon_strategy as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ogerpon_strategy / cg engine unavailable: {exc}")
    return m


@pytest.fixture
def OS():
    from ptcg_ai.ml_policy import ogerpon_option_state as m
    return m


class _Card:
    def __init__(self, card_id, serial):
        self.id = card_id
        self.serial = serial


class _Pokemon:
    _n = [7000]

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


class _State:
    def __init__(self, players, your_index=0, result=-1, turn=3):
        self.players = players
        self.yourIndex = your_index
        self.result = result
        self.turn = turn


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
    """type だけを持つダミー選択肢(ATTACK等、「競合候補」を作るためだけに使う)。"""

    def __init__(self, otype):
        self.type = int(otype)
        self.area = self.index = self.inPlayArea = self.inPlayIndex = None
        self.serial = None
        self.cardId = None


class _RetreatOption:
    def __init__(self):
        from cg.api import OptionType
        self.type = int(OptionType.RETREAT)
        self.serial = None
        self.inPlayArea = self.inPlayIndex = self.area = self.index = None


class _CardOption:
    def __init__(self, area, index):
        from cg.api import OptionType
        self.type = int(OptionType.CARD)
        self.area = area
        self.index = index
        self.inPlayArea = None
        self.inPlayIndex = None
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


def _card_select(options, context_name):
    from cg.api import SelectContext, SelectType
    return _Select(int(SelectType.CARD), options, context=int(getattr(SelectContext, context_name)))


# --- detect_trigger ---------------------------------------------------------------

def test_no_trigger_for_non_ogerpon_deck(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, ALAKAZAM_DECK, OS.IDLE) is None


def test_no_trigger_for_multi_select(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([opt, _SimpleOption(OptionType.ATTACK)])
    sel.maxCount = 2
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) is None


def test_main_attach_trigger_detected(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)   # 攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)       # あと1、投資価値あり
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) == mod.TRIGGER_MAIN_ATTACH


def test_main_attach_not_triggered_when_only_option(mod, OS):
    """競合候補が無い(このATTACHしか選べない)なら発火しない。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    sel = _main_select([opt])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) is None


def test_promote_trigger_detected(mod, OS):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    ready_ex = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(bench=[ready_bulu, ready_ex]), _Player()])
    opts = [_CardOption(_area_int("BENCH"), 0), _CardOption(_area_int("BENCH"), 1)]
    sel = _card_select(opts, "TO_ACTIVE")
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) == mod.TRIGGER_PROMOTE


def test_promote_not_triggered_without_ready_ex(mod, OS):
    """攻撃可能なexが無ければ(ブルルだけ)発火しない(比較する相手がいない)。"""
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    not_ready_ex = _Pokemon(OGERPON_EX, [])
    state = _State([_Player(bench=[ready_bulu, not_ready_ex]), _Player()])
    opts = [_CardOption(_area_int("BENCH"), 0), _CardOption(_area_int("BENCH"), 1)]
    sel = _card_select(opts, "TO_ACTIVE")
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) is None


def test_retreat_trigger_detected(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)  # 攻撃可能・続投できる
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)      # 攻撃可能・中継候補
    state = _State([_Player(active=[active], bench=[bulu]), _Player()])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK), _RetreatOption()])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) == mod.TRIGGER_RETREAT


def test_retreat_not_triggered_when_active_cannot_attack(mod, OS):
    """現在のexが攻撃不能なら「続投」との競合ではない。"""
    active = _Pokemon(OGERPON_EX, [])  # 攻撃不能
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[active], bench=[bulu]), _Player()])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK), _RetreatOption()])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.IDLE) is None


def test_continue_trigger_when_mid_rotation(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(active=[active]), _Player()])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    for mode in (OS.BUILD, OS.READY, OS.ACTIVE):
        assert mod.detect_trigger(obs, 0, OGERPON_DECK, mode) == mod.TRIGGER_CONTINUE


def test_continue_takes_priority_over_other_triggers(mod, OS):
    """既に中継中なら、他の条件(retreatなど)に該当してもcontinueとして扱う。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[active], bench=[bulu]), _Player()])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK), _RetreatOption()])
    obs = _Obs(state, sel)
    assert mod.detect_trigger(obs, 0, OGERPON_DECK, OS.ACTIVE) == mod.TRIGGER_CONTINUE


def test_exception_returns_none(mod, OS):
    assert mod.detect_trigger(None, 0, OGERPON_DECK, OS.IDLE) is None


# --- build_candidates --------------------------------------------------------------

def test_build_candidates_for_main_attach(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    attach_opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([attach_opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    baseline = [1]  # ATTACKを選んだ、というbaselineの体
    pair = mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_MAIN_ATTACH, baseline)
    assert pair is not None
    ex_c, single_c = pair
    assert ex_c.option_name == mod.EX_TEMPO
    assert ex_c.first_action == baseline
    assert single_c.option_name == mod.SINGLE_PRIZE_ROTATION
    assert single_c.first_action == [0]
    assert single_c.target_card_id == TAPU_BULU
    assert single_c.target_serial == bulu.serial
    assert single_c.turns_until_ready == 1
    assert single_c.safety_flags["legal"] is True


def test_build_candidates_for_promote(mod, OS):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    ready_ex = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(bench=[ready_bulu, ready_ex], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    opts = [_CardOption(_area_int("BENCH"), 0), _CardOption(_area_int("BENCH"), 1)]
    sel = _card_select(opts, "TO_ACTIVE")
    obs = _Obs(state, sel)
    pair = mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_PROMOTE, [1])
    assert pair is not None
    _, single_c = pair
    assert single_c.first_action == [0]
    assert single_c.target_card_id == TAPU_BULU


def test_build_candidates_for_retreat(mod, OS):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[active], bench=[bulu], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK), _RetreatOption()])
    obs = _Obs(state, sel)
    pair = mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_RETREAT, [0])
    assert pair is not None
    _, single_c = pair
    assert single_c.first_action == [1]
    assert single_c.target_card_id == TAPU_BULU


def test_build_candidates_none_for_continue_trigger(mod, OS):
    """continueトリガーには具体的な代替行動が無いので比較不能。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(active=[active], prize_count=6), _Player(prize_count=6)])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    assert mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_CONTINUE, [0]) is None


def test_build_candidates_none_when_target_missing(mod, OS):
    state = _State([_Player(prize_count=6), _Player(prize_count=6)])
    from cg.api import OptionType
    sel = _main_select([_SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    assert mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_MAIN_ATTACH, [0]) is None


def test_build_candidates_exception_returns_none(mod, OS):
    assert mod.build_candidates(None, 0, {}, OGERPON_DECK, mod.TRIGGER_MAIN_ATTACH, [0]) is None
