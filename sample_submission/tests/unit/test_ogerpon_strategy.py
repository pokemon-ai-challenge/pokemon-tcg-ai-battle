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


def test_ex_tempo_resolves_its_own_target_when_attach(mod, OS):
    """P1-12: EX_TEMPO側のbaseline_actionがATTACHなら、その対象も解決する
    (以前はSINGLE_PRIZE_ROTATION側だけtarget系フィールドが埋まっていた)。
    """
    active = _Pokemon(OGERPON_EX, [GRASS] * 2)   # あと1で攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    attach_to_active = _AttachOption(_area_int("HAND"), 0, _area_int("ACTIVE"), 0)
    attach_to_bulu = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    sel = _main_select([attach_to_active, attach_to_bulu])
    obs = _Obs(state, sel)
    baseline = [0]  # activeへの手貼りを選んだ、というbaselineの体
    pair = mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_MAIN_ATTACH, baseline)
    assert pair is not None
    ex_c, _ = pair
    assert ex_c.target_serial == active.serial
    assert ex_c.target_card_id == OGERPON_EX
    assert ex_c.turns_until_ready == 1
    assert ex_c.safety_flags.get("legal") is True


def test_ex_tempo_target_stays_empty_for_attack_action(mod, OS):
    """ATTACK等、対象ポケモンという概念を持たない型ではtargetは解決されない(既定値のまま)。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS, 1)], prize_count=6),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=6)])
    attach_opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    from cg.api import OptionType
    sel = _main_select([attach_opt, _SimpleOption(OptionType.ATTACK)])
    obs = _Obs(state, sel)
    pair = mod.build_candidates(obs, 0, {}, OGERPON_DECK, mod.TRIGGER_MAIN_ATTACH, [1])
    assert pair is not None
    ex_c, _ = pair
    assert ex_c.target_serial is None
    assert ex_c.safety_flags == {}


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


# ===========================================================================
# compare_options: design.md §11.2のQ比較(Phase3 item4、shadow-only)
# ===========================================================================

class _FakeQModel:
    """predict_membersだけを差し替えたテストダブル。ensemble各モデルのwin予測を
    (ex用リスト, single用リスト)で直接指定できる。"""

    def __init__(self, wins_ex, wins_single, loop_completes_single=None):
        self._wins_ex = wins_ex
        self._wins_single = wins_single
        self._loop = loop_completes_single or [0.5] * len(wins_single)

    def predict_members(self, cont, slots, opt_features, option_name):
        if option_name == "EX_TEMPO":
            return [{"win": w, "loop_complete": 0.5} for w in self._wins_ex]
        return [{"win": w, "loop_complete": lc} for w, lc in zip(self._wins_single, self._loop)]


def _encoded():
    return {"continuous_features": [0.0], "slot_card_ids": [0] * 12,
           "option_features": {"EX_TEMPO": [0.0], "SINGLE_PRIZE_ROTATION": [0.0]}}


def _candidate(required_ko_gain=0.0):
    from ptcg_ai.ml_policy.ogerpon_strategy import SINGLE_PRIZE_ROTATION, StrategyCandidate
    return StrategyCandidate(option_name=SINGLE_PRIZE_ROTATION, first_action=[0],
                             trigger_kind="main_attach", required_ko_gain=required_ko_gain)


def test_compare_options_neutral_fallback_when_model_not_ready(mod):
    class _NotReady:
        def predict_members(self, *a, **k):
            return []
    result = mod.compare_options(_NotReady(), _encoded(), _candidate())
    assert result["chosen_option"] == mod.EX_TEMPO
    assert result["delta"] == 0.0
    assert result["p_ex"] == result["p_single"] == 0.5


def test_compare_options_delta_std_is_paired_not_independent(mod):
    """delta_stdは「同じモデルの2Option差」の標準偏差(独立な2つのstdの合成ではない)。

    2モデルとも win_single - win_ex = 0.3 で揃っていれば(値自体は違っても差は一定)、
    delta_std は 0 になるべき(design.md §11.2、外部設計どおりのペア差計算の回帰テスト)。
    """
    model = _FakeQModel(wins_ex=[0.4, 0.6], wins_single=[0.7, 0.9])
    result = mod.compare_options(model, _encoded(), _candidate())
    assert result["delta_std"] == pytest.approx(0.0, abs=1e-9)
    assert result["delta"] == pytest.approx(0.3, abs=1e-9)


def test_compare_options_strict_override_when_lcb_delta_above_threshold(mod):
    model = _FakeQModel(wins_ex=[0.3, 0.3], wins_single=[0.5, 0.5])  # delta=0.2, std=0
    result = mod.compare_options(model, _encoded(), _candidate())
    assert result["chosen_option"] == mod.SINGLE_PRIZE_ROTATION
    assert result["reason"] == "strict_override"


def test_compare_options_stays_ex_tempo_when_uncertainty_erases_margin(mod):
    """マージン(delta=0.15)自体はしきい値(0.02)を大きく超えていても、モデル間で
    大きく食い違う(delta_std=0.2)場合はlcb_delta=-0.05となりしきい値を割り込み、
    EX_TEMPOのまま(design.md §11.2の不確実性ペナルティ)。"""
    model = _FakeQModel(wins_ex=[0.1, 0.7], wins_single=[0.45, 0.65])
    result = mod.compare_options(model, _encoded(), _candidate())
    assert result["delta"] == pytest.approx(0.15, abs=1e-9)
    assert result["lcb_delta"] < result["thresholds"]["strict_override_threshold"]
    assert result["chosen_option"] == mod.EX_TEMPO


def test_compare_options_near_tie_disabled_by_default(mod):
    """near_tie_enabled=False(初期値)では、要件を満たしていてもSINGLEを選ばない。"""
    model = _FakeQModel(wins_ex=[0.5, 0.5], wins_single=[0.49, 0.49],
                        loop_completes_single=[0.9, 0.9])
    result = mod.compare_options(model, _encoded(), _candidate(required_ko_gain=1.0))
    assert result["chosen_option"] == mod.EX_TEMPO


def test_compare_options_near_tie_when_explicitly_enabled(mod):
    thresholds = dict(mod.DEFAULT_DECISION_THRESHOLDS)
    thresholds["near_tie_enabled"] = True
    model = _FakeQModel(wins_ex=[0.5, 0.5], wins_single=[0.49, 0.49],
                        loop_completes_single=[0.9, 0.9])
    result = mod.compare_options(model, _encoded(), _candidate(required_ko_gain=1.0), thresholds)
    assert result["chosen_option"] == mod.SINGLE_PRIZE_ROTATION
    assert result["reason"] == "near_tie"


def test_compare_options_near_tie_requires_positive_required_ko_gain(mod):
    thresholds = dict(mod.DEFAULT_DECISION_THRESHOLDS)
    thresholds["near_tie_enabled"] = True
    model = _FakeQModel(wins_ex=[0.5, 0.5], wins_single=[0.49, 0.49],
                        loop_completes_single=[0.9, 0.9])
    result = mod.compare_options(model, _encoded(), _candidate(required_ko_gain=0.0), thresholds)
    assert result["chosen_option"] == mod.EX_TEMPO
