"""og_bulu_finish_only(ブルル完成専用モジュール)のユニットテスト。

0/1エネから無理に育てず、2/3エネまで育ったブルルの完成だけを後押しする最終候補。
旧 ogerpon_planner(enabled 共通ゲート)には依存しない専用コード(finish_only_*)を検証する。
"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX = 96
TAPU_BULU = 920
GRASS_ENERGY = 1   # 基本【草】エネルギー(ブルルのコストを満たす)
FIRE_ENERGY = 2    # 基本【炎】エネルギー(ブルルのコスト[草,草,無色,無色]に対し非生産的になりうる)
OGERPON_DECK = [OGERPON_EX] * 4 + [TAPU_BULU] + [7] * 55
ALAKAZAM_DECK = [741, 742, 743] * 4 + [5] * 48


@pytest.fixture
def planner():
    try:
        from ptcg_ai.ml_policy import ogerpon_planner as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ogerpon_planner / cg engine unavailable: {exc}")
    return mod


class _Card:
    def __init__(self, card_id, serial):
        self.id = card_id
        self.serial = serial


class _Pokemon:
    _n = [2000]

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
    def __init__(self, area, index, in_play_area, in_play_index, card_id=GRASS_ENERGY, otype=None):
        from cg.api import OptionType
        self.type = int(otype if otype is not None else OptionType.ATTACH)
        self.area = area
        self.index = index
        self.inPlayArea = in_play_area
        self.inPlayIndex = in_play_index
        self.serial = None
        self.cardId = card_id


class _CardOption:
    """CARD/TO_ACTIVE,SWITCH 用の選択肢(area/index が対象ポケモンを指す)。"""
    def __init__(self, area, index, otype=None):
        from cg.api import OptionType
        self.type = int(otype if otype is not None else OptionType.CARD)
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


def _main_select(options):
    from cg.api import SelectType
    return _Select(int(SelectType.MAIN), options)


def _card_select(options, context_name):
    from cg.api import SelectContext, SelectType
    return _Select(int(SelectType.CARD), options, context=int(getattr(SelectContext, context_name)))


def _area_int(name):
    from cg.api import AreaType
    return int(getattr(AreaType, name))


def _fo_enabled(**overrides):
    cfg = {"enabled": True}
    cfg.update(overrides)
    return {"ogerpon_finish_only": cfg}


def _bulu_state(active_energies, bulu_energies, hand_card_id=GRASS_ENERGY, prize_count=6):
    active = _Pokemon(OGERPON_EX, active_energies)
    bulu = _Pokemon(TAPU_BULU, bulu_energies)
    mine = _Player(active=[active], bench=[bulu], hand=[_Card(hand_card_id, 900)],
                   prize_count=prize_count)
    opp = _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=prize_count)
    state = _State([mine, opp])
    return state, active, bulu


# --- エネ段階ゲート(0/1エネ: 無視、2エネ: 小、3エネ: 強、4エネ以上: 優先しない) -----------

def test_zero_energy_no_injection_no_bonus(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [])  # active攻撃可能、bulu=0エネ
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) is None
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) == 0.0


def test_one_energy_no_injection_no_bonus(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) is None
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) == 0.0


def test_two_energy_injected_with_small_positive_bonus(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 2)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) == 0
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert 0.0 < bonus.get(0, 0.0) <= planner.FINISH_ONLY_DEFAULTS["cap_tier2"] + 1e-9


def test_three_energy_injected_with_strong_positive_bonus(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 3)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) == 0
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert 0.0 < bonus.get(0, 0.0) <= planner.FINISH_ONLY_DEFAULTS["cap_tier3"] + 1e-9


def test_tier3_bonus_greater_than_tier2_bonus(planner):
    cfg = _fo_enabled()
    s2, _, _ = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 2)
    s3, _, _ = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 3)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    b2 = planner.finish_only_attach_bonus(_Obs(s2, _main_select([opt])), 0, [0], cfg, OGERPON_DECK)
    b3 = planner.finish_only_attach_bonus(_Obs(s3, _main_select([opt])), 0, [0], cfg, OGERPON_DECK)
    assert b3.get(0, 0.0) > b2.get(0, 0.0)


def test_four_plus_energy_not_prioritised_bonus_non_positive(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 4)  # 既に完成
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) is None
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) <= 0.0


# --- 安全条件 -----------------------------------------------------------------

def test_safety_a_active_not_ready_blocks_bonus_and_injection(planner):
    """(a) 現在のバトルポケモンが手貼り前から攻撃可能でなければ発火しない。"""
    state, active, bulu = _bulu_state([], [GRASS_ENERGY] * 3)  # active攻撃不能
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) is None
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) == 0.0


def test_safety_c_non_productive_energy_type_blocks_bonus(planner):
    """(c) 付けるエネルギーが実際にコストへ効かない(色が合わず無色枠も埋まっている)なら発火しない。"""
    # bulu は既に無色2枠を非草エネで埋めている(コスト[草,草,無色,無色])。
    # ここへさらに炎エネを足しても必要エネは減らない(草2枚はどのみち別途必要)。
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [FIRE_ENERGY, FIRE_ENERGY],
                                      hand_card_id=FIRE_ENERGY)
    assert planner._true_shortfall(bulu, [FIRE_ENERGY, FIRE_ENERGY]) == \
        planner._true_shortfall(bulu, [FIRE_ENERGY, FIRE_ENERGY, FIRE_ENERGY])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0, card_id=FIRE_ENERGY)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, OGERPON_DECK) is None
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) == 0.0


def test_safety_c_matching_energy_type_is_productive(planner):
    """草エネはブルルのコストへ実際に効く(対照実験)。"""
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [FIRE_ENERGY, FIRE_ENERGY])
    assert planner._true_shortfall(bulu, [FIRE_ENERGY, FIRE_ENERGY]) > \
        planner._true_shortfall(bulu, [FIRE_ENERGY, FIRE_ENERGY, GRASS_ENERGY])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0, card_id=GRASS_ENERGY)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) > 0.0


def test_safety_d_active_target_never_gets_finish_only_bonus(planner):
    """(d) バトル場自身への手貼りは finish_only の対象外(常にbonus 0)。"""
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 2)  # 非ex/exに関係なく対象外
    state = _State([_Player(active=[active], hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("ACTIVE"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) == 0.0


# --- 交代(TO_ACTIVE/SWITCH): 4エネ完成 かつ 支払い可能 かつ 交代後攻撃可能 のときのみ ------

def test_switch_bonus_only_when_bulu_ready(planner):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    not_ready_bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 2)
    mine = _Player(bench=[ready_bulu, not_ready_bulu])
    state = _State([mine, _Player()])
    opts = [_CardOption(_area_int("BENCH"), 0), _CardOption(_area_int("BENCH"), 1)]
    obs = _Obs(state, _card_select(opts, "TO_ACTIVE"))
    cfg = _fo_enabled()
    adj = planner.finish_only_switch_adjustments(obs, 0, cfg, OGERPON_DECK)
    assert adj[0] > 0.0
    assert adj[1] == 0.0  # 攻撃不能な壁交代のbonusは無い


def test_switch_context_covers_switch_too(planner):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    mine = _Player(bench=[ready_bulu])
    state = _State([mine, _Player()])
    opts = [_CardOption(_area_int("BENCH"), 0)]
    obs = _Obs(state, _card_select(opts, "SWITCH"))
    cfg = _fo_enabled()
    adj = planner.finish_only_switch_adjustments(obs, 0, cfg, OGERPON_DECK)
    assert adj[0] > 0.0


def test_switch_ignores_other_contexts(planner):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    mine = _Player(bench=[ready_bulu])
    state = _State([mine, _Player()])
    opts = [_CardOption(_area_int("BENCH"), 0)]
    obs = _Obs(state, _card_select(opts, "DISCARD"))
    cfg = _fo_enabled()
    assert planner.finish_only_switch_adjustments(obs, 0, cfg, OGERPON_DECK) is None


# --- 有効化ゲート ---------------------------------------------------------------

def test_disabled_by_default(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 3)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    assert planner.finish_only_select_attach_candidate(obs, 0, {}, OGERPON_DECK) is None
    assert planner.finish_only_attach_bonus(obs, 0, [0], {}, OGERPON_DECK) == {}
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    switch_obs = _Obs(_State([_Player(bench=[ready_bulu]), _Player()]),
                      _card_select([_CardOption(_area_int("BENCH"), 0)], "TO_ACTIVE"))
    assert planner.finish_only_switch_adjustments(switch_obs, 0, {}, OGERPON_DECK) is None


def test_disabled_for_non_ogerpon_deck(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 3)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled()
    assert planner.finish_only_select_attach_candidate(obs, 0, cfg, ALAKAZAM_DECK) is None
    assert planner.finish_only_attach_bonus(obs, 0, [0], cfg, ALAKAZAM_DECK) == {}


def test_legacy_ogerpon_planner_stays_off_when_finish_only_used(planner):
    """finish_only用configでは旧 ogerpon_planner.enabled が False のままで、
    score_adjustments (CARD/TO_ACTIVE,SWITCHの旧・壁promotionロジック)は発火しない。"""
    import json

    cfg_path = SAMPLE_SUBMISSION_ROOT / "configs" / "og_bulu_finish_only.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["ogerpon_planner"]["enabled"] is False
    assert cfg["ogerpon_finish_only"]["enabled"] is True
    assert not planner.is_active(cfg, OGERPON_DECK)


# --- 例外安全・上限保証 ----------------------------------------------------------

def test_exception_falls_back_to_none_or_empty(planner):
    cfg = _fo_enabled()
    assert planner.finish_only_attach_bonus(None, 0, [0], cfg, OGERPON_DECK) == {}
    assert planner.finish_only_select_attach_candidate(None, 0, cfg, OGERPON_DECK) is None
    assert planner.finish_only_switch_adjustments(None, 0, cfg, OGERPON_DECK) is None


def test_bonus_never_exceeds_cap_tier3(planner):
    state, active, bulu = _bulu_state([GRASS_ENERGY] * 3, [GRASS_ENERGY] * 3, prize_count=2)
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _fo_enabled(cap_tier3=0.08, w_tier3=0.06, w_required_ko_increase=0.5)
    bonus = planner.finish_only_attach_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) <= 0.08 + 1e-9


def test_pipeline_wires_finish_only_independently_of_legacy_flags():
    """og_bulu_finish_only は main_enabled/attach_enabled(旧)をONにしなくても発火する。"""
    import inspect

    from ptcg_ai.ml_policy import ml_policy_agent

    src = inspect.getsource(ml_policy_agent._try_pipeline)
    assert "fo_on" in src and "finish_only_config" in src


def test_select_action_still_runs_lethal_before_pipeline():
    import inspect

    from ptcg_ai.ml_policy import ml_policy_agent

    src = inspect.getsource(ml_policy_agent._select_action)
    assert src.index("_try_lethal") < src.index("_try_pipeline")
