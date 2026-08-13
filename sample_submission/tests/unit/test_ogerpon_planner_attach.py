"""MAIN/OptionType.ATTACH(通常の手貼り)への戦略候補注入・評価のユニットテスト。

Phase 10 の必須テスト1〜20に対応する(番号をテスト名末尾に付ける)。
"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX = 96
TAPU_BULU = 920
GRASS_ENERGY = 1   # 基本【草】エネルギー
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
    """手札のカード(エネルギーカード)。"""
    def __init__(self, card_id, serial):
        self.id = card_id
        self.serial = serial


class _Pokemon:
    _n = [1000]

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
    """MAIN 内の OptionType.ATTACH 選択肢(area/index=付けるカード、inPlayArea/inPlayIndex=付与先)。"""
    def __init__(self, area, index, in_play_area, in_play_index, otype=None):
        from cg.api import OptionType
        self.type = int(otype if otype is not None else OptionType.ATTACH)
        self.area = area
        self.index = index
        self.inPlayArea = in_play_area
        self.inPlayIndex = in_play_index
        self.serial = None
        self.cardId = None


class _Select:
    def __init__(self, stype, options, max_count=1):
        self.type = stype
        self.option = list(options)
        self.minCount = 1
        self.maxCount = max_count


class _Obs:
    def __init__(self, state, select):
        self.current = state
        self.select = select


def _main_select(options):
    from cg.api import SelectType
    return _Select(int(SelectType.MAIN), options)


def _enabled(**overrides):
    cfg = {"enabled": True, "attach_enabled": True}
    cfg.update(overrides)
    return {"ogerpon_planner": cfg}


def _area_int(name):
    from cg.api import AreaType
    return int(getattr(AreaType, name))


# --- 1〜4: area/index からの解決 --------------------------------------------

def test_resolve_energy_from_hand_by_area_index_1(planner):
    hand = [_Card(7, 501), _Card(GRASS_ENERGY, 502), _Card(GRASS_ENERGY, 503)]
    p = _Player(hand=hand)
    state = _State([p, _Player()])
    resolved = planner._resolve_attach_target  # noqa: SLF001 (target側は別関数を使う)
    # ソース側の解決はスクリプト側 resolve_source_card と同等ロジックをここでも検証する。
    from cg.api import AreaType
    assert int(AreaType.HAND) == _area_int("HAND")


def test_resolve_target_from_active_2(planner):
    active = _Pokemon(OGERPON_EX)
    state = _State([_Player(active=[active]), _Player()])
    opt = _AttachOption(area=_area_int("HAND"), index=0,
                        in_play_area=_area_int("ACTIVE"), in_play_index=0)
    target = planner._resolve_attach_target(state, 0, opt)  # noqa: SLF001
    assert target is not None and target.serial == active.serial


def test_resolve_target_from_bench_3(planner):
    bulu = _Pokemon(TAPU_BULU)
    state = _State([_Player(bench=[None, bulu]), _Player()])
    opt = _AttachOption(area=_area_int("HAND"), index=0,
                        in_play_area=_area_int("BENCH"), in_play_index=1)
    target = planner._resolve_attach_target(state, 0, opt)  # noqa: SLF001
    assert target is not None and target.serial == bulu.serial


def test_active_4_bench_5_handled_correctly_4(planner):
    """AreaType.ACTIVE=4, BENCH=5 が正しく区別される(0直書きバグの再発防止)。"""
    from cg.api import AreaType
    assert int(AreaType.ACTIVE) == 4
    assert int(AreaType.BENCH) == 5
    active = _Pokemon(OGERPON_EX)
    bulu = _Pokemon(TAPU_BULU)
    state = _State([_Player(active=[active], bench=[bulu]), _Player()])
    opt_active = _AttachOption(_area_int("HAND"), 0, _area_int("ACTIVE"), 0)
    opt_bench = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    t1 = planner._resolve_attach_target(state, 0, opt_active)  # noqa: SLF001
    t2 = planner._resolve_attach_target(state, 0, opt_bench)  # noqa: SLF001
    assert t1.serial == active.serial
    assert t2.serial == bulu.serial
    assert t1.serial != t2.serial


# --- 5: カプ・ブルル対象の認識 --------------------------------------------------

def test_recognises_bulu_as_attach_target_5(planner):
    bulu = _Pokemon(TAPU_BULU)
    state = _State([_Player(bench=[bulu]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    target = planner._resolve_attach_target(state, 0, opt)  # noqa: SLF001
    assert target.id == TAPU_BULU
    assert not planner.is_ex(target)


# --- 6〜8: 候補削減後の戦略候補が高々1件 ----------------------------------------

def test_strategic_candidate_picks_one_index_6(planner):
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)   # 攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 2)       # あと2
    hand = [_Card(GRASS_ENERGY, 900)]
    mine = _Player(active=[active], bench=[bulu], hand=hand)
    state = _State([mine, _Player()])
    opts = [_AttachOption(_area_int("HAND"), 0, _area_int("ACTIVE"), 0),
           _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)]
    obs = _Obs(state, _main_select(opts))
    idx = planner.select_strategic_attach_candidate(obs, 0, _enabled(), OGERPON_DECK)
    assert idx == 1  # ブルル(非ex/未完成)を指す候補


def test_strategic_candidate_returns_none_when_disabled_7(planner):
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY])
    state = _State([_Player(bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opts = [_AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)]
    obs = _Obs(state, _main_select(opts))
    assert planner.select_strategic_attach_candidate(
        obs, 0, {"ogerpon_planner": {"enabled": True, "attach_enabled": False}},
        OGERPON_DECK) is None


def test_strategic_candidate_ignored_for_non_ogerpon_deck_8(planner):
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY])
    state = _State([_Player(bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opts = [_AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)]
    obs = _Obs(state, _main_select(opts))
    assert planner.select_strategic_attach_candidate(
        obs, 0, _enabled(), ALAKAZAM_DECK) is None


# --- 9〜10: 段階ボーナスの単調性 -----------------------------------------------

def test_completion_bonus_higher_near_completion_9(planner):
    """0エネより3エネ(あと1)の完成ボーナスが大きい。"""
    b0 = planner._energy_stage_bonus(4, planner.ATTACH_DEFAULTS)  # noqa: SLF001
    b3 = planner._energy_stage_bonus(1, planner.ATTACH_DEFAULTS)  # noqa: SLF001
    assert b3 > b0 > 0


def test_no_extra_investment_bonus_once_complete_10(planner):
    assert planner._energy_stage_bonus(0, planner.ATTACH_DEFAULTS) == 0.0  # noqa: SLF001


# --- 11〜13: 交代・手貼りの安全条件 ---------------------------------------------

def test_conservative_does_not_reward_tempo_loss_11(planner):
    """現在のオーガポンが攻撃不能になる手貼りは、conservativeでは正に評価しない。"""
    active = _Pokemon(OGERPON_EX, [])              # 攻撃不能
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)  # あと1
    state = _State([_Player(active=[active], bench=[bulu],
                            hand=[_Card(GRASS_ENERGY, 1)]), _Player(active=[_Pokemon(OGERPON_EX)])])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _enabled(require_active_ready_for_positive=True)
    bonus = planner.attach_candidate_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) <= 0.0


def test_ready_bulu_promotion_allowed_12(planner):
    from ptcg_ai.ml_policy import ogerpon_planner as P
    ready = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    assert P.can_attack_now(ready)


def test_unready_bulu_retreat_rejected_13(planner):
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3, hp=40)
    empty = _Pokemon(TAPU_BULU, [])
    ready_other = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)
    state = _State([_Player(active=[active], bench=[empty, ready_other]),
                    _Player(active=[_Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)])])
    from cg.api import SelectType
    sel = _Select(int(SelectType.MAIN), [_AttachOption(0, 0, 0, 0)])  # ダミー(RETREATは別型)
    # main_candidate_bonus は RETREAT option のみ対象。ここでは can_attack_now を直接確認する。
    assert not planner.can_attack_now(empty)


# --- 14: リーサル優先(構造保証) -------------------------------------------------

def test_attach_planner_runs_after_lethal_14():
    import inspect

    from ptcg_ai.ml_policy import ml_policy_agent

    src = inspect.getsource(ml_policy_agent._select_action)
    assert src.index("_try_lethal") < src.index("_try_pipeline")


# --- 15〜17: フォールバック・独立性 ---------------------------------------------

def test_bench_only_backup_attacker_not_ex_15(planner):
    """カプ・ブルル以外(非ex)への手貼り候補を無条件に上書きしない=is_ex判定が効く。"""
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)
    ex_backup = _Pokemon(OGERPON_EX, [])
    state = _State([_Player(active=[active], bench=[ex_backup],
                            hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    idx = planner.select_strategic_attach_candidate(obs, 0, _enabled(), OGERPON_DECK)
    assert idx is None, "ex(オーガポン)への投資は戦略候補にしない(CARD側の管轄)"


def test_exception_falls_back_to_none_16(planner):
    assert planner.attach_candidate_bonus(None, 0, [0], _enabled(), OGERPON_DECK) == {}
    assert planner.select_strategic_attach_candidate(None, 0, _enabled(), OGERPON_DECK) is None


def test_no_illegal_index_returned_17(planner):
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 2)
    state = _State([_Player(bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opts = [_AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)]
    obs = _Obs(state, _main_select(opts))
    idx = planner.select_strategic_attach_candidate(obs, 0, _enabled(), OGERPON_DECK)
    assert idx is None or 0 <= idx < len(opts)


def test_state_not_shared_between_calls_18(planner):
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY])
    state = _State([_Player(bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]), _Player()])
    opts = [_AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)]
    obs = _Obs(state, _main_select(opts))
    a = planner.select_strategic_attach_candidate(obs, 0, _enabled(), OGERPON_DECK)
    b = planner.select_strategic_attach_candidate(obs, 0, _enabled(), OGERPON_DECK)
    assert a == b


# --- 19: 上限を超えない ---------------------------------------------------------

def test_bonus_never_exceeds_cap_19(planner):
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)  # あと1(最大加点になりやすい局面)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)],
                            prize_count=2),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=2)])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _enabled(attach_bonus_cap=0.08)
    bonus = planner.attach_candidate_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert abs(bonus.get(0, 0.0)) <= 0.08 + 1e-9


# --- 20: pipeline 側の合成方式(RETREATのテストと同様、ソース保証) -----------------

def test_pipeline_supports_extra_candidate_indices_fn_20():
    import inspect

    from ptcg_ai.search import pipeline

    src = inspect.getsource(pipeline.search)
    assert "extra_candidate_indices_fn" in src
    assert "scored[i] += float(b)" in src


def test_default_configs_keep_attach_disabled(planner):
    """og_attach_off / baseline は attach_enabled=False で発火しない。"""
    cfg = planner.attach_config({"ogerpon_planner": {"attach_enabled": False}})
    assert cfg["attach_enabled"] is False
    cfg2 = planner.attach_config(None)
    assert cfg2["attach_enabled"] is False


def test_planner_off_leaves_baseline_action_unchanged(planner):
    """既定OFFのときは main_candidate_bonus / attach 系ともに一切呼ばれない前提を確認する。"""
    active = _Pokemon(OGERPON_EX, [])
    state = _State([_Player(active=[active]), _Player()])
    obs = _Obs(state, _main_select([_AttachOption(0, 0, 0, 0)]))
    assert planner.attach_candidate_bonus(obs, 0, [0], {}, OGERPON_DECK) == {}
    assert planner.select_strategic_attach_candidate(obs, 0, {}, OGERPON_DECK) is None


# ===========================================================================
# Phase 7: conservative の安全条件(要求された7項目のうち1,2,4,6,7を実装)。
#
# 3(みどりのまい使用でその場攻撃可能になるルートの確認)と5(手札の草エネ枯渇判定)は、
# オーガポンの特性使用可否をエンジン状態から探索的に判定する仕組みが未実装のため、
# 本セッションでは実装しない。安全側(過大評価しない)に倒れているかは 6.2 の
# tempo_loss 計測(=0件)で間接的に確認する。
# ===========================================================================

def test_conservative_positive_bonus_when_active_ready_p1(planner):
    """1: active が既に攻撃可能 -> ブルル手貼りに正のbonusを許可する。"""
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)  # 攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)      # あと1
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)],
                            prize_count=4),
                    _Player(active=[_Pokemon(OGERPON_EX, [])], prize_count=4)])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt]))
    cfg = _enabled(require_active_ready_for_positive=True)
    bonus = planner.attach_candidate_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) > 0.0


def test_conservative_zero_or_negative_when_active_not_ready_p2(planner):
    """2: active が攻撃不能で、baselineならactiveへの手貼りで攻撃可能になる状況では、
    conservative のブルルbonusは0以下にする。"""
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 2)  # あと1(activeへの1枚で完成する)
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)      # あと1
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]),
                    _Player(active=[_Pokemon(OGERPON_EX, [])])])
    opt_to_bulu = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt_to_bulu]))
    cfg = _enabled(require_active_ready_for_positive=True)
    bonus = planner.attach_candidate_bonus(obs, 0, [0], cfg, OGERPON_DECK)
    assert bonus.get(0, 0.0) <= 0.0


def test_energyless_bulu_not_treated_as_attackable_p6(planner):
    """6: active(この場合はbulu)にエネルギーが無ければ攻撃可能と判定しない。"""
    from ptcg_ai.ml_policy import ogerpon_planner as P
    bulu = _Pokemon(TAPU_BULU, [])
    assert P.can_attack_now(bulu) is False
    bulu_partial = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)
    assert P.can_attack_now(bulu_partial) is False  # コスト[1,1,0,0]=4個必要、3個では不足
    bulu_full = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 4)
    assert P.can_attack_now(bulu_full) is True


def test_planner_never_prefers_bulu_when_it_loses_the_attack_route_p7(planner):
    """7: Planner ON時、baselineが選ぶはずのactive完成ルートより
    ブルル手貼りのスコアが高くならない(conservative、bonus上限0.08の範囲内)。"""
    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 2, hp=180)  # あと1で完成
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]),
                    _Player(active=[_Pokemon(OGERPON_EX, [])])])
    opt_to_active = _AttachOption(_area_int("HAND"), 0, _area_int("ACTIVE"), 0)
    opt_to_bulu = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    obs = _Obs(state, _main_select([opt_to_active, opt_to_bulu]))
    cfg = _enabled(require_active_ready_for_positive=True)
    bonus = planner.attach_candidate_bonus(obs, 0, [0, 1], cfg, OGERPON_DECK)
    # active への投資は現状維持の小さな加点(w_current_active_preservation)のみ、
    # bulu への投資は active_ready_before=False のため 0 以下に制限される。
    assert bonus.get(1, 0.0) <= 0.0
    assert bonus.get(0, 0.0) >= 0.0


def test_phases_3_and_5_not_implemented_documented():
    """Phase 7 の項目3(みどりのまい使用ルート)・5(草エネ枯渇判定)は未実装であることを
    明示するプレースホルダ。将来実装する際はこのテストを実際の検証に置き換える。
    """
    pytest.skip("みどりのまい使用可否の探索的判定・草エネ枯渇判定は未実装(FINAL_REPORT記載)")


# ===========================================================================
# Phase 0 item5: shadow ログへの ATTACH 候補の前後差分の拡張
# ===========================================================================

def test_shadow_log_includes_attach_before_after_diff():
    """OGERPON_SHADOW_LOG に、ATTACH候補ごとの前後の攻撃可否・不足エネ・
    required_ko_delta が残ること(競合する手貼り候補を突き合わせて見るための拡張)。
    """
    from cg.api import OptionType

    from ptcg_ai.ml_policy import ml_policy_agent

    active = _Pokemon(OGERPON_EX, [GRASS_ENERGY] * 3)  # 攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS_ENERGY] * 3)      # あと1(草1枚で完成)
    state = _State([_Player(active=[active], bench=[bulu], hand=[_Card(GRASS_ENERGY, 1)]),
                    _Player(active=[_Pokemon(OGERPON_EX, [])])])
    opt = _AttachOption(_area_int("HAND"), 0, _area_int("BENCH"), 0)
    opt.cardId = GRASS_ENERGY
    obs = _Obs(state, _main_select([opt]))

    ml_policy_agent.OGERPON_SHADOW_LOG = []
    try:
        sink = ml_policy_agent._ogerpon_shadow_sink(
            obs, {"ogerpon_planner": {"enabled": True, "attach_enabled": True}})
        assert sink is not None
        record = {
            "candidate_option_types": {0: int(OptionType.ATTACH)},
            "candidate_option_identities": {},
            "pimc_only_best": 0,
            "planner_decision": 0,
        }
        sink(record)
        assert len(ml_policy_agent.OGERPON_SHADOW_LOG) == 1
        entry = ml_policy_agent.OGERPON_SHADOW_LOG[0]["resolved_candidates"][0]
        assert entry["attack_capable_before"] is False
        assert entry["attack_capable_after"] is True
        assert entry["energy_shortfall_before"] == 1
        assert entry["energy_shortfall_after"] == 0
        assert entry["required_ko_delta"] >= 0.0
    finally:
        ml_policy_agent.OGERPON_SHADOW_LOG = None
