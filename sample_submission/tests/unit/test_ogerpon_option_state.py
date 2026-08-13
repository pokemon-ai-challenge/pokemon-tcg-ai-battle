"""ogerpon_option_state(Phase1: カプ・ブルル中継戦略の固定Option Controller)のユニットテスト。

design.md §15.1 のうち、静的な盤面スナップショット+obs.logsだけで判定できる項目
(状態遷移・reset・abort・例外安全)を対象にする。Q-criticが無いのでOption選択の
判断そのもの(いつ SINGLE_PRIZE_ROTATION を選ぶか)は対象外(Phase3)。
"""

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
def mod():
    try:
        from ptcg_ai.ml_policy import ogerpon_option_state as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ogerpon_option_state / cg engine unavailable: {exc}")
    return m


class _Pokemon:
    _n = [3000]

    def __init__(self, card_id, energies=(), hp=None, max_hp=None):
        self.id = card_id
        self.energies = list(energies)
        self.maxHp = max_hp if max_hp is not None else (210 if card_id == OGERPON_EX else 140)
        self.hp = hp if hp is not None else self.maxHp
        _Pokemon._n[0] += 1
        self.serial = _Pokemon._n[0]


class _Player:
    def __init__(self, active=(), bench=(), prize_count=6):
        self.active = list(active)
        self.bench = list(bench)
        self.prize = [object()] * prize_count


class _State:
    def __init__(self, players, your_index=0, result=-1, turn=1):
        self.players = players
        self.yourIndex = your_index
        self.result = result
        self.turn = turn


class _Log:
    def __init__(self, type_, playerIndex=None, serial=None):
        self.type = type_
        self.playerIndex = playerIndex
        self.serial = serial


class _Obs:
    def __init__(self, state, logs=()):
        self.current = state
        self.logs = list(logs)
        self.select = None


def _attack_log(me, serial):
    from cg.api import LogType
    return _Log(LogType.ATTACK, playerIndex=me, serial=serial)


# --- 基本遷移: IDLE -> BUILD -> READY -> ACTIVE -> COMPLETE --------------------

def test_idle_stays_idle_without_force_start(mod):
    bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player(bench=[bulu]), _Player()], turn=2)
    obs = _Obs(state)
    prev = mod.OgerponOptionState()
    nxt = mod.advance_state(prev, obs, 0, {}, force_start=False)
    assert nxt.mode == mod.IDLE


def test_idle_to_build_with_force_start(mod):
    bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player(bench=[bulu]), _Player()], turn=2)
    obs = _Obs(state)
    prev = mod.OgerponOptionState()
    nxt = mod.advance_state(prev, obs, 0, {}, force_start=True)
    assert nxt.mode == mod.BUILD
    assert nxt.target_serial == bulu.serial
    assert nxt.target_card_id == TAPU_BULU
    assert nxt.build_deadline_turn == 2 + 4 * 2


def test_build_to_ready_when_bulu_can_attack(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)  # 攻撃可能(コスト[草,草,無色,無色])
    state = _State([_Player(bench=[bulu]), _Player()], turn=4)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU,
                                  started_turn=2, build_deadline_turn=10)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.READY


def test_build_stays_build_when_not_ready(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 2)  # まだ攻撃不可
    state = _State([_Player(bench=[bulu]), _Player()], turn=4)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU,
                                  started_turn=2, build_deadline_turn=10)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.BUILD


def test_ready_to_active_when_promoted(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[bulu]), _Player()], turn=6)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.READY, phase=mod.READY,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU,
                                  started_turn=2)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ACTIVE


def test_ready_stays_ready_when_not_yet_promoted(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    other_active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player(active=[other_active], bench=[bulu]), _Player()], turn=6)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.READY, phase=mod.READY,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.READY


def test_active_to_complete_when_bulu_faints(mod):
    """気絶して場のどこにも見つからなくなったら COMPLETE(想定どおりの完遂)。"""
    state = _State([_Player(active=[]), _Player()], turn=8)  # ブルルはもう場に居ない
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.ACTIVE, phase=mod.ACTIVE,
                                  target_serial=12345, target_card_id=TAPU_BULU,
                                  bulu_attack_count=1)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.COMPLETE
    assert nxt.bulu_attack_count == 1  # 消滅した手番自体の攻撃ログは無いので加算されない


def test_active_counts_attacks_from_logs(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[bulu]), _Player()], turn=8)
    obs = _Obs(state, logs=[_attack_log(0, bulu.serial)])
    prev = mod.OgerponOptionState(mode=mod.ACTIVE, phase=mod.ACTIVE,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU,
                                  bulu_attack_count=0)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ACTIVE
    assert nxt.bulu_attack_count == 1


def test_active_ignores_attack_logs_from_opponent(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[bulu]), _Player()], turn=8)
    obs = _Obs(state, logs=[_attack_log(1, bulu.serial)])  # playerIndex=1(相手)のログ
    prev = mod.OgerponOptionState(mode=mod.ACTIVE, phase=mod.ACTIVE,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.bulu_attack_count == 0


# --- ABORT 条件 -----------------------------------------------------------------

def test_build_aborts_when_target_vanishes(mod):
    state = _State([_Player(bench=[]), _Player()], turn=4)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD,
                                  target_serial=999, target_card_id=TAPU_BULU,
                                  build_deadline_turn=10)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ABORT
    assert nxt.abort_reason == "target_lost_or_replaced"


def test_build_aborts_when_target_replaced_by_different_card(mod):
    """同じserialが別カードに再利用されていたら(別個体への置換)ABORTする。"""
    replaced = _Pokemon(OGERPON_EX, [])
    replaced.serial = 999  # 元の対象と同じserial値だが別カード
    state = _State([_Player(bench=[replaced]), _Player()], turn=4)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD,
                                  target_serial=999, target_card_id=TAPU_BULU,
                                  build_deadline_turn=10)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ABORT
    assert nxt.abort_reason == "target_lost_or_replaced"


def test_build_aborts_past_deadline(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 2)  # まだ未完成
    state = _State([_Player(bench=[bulu]), _Player()], turn=11)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU,
                                  started_turn=2, build_deadline_turn=10)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ABORT
    assert nxt.abort_reason == "build_deadline_exceeded"


def test_active_aborts_if_retreated_unexpectedly(mod):
    """ACTIVE中の自発的な交代は原則拒否する設計なので、ベンチへ移動していたら想定外としてABORT。"""
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player(active=[], bench=[bulu]), _Player()], turn=8)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.ACTIVE, phase=mod.ACTIVE,
                                  target_serial=bulu.serial, target_card_id=TAPU_BULU)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.ABORT
    assert nxt.abort_reason == "retreated_unexpectedly"


def test_terminal_states_stay_until_reset(mod):
    for mode in (mod.COMPLETE, mod.ABORT):
        state = _State([_Player(), _Player()], turn=9)
        obs = _Obs(state)
        prev = mod.OgerponOptionState(mode=mode, phase=mode)
        nxt = mod.advance_state(prev, obs, 0, {})
        assert nxt.mode == mode


def test_match_end_leaves_state_untouched(mod):
    state = _State([_Player(), _Player()], turn=9, result=0)
    obs = _Obs(state)
    prev = mod.OgerponOptionState(mode=mod.BUILD, phase=mod.BUILD, target_serial=1)
    nxt = mod.advance_state(prev, obs, 0, {})
    assert nxt.mode == mod.BUILD  # 試合終了時は判断を進めない(現状維持)


# --- 個体識別: カプ・ブルル以外は対象にしない -----------------------------------

def test_does_not_target_arbitrary_non_ex_on_build_start(mod):
    other = _Pokemon(741, [], max_hp=140)  # カプ・ブルルではない非ex
    state = _State([_Player(bench=[other]), _Player()], turn=2)
    obs = _Obs(state)
    prev = mod.OgerponOptionState()
    nxt = mod.advance_state(prev, obs, 0, {}, force_start=True)
    assert nxt.mode == mod.IDLE, "カプ・ブルル以外は対象に取ってはいけない"


# --- 例外安全・reset --------------------------------------------------------------

def test_exception_falls_back_to_prev_state(mod):
    prev = mod.OgerponOptionState(mode=mod.BUILD, target_serial=1)
    broken_obs = object()  # .current が無い壊れたobs
    nxt = mod.advance_state(prev, broken_obs, 0, {})
    assert nxt is prev


def test_reset_clears_module_global_state(mod):
    bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player(bench=[bulu]), _Player()], turn=2)
    obs = _Obs(state)
    try:
        mod.advance(obs, 0, {}, force_start=True)
        assert mod.get_state().mode == mod.BUILD
    finally:
        mod.reset()
    assert mod.get_state().mode == mod.IDLE
    assert mod.get_state().target_serial is None


def test_config_overrides_max_build_turns(mod):
    bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player(bench=[bulu]), _Player()], turn=2)
    obs = _Obs(state)
    prev = mod.OgerponOptionState()
    cfg = {"ogerpon_option_state": {"max_build_turns": 1}}
    nxt = mod.advance_state(prev, obs, 0, cfg, force_start=True)
    assert nxt.build_deadline_turn == 2 + 1 * 2


def test_force_start_test_mode_config_flag(mod):
    """force_start引数無しでも、config の force_start_test_mode=True なら BUILD へ入る。"""
    bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player(bench=[bulu]), _Player()], turn=2)
    obs = _Obs(state)
    prev = mod.OgerponOptionState()
    cfg = {"ogerpon_option_state": {"force_start_test_mode": True}}
    nxt = mod.advance_state(prev, obs, 0, cfg)
    assert nxt.mode == mod.BUILD
