"""eval_bulu_loop.py の純粋関数(self-playを回さずに検証できる部分)のテスト。

design.md(ogerpon_single_prize_mlp_design.md) Phase1 item4-6: 「tracker判定を
純関数または小さな状態クラスへ分離し、self-playを回さずに境界条件を単体テストする」
に対応する。状態遷移そのもの(BUILD/READY/ACTIVE/...)は
sample_submission/tests/unit/test_ogerpon_option_state.py で別途検証済み。
"""

import sys
from pathlib import Path

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _RL_DIR.parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _p in (str(_RL_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OGERPON_EX = 96
TAPU_BULU = 920
GRASS = 1


@pytest.fixture
def mod():
    try:
        import eval_bulu_loop as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"eval_bulu_loop / cg engine unavailable: {exc}")
    return m


class _Pokemon:
    _n = [5000]

    def __init__(self, card_id, energies=(), hp=None, max_hp=None):
        self.id = card_id
        self.energies = list(energies)
        self.maxHp = max_hp if max_hp is not None else (210 if card_id == OGERPON_EX else 140)
        self.hp = hp if hp is not None else self.maxHp
        _Pokemon._n[0] += 1
        self.serial = _Pokemon._n[0]


class _Player:
    def __init__(self, active=(), bench=()):
        self.active = list(active)
        self.bench = list(bench)


# --- detect_ko_after_bulu_attack ------------------------------------------------

def test_ko_detected_when_target_vanishes(mod):
    opp = _Player(active=[], bench=[])
    assert mod.detect_ko_after_bulu_attack(999, opp) is True


def test_ko_not_detected_when_target_still_active(mod):
    target = _Pokemon(OGERPON_EX)
    opp = _Player(active=[target], bench=[])
    assert mod.detect_ko_after_bulu_attack(target.serial, opp) is False


def test_ko_not_detected_when_target_retreated_to_bench(mod):
    """気絶ではなくベンチへ退避しただけならKOではない。"""
    target = _Pokemon(OGERPON_EX)
    opp = _Player(active=[], bench=[target])
    assert mod.detect_ko_after_bulu_attack(target.serial, opp) is False


def test_ko_not_detected_when_no_target_serial(mod):
    opp = _Player(active=[], bench=[])
    assert mod.detect_ko_after_bulu_attack(None, opp) is False


# --- classify_emergency_retreat --------------------------------------------------

def test_emergency_when_cannot_attack(mod):
    bulu = _Pokemon(TAPU_BULU, [])  # 攻撃不能
    opp_active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    assert mod.classify_emergency_retreat(bulu, opp_active, {}) is True


def test_emergency_when_facing_certain_ko(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4, hp=1)  # 攻撃可能だが確実に倒される
    opp_active = _Pokemon(TAPU_BULU, [GRASS] * 4)  # 攻撃可能・相手も十分なエネ
    assert mod.classify_emergency_retreat(bulu, opp_active, {}) is True


def test_not_emergency_when_safe_and_ready(mod):
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4, hp=140)  # 攻撃可能・満タン
    opp_active = _Pokemon(OGERPON_EX, [])  # エネ0 = 今は確実に払えるコストが無い
    assert mod.classify_emergency_retreat(bulu, opp_active, {}) is False


def test_emergency_when_target_unknown(mod):
    assert mod.classify_emergency_retreat(None, None, {}) is True
