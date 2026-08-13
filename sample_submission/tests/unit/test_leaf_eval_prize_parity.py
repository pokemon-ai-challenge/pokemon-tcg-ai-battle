"""leaf_eval のサイド価値(必要KO数)項のユニットテスト。

再現したい実戦の問題:
    オーガポンデッキ(オーガポンex ×4 = サイド2枚 / カプ・ブルル ×1 = サイド1枚)で、
    サイドが偶数のときにカプ・ブルルを使わずオーガポンexだけで戦って負ける。

従来の評価関数はサイドの**枚数**しか見ないため、
「全部 ex の盤面」と「1枚ポケモンを混ぜた盤面」を区別できない(＝カプ・ブルルを
前に出す価値がゼロに見える)。この項はその区別を評価値に入れる。
"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX_ID = 96      # オーガポン みどりのめんex (ex, サイド2枚)
TAPU_BULU_ID = 920      # カプ・ブルル (非ex, サイド1枚)


@pytest.fixture
def leaf_eval():
    try:
        from ptcg_ai.search import leaf_eval as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"leaf_eval / cg engine unavailable: {exc}")
    return mod


class _Pokemon:
    def __init__(self, card_id, hp=200, max_hp=200, energies=()):
        self.id = card_id
        self.hp = hp
        self.maxHp = max_hp
        self.energies = list(energies)


class _Player:
    def __init__(self, active, bench=(), prize_count=6, hand_count=5):
        self.active = list(active)
        self.bench = list(bench)
        self.prize = [object()] * prize_count
        self.handCount = hand_count


class _State:
    def __init__(self, players, result=-1):
        self.players = players
        self.result = result


def _state(my_active_ids, my_bench_ids, my_prizes, opp_prizes, opp_active_ids=(OGERPON_EX_ID,)):
    mine = _Player([_Pokemon(i) for i in my_active_ids],
                   [_Pokemon(i) for i in my_bench_ids], my_prizes)
    opp = _Player([_Pokemon(i) for i in opp_active_ids], [], opp_prizes)
    return _State([mine, opp])


def test_prize_value_distinguishes_ex_from_non_ex(leaf_eval):
    assert leaf_eval._prize_value(_Pokemon(OGERPON_EX_ID)) == 2
    assert leaf_eval._prize_value(_Pokemon(TAPU_BULU_ID)) == 1


def test_kos_needed_all_ex_vs_mixed_board(leaf_eval):
    """残サイド6。全部 ex なら相手は3KOで勝てるが、1枚ポケモンを1体挟むと4KO必要。"""
    all_ex = _Player([_Pokemon(OGERPON_EX_ID)],
                     [_Pokemon(OGERPON_EX_ID), _Pokemon(OGERPON_EX_ID)], 6)
    mixed = _Player([_Pokemon(TAPU_BULU_ID)],
                    [_Pokemon(OGERPON_EX_ID), _Pokemon(OGERPON_EX_ID)], 6)
    assert leaf_eval._kos_needed_against(all_ex) == pytest.approx(3.0)
    assert leaf_eval._kos_needed_against(mixed) == pytest.approx(4.0)


def test_default_evaluator_is_blind_to_prize_parity(leaf_eval):
    """既定(係数0)では従来どおり、1枚ポケモンの有無で評価が変わらない = 後方互換。"""
    ev = leaf_eval.HandcraftedEvaluator()
    all_ex = _state([OGERPON_EX_ID], [OGERPON_EX_ID, OGERPON_EX_ID], 6, 6)
    mixed = _state([TAPU_BULU_ID], [OGERPON_EX_ID, OGERPON_EX_ID], 6, 6)
    # HP/エネルギーは同一に作ってあるので、従来評価では完全一致するはず。
    assert ev.evaluate(all_ex, 0) == pytest.approx(ev.evaluate(mixed, 0))


def test_prize_parity_term_prefers_one_prize_attacker(leaf_eval):
    """係数を有効にすると、1枚ポケモンを前に出した盤面の方が高く評価される。

    これがユーザー報告「サイドが偶数のときカプ・ブルルを使わずオーガポンだけで戦って負ける」
    に対応する修正の中核。
    """
    ev = leaf_eval.HandcraftedEvaluator(prize_parity_coeff=0.30)
    all_ex = _state([OGERPON_EX_ID], [OGERPON_EX_ID, OGERPON_EX_ID], 6, 6)
    mixed = _state([TAPU_BULU_ID], [OGERPON_EX_ID, OGERPON_EX_ID], 6, 6)
    assert ev.evaluate(mixed, 0) > ev.evaluate(all_ex, 0)


def test_build_evaluator_defaults_to_disabled(leaf_eval):
    ev = leaf_eval.build_evaluator({"kind": "handcrafted"})
    assert ev.prize_parity_coeff == 0.0


def test_build_evaluator_reads_coeff_from_config(leaf_eval):
    ev = leaf_eval.build_evaluator({"kind": "handcrafted", "prize_parity_coeff": 0.25})
    assert ev.prize_parity_coeff == pytest.approx(0.25)


def test_unknown_card_falls_back_to_one_prize(leaf_eval):
    """カードデータを引けないIDでも例外にせず、安全側(1枚)に倒す。"""
    assert leaf_eval._prize_value(_Pokemon(9999999)) == 1


def test_evaluator_never_raises_on_broken_state(leaf_eval):
    ev = leaf_eval.HandcraftedEvaluator(prize_parity_coeff=0.30)
    assert ev.evaluate(None, 0) == pytest.approx(0.5)
