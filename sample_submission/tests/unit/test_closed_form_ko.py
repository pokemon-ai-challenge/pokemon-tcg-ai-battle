"""`ptcg_ai.search.closed_form_ko`(閉形式KO判定)のユニットテスト。

検証の軸は2つ:

1. **打点の式が正しいこと**: まんようしぐれ(attackId 120)= 30 + 30 ×(両バトルポケモンの
   エネ数)。数値は実リプレイのログと突き合わせた実測値を使う
   (`kaggle_replays/_probe_closed_form_ko.py`。250件中218件が素の式 or 弱点×2 で一致、
   残る32件は全て「ダメージ無効化」局面で本モジュールが判定不能を返す側)。
2. **保守性(=間違った True/False を返さないこと)**: 抵抗持ち・ダメージ改変カードが場に
   ある・未知のワザ、のいずれでも ``None``(判定不能)に落ちること。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_OGERPON = 96          # オーガポン みどりのめん ex(ワザ=まんようしぐれ 120、草・テラスタル)
_GRASS = 1             # EnergyType.GRASS
_GRASS_ENERGY_CARD = 1  # 基本【草】エネルギー(カードID)
_CRUSTLE = 345         # 「ふしぎないわやど」= ex のワザのダメージを完全に無効化する特性
_CHANDELURE = 98       # ワザ「マインドルーラー」= 未知パターン(手札枚数依存)
_NEUTRAL_OPPONENT = 506  # クマシュン(弱点=鋼・抵抗なし・特性なし)= 弱点補正が絡まない相手
_GRASS_WEAK_OPPONENT = 648  # マーニーのオーロンゲex(草弱点=打点×2)


@pytest.fixture(scope="module")
def cfk():
    try:
        from ptcg_ai.search import closed_form_ko as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"closed_form_ko / cg engine unavailable: {exc}")
    return mod


# ---------------------------------------------------------------------------
# 盤面スタブ(closed_form_ko はダックタイピングでしか触らないので軽量に作る)
# ---------------------------------------------------------------------------

def _mon(card_id: int, hp: int, energy_count: int, *, energy_type=_GRASS,
         tools=(), energy_cards=()):
    return SimpleNamespace(
        id=card_id, serial=card_id * 10, hp=hp, maxHp=hp, appearThisTurn=False,
        energies=[energy_type] * energy_count,
        energyCards=list(energy_cards), tools=list(tools), preEvolution=[],
    )


def _card(card_id: int):
    return SimpleNamespace(id=card_id, serial=card_id * 100, playerIndex=0)


def _player(active=None, bench=(), hand=(), deck_count=10):
    return SimpleNamespace(
        active=[active] if active is not None else [],
        bench=list(bench), benchMax=5, deckCount=deck_count,
        discard=[], prize=[None] * 3, handCount=len(hand), hand=list(hand),
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def _state(me_player, opp_player, *, me=0, stadium=None, energy_attached=False):
    players = [me_player, opp_player] if me == 0 else [opp_player, me_player]
    return SimpleNamespace(
        turn=9, turnActionCount=1, yourIndex=me, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=energy_attached,
        retreated=False, result=-1,
        stadium=[] if stadium is None else [_card(stadium)],
        looking=None, players=players,
    )


def _board(my_energy: int, opp_energy: int, opp_hp: int, *, opp_id=_NEUTRAL_OPPONENT,
           my_bench=(), opp_bench=(), hand=(), stadium=None, energy_attached=False,
           my_id=_OGERPON, my_hp=210):
    """自分=オーガポン、相手=``opp_id`` の最小盤面。"""
    return _state(
        _player(_mon(my_id, my_hp, my_energy), bench=my_bench, hand=hand),
        _player(_mon(opp_id, opp_hp, opp_energy), bench=opp_bench),
        stadium=stadium, energy_attached=energy_attached,
    )


# ---------------------------------------------------------------------------
# 1. 打点の式(実リプレイの実測値と同じ数値で固定する)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "my_energy, opp_energy, expected, source",
    [
        (6, 0, 210, "ep93517227 row78 T5 実ログ210"),
        (8, 0, 270, "ep93517227 row106 T7 実ログ270"),
        (8, 1, 300, "ep93517227 row126 T9 実ログ300"),
        (4, 1, 180, "ep93517227 row148 T11 実ログ180"),
        (3, 6, 300, "ep93510303 row84 T6 実ログ300"),
        (7, 4, 360, "ep93510303 row141 T10 実ログ360"),
        (5, 8, 420, "ep93528233 row99 T6 実ログ420"),
        (14, 0, 450, "ep93542157 row109 T16 実ログ450"),
        (3, 2, 180, "ep93538707 row76 T5 実ログ180"),
        (4, 2, 210, "ep93538707 row104 T7 実ログ210"),
    ],
)
def test_myriad_leaf_shower_matches_replay_damage(cfk, my_energy, opp_energy, expected, source):
    """まんようしぐれ = 30 + 30×(両バトル場のエネ数)。実ログのダメージ値と一致すること。"""
    state = _board(my_energy, opp_energy, 300)
    assert cfk.estimate_attack_damage(state, 0, cfk.MYRIAD_LEAF_SHOWER_ID) == expected, source
    assert cfk.estimate_current_damage(state, 0) == expected, source


def test_known_attack_constants_match_card_data(cfk):
    """既知パターンの定数がカードデータ(all_attack)とズレていないこと。"""
    assert cfk.verify_known_attacks() == []


def test_extra_energy_adds_30_each(cfk):
    state = _board(4, 1, 300)
    assert cfk.estimate_current_damage(state, 0) == 180
    assert cfk.estimate_current_damage(state, 0, extra_energy=1) == 210
    assert cfk.estimate_current_damage(state, 0, extra_energy=3) == 270


def test_unpayable_cost_returns_none(cfk):
    """コスト(草3)を払えないなら打点は判定不能。エネを足して払えるようになれば値が出る。"""
    state = _board(2, 0, 300)
    report: dict = {}
    assert cfk.estimate_current_damage(state, 0, report=report) is None
    assert report["reason"] == "no_payable_attack"
    assert cfk.estimate_current_damage(state, 0, extra_energy=1) == 120


def test_unknown_attacker_returns_none(cfk):
    """既知パターン外のワザを持つポケモンがアクティブなら判定不能。"""
    state = _board(5, 0, 300, my_id=_CHANDELURE, my_hp=130)
    report: dict = {}
    assert cfk.estimate_current_damage(state, 0, report=report) is None
    assert report["reason"] == "unknown_attack"


def test_nighttime_mine_surcharge_blocks_tera_attack(cfk):
    """夜の鉱山(1266)下ではテラスタルのワザに【無】1個ぶん上乗せ=草3では撃てない。"""
    state = _board(3, 0, 300, stadium=1266)
    assert cfk.estimate_current_damage(state, 0) is None
    assert cfk.estimate_current_damage(state, 0, extra_energy=1) == 30 + 30 * 4


# ---------------------------------------------------------------------------
# 2. can_ko_now / can_ko_with_more_energy(3値)
# ---------------------------------------------------------------------------

def test_can_ko_now_true_when_damage_reaches_hp(cfk):
    """打点300 >= 残HP70 なら True(実例: ep93517227 row122 の局面)。"""
    state = _board(8, 1, 70)
    report: dict = {}
    assert cfk.can_ko_now(state, 0, report=report) is True
    assert report["damage"] == 300
    assert report["remaining_hp"] == 70


def test_can_ko_now_exact_boundary_is_true(cfk):
    """ちょうど残HP と同値ならKO成立(> ではなく >=)。"""
    assert cfk.can_ko_now(_board(4, 1, 180), 0) is True
    assert cfk.can_ko_now(_board(4, 1, 190), 0) is False


def test_ko_margin_makes_true_stricter(cfk):
    """``ko_margin`` を積むと True 側だけが厳しくなる(False の閾値は動かない)。"""
    state = _board(4, 1, 180)  # 打点180 = 残HP180
    assert cfk.can_ko_now(state, 0, {"ko_margin": 0}) is True
    assert cfk.can_ko_now(state, 0, {"ko_margin": 30}) is None  # 余裕不足=判定不能


def test_can_ko_with_more_energy_is_monotonic(cfk):
    state = _board(3, 0, 200)
    assert cfk.can_ko_with_more_energy(state, 0, 0) is False   # 120 < 200
    assert cfk.can_ko_with_more_energy(state, 0, 2) is False   # 180 < 200
    assert cfk.can_ko_with_more_energy(state, 0, 3) is True    # 210 >= 200


# ---------------------------------------------------------------------------
# 3. 保守性: 間違った True/False を返さないこと
# ---------------------------------------------------------------------------

def test_weakness_never_used_for_true_but_blocks_false(cfk):
    """草弱点(×2)は True 判定に使わない=下界のまま。False 判定は上界(×2)で見る。

    実リプレイで×2が確認できたマーニーのオーロンゲex(648、草弱点)を使う
    (ep93471752 row99: 素300 → 実ログ600)。
    """
    from ptcg_ai.shared import card_cache
    weak_id = _GRASS_WEAK_OPPONENT
    assert int(card_cache.get_card(weak_id).weakness) == _GRASS  # 前提

    # 素の打点150 / 弱点込み300。残HP 200 は「素では届かない・弱点込みなら届く」帯。
    state = _board(4, 0, 200, opp_id=weak_id)
    assert cfk.estimate_current_damage(state, 0) == 150
    assert cfk.can_ko_now(state, 0) is None  # False とは言い切れない=判定不能

    # 残HP 400 は弱点込み(300)でも届かない=確実に False。
    assert cfk.can_ko_now(_board(4, 0, 400, opp_id=weak_id), 0) is False

    # 素の打点だけで足りるなら弱点を使わずに True。
    assert cfk.can_ko_now(_board(4, 0, 140, opp_id=weak_id), 0) is True


def test_damage_modifier_on_board_forces_none(cfk):
    """ダメージ無効化特性(イワパレス345「ふしぎないわやど」)が場にあれば判定不能。

    実測: ep93542157 / 93547571 / 93503044 では、閉形式の見積り 150〜570 に対し
    実ログのダメージが 0(完全無効化)だった。この局面で True を返してはならない。
    """
    state = _board(8, 1, 70, opp_id=_CRUSTLE)
    report: dict = {}
    assert cfk.can_ko_now(state, 0, report=report) is None
    assert report["reason"] == "damage_modifier_in_play"
    assert report["modifier_card_id"] == _CRUSTLE


def test_damage_modifier_on_opponent_bench_also_forces_none(cfk):
    """ベンチ発動のオーラもバトル場のダメージを変えうるのでベンチまで見る。"""
    state = _board(8, 1, 70, opp_bench=[_mon(_CRUSTLE, 150, 0)])
    assert cfk.can_ko_now(state, 0) is None


def test_battle_cage_is_not_treated_as_damage_modifier(cfk):
    """バトルケージ(1264)は「ベンチにダメカンが置かれるのを防ぐ(ワザのダメージは受ける)」

    ので判定不能にしない。実測でこの取りこぼしが250件中13件あった。
    """
    state = _board(8, 1, 70, stadium=1264)
    assert cfk.can_ko_now(state, 0) is True


def test_no_opponent_active_returns_none(cfk):
    state = _state(_player(_mon(_OGERPON, 210, 5)), _player(None))
    report: dict = {}
    assert cfk.can_ko_now(state, 0, report=report) is None
    assert report["reason"] == "no_opponent_active"


# ---------------------------------------------------------------------------
# 4. is_ko_impossible_this_turn(健全な上界判定)
# ---------------------------------------------------------------------------

def test_ko_impossible_when_no_attack_is_payable(cfk):
    """場の全エネを1体に集めても草3を払えないなら「攻撃でKOする手は存在しない」。"""
    state = _board(1, 0, 60, my_bench=[_mon(_OGERPON, 210, 1)], hand=[])
    report: dict = {}
    assert cfk.is_ko_impossible_this_turn(state, 0, report=report) is True
    assert report["reason"] == "no_payable_attack"


def test_ko_not_impossible_when_hand_energy_can_enable_attack(cfk):
    """手札の草エネで払えるようになるなら「不可能」とは言わない。"""
    state = _board(1, 0, 60, my_bench=[_mon(_OGERPON, 210, 1)],
                   hand=[_card(_GRASS_ENERGY_CARD)] * 3)
    assert cfk.is_ko_impossible_this_turn(state, 0) is False


def test_ko_impossible_when_upper_bound_damage_cannot_reach(cfk):
    """上界打点(場の総エネ+手札から足せるぶん)でも相手の最小残HPに届かないなら True。"""
    state = _board(3, 0, 5000, hand=[])
    report: dict = {}
    assert cfk.is_ko_impossible_this_turn(state, 0, report=report) is True
    assert report["max_damage"] == 30 + 30 * 3
    assert report["min_opponent_hp"] == 5000


def test_ko_impossible_uses_min_hp_over_all_opponent_pokemon(cfk):
    """ボスの指令で引きずり出せる相手も含めて最小残HPで見る(=不可能と言いにくくする)。"""
    state = _board(3, 0, 5000, opp_bench=[_mon(_NEUTRAL_OPPONENT, 60, 0)], hand=[])
    assert cfk.is_ko_impossible_this_turn(state, 0) is False


def test_ko_impossible_is_none_when_damage_modifier_present(cfk):
    state = _board(1, 0, 60, opp_id=_CRUSTLE, hand=[])
    assert cfk.is_ko_impossible_this_turn(state, 0) is None


def test_ko_impossible_is_none_for_unknown_attacker(cfk):
    state = _board(3, 0, 5000, my_id=_CHANDELURE, my_hp=130, hand=[])
    report: dict = {}
    assert cfk.is_ko_impossible_this_turn(state, 0, report=report) is None
    assert report["reason"] == "unknown_attack"


# ---------------------------------------------------------------------------
# 5. 例外・欠損入力で落ちないこと(意思決定を止めない)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", ["can_ko_now", "is_ko_impossible_this_turn"])
def test_none_state_is_safe(cfk, fn):
    assert getattr(cfk, fn)(None, 0) is None


def test_broken_state_is_safe(cfk):
    broken = SimpleNamespace()
    assert cfk.can_ko_now(broken, 0) is None
    assert cfk.is_ko_impossible_this_turn(broken, 0) is None
    assert cfk.estimate_current_damage(broken, 0) is None
