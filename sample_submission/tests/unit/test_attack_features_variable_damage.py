"""クラスタ② 盤面評価／担当B

attack_features.resolve_damage / can_ko の「盤面依存の可変ダメージ技」回帰テスト
（2026-08-12、カミツオロチexデッキ対応）。

対象は次の3ワザ（すべて実カードID、attack.text はカード原文そのまま）:

    150 カミツオロチex  Syrup Storm (damage=30)
        "This attack does 30 more damage for each {G} Energy attached to all of your Pokémon."
        -> 30 + 30 * (自分の場全員についている【草】エネルギーの合計本数)

    93  カミッチュ       Do the Wave (damage=0)
        "This attack does 20 damage for each of your Benched Pokémon."
        -> 20 * (自分のベンチ数)
        修正前は _FIXED_DAMAGE_PATTERN("does (\\d+) damage")に誤マッチし、ベンチ数に
        関係なく固定20と評価されていた（本ファイルの regression 対象）。

    96  オーガポンみどりのめんex  Myriad Leaf Shower (damage=30)
        "This attack does 30 more damage for each Energy attached to both Active Pokémon."
        -> 30 + 30 * (自分と相手、両方のバトル場についているエネルギーの合計本数)

710 メガニウム「おいしげる」（基本【草】エネルギーを2個ぶんとして扱う）については、cg
エンジンが Pokemon.energies に既に倍化後の実効本数を入れて返すことを自己対戦で実測して
確認済み（energyCards 5枚 に対し energies が10要素になった）。したがって
attack_features 側はメガニウムの有無を判定せず、Pokemon.energies をそのまま数える
だけでよい。本ファイルのメガニウム系テストは、その実測結果を模して energies を
2倍で構築することで再現する（実際にゲームエンジンを介さずに検証するため）。

Run from sample_submission/:
    python -m pytest tests/unit/test_attack_features_variable_damage.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import EnergyType, Pokemon, all_attack
from ptcg_ai.board_evaluation.attack_features import can_ko, resolve_damage

# ---------------------------------------------------------------------------
# 実カードID
# ---------------------------------------------------------------------------

KAMITSUOROCHI_EX = 150      # Syrup Storm: 30 + 30/【草】エネ
KAMICCHU = 93               # Do the Wave: 20 x ベンチ数
OGERPON_TEAL_EX = 96        # Myriad Leaf Shower: 30 + 30/両バトル場エネ
MEGANIUM = 710              # おいしげる（本ファイルではエンジンの実測挙動を模して energies を倍化するだけで、
                             # このIDそのものへの依存はコード側に無い）
HOUDINI_ALAKAZAM = 743      # Powerful Hand: 対照（手札依存の効果ダメージ、既存挙動）
DUMMY_MON = 999              # 弱点/抵抗力の判定に関与しない任意の防御側/添え物ポケモン

_ATTACKS = {a.attackId: a for a in all_attack()}


def _attack_of(card_id: int):
    """カードIDが持つ唯一の attack を返す（本ファイルで使う4枚はいずれもワザ1つだけ）。"""
    from ptcg_ai.shared import card_cache

    card = card_cache.get_card(card_id)
    assert len(card.attacks) >= 1
    return card_cache.get_attack(card.attacks[0])


SYRUP_STORM = _attack_of(KAMITSUOROCHI_EX)
DO_THE_WAVE = _attack_of(KAMICCHU)
MYRIAD_LEAF_SHOWER = _attack_of(OGERPON_TEAL_EX)
POWERFUL_HAND = _attack_of(HOUDINI_ALAKAZAM)


def mkmon(
    card_id: int,
    *,
    energies: list[EnergyType] | None = None,
    serial: int = 1,
    hp: int = 300,
    max_hp: int = 300,
) -> Pokemon:
    """テスト用の最小 Pokemon。energyCards は使わない（可変ダメージ推定は energies だけ見る）。"""
    return Pokemon(
        id=card_id, serial=serial, hp=hp, maxHp=max_hp, appearThisTurn=False,
        energies=energies or [], energyCards=[], tools=[], preEvolution=[],
    )


def grass_energy_side(n: int, *, meganium: bool = False, serial_base: int = 1) -> list[Pokemon | None]:
    """カミツオロチexが先頭(バトル場)、【草】エネルギーを n 枚背負わせる。

    meganium=True のときは、エンジンが実際に返す実効値(2倍)を模して energies を
    n*2 個の GRASS で構築する（おいしげる自体のカード効果は attack_features 側では
    一切見ない。上のモジュール docstring 参照）。
    """
    energy_count = n * 2 if meganium else n
    attacker = mkmon(
        KAMITSUOROCHI_EX, energies=[EnergyType.GRASS] * energy_count, serial=serial_base
    )
    side: list[Pokemon | None] = [attacker]
    if meganium:
        side.append(mkmon(MEGANIUM, serial=serial_base + 1))
    return side


# ---------------------------------------------------------------------------
# 150 カミツオロチex: 30 + 30 * 【草】エネ合計
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grass_count,expected", [(0, 30), (2, 90), (5, 180)])
def test_kamitsuorochi_scales_with_grass_energy(grass_count, expected):
    side = grass_energy_side(grass_count)
    attacker = side[0]
    damage = resolve_damage(SYRUP_STORM, attacker, None, None, attacker_side_pokemon=side)
    assert damage == expected


@pytest.mark.parametrize("grass_count,expected", [(0, 30), (2, 150), (5, 330)])
def test_kamitsuorochi_with_meganium_doubles_grass_energy(grass_count, expected):
    """メガニウム下では実質 30 + 60*n（HP330のexをエネ5枚でワンパンできる想定値）。"""
    side = grass_energy_side(grass_count, meganium=True)
    attacker = side[0]
    damage = resolve_damage(SYRUP_STORM, attacker, None, None, attacker_side_pokemon=side)
    assert damage == expected


def test_kamitsuorochi_can_ko_wires_attacker_side_pokemon():
    """can_ko() 経由でも盤面依存の可変ダメージが正しく効く（best_attack_damage だけでなく
    can_ko 特徴も過小評価されていたのが今回の主目的）。"""
    side = grass_energy_side(5, meganium=True)  # 330打点
    attacker = side[0]
    defender = mkmon(DUMMY_MON, serial=50, hp=330, max_hp=330)  # ちょうどHP330のex
    assert can_ko(
        SYRUP_STORM, attacker, defender, None, None, attacker_side_pokemon=side
    ) is True

    weaker_side = grass_energy_side(2)  # 90打点、330には届かない
    weaker_attacker = weaker_side[0]
    assert can_ko(
        SYRUP_STORM, weaker_attacker, defender, None, None, attacker_side_pokemon=weaker_side
    ) is False


# ---------------------------------------------------------------------------
# 93 カミッチュ: 20 * ベンチ数（_FIXED_DAMAGE_PATTERN への誤マッチ回帰）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bench_count,expected", [(0, 0), (3, 60), (5, 100)])
def test_kamicchu_scales_with_bench_count(bench_count, expected):
    attacker = mkmon(KAMICCHU, serial=1)
    bench = [mkmon(917, serial=10 + i) for i in range(bench_count)]
    side: list[Pokemon | None] = [attacker] + bench
    damage = resolve_damage(DO_THE_WAVE, attacker, None, None, attacker_side_pokemon=side)
    assert damage == expected


def test_kamicchu_zero_bench_is_not_misparsed_as_fixed_twenty():
    """回帰の主目的: "does 20 damage for each of your Benched Pokémon" が
    _FIXED_DAMAGE_PATTERN の "does (\\d+) damage" に誤マッチして、ベンチ0体でも
    固定20点と誤判定されていたバグ（修正前の実測値）が直っていることを確認する。"""
    attacker = mkmon(KAMICCHU, serial=1)
    side: list[Pokemon | None] = [attacker]  # ベンチ0体
    damage = resolve_damage(DO_THE_WAVE, attacker, None, None, attacker_side_pokemon=side)
    assert damage == 0  # 修正前は 20 を返していた


# ---------------------------------------------------------------------------
# 96 オーガポンみどりのめんex: 30 + 30 * 両バトル場エネ合計
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_energy,expected", [(0, 30), (3, 120)])
def test_ogerpon_scales_with_both_actives_energy(total_energy, expected):
    # 自分側に total_energy 分、相手側は0にして合計を作る（対称性は別テストで見る必要はない）。
    attacker = mkmon(OGERPON_TEAL_EX, energies=[EnergyType.COLORLESS] * total_energy, serial=1)
    defender_active = mkmon(DUMMY_MON, serial=2, energies=[])
    side: list[Pokemon | None] = [attacker]
    damage = resolve_damage(
        MYRIAD_LEAF_SHOWER, attacker, None, None, attacker_side_pokemon=side, defender=defender_active
    )
    assert damage == expected


def test_ogerpon_counts_defender_active_energy_too():
    """両者のエネを分けて持たせても合計で効く(自分1枚+相手2枚=3枚 -> 120)。"""
    attacker = mkmon(OGERPON_TEAL_EX, energies=[EnergyType.COLORLESS] * 1, serial=1)
    defender_active = mkmon(DUMMY_MON, serial=2, energies=[EnergyType.COLORLESS] * 2)
    side: list[Pokemon | None] = [attacker]
    damage = resolve_damage(
        MYRIAD_LEAF_SHOWER, attacker, None, None, attacker_side_pokemon=side, defender=defender_active
    )
    assert damage == 120


# ---------------------------------------------------------------------------
# 後方互換性: 新引数を渡さない既存呼び出しの戻り値は不変
# ---------------------------------------------------------------------------


def test_backward_compatible_call_without_new_args_unchanged():
    """attacker_side_pokemon / defender_active_pokemon を渡さない4引数呼び出しは、
    盤面依存パターンの判定を一切行わない（完全後方互換）。"""
    attacker = mkmon(KAMITSUOROCHI_EX, energies=[EnergyType.GRASS] * 5, serial=1)
    # 盤面コンテキストを渡さなければ、attack.damage(=30)がそのまま使われる
    # (可変ダメージパターンは判定されない)。
    damage = resolve_damage(SYRUP_STORM, attacker, None, None)
    assert damage == 30

    attacker2 = mkmon(KAMICCHU, serial=1)
    damage2 = resolve_damage(DO_THE_WAVE, attacker2, None, None)
    # 盤面コンテキストを渡さない場合は Benched パターンを判定せず、_FIXED_DAMAGE_PATTERN
    # ("does (\d+) damage") への誤マッチという修正前からの挙動がそのまま残る
    # （= 完全後方互換。呼び出し側が attacker_side_pokemon を渡して初めて正しい0になる。
    # 実際の5箇所の呼び出し口は全て今回 attacker_side_pokemon を渡すよう更新済み）。
    assert damage2 == 20


def test_backward_compatible_defender_only_call_unchanged_for_unrelated_attack():
    """defender だけを渡す既存呼び出し（A8/A10 で追加済みの防壁判定用）は、
    カミツオロチ系と無関係な技には一切影響しない。"""
    attacker = mkmon(HOUDINI_ALAKAZAM, serial=1)
    defender = mkmon(DUMMY_MON, serial=2)
    damage = resolve_damage(
        POWERFUL_HAND, attacker, None, None, attacker_hand_size=3, defender=defender
    )
    assert damage == 3 * 10 * 2  # 手札3枚 x ダメカン2個 x 10点 = 60（既存の効果ダメージ推定）


# ---------------------------------------------------------------------------
# 743 フーディン(Powerful Hand): 既存挙動が不変であることの対照テスト
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hand_size,expected", [(0, 0), (3, 60), (10, 200)])
def test_houdini_powerful_hand_unaffected_by_board_context(hand_size, expected):
    """盤面依存パターン群はカミツオロチ系のテキストにしか一致しないため、
    attacker_side_pokemon を渡しても渡さなくても Powerful Hand の推定は変わらない。"""
    attacker = mkmon(HOUDINI_ALAKAZAM, serial=1)
    side: list[Pokemon | None] = [attacker]
    defender = mkmon(DUMMY_MON, serial=2)

    without_ctx = resolve_damage(
        POWERFUL_HAND, attacker, None, None, attacker_hand_size=hand_size
    )
    with_ctx = resolve_damage(
        POWERFUL_HAND, attacker, None, None, attacker_hand_size=hand_size,
        attacker_side_pokemon=side, defender=defender,
    )
    assert without_ctx == with_ctx == expected
