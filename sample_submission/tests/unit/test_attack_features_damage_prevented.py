"""クラスタ② 盤面評価／担当B

attack_features.damage_prevented / damage_is_effect_based の回帰テスト。

board_evaluation/attack_features.py 冒頭のコメントにある通り、防御側の特性・特殊エネルギーに
よるダメージ無効化は「頻出パターンだけを保守的に扱う」実装であり、カード効果テキストの
語順やカード固有の条件節を誤読すると新しい誤発火（本来通るはずのダメージ/効果を止めてしまう）
を生みやすい。特に 83 Farigiraf ex は「相手のたねポケモンex」限定なのに「相手のポケモンex」と
誤読すると誤発火する、というのが本ファイルの主目的の回帰ケース。

すべて実カードID（all_card_data() で実在確認済み。card_id と実際のスキル/攻撃テキストは
本ファイル末尾のコメントを参照）を使う。fake のカードテキストは一切使わない。

Run from sample_submission/:
    python -m pytest tests/unit/test_attack_features_damage_prevented.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import Card, Pokemon, all_attack
from ptcg_ai.board_evaluation.attack_features import damage_is_effect_based, damage_prevented

# ---------------------------------------------------------------------------
# 実カードID(all_card_data()で実在確認済み。コメントは確認時点のカード名/効果の要約)
# ---------------------------------------------------------------------------

# 無効化系の特性を持つ防御側(コーディネータ指示の15枚)
POLTCHAGEIST = 28          # ベンチ限定・自衛・種別2(damage+effects)
RABSCA = 74                # ベンチ庇い・種別2
FARIGIRAF_EX = 83          # 自衛・攻撃側「たねex」限定・種別1
CORNERSTONE_OGERPON_EX = 117  # 自衛・攻撃側「特性持ち」限定・種別1。かつ basic/ex/tera 属性持ち
DREDNAW = 158              # 自衛・ダメージ量依存(200以上)につき常にFalse・種別1
MILOTIC_EX = 207           # 自衛・攻撃側Tera限定・種別2
SYLVEON = 330              # 自衛・攻撃側ex限定・種別1
SHAYMIN = 343              # ベンチ庇い・ルールボックス無し限定・種別1
CRUSTLE = 345              # 自衛・攻撃側ex限定・種別1
MISTYS_MAGIKARP = 362      # ベンチ限定・自衛・種別2
CARRACOSTA = 504           # 自衛・攻撃側「特殊エネルギー付き」限定・種別2
ANTIQUE_PLUME_FOSSIL = 1138  # ベンチ限定・自衛・種別1

# 追加軸(種別3=効果のみ)の特殊エネルギー
MIST_ENERGY = 11            # 無条件で効果のみ無効化(種別3)
ROCK_FIGHTING_ENERGY = 20   # {F}ポケモン限定で効果のみ無効化(種別3)

# 攻撃側として使う実カード(いずれも上記の無効化特性を一切持たない、条件判定用の駒)
HIPPOPOTAS = 22             # basic, ex/tera無し, 特性無し(素の攻撃側)
CORNERSTONE_AS_ATTACKER = 117  # basic な ex かつ tera かつ特性持ち(基本たねexとして流用)
MILOTIC_EX_AS_ATTACKER = 207   # basic ではない ex(たねでないex)
ALAKAZAM = 743              # 特性持ち(Psychic Draw)。basic ではない。攻撃 Powerful Hand は効果ダメージ
DUDUNSPARCE = 66            # 特性は持つが、攻撃 Land Crush は通常のワザダメージ(damage=90)

BASIC_WATER_ENERGY = 1      # 基本エネルギー(特殊エネルギーではない)

_ATTACKS = {a.attackId: a for a in all_attack()}


def mkmon(card_id: int, *, energy_cards: list[Card] | None = None, serial: int = 1) -> Pokemon:
    """テスト用の最小 Pokemon。damage_prevented が見るのは id と energyCards のみ。"""
    return Pokemon(
        id=card_id, serial=serial, hp=100, maxHp=100, appearThisTurn=False,
        energies=[], energyCards=energy_cards or [], tools=[], preEvolution=[],
    )


def mkcard(card_id: int, *, serial: int = 1) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=0)


HIPPO = mkmon(HIPPOPOTAS)
CORNERSTONE = mkmon(CORNERSTONE_AS_ATTACKER)
MILOTIC_ATTACKER = mkmon(MILOTIC_EX_AS_ATTACKER)
ALAKAZAM_MON = mkmon(ALAKAZAM)


@pytest.fixture(autouse=True)
def _clear_disable_env(monkeypatch):
    """他のテストで PTCG_DISABLE_DEFENDER_ABILITY が残らないようにする。"""
    monkeypatch.delenv("PTCG_DISABLE_DEFENDER_ABILITY", raising=False)


# ---------------------------------------------------------------------------
# 345 Crustle / 330 Sylveon: 単純な「攻撃側ex限定」(種別1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("defender_id", [CRUSTLE, SYLVEON])
def test_ex_only_barrier_blocks_ex_attacker(defender_id):
    assert damage_prevented(CORNERSTONE, mkmon(defender_id)) is True


@pytest.mark.parametrize("defender_id", [CRUSTLE, SYLVEON])
def test_ex_only_barrier_lets_non_ex_attacker_through(defender_id):
    assert damage_prevented(HIPPO, mkmon(defender_id)) is False


# ---------------------------------------------------------------------------
# 83 Farigiraf ex: 「相手のたねポケモンex」限定。回帰テストの主目的。
# 修正前の実装は "{ex}" しか見ておらず、たねでないexの攻撃側も誤って無効化していた。
# ---------------------------------------------------------------------------


def test_farigiraf_blocks_basic_ex_attacker():
    """攻撃側が「たねの ex」(Cornerstone Ogerpon exはbasic=True) -> True。"""
    assert damage_prevented(CORNERSTONE, mkmon(FARIGIRAF_EX)) is True


def test_farigiraf_does_not_block_evolved_ex_attacker():
    """攻撃側が「たねでない ex」(Milotic exはbasic=False) -> False(旧実装はここで誤発火していた)。"""
    assert damage_prevented(MILOTIC_ATTACKER, mkmon(FARIGIRAF_EX)) is False


def test_farigiraf_does_not_block_basic_non_ex_attacker():
    assert damage_prevented(HIPPO, mkmon(FARIGIRAF_EX)) is False


# ---------------------------------------------------------------------------
# 117 Cornerstone Mask Ogerpon ex: 攻撃側「特性持ち」限定(種別1=ダメージのみ)
# ---------------------------------------------------------------------------


def test_cornerstone_blocks_ability_attacker_damage():
    assert damage_prevented(ALAKAZAM_MON, mkmon(CORNERSTONE_OGERPON_EX), damage_is_effect=False) is True


def test_cornerstone_does_not_block_non_ability_attacker():
    assert damage_prevented(HIPPO, mkmon(CORNERSTONE_OGERPON_EX), damage_is_effect=False) is False


def test_cornerstone_is_damage_only_effect_passes_through():
    """種別1(ダメージのみ)なので、同じ攻撃側でも damage_is_effect=True の効果ダメージは止めない。

    これが追加軸の主目的: 743 Alakazam の Powerful Hand(手札分のダメージカウンター=効果)は
    Cornerstoneでは止まらない。旧仕様(種別を区別しない)だとここで誤って止めてしまっていた。
    """
    assert damage_prevented(ALAKAZAM_MON, mkmon(CORNERSTONE_OGERPON_EX), damage_is_effect=True) is False


# ---------------------------------------------------------------------------
# 207 Milotic ex: 攻撃側Tera限定(種別2=ダメージ+効果)
# ---------------------------------------------------------------------------


def test_milotic_blocks_tera_attacker():
    assert damage_prevented(CORNERSTONE, mkmon(MILOTIC_EX)) is True


def test_milotic_does_not_block_non_tera_attacker():
    assert damage_prevented(HIPPO, mkmon(MILOTIC_EX)) is False


def test_milotic_blocks_effect_damage_too():
    """種別2なので damage_is_effect=True でも止まる。"""
    assert damage_prevented(CORNERSTONE, mkmon(MILOTIC_EX), damage_is_effect=True) is True


# ---------------------------------------------------------------------------
# 504 Carracosta: 攻撃側「特殊エネルギー付き」限定(種別2=ダメージ+効果)
# ---------------------------------------------------------------------------


def test_carracosta_blocks_attacker_with_special_energy():
    attacker = mkmon(9001, energy_cards=[mkcard(MIST_ENERGY)])
    assert damage_prevented(attacker, mkmon(CARRACOSTA)) is True


def test_carracosta_lets_basic_energy_only_attacker_through():
    attacker = mkmon(9002, energy_cards=[mkcard(BASIC_WATER_ENERGY)])
    assert damage_prevented(attacker, mkmon(CARRACOSTA)) is False


@pytest.mark.parametrize("damage_is_effect", [False, True])
def test_carracosta_blocks_both_damage_and_effect(damage_is_effect):
    """種別2なので damage_is_effect の True/False どちらでも、条件さえ満たせば止まる。"""
    attacker = mkmon(9003, energy_cards=[mkcard(MIST_ENERGY)])
    assert damage_prevented(attacker, mkmon(CARRACOSTA), damage_is_effect=damage_is_effect) is True


# ---------------------------------------------------------------------------
# 158 Drednaw: ダメージ量依存(200以上) -> この関数では判定不能につき常にFalse(安全側)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attacker", [HIPPO, CORNERSTONE])
def test_drednaw_always_false(attacker):
    assert damage_prevented(attacker, mkmon(DREDNAW)) is False
    assert damage_prevented(attacker, mkmon(DREDNAW), damage_is_effect=True) is False


# ---------------------------------------------------------------------------
# 28 / 362 / 1138: 「特性の持ち主自身がベンチにいる間だけ」自衛
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("defender_id", [POLTCHAGEIST, MISTYS_MAGIKARP, ANTIQUE_PLUME_FOSSIL])
def test_bench_only_self_protect_true_when_benched(defender_id):
    assert damage_prevented(CORNERSTONE, mkmon(defender_id), defender_is_benched=True) is True


@pytest.mark.parametrize("defender_id", [POLTCHAGEIST, MISTYS_MAGIKARP, ANTIQUE_PLUME_FOSSIL])
def test_bench_only_self_protect_false_when_active(defender_id):
    assert damage_prevented(CORNERSTONE, mkmon(defender_id), defender_is_benched=False) is False


# ---------------------------------------------------------------------------
# 74 Rabsca: 味方のベンチポケモンを庇う型
# ---------------------------------------------------------------------------


def test_rabsca_protects_benched_ally():
    rabsca = mkmon(RABSCA, serial=1)
    target = mkmon(HIPPOPOTAS, serial=2)  # 適当な味方ベンチポケモン
    side = [rabsca, target]
    assert damage_prevented(CORNERSTONE, target, defender_side_pokemon=side, defender_is_benched=True) is True


def test_rabsca_without_side_list_is_false():
    """defender_side_pokemon=None(省略)なら味方庇い型は評価しない(安全側でFalseに倒す)。"""
    target = mkmon(HIPPOPOTAS, serial=2)
    assert damage_prevented(CORNERSTONE, target, defender_side_pokemon=None, defender_is_benched=True) is False


def test_rabsca_does_not_protect_active_target():
    rabsca = mkmon(RABSCA, serial=1)
    target = mkmon(HIPPOPOTAS, serial=2)
    side = [rabsca, target]
    assert damage_prevented(CORNERSTONE, target, defender_side_pokemon=side, defender_is_benched=False) is False


# ---------------------------------------------------------------------------
# 343 Shaymin: 味方のベンチポケモン(ルールボックス無し)を庇う型
# ---------------------------------------------------------------------------


def test_shaymin_protects_benched_ally_without_rule_box():
    shaymin = mkmon(SHAYMIN, serial=1)
    plain_target = mkmon(HIPPOPOTAS, serial=2)  # ex/megaExではない
    side = [shaymin, plain_target]
    assert damage_prevented(CORNERSTONE, plain_target, defender_side_pokemon=side, defender_is_benched=True) is True


def test_shaymin_does_not_protect_ex_ally():
    """defenderがex(ルールボックス持ち)ならShayminの庇いは効かない。

    attacker には HIPPO(非Tera)を使う: Milotic ex 自身の自衛特性(攻撃側Tera限定)と
    このテストが混同しないようにするため。
    """
    shaymin = mkmon(SHAYMIN, serial=1)
    ex_target = mkmon(MILOTIC_EX, serial=3)
    side = [shaymin, ex_target]
    assert damage_prevented(HIPPO, ex_target, defender_side_pokemon=side, defender_is_benched=True) is False


def test_shaymin_without_side_list_is_false():
    plain_target = mkmon(HIPPOPOTAS, serial=2)
    assert damage_prevented(CORNERSTONE, plain_target, defender_side_pokemon=None, defender_is_benched=True) is False


# ---------------------------------------------------------------------------
# 11 Mist Energy / 20 Rock Fighting Energy: ポケモンの特性ではなく装着エネルギー自身が
# 持つ「効果のみを防ぐ」(種別3)防壁。ability_text ではなく defender.energyCards を見る必要がある。
# ---------------------------------------------------------------------------


def test_mist_energy_blocks_effect_damage_unconditionally():
    defender = mkmon(9101, energy_cards=[mkcard(MIST_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=True) is True


def test_mist_energy_does_not_block_attack_damage():
    """種別3(効果のみ)なので、ワザ本体のダメージ(damage_is_effect=False)は止めない。"""
    defender = mkmon(9102, energy_cards=[mkcard(MIST_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=False) is False


def test_rock_fighting_energy_blocks_effect_damage_on_fighting_defender():
    """defenderが{F}(闘)タイプなら効果を無効化する。

    117 Cornerstone Mask Ogerpon ex は energyType=FIGHTING なので、defenderのカードとして流用する
    (ロック闘エネルギーはどのポケモンにも付けられるので、カード自体の他の特性は無関係)。
    """
    defender = mkmon(CORNERSTONE_OGERPON_EX, energy_cards=[mkcard(ROCK_FIGHTING_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=True) is True


def test_rock_fighting_energy_does_not_block_non_fighting_defender():
    """defenderが{F}タイプでなければ、Rock Fighting Energyが付いていても無効化されない。

    28 Poltchageist は energyType=GRASS。
    """
    defender = mkmon(POLTCHAGEIST, energy_cards=[mkcard(ROCK_FIGHTING_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=True) is False


def test_rock_fighting_energy_does_not_block_attack_damage():
    defender = mkmon(CORNERSTONE_OGERPON_EX, energy_cards=[mkcard(ROCK_FIGHTING_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=False) is False


# ---------------------------------------------------------------------------
# damage_is_effect_based: ワザのダメージが「効果によるダメージ」かどうかの分類器
# ---------------------------------------------------------------------------


def test_powerful_hand_is_effect_based():
    """743 Alakazam の Powerful Hand: damage=0、手札分のダメージカウンターを置く効果技。"""
    attack = _ATTACKS[1072]
    assert attack.name == "Powerful Hand"
    assert damage_is_effect_based(attack) is True


def test_land_crush_is_not_effect_based():
    """66 Dudunsparce の Land Crush: damage=90 の通常のワザダメージ。"""
    attack = _ATTACKS[76]
    assert attack.name == "Land Crush"
    assert attack.damage == 90
    assert damage_is_effect_based(attack) is False


# ---------------------------------------------------------------------------
# PTCG_DISABLE_DEFENDER_ABILITY=1: A/B比較用の全無効化フラグ(既存挙動の維持)
# ---------------------------------------------------------------------------


def test_env_flag_disables_all_barriers(monkeypatch):
    monkeypatch.setenv("PTCG_DISABLE_DEFENDER_ABILITY", "1")

    assert damage_prevented(CORNERSTONE, mkmon(CRUSTLE)) is False
    assert damage_prevented(CORNERSTONE, mkmon(FARIGIRAF_EX)) is False
    assert damage_prevented(CORNERSTONE, mkmon(POLTCHAGEIST), defender_is_benched=True) is False

    rabsca = mkmon(RABSCA, serial=1)
    target = mkmon(HIPPOPOTAS, serial=2)
    side = [rabsca, target]
    assert damage_prevented(CORNERSTONE, target, defender_side_pokemon=side, defender_is_benched=True) is False

    defender = mkmon(9201, energy_cards=[mkcard(MIST_ENERGY)])
    assert damage_prevented(HIPPO, defender, damage_is_effect=True) is False
