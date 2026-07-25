"""学習時の特徴量抽出を固定するための、盤面評価ロジックの凍結コピー。

なぜ別コピーが要るか:
    policy_model / value_model は ``learning.encoder`` が生成する特徴量で学習済みで、
    その重み(policy_weights.json / value_weights.json)は「学習時点の特徴量定義」に
    紐づいている。学習時点の ``board_evaluation.attack_features`` は、手札枚数依存の
    可変ダメージ技("... damage counters ... for each card in your hand")を
    「ダメージカウンター個数 = ダメージ点数」として推定していた。

    その後 ``board_evaluation.attack_features`` 側で「1カウンター=10ダメージ」に直す
    バグ修正(_DAMAGE_PER_COUNTER 導入)が入った。これはルールベース(rule_based)が
    実行時に読む値としては正しい修正だが、encoder が同じ関数を(直接 / board_features 経由で)
    食うと、学習時と推論時で特徴量がズレる(train/inference skew)。該当技を含む局面で
    モデル出力が変わり、golden test(test_policy_model / test_value_model)も割れる。

    そこで encoder が使うダメージ解決とその派生(次ターン被KO判定)だけを、学習時点の
    定義でこのモジュールに凍結する。board_evaluation 側の修正は rule_based では生かした
    まま、ml 特徴量は学習時と同一に保つ。恒久解は「修正後の特徴量で policy/value を
    再学習」で、その際は encoder のインポートを board_evaluation に戻してこのモジュールを消す。

弱点/抵抗力の解決や次ターン被KO判定のロジックは学習時点から変わっていないため、そのまま
同じ内容を写している(将来 board_evaluation 側のこれらが変わっても、学習済みモデル用の
ここは意図的に追従しない)。カード固有のID/名前には依存しない点も本家と同じ。
"""

import re

from cg.api import Attack, EnergyType, Pokemon, State

from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.shared import card_cache

# 現行ルール準拠の簡易モデル: 弱点は2倍、抵抗力は-30（学習時点の定義のまま凍結）。
_WEAKNESS_MULTIPLIER = 2
_RESISTANCE_REDUCTION = 30

# 相手が今エネルギー1枚足りないだけの技は、次の相手ターンには使えるようになるとみなす。
_NEXT_TURN_ENERGY_ALLOWANCE = 1

# Attack.damage が 0（可変ダメージなど）の場合に、Attack.text から推定を試みるパターン。
_FIXED_DAMAGE_PATTERN = re.compile(r"does (\d+) damage", re.IGNORECASE)
_PER_HAND_CARD_PATTERN = re.compile(r"(\d+) damage counters? .*? for each card in your hand", re.IGNORECASE)


def _estimate_variable_damage(attack: Attack, attacker_hand_size: int | None) -> int:
    """attack.damage が 0 の可変ダメージ技を、attack.text から推定する（不明なら0）。

    学習時点の挙動を凍結: 手札枚数依存技では抽出値をそのまま点数として扱う
    (board_evaluation.attack_features の ×10 バグ修正は ml 特徴量には反映しない)。
    """
    text = attack.text or ""

    if attacker_hand_size is not None:
        match = _PER_HAND_CARD_PATTERN.search(text)
        if match:
            return int(match.group(1)) * attacker_hand_size

    match = _FIXED_DAMAGE_PATTERN.search(text)
    if match:
        return int(match.group(1))

    return 0


def resolve_damage(
    attack: Attack,
    attacker: Pokemon,
    defender_weakness: EnergyType | None,
    defender_resistance: EnergyType | None,
    attacker_hand_size: int | None = None,
) -> int:
    """弱点・抵抗力を考慮した与ダメージ（学習時点の定義で凍結）。"""
    damage = attack.damage
    if damage <= 0:
        damage = _estimate_variable_damage(attack, attacker_hand_size)

    attacker_type = card_cache.get_card(attacker.id).energyType

    if defender_weakness is not None and attacker_type == defender_weakness:
        damage *= _WEAKNESS_MULTIPLIER
    if defender_resistance is not None and attacker_type == defender_resistance:
        damage = max(0, damage - _RESISTANCE_REDUCTION)

    return damage


def can_ko(
    attack: Attack,
    attacker: Pokemon,
    defender: Pokemon,
    defender_weakness: EnergyType | None,
    defender_resistance: EnergyType | None,
    attacker_hand_size: int | None = None,
) -> bool:
    """このワザで相手をきぜつさせられるか（学習時点の定義で凍結）。"""
    damage = resolve_damage(attack, attacker, defender_weakness, defender_resistance, attacker_hand_size)
    return damage >= defender.hp


def is_likely_ko_next_turn(pokemon: Pokemon, state: State, your_index: int) -> bool:
    """このポケモンが次の相手ターンで倒されやすいか（学習時点の定義で凍結）。

    board_features.is_likely_ko_next_turn と同じロジックだが、上の凍結 resolve_damage を
    使う点だけが異なる（可変ダメージ推定を学習時点に固定するため）。
    """
    opponent = state.players[1 - your_index]
    if not opponent.active or opponent.active[0] is None:
        return False
    attacker = opponent.active[0]
    attacker_card = card_cache.get_card(attacker.id)
    defender_card = card_cache.get_card(pokemon.id)
    attacker_hand_size = opponent.handCount

    for attack_id in attacker_card.attacks:
        attack = card_cache.get_attack(attack_id)
        shortfall = energy_requirements.energy_shortfall(attack, attacker.energies)
        if sum(shortfall.values()) > _NEXT_TURN_ENERGY_ALLOWANCE:
            continue
        damage = resolve_damage(
            attack, attacker, defender_card.weakness, defender_card.resistance, attacker_hand_size
        )
        if damage >= pokemon.hp:
            return True
    return False
