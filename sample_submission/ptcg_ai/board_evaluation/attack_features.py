"""クラスタ② 盤面評価／担当B

技の打点・与ダメージの解決に関する共通部品。デッキ固有のカードID/カード名をハードコードは
しないが、弱点・抵抗力の判定には「攻撃側ポケモンのタイプ」(CardData.energyType) が要るため、
shared.card_cache 経由で attacker.id から引く（card_cache はカードIDをキーにした汎用の
生データキャッシュであり、特定デッキのカードを直書きするものではない）。

cg.api の Attack.damage は、手札枚数などで変動する可変ダメージ技（例:
「Place 2 damage counters ... for each card in your hand」）では 0 のまま返ってくる
（実際の計算はエンジン側が実行時に行うため）。これを damage=0 の技として無視すると
主力技の価値を大きく見誤るため、Attack.text（英語の効果テキスト）から既知の言い回しに
一致するものだけ、汎用的な正規表現でダメージを推定する。カード固有のID/名前には一切
依存しないため、担当Aのデータ投入を待たず、デッキが変わっても機能する。
未知の言い回しは推定せず 0 のまま返す（安全側）。

注意: _FIXED_DAMAGE_PATTERN は「攻撃の基本ダメージ」を意図した言い回しにのみ一致することを
attackId=183（Cruel Arrow）で確認したものであり、他の未検証テキストへの汎化は保証しない。
「If the Defending Pokémon is Basic, this attack does 20 more damage」のような、
本体ダメージではなく条件付き追加ダメージの言い回しに一致して数値を誤って拾う可能性がある
（該当テキストが is 単数系の「Basic」を使うなど、まだ確認していない他パターンとの区別は
ここでは行っていない）。新しい0ダメージ技を確認するたびに、実際のテキストで検証してから
パターンを足すこと。
"""

import re

from cg.api import Attack, EnergyType, Pokemon

from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.shared import card_cache

# 現行ルール準拠の簡易モデル: 弱点は2倍、抵抗力は-30（攻撃の追加効果によるダメージ増減は考慮しない）。
_WEAKNESS_MULTIPLIER = 2
_RESISTANCE_REDUCTION = 30

# 公式ルール上、ダメージカウンター1個 = 10ダメージ。
# 「N damage counters」表記の技は、直接ダメージ点数を書く技（例: "does 100 damage"）とは
# 単位が異なるため、抽出した数値をそのままダメージ点数として使ってはいけない。
_DAMAGE_PER_COUNTER = 10

# Attack.damage が 0（可変ダメージなど）の場合に、Attack.text から推定を試みるパターン。
# 上から順に試し、最初にマッチしたものを採用する。新しい言い回しが見つかったら追記していく。
_FIXED_DAMAGE_PATTERN = re.compile(r"does (\d+) damage", re.IGNORECASE)
_PER_HAND_CARD_PATTERN = re.compile(r"(\d+) damage counters? .*? for each card in your hand", re.IGNORECASE)


def _estimate_variable_damage(attack: Attack, attacker_hand_size: int | None) -> int:
    """attack.damage が 0 の可変ダメージ技を、attack.text から推定する（不明なら0）。"""
    text = attack.text or ""

    if attacker_hand_size is not None:
        match = _PER_HAND_CARD_PATTERN.search(text)
        if match:
            # 抽出した数値は「ダメージカウンター」の個数であり、ダメージ点数そのものではない
            # （1個=10ダメージ）。手札1枚あたりの点数に換算してから手札枚数を掛ける。
            return int(match.group(1)) * _DAMAGE_PER_COUNTER * attacker_hand_size

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
    """弱点・抵抗力を考慮した実際の与ダメージを計算する。

    弱点/抵抗力は「攻撃側ポケモンのタイプ」(CardData.energyType) と防御側の
    weakness/resistance を比較して判定する。attack.damage が 0 の可変ダメージ技は、
    attacker_hand_size（攻撃側の手札枚数、PlayerState.handCount）が渡されていれば
    _estimate_variable_damage で推定する。
    """
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
    """このワザで相手をきぜつさせられるか（残りHP <= 与ダメージ）を判定する。

    エネルギー充足は判定しない（呼び出し側で is_energy_sufficient を先に確認すること）。
    """
    damage = resolve_damage(attack, attacker, defender_weakness, defender_resistance, attacker_hand_size)
    return damage >= defender.hp


def can_ko_with_any_available_attack(
    attacker: Pokemon,
    defender: Pokemon,
    defender_weakness: EnergyType | None,
    defender_resistance: EnergyType | None,
    attacker_hand_size: int | None = None,
) -> bool:
    """attacker が今持っているエネルギーだけで使えるワザのうち、defender をきぜつさせられる
    ものが1つでもあるかを判定する。

    「使用可能などのワザでもKOできるか」を判定する処理（attack.py の KO 判定、
    priorities/retreat.py のトリガーB、pokemon_value.py の即KOボーナスで共通して必要）を
    1箇所に集約する。attacker が今バトル場にいるかどうかは問わない（ベンチ候補にもそのまま使える）。
    """
    attacker_card = card_cache.get_card(attacker.id)
    for attack_id in attacker_card.attacks:
        attack = card_cache.get_attack(attack_id)
        if not energy_requirements.is_energy_sufficient(attack, attacker.energies):
            continue
        if can_ko(attack, attacker, defender, defender_weakness, defender_resistance, attacker_hand_size):
            return True
    return False
