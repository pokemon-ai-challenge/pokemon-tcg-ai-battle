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

import os
import re
from functools import lru_cache

from cg.api import Attack, CardType, EnergyType, Pokemon

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


def damage_is_effect_based(attack: Attack) -> bool:
    """このワザの与ダメージが「ワザ本体のダメージ」ではなく「ワザの効果によるダメージ」か。

    ポケカのルール上この2つは別物で、防御側の防壁カードもテキストで書き分けている
    （11 Mist Energy「Damage is not an effect.」）。damage_prevented の damage_is_effect
    引数に渡す値の算出方法をここに置く。Attack.damage が可変ダメージ技のために 0 のまま
    返ってくるケース（_estimate_variable_damage と同じ前提）のうち、手札枚数に応じて
    ダメージカウンターを置く言い回しに一致するものだけを効果ダメージと判定する。
    未知の0ダメージ技は判定できないので False（安全側。呼び出し側の分類ロジックは別タスク）。

    **渡された attack をそのまま読む（attackId から引き直さない）。**
    引き直す実装も試したが、(1) 実測 0.10µs で `_ability_text_for_card_id` のような
    メモ化の利得が無く、(2) `card_cache.get_attack` を差し替えるテストのモックを壊し、
    (3) 呼び出し側が渡したオブジェクトと引き直した正準データが食い違い得る、の3点で
    損しかしなかった。カードIDから引くべきなのは「テキストの正規化が重い」場合だけ。
    """
    if attack.damage > 0:
        return False
    return bool(_PER_HAND_CARD_PATTERN.search(attack.text or ""))


def resolve_damage(
    attack: Attack,
    attacker: Pokemon,
    defender_weakness: EnergyType | None,
    defender_resistance: EnergyType | None,
    attacker_hand_size: int | None = None,
    defender: Pokemon | None = None,
    defender_side_pokemon: list[Pokemon | None] | None = None,
    defender_is_benched: bool = False,
    damage_is_effect: bool = False,
) -> int:
    """弱点・抵抗力を考慮した実際の与ダメージを計算する。

    弱点/抵抗力は「攻撃側ポケモンのタイプ」(CardData.energyType) と防御側の
    weakness/resistance を比較して判定する。attack.damage が 0 の可変ダメージ技は、
    attacker_hand_size（攻撃側の手札枚数、PlayerState.handCount）が渡されていれば
    _estimate_variable_damage で推定する。

    defender_side_pokemon / defender_is_benched / damage_is_effect は damage_prevented への
    素通し引数。既存の呼び出し（defender=None）では従来どおり特性を考慮しない。
    """
    # 防御側の特性で完全に無効化されるなら 0（defender を渡された場合のみ判定する。
    # 既存の呼び出しは defender=None で従来どおりの挙動）。
    if defender is not None and damage_prevented(
        attacker, defender, defender_side_pokemon, defender_is_benched, damage_is_effect
    ):
        return 0

    damage = attack.damage
    if damage <= 0:
        damage = _estimate_variable_damage(attack, attacker_hand_size)

    attacker_type = card_cache.get_card(attacker.id).energyType

    if defender_weakness is not None and attacker_type == defender_weakness:
        damage *= _WEAKNESS_MULTIPLIER
    if defender_resistance is not None and attacker_type == defender_resistance:
        damage = max(0, damage - _RESISTANCE_REDUCTION)

    return damage



# ---------------------------------------------------------------------------
# 防御側の特性によるダメージ無効化・きぜつ耐性（2026-08-06 追加）
#
# resolve_damage は弱点・抵抗力しか見ておらず、**防御側の特性を一切参照していなかった**。
# そのため、例えばイワパレスの
#     "Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon {ex}"
# に対して、ex アタッカーの打点を満額で計算していた（実際は 0）。
#
# この誤りは best_attack_damage / can_ko_defender として**盤面12体すべての特徴**に入り、
# 方策・価値関数・探索の葉評価が同じ嘘を見ることになる。marnie(主砲がオーロンゲ ex) の
# crustle 戦が最悪マッチアップ(0.583)である最有力の説明。
#
# カード効果テキストの汎用的な解釈は不可能なので、**頻出パターンだけを保守的に**扱う。
# 未知の言い回しは何もしない（従来と同じ挙動）。
#
# 追記（同日）: 「ワザのダメージ」と「ワザの効果によるダメージ」はルール上別物で、防壁カードも
# テキストで書き分けている（"prevent all damage ... attacks" はダメージのみ、"prevent all
# damage from and effects of attacks" は両方、"prevent all effects of attacks" は効果のみ。
# 11 Mist Energy の "Damage is not an effect." が根拠）。この区別を無視すると、例えば
# Cornerstone Mask Ogerpon ex（ダメージのみを防ぐ）が Alakazam の Powerful Hand
# （手札に応じたダメージカウンター＝効果ダメージ）まで止めてしまう、という新しい誤発火を生む。
# また 11 Mist Energy / 20 Rock Fighting Energy のような「効果のみを防ぐ」防壁は、ポケモンの
# 特性ではなく装着された特殊エネルギーカード自身のテキストなので、defender.energyCards を
# 別途走査する必要がある。
# ---------------------------------------------------------------------------


def _defender_ability_disabled() -> bool:
    """PTCG_DISABLE_DEFENDER_ABILITY=1 なら、防御側特性の考慮を無効化する（A/B比較用）。

    実験用のA/B切り替えであり、本番では設定しない。修正前後の効果を測るための
    一時的なフラグで、既定（未設定）では修正後の挙動（特性を考慮する）になる。
    呼び出しごとに os.environ を引く（import 時に1回だけ読むと、テストやプロセス内で
    切り替えられなくなるため）。この関数は1決定あたり数十回しか呼ばれないので、
    毎回読むコストは無視できる。
    """
    return os.environ.get("PTCG_DISABLE_DEFENDER_ABILITY") == "1"


# カードテキストの表記ゆれ（'’' と \xa0 など全角風の空白、連続する空白/改行）を吸収してから
# 部分文字列マッチするための正規化。語順に依存した判定をしないための前提でもある
# （固定の1本のパターンではなく、独立した条件の組み合わせで判定するため）。
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_ability_text(raw: str) -> str:
    text = raw.lower().replace("é", "e").replace("’", "'")
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


@lru_cache(maxsize=None)
def _ability_text_for_card_id(card_id: int) -> str:
    """カードIDから特性/カードテキストを正規化して返す（未知IDやエラー時は空文字）。

    ポケモンの特性だけでなく、特殊エネルギーカードが持つ「効果を防ぐ」テキスト
    （11 Mist Energy 等）も同じ CardData.skills に載っているため、この関数はポケモン・
    エネルギーどちらの cardId にも使える。

    card_id だけの純関数（カードデータは静的）なので lru_cache でメモ化する。特徴量ビルドで
    盤面12スロット×2ワザ相当の頻度で呼ばれ、テキスト正規化（.lower/replace/正規表現）が
    ここでの支配的なコストだったため、最も効果が大きい（2026-08-06 メモ化対応）。
    """
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001 - 未知IDは「テキストなし」に倒す
        return ""
    raw = " ".join((sk.text or "") for sk in (getattr(card, "skills", None) or []))
    return _normalize_ability_text(raw)


def _skill_text(pokemon: Pokemon | None) -> str:
    """ポケモンの特性テキストを正規化して返す。"""
    if pokemon is None:
        return ""
    return _ability_text_for_card_id(pokemon.id)


@lru_cache(maxsize=None)
def _is_ex_for_card_id(card_id: int) -> bool:
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(card, "ex", False) or getattr(card, "megaEx", False))


def _is_ex(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    return _is_ex_for_card_id(pokemon.id)


@lru_cache(maxsize=None)
def _is_basic_for_card_id(card_id: int) -> bool:
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(card, "basic", False))


def _is_basic(pokemon: Pokemon | None) -> bool:
    """たねポケモンか。CardData.basic をそのまま見る（stage1/stage2 との排他フラグ）。"""
    if pokemon is None:
        return False
    return _is_basic_for_card_id(pokemon.id)


@lru_cache(maxsize=None)
def _has_ability_for_card_id(card_id: int) -> bool:
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(card, "skills", None))


def _has_ability(pokemon: Pokemon | None) -> bool:
    """カードが特性を持つか。

    CardData.skills はワザ（attacks は別途 attackId のリストで持つ）とは構造的に分離されて
    おり、skills に載っているのは常に特性・カード固有ルールテキストの類である。したがって
    非空かどうかで判定してよい（Fossilの「as if it were a Pokémon」のような戦闘中に発動しない
    ルールテキストも技術的には skills に混ざるが、そうしたカードは attacks が空で attacker に
    はなり得ないため、ここでの用途では区別する必要がない）。
    """
    if pokemon is None:
        return False
    return _has_ability_for_card_id(pokemon.id)


@lru_cache(maxsize=None)
def _is_tera_for_card_id(card_id: int) -> bool:
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(card, "tera", False))


def _is_tera(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    return _is_tera_for_card_id(pokemon.id)


def _has_special_energy(pokemon: Pokemon | None) -> bool:
    """Pokemon.energies ではなく energyCards（装着エネルギーカードそのもの）を見て、
    特殊エネルギー（CardType.SPECIAL_ENERGY）が1枚でも付いているかを判定する。
    energies は EnergyType の集計値でしかなく、基本/特殊の区別が付かないため使えない。
    """
    if pokemon is None:
        return False
    for card in getattr(pokemon, "energyCards", None) or []:
        try:
            energy_data = card_cache.get_card(card.id)
        except Exception:  # noqa: BLE001 - 未知IDは「特殊エネルギーではない」に倒す
            continue
        if getattr(energy_data, "cardType", None) == CardType.SPECIAL_ENERGY:
            return True
    return False


@lru_cache(maxsize=None)
def _has_rule_box_for_card_id(card_id: int) -> bool:
    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(card, "ex", False) or getattr(card, "megaEx", False))


def _has_rule_box(pokemon: Pokemon | None) -> bool:
    """ex / megaEx などルールボックスを持つか。

    CardData には ex / megaEx の bool しかなく、V/VMAX/VSTAR 用の専用フィールドは無い
    （本コンペのカードプールには現状 ex 系以外のルールボックス持ちが確認できていない）。
    将来 V 系が追加された場合はここに条件を足す必要がある。
    """
    if pokemon is None:
        return False
    return _has_rule_box_for_card_id(pokemon.id)


def _barrier_kind(text: str) -> int | None:
    """防壁テキストの種別。1=ワザのダメージのみ、2=ダメージと効果の両方、3=効果のみ。

    「ワザのダメージ」と「ワザの効果によるダメージ」はルール上別物で
    ("Damage is not an effect." — 11 Mist Energy)、防壁テキストもこれを書き分けている。
    語順に依存しないよう部分文字列の組み合わせだけで判定し、該当なければ None（防壁ではない）。
    """
    if "damage from and effects of attacks" in text or "damage and effects of attacks" in text:
        return 2
    if "prevent all effects of attacks" in text:
        return 3
    if "prevent all damage" in text:
        return 1
    return None


def _attacker_conditions_met(text: str, attacker: Pokemon | None) -> bool:
    """(d) 攻撃側への条件。テキストに含まれる条件をすべて満たさなければ False。

    「if that damage is ...」(158 Drednaw) はダメージ量依存でこの関数の入力（カードテキスト）
    だけでは判定できないため、安全側で常に False を返す（＝無効化しない。従来どおりの
    打点計算になる）。
    """
    if "basic pokemon {ex}" in text:
        if not (_is_basic(attacker) and _is_ex(attacker)):
            return False
    elif "{ex}" in text:
        if not _is_ex(attacker):
            return False
    if "that have an ability" in text and not _has_ability(attacker):
        return False
    if "tera pokemon" in text and not _is_tera(attacker):
        return False
    if "that have any special energy attached" in text and not _has_special_energy(attacker):
        return False
    if "if that damage is" in text:
        return False
    return True


def _pokemon_ability_prevents(
    text: str,
    ability_holder_is_self: bool,
    attacker: Pokemon | None,
    defender: Pokemon | None,
    defender_is_benched: bool,
    damage_is_effect: bool,
) -> bool:
    """ポケモンの特性テキスト1つ分が、この攻撃(attacker→defender)を防ぐか。

    ability_holder_is_self: この特性の持ち主が defender 自身かどうか。
        True  … 28/158/330等の自衛型（"to this pokemon"）。
        False … 74/343等のベンチ庇い型（"to your benched pokemon"）。特性の持ち主は
                defender_side_pokemon 中の別のポケモンであり得るが、守られる対象は
                常に defender 自身。
    """
    if not text:
        return False

    kind = _barrier_kind(text)
    if kind is None:
        return False
    allowed_kinds = (2, 3) if damage_is_effect else (1, 2)
    if kind not in allowed_kinds:
        return False

    if ability_holder_is_self:
        if "to this pokemon" not in text:
            return False
        # (c) 発動場所: 「as long as this pokemon is on your bench」があれば、特性の持ち主
        # （= defender 自身）がベンチにいるときだけ有効。
        if "as long as this pokemon is on your bench" in text and not defender_is_benched:
            return False
    else:
        if "to your benched pokemon" not in text:
            return False
        # ベンチ庇い型は、守られる対象（defender）自身がベンチにいなければ意味を持たない。
        if not defender_is_benched:
            return False

    if not _attacker_conditions_met(text, attacker):
        return False

    # (e) 防御側条件: 「that don't have a rule box」なら defender がルールボックスを
    # 持たないときだけ有効。
    if "that don't have a rule box" in text and _has_rule_box(defender):
        return False

    return True


# {F} のような1文字のタイプ記号からEnergyTypeへのマップ。20 Rock Fighting Energy の
# "done to the {F} Pokémon this card is attached to" のように、防ぐ相手をタイプで限定する
# エネルギーカードのテキストを読むために使う。カードID/名前ではなく、公式のタイプ記号という
# 汎用的な語彙をハードコードしているだけなので、他の同種カードにもそのまま使える。
_ENERGY_SYMBOL_TO_TYPE: dict[str, EnergyType] = {
    "c": EnergyType.COLORLESS,
    "g": EnergyType.GRASS,
    "r": EnergyType.FIRE,
    "w": EnergyType.WATER,
    "l": EnergyType.LIGHTNING,
    "p": EnergyType.PSYCHIC,
    "f": EnergyType.FIGHTING,
    "d": EnergyType.DARKNESS,
    "m": EnergyType.METAL,
    "n": EnergyType.DRAGON,
}

_ENERGY_TYPE_RESTRICTION_PATTERN = re.compile(r"\{([a-z])\} pokemon")


def _energy_type_restriction_met(text: str, defender: Pokemon | None) -> bool:
    """"...done to the {F} Pokémon this card is attached to" のように、守る相手をタイプで
    限定する記法を読む。記法が無ければ無条件で True。記号が未知、または defender の
    タイプが引けない場合は安全側で False にする（過剰に無効化しない）。
    """
    match = _ENERGY_TYPE_RESTRICTION_PATTERN.search(text)
    if not match:
        return True
    energy_type = _ENERGY_SYMBOL_TO_TYPE.get(match.group(1))
    if energy_type is None or defender is None:
        return False
    try:
        return card_cache.get_card(defender.id).energyType == energy_type
    except Exception:  # noqa: BLE001
        return False


def _energy_prevents(
    attacker: Pokemon | None,
    defender: Pokemon | None,
    damage_is_effect: bool,
) -> bool:
    """defender に装着された特殊エネルギー自身が持つ防壁（11 Mist Energy / 20 Rock Fighting
    Energy 等）。ポケモンの特性ではなくエネルギーカードのテキストなので、_skill_text(defender)
    では拾えず、defender.energyCards を個別に見る必要がある。
    """
    if defender is None:
        return False
    for card in getattr(defender, "energyCards", None) or []:
        text = _ability_text_for_card_id(card.id)
        if not text:
            continue
        kind = _barrier_kind(text)
        if kind is None:
            continue
        allowed_kinds = (2, 3) if damage_is_effect else (1, 2)
        if kind not in allowed_kinds:
            continue
        if not _attacker_conditions_met(text, attacker):
            continue
        if not _energy_type_restriction_met(text, defender):
            continue
        return True
    return False


def damage_prevented(
    attacker: Pokemon | None,
    defender: Pokemon | None,
    defender_side_pokemon: list[Pokemon | None] | None = None,
    defender_is_benched: bool = False,
    damage_is_effect: bool = False,
) -> bool:
    """防御側の特性・特殊エネルギーで、この攻撃側からのダメージ（または効果）が
    完全に無効化されるか。

    damage_is_effect は「ワザのダメージ」(False, 既定) と「ワザの効果によるダメージ」(True)
    のどちらを判定しているかを表す。この2つはルール上別物で、防壁テキストも書き分けている
    （"prevent all damage ... attacks" はダメージのみ、"prevent all damage from and effects
    of attacks" は両方、"prevent all effects of attacks" は効果のみ）。既定 False なので、
    既存の2引数呼び出しの意味は変わらない。

    defender_side_pokemon（防御側の場のポケモン、バトル場＋ベンチ。None なら評価しない）と
    defender_is_benched（defender がベンチにいるか）を渡すと、ベンチ庇い型
    （74 Rabsca / 343 Shaymin）や「特性の持ち主自身がベンチにいる間だけ有効」型
    （28 / 362 / 1138）も判定できる。

    カードテキストの解釈は頻出パターンだけを保守的に扱う。未知の言い回しは何もしない
    （False）。PTCG_DISABLE_DEFENDER_ABILITY=1 が設定されている場合は常に False を返す
    （修正前の挙動に戻すA/B切り替え。実験用であり、本番では設定しない）。
    """
    if _defender_ability_disabled():
        return False

    if _pokemon_ability_prevents(
        _skill_text(defender), True, attacker, defender, defender_is_benched, damage_is_effect
    ):
        return True

    if defender_side_pokemon is not None:
        for ally in defender_side_pokemon:
            if ally is None:
                continue
            if _pokemon_ability_prevents(
                _skill_text(ally), False, attacker, defender, defender_is_benched, damage_is_effect
            ):
                return True

    if _energy_prevents(attacker, defender, damage_is_effect):
        return True

    return False


def survives_via_ability(defender: Pokemon | None, damage: int) -> bool:
    """きぜつするはずのダメージを、防御側の特性で耐えるか（例: Sturdy）。

    「満タンなら、きぜつするダメージを受けても HP10 で残る」型。
    満タン判定を誤ると過大評価になるので、maxHp が取れない場合は False。

    PTCG_DISABLE_DEFENDER_ABILITY=1 が設定されている場合は常に False を返す
    （修正前の挙動に戻すA/B切り替え。実験用であり、本番では設定しない）。
    """
    if _defender_ability_disabled():
        return False
    if defender is None or damage < defender.hp:
        return False
    text = _skill_text(defender)
    if not text:
        return False
    if "would be knocked out by damage from an attack, it is not knocked out" not in text:
        return False
    max_hp = getattr(defender, "maxHp", None) or 0
    return bool(max_hp and defender.hp >= max_hp)


def can_ko(
    attack: Attack,
    attacker: Pokemon,
    defender: Pokemon,
    defender_weakness: EnergyType | None,
    defender_resistance: EnergyType | None,
    attacker_hand_size: int | None = None,
    defender_side_pokemon: list[Pokemon | None] | None = None,
    defender_is_benched: bool = False,
    damage_is_effect: bool = False,
) -> bool:
    """このワザで相手をきぜつさせられるか（残りHP <= 与ダメージ）を判定する。

    防御側の特性・特殊エネルギーによるダメージ無効化・きぜつ耐性を考慮する
    （2026-08-06 追加）。defender_side_pokemon / defender_is_benched / damage_is_effect は
    damage_prevented への素通し引数（既定値のままなら従来どおりの挙動）。
    """
    if damage_prevented(attacker, defender, defender_side_pokemon, defender_is_benched, damage_is_effect):
        return False
    damage = resolve_damage(attack, attacker, defender_weakness, defender_resistance, attacker_hand_size)
    if survives_via_ability(defender, damage):
        return False
    return damage >= defender.hp
