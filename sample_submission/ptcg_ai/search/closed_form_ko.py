"""閉形式のKO可否判定(探索なし・O(1))。

なぜ必要か
----------
`ability_draw_brake`(r11 Fix-I)と `low_deck_draw_brake`(r9 Fix-D)は「今ターンKOできるか」を
`ko_search.can_ko_this_turn` / `ability_draw_eval.already_ko_without_more_energy` という
**エンジン探索**で判定している。しかし本デッキ(og_v032)の実盤面は「みどりのまい」を持つ
オーガポン みどりのめん ex が複数体並ぶため分岐が非常に大きく、300ms の予算内に探索が
完走しない。両ガードは「判定不能(inconclusive)なら安全側で介入しない」設計なので、
**実対戦120試合で発火0回**という実測になった(=ガードが実質的に死んでいる)。

このモジュールはその律速を外すためのファストパス。本デッキのアタッカーは
オーガポン みどりのめん ex(cardId 96)の1種類だけで、そのワザ「まんようしぐれ」
(attackId 120、`all_attack()` の実データで確認)は

    ダメージ = 30 + 30 × (自分のバトルポケモンのエネ数 + 相手のバトルポケモンのエネ数)

という閉形式なので、相手アクティブの残HP(`Pokemon.hp` = Current HP)と突き合わせれば
探索なしでKO可否が O(1) で判定できる。実リプレイ突合は
`kaggle_replays/_probe_closed_form_ko.py`。

安全側(conservative)の設計 — ここが本モジュールの肝
--------------------------------------------------
戻り値は `bool | None` の3値で、``None`` は「判定不能=呼び出し側は従来の探索へフォールバック
せよ」を意味する。**弱点・抵抗・軽減効果までは式に入れない**代わりに、それらが結論を
ひっくり返しうる状況では ``None`` を返して黙る。具体的には:

1. **既知パターンのみ**: まんようしぐれ(120)以外のワザ / 未知のポケモンが絡む局面は ``None``。
   自分のアクティブが複数ワザを持ち1つでも未知ならその時点で ``None``(最大打点が確定しない)。
2. **弱点は True 側で一切使わない**: 弱点は打点を増やす方向にしか働かないので、
   「KOできる(True)」の判定では弱点を無視した**下界**で比較する(過小評価=安全)。
   逆に「KOできない(False)」の判定では弱点×2を掛けた**上界**で比較する(過大評価=安全)。
3. **抵抗は数値が分からないので降参**: `CardData.resistance` は色しか持たず軽減量が不明。
   相手アクティブが草(GRASS)抵抗を持つなら True 側の判定は ``None``。
4. **ダメージ改変カードが場にあれば降参**: 場(両者のポケモン・ついているどうぐ/特殊エネ・
   スタジアム)のカードテキストに「ダメージを増減/無効化する」記述があれば、量を評価せず
   ``None`` を返す(`_damage_modifier_risk`)。判定はカードIDのハードコードではなく
   `CardData.skills[].text` の走査なので、カードプールが増えても自動的に追随する。
5. **余裕マージン**: `ko_margin`(既定0)を足した上で比較する。上記1〜4で未知要因は
   すべて ``None`` に落としてあるため既定は0(算術は厳密)。それでも保険を厚くしたい
   config のために外から積めるようにしてある。

`is_ko_impossible_this_turn` は「攻撃でKOする手が**そもそも存在しない**」ことの健全な上界判定。
自分の場の総エネ+手札から追加できるエネの上界、相手の場の最大エネ、相手の場の最小残HPを
それぞれ独立に最悪ケースで取るので、True(不可能)を返すのは本当に不可能なときだけ。
"""

from __future__ import annotations

import re

from cg.api import CardType, EnergyType

from ptcg_ai.shared import card_cache

# --- 既知パターン --------------------------------------------------------------------

TEAL_MASK_OGERPON_EX_ID = 96      # オーガポン みどりのめん ex
MYRIAD_LEAF_SHOWER_ID = 120       # まんようしぐれ(30 + 30×両バトルポケモンのエネ数)
MYRIAD_LEAF_SHOWER_BASE = 30
MYRIAD_LEAF_SHOWER_PER_ENERGY = 30

# 「まんようしぐれ型」= 30 + per_energy ×(自分のバトル場エネ + 相手のバトル場エネ)。
# 値は `cg.api.all_attack()` の実データ(damage=30 / text="This attack does 30 more damage
# for each Energy attached to both Active Pokémon.")と一致することを起動時に検証する
# (`verify_known_attacks`)。
_BOTH_ACTIVE_ENERGY_ATTACKS: dict[int, tuple[int, int]] = {
    MYRIAD_LEAF_SHOWER_ID: (MYRIAD_LEAF_SHOWER_BASE, MYRIAD_LEAF_SHOWER_PER_ENERGY),
}

TAPU_BULU_ID = 920            # カプ・ブルル(たね/非ex/特性なし/HP140/にげ3)
WOOD_HAMMER_ID = 1326         # ウッドハンマー(固定220、自分に30)
WOOD_HAMMER_DAMAGE = 220
WOOD_HAMMER_SELF_DAMAGE = 30

# 「固定打点型」= 盤面に依存しない定数ダメージ。値は (打点, 自傷ダメージ)。
#
# なぜ追加が必要か(壁デッキ対策 `wall_attacker_route` の前提):
# オーガポン みどりのめん ex(96)は **ex かつ 特性持ち** なので、イワパレス(345、
# 「相手のポケモンexのワザのダメージを完全に無効」)と いしずえのめんex(117、
# 「特性を持つ相手のポケモンのワザのダメージを完全に無効」)の**両方**に対して打点0になる。
# その回答として入れる非exアタッカー=カプ・ブルル(920)のワザ「ウッドハンマー」
# (attackId 1326。`all_attack()` の実データで damage=220 /
# text="This Pokémon also does 30 damage to itself." を確認済み)は、この表に無いと
# **未知のワザ**として扱われ、`estimate_current_damage` / `is_ko_impossible_this_turn` が
# 「自分の場に1体でもブルルが居るだけで常に判定不能(None)」になる。つまりブルルを1枚
# 挿すと r13 の閉形式ファストパス(`ability_draw_brake` / `low_deck_draw_brake`)が
# デッキ全体で死ぬ。ここに登録することでそれを防ぎ、同時にブルルがバトル場に居る間の
# KO判定も O(1) で効くようになる。
#
# 自傷ダメージは「相手をKOできるか」には影響しない(この表の第2要素は
# `self_damage_of` 経由で、自滅を避ける側の判定にだけ使う)。
_FIXED_DAMAGE_ATTACKS: dict[int, tuple[int, int]] = {
    WOOD_HAMMER_ID: (WOOD_HAMMER_DAMAGE, WOOD_HAMMER_SELF_DAMAGE),
}


def is_known_attack(attack_id) -> bool:
    """このモジュールが打点を閉形式で計算できるワザか。"""
    try:
        aid = int(attack_id)
    except Exception:  # noqa: BLE001
        return False
    return aid in _BOTH_ACTIVE_ENERGY_ATTACKS or aid in _FIXED_DAMAGE_ATTACKS


def self_damage_of(attack_id) -> int:
    """既知の固定打点ワザの**自傷ダメージ**。未知/自傷なしは 0。

    「自分のアタッカーが自滅する撃ち方を避ける」用途にだけ使う(相手のKO可否には無関係)。
    """
    try:
        return int(_FIXED_DAMAGE_ATTACKS.get(int(attack_id), (0, 0))[1])
    except Exception:  # noqa: BLE001
        return 0

# スタジアムによるワザコスト増(`_metrics_gate._STADIUM_COST_SURCHARGE` と同じ事実)。
# 夜の鉱山(1266)=「場のテラスタルのポケモンのワザは【無】1個ぶん多くかかる」。
_TERA_COST_SURCHARGE_STADIUM_IDS = frozenset({1266})

# 任意色として使えるエネルギー(コスト充足判定用)。
_WILD_ENERGY = frozenset({int(EnergyType.RAINBOW), int(EnergyType.TEAM_ROCKET)})

# ダメージを増減/無効化しうるカードテキストの手掛かり。1つでも当たったら**量を評価せず
# 判定不能**にする(このモジュールは「分からないときは黙る」方針)。
#
# 唯一の除外: 「prevent all damage **counters**」(バトルケージ1264 = 「ベンチにダメカンが
# 置かれるのを防ぐ。(ワザのダメージは受ける)」)。カードテキスト自身が「ワザのダメージは
# 通る」と明記しているうえ、こちらの攻撃はバトル場へのワザのダメージなので影響しない。
# 実測でこの取りこぼしが多かった(リプレイ250件中13件)ため負の先読みで除外する。
_DAMAGE_MODIFIER_HINTS = re.compile(
    r"(less damage"
    r"|more damage"
    r"|prevent all damage(?! counters)"
    r"|prevent that damage"
    r"|prevent all of the damage"
    r"|damage done to"
    r"|no damage from"
    r"|takes? no damage"
    r"|double the damage"
    r"|damage is doubled"
    r"|weakness)",
    re.IGNORECASE,
)


class _Reason:
    """`report` に書く判定不能理由(文字列定数)。"""

    NO_STATE = "no_state"
    NO_ACTIVE = "no_active"
    NO_OPPONENT_ACTIVE = "no_opponent_active"
    UNKNOWN_ATTACK = "unknown_attack"
    NO_PAYABLE_ATTACK = "no_payable_attack"
    OPPONENT_RESISTANCE = "opponent_resistance"
    DAMAGE_MODIFIER_IN_PLAY = "damage_modifier_in_play"
    UNKNOWN_HP = "unknown_hp"
    ERROR = "error"


def _note(report: dict | None, key: str, value=True) -> None:
    if report is not None:
        report[key] = value


# --- 盤面アクセサ --------------------------------------------------------------------

def _active_pokemon(state, player: int):
    """``player`` のバトルポケモン。伏せ/不在/例外は None。"""
    try:
        active = state.players[player].active or []
    except Exception:  # noqa: BLE001
        return None
    for mon in active:
        if mon is not None:
            return mon
    return None


def _active_energy_count(state, player: int) -> int | None:
    """``player`` のバトルポケモンについているエネルギーの個数。判定不能は None。

    `Pokemon.energies` は「そのカードが供給しているエネルギーの配列」なので、2個ぶんを
    供給する特殊エネは2要素として並ぶ = ``len`` がそのまま「ついているエネルギーの数」。
    """
    mon = _active_pokemon(state, player)
    if mon is None:
        return None
    try:
        return len(mon.energies or [])
    except Exception:  # noqa: BLE001
        return None


def _in_play_pokemon(state, player: int) -> list:
    """``player`` の場のポケモン(バトル場+ベンチ)。"""
    out = []
    try:
        for mon in (state.players[player].active or []):
            if mon is not None:
                out.append(mon)
        for mon in (state.players[player].bench or []):
            if mon is not None:
                out.append(mon)
    except Exception:  # noqa: BLE001
        return out
    return out


def _card(card_id) -> object | None:
    if card_id is None:
        return None
    try:
        return card_cache.get_card(int(card_id))
    except Exception:  # noqa: BLE001
        return None


def _stadium_id(state) -> int | None:
    try:
        stadium = state.stadium or []
        return int(stadium[0].id) if stadium and stadium[0] is not None else None
    except Exception:  # noqa: BLE001
        return None


# --- ダメージ改変リスクの検出 ----------------------------------------------------------

def _has_damage_modifier_text(card_id) -> bool:
    card = _card(card_id)
    if card is None:
        return False
    try:
        for skill in (card.skills or []):
            if _DAMAGE_MODIFIER_HINTS.search(skill.text or ""):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _damage_modifier_risk(state, me: int) -> int | None:
    """場に「ダメージを増減/無効化しうる」カードがあれば、その cardId を1つ返す。

    走査対象は両プレイヤーの場のポケモン本体・ついているどうぐ・ついている特殊エネルギー、
    およびスタジアム。ベンチ発動のオーラ(ボーマンダ系の軽減特性など)もバトル場のダメージを
    変えるので、ベンチも必ず見る。1件でも見つかれば呼び出し側は判定不能にする。
    """
    try:
        stadium = _stadium_id(state)
        if stadium is not None and _has_damage_modifier_text(stadium):
            return stadium
        for player in (me, 1 - me):
            for mon in _in_play_pokemon(state, player):
                if _has_damage_modifier_text(getattr(mon, "id", None)):
                    return int(mon.id)
                for tool in (getattr(mon, "tools", None) or []):
                    if tool is not None and _has_damage_modifier_text(getattr(tool, "id", None)):
                        return int(tool.id)
                for card in (getattr(mon, "energyCards", None) or []):
                    if card is None:
                        continue
                    data = _card(getattr(card, "id", None))
                    if data is None or data.cardType != CardType.SPECIAL_ENERGY:
                        continue
                    if _has_damage_modifier_text(card.id):
                        return int(card.id)
    except Exception:  # noqa: BLE001
        return None
    return None


# --- ワザコストの充足判定 --------------------------------------------------------------

def _can_pay_cost(cost: list[int], have: list[int]) -> bool:
    """`cost`(必要エネの色配列)を `have`(ついているエネの色配列)で払えるか。"""
    pool = list(have)
    for need in cost:
        if int(need) == int(EnergyType.COLORLESS):
            continue
        for i, got in enumerate(pool):
            if int(got) == int(need) or int(got) in _WILD_ENERGY:
                pool.pop(i)
                break
        else:
            return False
    colorless = sum(1 for c in cost if int(c) == int(EnergyType.COLORLESS))
    return len(pool) >= colorless


def _cost_surcharge(state, mon) -> int:
    """スタジアム効果で増える【無】エネルギーの個数(夜の鉱山×テラスタル)。"""
    if _stadium_id(state) not in _TERA_COST_SURCHARGE_STADIUM_IDS:
        return 0
    card = _card(getattr(mon, "id", None))
    return 1 if (card is not None and bool(card.tera)) else 0


# --- 打点の閉形式見積り ----------------------------------------------------------------

def estimate_attack_damage(
    state, me: int, attack_id: int, extra_energy: int = 0, report: dict | None = None,
) -> int | None:
    """``attack_id`` のワザを今使ったときの打点(弱点・抵抗・軽減を**含まない**素の値)。

    既知パターン(まんようしぐれ型 / 固定打点型)以外、または必要な盤面情報が読めない場合は
    None。``extra_energy`` は「自分のバトルポケモンにこれから追加するエネルギーの個数」。
    """
    try:
        fixed = _FIXED_DAMAGE_ATTACKS.get(int(attack_id))
        if fixed is not None:
            return int(fixed[0])  # 盤面非依存の定数打点(エネを足しても変わらない)
        entry = _BOTH_ACTIVE_ENERGY_ATTACKS.get(int(attack_id))
        if entry is None:
            _note(report, "reason", _Reason.UNKNOWN_ATTACK)
            return None
        base, per_energy = entry
        mine = _active_energy_count(state, me)
        theirs = _active_energy_count(state, 1 - me)
        if mine is None:
            _note(report, "reason", _Reason.NO_ACTIVE)
            return None
        if theirs is None:
            _note(report, "reason", _Reason.NO_OPPONENT_ACTIVE)
            return None
        return base + per_energy * (mine + max(0, int(extra_energy)) + theirs)
    except Exception:  # noqa: BLE001 - 見積り失敗が意思決定を止めてはならない
        _note(report, "reason", _Reason.ERROR)
        return None


def estimate_current_damage(
    state, me: int, extra_energy: int = 0, report: dict | None = None,
) -> int | None:
    """自分のバトルポケモンが**今すぐ使えるワザ**の最大打点(素の値)。判定不能は None。

    - コストを払えるワザだけを対象にする(夜の鉱山のコスト増も加味)。
    - 自分のバトルポケモンが持つワザに**1つでも未知パターン**があれば None
      (最大打点が確定しないため。「知っているワザだけで最大を取る」と過小評価にも
      過大評価にもなりうるので黙る方が安全)。
    - 払えるワザが1つも無ければ None(=「攻撃できない」と「判定不能」を区別しない。
      呼び出し側はどちらでもフォールバックすればよい)。
    """
    try:
        mon = _active_pokemon(state, me)
        if mon is None:
            _note(report, "reason", _Reason.NO_ACTIVE)
            return None
        card = _card(getattr(mon, "id", None))
        if card is None or not (card.attacks or []):
            _note(report, "reason", _Reason.UNKNOWN_ATTACK)
            return None
        extra = max(0, int(extra_energy))
        have = [int(e) for e in (mon.energies or [])] + [int(EnergyType.GRASS)] * extra
        surcharge = [int(EnergyType.COLORLESS)] * _cost_surcharge(state, mon)
        best: int | None = None
        for attack_id in card.attacks:
            if not is_known_attack(attack_id):
                _note(report, "reason", _Reason.UNKNOWN_ATTACK)
                return None  # 未知のワザが混ざる=最大打点が確定しない
            try:
                cost = [int(e) for e in card_cache.get_attack(int(attack_id)).energies]
            except Exception:  # noqa: BLE001
                _note(report, "reason", _Reason.UNKNOWN_ATTACK)
                return None
            if not _can_pay_cost(cost + surcharge, have):
                continue
            damage = estimate_attack_damage(state, me, int(attack_id), extra, report)
            if damage is None:
                return None
            best = damage if best is None else max(best, damage)
        if best is None:
            _note(report, "reason", _Reason.NO_PAYABLE_ATTACK)
        return best
    except Exception:  # noqa: BLE001
        _note(report, "reason", _Reason.ERROR)
        return None


# --- KO可否 ----------------------------------------------------------------------------

def _remaining_hp(mon) -> int | None:
    """残HP。`Pokemon.hp` は Current HP(=残り)なのでそのまま使う。"""
    try:
        hp = int(mon.hp)
    except Exception:  # noqa: BLE001
        return None
    return hp if hp > 0 else None


def _my_attack_energy_type(state, me: int) -> int | None:
    """自分のバトルポケモンの弱点判定に使うタイプ(= ポケモンの `energyType`)。"""
    mon = _active_pokemon(state, me)
    card = _card(getattr(mon, "id", None)) if mon is not None else None
    if card is None:
        return None
    try:
        return int(card.energyType)
    except Exception:  # noqa: BLE001
        return None


def can_ko_with_more_energy(
    state, me: int, extra_energy: int, config: dict | None = None,
    report: dict | None = None,
) -> bool | None:
    """自分のバトルポケモンにエネを ``extra_energy`` 個足したとき、相手のバトルポケモンを
    今すぐの攻撃でKOできるか。``True``/``False``/``None``(判定不能)の3値。

    - ``True``: 弱点を**使わない**下界打点が「残HP + ko_margin」以上。
    - ``False``: 弱点×2の上界打点でも残HPに**届かない**。
    - ``None``: 未知のワザ / 抵抗持ち / 場にダメージ改変カード / 情報不足。
    """
    try:
        if state is None:
            _note(report, "reason", _Reason.NO_STATE)
            return None
        opponent = _active_pokemon(state, 1 - me)
        if opponent is None:
            _note(report, "reason", _Reason.NO_OPPONENT_ACTIVE)
            return None
        hp = _remaining_hp(opponent)
        if hp is None:
            _note(report, "reason", _Reason.UNKNOWN_HP)
            return None

        modifier = _damage_modifier_risk(state, me)
        if modifier is not None:
            _note(report, "reason", _Reason.DAMAGE_MODIFIER_IN_PLAY)
            _note(report, "modifier_card_id", modifier)
            return None

        damage = estimate_current_damage(state, me, extra_energy, report)
        if damage is None:
            return None

        margin = int((config or {}).get("ko_margin", 0))
        my_type = _my_attack_energy_type(state, me)
        opp_card = _card(getattr(opponent, "id", None))
        weak = (opp_card is not None and opp_card.weakness is not None
                and my_type is not None and int(opp_card.weakness) == my_type)
        resist = (opp_card is not None and opp_card.resistance is not None
                  and my_type is not None and int(opp_card.resistance) == my_type)

        _note(report, "damage", damage)
        _note(report, "remaining_hp", hp)
        _note(report, "weakness", weak)
        _note(report, "resistance", resist)

        if not resist and damage >= hp + margin:
            return True  # 弱点抜きの下界で足りている=確実
        upper = damage * 2 if weak else damage
        if upper < hp:
            return False  # 弱点込みの上界でも届かない=確実に無理
        _note(report, "reason", _Reason.OPPONENT_RESISTANCE if resist else "within_margin")
        return None
    except Exception:  # noqa: BLE001
        _note(report, "reason", _Reason.ERROR)
        return None


def can_ko_now(state, me: int, config: dict | None = None,
               report: dict | None = None) -> bool | None:
    """今のエネルギーのまま、相手のバトルポケモンを今すぐの攻撃でKOできるか(3値)。"""
    return can_ko_with_more_energy(state, me, 0, config, report)


# --- 「このターン攻撃でKOする手は存在しない」の健全な上界判定 -----------------------------

def _hand_attachable_energy(state, me: int) -> int:
    """今ターン、手札から自分の場に**追加できる**エネルギーの個数の上界。

    - 手貼り(`state.energyAttached` が False なら1回)
    - みどりのまい型の特性(自分の場のオーガポン1体につき1回。使用済みかは観測できないので
      **未使用**と仮定=上界側)
    どちらも手札にあるエネルギーカードの枚数で頭打ちになる。
    """
    try:
        hand = state.players[me].hand or []
    except Exception:  # noqa: BLE001
        return 0
    energy_in_hand = 0
    for card in hand:
        data = _card(getattr(card, "id", None))
        if data is not None and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            energy_in_hand += 1
    slots = 0 if bool(getattr(state, "energyAttached", False)) else 1
    for mon in _in_play_pokemon(state, me):
        if int(getattr(mon, "id", -1)) == TEAL_MASK_OGERPON_EX_ID:
            slots += 1
    return min(energy_in_hand, slots)


def max_attacker_energy(state, me: int) -> int | None:
    """このターン、自分のバトルポケモンに乗りうるエネルギー数の上界。

    にげる/入れ替え・Nの計画(ベンチ→バトル場へエネ移動)まで含めた最悪ケースとして、
    「自分の場の総エネ + 手札から追加できるエネ」を採る(実際にはここまで集められない)。
    エネ移動カードの種類を前提にしない=未知の移動手段があっても上界を割らない。
    """
    try:
        total = 0
        for mon in _in_play_pokemon(state, me):
            total += len(getattr(mon, "energies", None) or [])
        return total + _hand_attachable_energy(state, me)
    except Exception:  # noqa: BLE001
        return None


def _any_attack_payable_at_best(state, me: int) -> bool | None:
    """自分の場のポケモンの**誰か**が、最良のエネ集約でワザのコストを払えるか。

    「自分の場の全エネ(色そのまま)+ 手札から足せる基本【草】エネ」を1体に集められると
    仮定した上界判定。False(=どう頑張ってもワザが撃てない)は健全。未知のワザが混ざる
    場合は None。
    """
    try:
        mons = _in_play_pokemon(state, me)
        if not mons:
            return False
        pool: list[int] = []
        for mon in mons:
            pool.extend(int(e) for e in (getattr(mon, "energies", None) or []))
        pool.extend([int(EnergyType.GRASS)] * _hand_attachable_energy(state, me))
        for mon in mons:
            card = _card(getattr(mon, "id", None))
            if card is None:
                return None
            surcharge = [int(EnergyType.COLORLESS)] * _cost_surcharge(state, mon)
            for attack_id in (card.attacks or []):
                try:
                    cost = [int(e) for e in card_cache.get_attack(int(attack_id)).energies]
                except Exception:  # noqa: BLE001
                    return None
                if _can_pay_cost(cost + surcharge, pool):
                    return True
        return False
    except Exception:  # noqa: BLE001
        return None


def is_ko_impossible_this_turn(
    state, me: int, config: dict | None = None, report: dict | None = None,
) -> bool | None:
    """「このターン、攻撃で相手をKOする手は**そもそも存在しない**」なら True。

    `ability_draw_brake` の (b)「最大まで足してもKO不可」/ `low_deck_draw_brake` の
    「今ターンKOできない」に対応する。以下をすべて最悪ケース(自分に有利な側)で見積り、
    それでも届かないときだけ True を返すので、True は健全(sound)。

      * そもそもワザのコストを払えるか = `_any_attack_payable_at_best`
        (場の全エネ+手札から足せるエネを1体に集められると仮定してなお払えないなら不可能)
      * 自分のバトルポケモンのエネ数 = `max_attacker_energy`(場の総エネ+手札から追加可能分)
      * 相手のバトルポケモンのエネ数 = 相手の場のポケモンの**最大**エネ数
        (ボスの指令等で誰を引きずり出しても超えない)
      * 相手の残HP = 相手の場のポケモンの**最小**残HP
      * 相手の場に草弱点が1体でもいれば打点×2

    判定不能(未知のワザ/ダメージ改変カード/情報不足)は None。
    """
    try:
        if state is None:
            _note(report, "reason", _Reason.NO_STATE)
            return None
        modifier = _damage_modifier_risk(state, me)
        if modifier is not None:
            _note(report, "reason", _Reason.DAMAGE_MODIFIER_IN_PLAY)
            _note(report, "modifier_card_id", modifier)
            return None

        my_mons = _in_play_pokemon(state, me)
        opp_mons = _in_play_pokemon(state, 1 - me)
        if not my_mons or not opp_mons:
            _note(report, "reason", _Reason.NO_ACTIVE if not my_mons else _Reason.NO_OPPONENT_ACTIVE)
            return None

        # 自分の場のポケモンが全部「既知パターンのワザだけ」でないと最大打点が確定しない。
        #
        # 混在(まんようしぐれ型 + 固定打点型)は base / per_energy をそれぞれ**最大値**で
        # 束ねる。得られる上界 base + per_energy×E は、
        #   まんようしぐれ  30 + 30×E   ≤ max(30,220) + max(30,0)×E
        #   ウッドハンマー  220         ≤ max(30,220) + max(30,0)×E
        # のどちらも必ず覆う(=健全な上界のまま)。上界が緩む方向にしか動かないので、
        # 「KOは不可能(True)」を返す条件は厳しくなる=安全側。
        per_energy = 0
        base = 0
        for mon in my_mons:
            card = _card(getattr(mon, "id", None))
            if card is None or not (card.attacks or []):
                _note(report, "reason", _Reason.UNKNOWN_ATTACK)
                return None
            for attack_id in card.attacks:
                entry = _BOTH_ACTIVE_ENERGY_ATTACKS.get(int(attack_id))
                fixed = _FIXED_DAMAGE_ATTACKS.get(int(attack_id))
                if entry is None and fixed is None:
                    _note(report, "reason", _Reason.UNKNOWN_ATTACK)
                    return None
                if entry is not None:
                    base = max(base, entry[0])
                    per_energy = max(per_energy, entry[1])
                else:
                    base = max(base, fixed[0])

        payable = _any_attack_payable_at_best(state, me)
        if payable is None:
            _note(report, "reason", _Reason.UNKNOWN_ATTACK)
            return None
        if payable is False:
            _note(report, "reason", _Reason.NO_PAYABLE_ATTACK)
            return True  # エネを最大限集めてもワザが撃てない=攻撃でKOする手は存在しない

        my_energy = max_attacker_energy(state, me)
        if my_energy is None:
            _note(report, "reason", _Reason.ERROR)
            return None
        opp_energy = 0
        min_hp: int | None = None
        weak_any = False
        # 弱点は「自分のどのポケモンが攻撃してもよい」前提で見る(1体でも弱点を突ける
        # 組み合わせがあれば ×2 側で上界を取る=「不可能」と言いにくくする方向)。
        my_types = set()
        for mon in my_mons:
            card = _card(getattr(mon, "id", None))
            if card is not None:
                my_types.add(int(card.energyType))
        for mon in opp_mons:
            opp_energy = max(opp_energy, len(getattr(mon, "energies", None) or []))
            hp = _remaining_hp(mon)
            if hp is not None:
                min_hp = hp if min_hp is None else min(min_hp, hp)
            card = _card(getattr(mon, "id", None))
            if (card is not None and card.weakness is not None
                    and int(card.weakness) in my_types):
                weak_any = True
        if min_hp is None:
            _note(report, "reason", _Reason.UNKNOWN_HP)
            return None

        upper = base + per_energy * (my_energy + opp_energy)
        if weak_any:
            upper *= 2
        _note(report, "max_damage", upper)
        _note(report, "min_opponent_hp", min_hp)
        return upper < min_hp
    except Exception:  # noqa: BLE001
        _note(report, "reason", _Reason.ERROR)
        return None


# --- 自己検証(既知パターンとカードデータの整合) -------------------------------------

def verify_known_attacks() -> list[str]:
    """既知パターンの定数が `cg.api.all_attack()` の実データと合っているかを検査する。

    テストとスモークから呼ぶ。ズレていれば理由の文字列リストを返す(空リスト=OK)。
    """
    problems: list[str] = []
    for attack_id, (base, per_energy) in _BOTH_ACTIVE_ENERGY_ATTACKS.items():
        try:
            attack = card_cache.get_attack(attack_id)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"attackId={attack_id} をカードデータから引けない: {exc!r}")
            continue
        if int(attack.damage) != int(base):
            problems.append(
                f"attackId={attack_id} の基礎打点が {attack.damage} で定数 {base} と不一致")
        if f"{per_energy} more damage" not in (attack.text or ""):
            problems.append(
                f"attackId={attack_id} のテキストに '{per_energy} more damage' が無い: "
                f"{attack.text!r}")
        if "both Active" not in (attack.text or ""):
            problems.append(
                f"attackId={attack_id} のテキストが両バトル場参照でない: {attack.text!r}")
    for attack_id, (damage, self_damage) in _FIXED_DAMAGE_ATTACKS.items():
        try:
            attack = card_cache.get_attack(attack_id)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"attackId={attack_id} をカードデータから引けない: {exc!r}")
            continue
        if int(attack.damage) != int(damage):
            problems.append(
                f"attackId={attack_id} の打点が {attack.damage} で定数 {damage} と不一致")
        text = attack.text or ""
        # 固定打点型は「盤面に応じて打点が変わる」記述が無いことが前提。追加ダメージ系の
        # 文言が付いたら(データ更新等)定数が嘘になるので検出する。
        if "more damage" in text:
            problems.append(
                f"attackId={attack_id} は固定打点のはずだが追加ダメージ記述がある: {text!r}")
        if self_damage and f"{self_damage} damage to itself" not in text:
            problems.append(
                f"attackId={attack_id} のテキストに自傷 {self_damage} の記述が無い: {text!r}")
    return problems
