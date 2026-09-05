"""壁デッキ検知 → 非exアタッカーを起用して育てる操縦ルートの**判定部**
(``ml_policy_agent`` の r14 ``wall_attacker_route``。config-gated、キー無し=完全不変)。

なぜ必要か(実測で確定した因果)
--------------------------------
crustle 系の壁デッキ(イワパレス 345 ×3 + オーガポン いしずえのめん ex 117 ×1 等)に対し、
og_v032 のアタッカーは オーガポン みどりのめん ex(96)**1種類だけ**で、これは
**ex かつ 特性(みどりのまい)持ち**。壁2種の特性はそれぞれ

  * イワパレス 345「Mysterious Rock Inn」:
    "Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon {ex}."
    = 相手が **ex** ならワザのダメージを完全に無効
  * いしずえのめん ex 117「Cornerstone Stance」:
    "Prevent all damage from attacks done to this Pokémon by your opponent's Pokémon that
    have an Ability."
    = 相手が **特性持ち** ならワザのダメージを完全に無効

なので、オーガポンの打点は両方の壁に対して**数学的に0**(打点計算ではなく無効化テキストで
確定する。だから探索は要らない)。

1枚を カプ・ブルル(920、たね/非ex/**特性なし**/HP140/にげ3/ウッドハンマー 草草無無 =
固定220、自傷30)に差し替えた版は crustle 勝率 17.75% → 22.75%(n=400、+5.0pt)。
しかし稼働診断(60試合)では **ブルル場出し率 88.3% / 4エネ到達率 6.7% / ウッドハンマー
0.0回/試合 / エネ0のまま終了 72%** で、**ブルルは置かれているだけで一度も攻撃していない**。
+5.0pt は「ポケモンが5枚になった副次効果」であって、アタッカーとしては機能していなかった。

既存の `energy_to_active_first`(r10 Fix-G)は「アクティブが攻撃不能ならアクティブにエネを
寄せる」ガードだが、**ブルルがそもそもアクティブに出てこない**ので発火しない。このモジュールは
その穴を埋める「エネをブルルへ寄せる → 前に出す → 殴る」の3点セットの判定部を提供する。

このファイルの位置づけ
----------------------
`ml_policy_agent.py` は他エージェントが並行編集中のため、判定ロジックは
`retreat_safety_eval.py` / `boss_target_eval.py` と同じ流儀で**独立モジュール**に置き、
`ml_policy_agent` 側には veto 連鎖に足す薄い関数だけを新規追加する(既存関数は書き換えない)。

安全側の設計
------------
* 壁の無効化判定は**カードIDのハードコードではなくテキスト**(`nullifies_damage_from`)。
  「prevent all damage ... this Pokémon」+ 条件(``{ex}`` / ``have an Ability``)の形だけを
  既知パターンとして扱い、それ以外の書き方は ``None``(判定不能)=発火しない。
  ``wall_card_ids`` は「どのカードを壁として見るか」の**ゲート**にすぎず、実際に無効化される
  かどうかは毎回テキストで確かめる。
* 情報が読めない/想定外のスキーマは全て「介入しない」側に倒す(戻り値 ``None``)。
"""

from __future__ import annotations

import re

from cg.api import AreaType, CardType, OptionType

from ptcg_ai.shared import card_cache

# 既定値(config で全て上書き可能)。``enabled`` が False の間は `evaluate` が即 None を返す。
DEFAULTS: dict = {
    "enabled": False,
    # 壁として監視するカードID(イワパレス / オーガポン いしずえのめん ex)。
    "wall_card_ids": (345, 117),
    # 壁に通る非exアタッカー(カプ・ブルル)。
    "attacker_card_ids": (920,),
    # そのアタッカーが撃ちたいワザの必要エネ数(ウッドハンマー=草草無無=4)。
    "attacker_attack_cost": 4,
    # 「現アタッカーでは壁に打点0」が**確定**していることを発火条件にするか。
    "require_zero_damage": True,
    # ベンチ→バトル場へエネを移すサポート(Nの筋書き 1221 = 最大2個)。
    "energy_move_card_ids": (1221,),
    "energy_move_max": 2,
}

# 「このポケモン自身へのワザのダメージを完全に無効化する」テキストの既知パターン。
# 実データ(all_card_data() の skills[].text)で 345 / 117 の両方が一致することを確認済み。
_PREVENT_ALL_DAMAGE = re.compile(r"prevent all (?:of the )?damage", re.IGNORECASE)
_TARGETS_ITSELF = re.compile(r"(?:done )?to this Pok[eé]mon", re.IGNORECASE)
# 無効化の対象条件。どちらにも当たらない「prevent all damage」は判定不能にする。
_ONLY_VS_EX = re.compile(r"Pok[eé]mon ?\{ex\}", re.IGNORECASE)
_ONLY_VS_ABILITY = re.compile(r"that have an Ability", re.IGNORECASE)


def resolve(config: dict | None) -> dict:
    """``config["wall_attacker_route"]`` を DEFAULTS とマージした dict を返す。"""
    merged = dict(DEFAULTS)
    merged.update(config or {})
    return merged


def _card(card_id):
    if card_id is None:
        return None
    try:
        return card_cache.get_card(int(card_id))
    except Exception:  # noqa: BLE001
        return None


def _has_ability(card) -> bool:
    """特性(skills)を持つ種族か。ワザは `attacks` 側なので skills には入らない。"""
    try:
        return bool(card.skills)
    except Exception:  # noqa: BLE001
        return False


def nullifies_damage_from(wall_card_id, attacker_card_id) -> bool | None:
    """``wall_card_id`` の特性が ``attacker_card_id`` のワザのダメージを**完全に無効化**するか。

    Returns:
        ``True``  = 無効化が確定(=打点0)。
        ``False`` = 既知パターンのどれにも当たらない(=無効化されない)。
        ``None``  = 「prevent all damage ... this Pokémon」ではあるが条件が未知の書き方
                    (=判定不能。呼び出し側は発火しない)。

    テキスト駆動にしているのは、カードプールが増えても同型の壁に自動追随させるため。
    """
    wall = _card(wall_card_id)
    attacker = _card(attacker_card_id)
    if wall is None or attacker is None:
        return None
    try:
        attacker_is_ex = bool(attacker.ex) or bool(attacker.megaEx)
        attacker_has_ability = _has_ability(attacker)
        unknown = False
        for skill in (wall.skills or []):
            text = getattr(skill, "text", "") or ""
            if not _PREVENT_ALL_DAMAGE.search(text) or not _TARGETS_ITSELF.search(text):
                continue
            if _ONLY_VS_EX.search(text):
                if attacker_is_ex:
                    return True
            elif _ONLY_VS_ABILITY.search(text):
                if attacker_has_ability:
                    return True
            else:
                unknown = True  # 無条件無効 or 未知の限定条件=量を評価せず黙る
        return None if unknown else False
    except Exception:  # noqa: BLE001
        return None


# --- 盤面アクセサ -----------------------------------------------------------------------

def _active_pokemon(state, player: int):
    try:
        for mon in (state.players[player].active or []):
            if mon is not None:
                return mon
    except Exception:  # noqa: BLE001
        return None
    return None


def _bench(state, player: int) -> list:
    try:
        return list(state.players[player].bench or [])
    except Exception:  # noqa: BLE001
        return []


def _energy_count(mon) -> int:
    """``mon`` についているエネルギーの個数。2個ぶん供給する特殊エネは2要素で並ぶ。"""
    try:
        return len(getattr(mon, "energies", None) or [])
    except Exception:  # noqa: BLE001
        return 0


def is_energy_card(card_id) -> bool:
    card = _card(card_id)
    if card is None:
        return False
    try:
        return card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
    except Exception:  # noqa: BLE001
        return False


def energy_attach_target(option, state, ability_card_ids) -> tuple | None:
    """この選択肢が「自分の場のポケモンにエネルギーを付ける手」なら ``(AreaType, index)`` を返す。

    `ml_policy_agent._energy_attach_target_area`(r10 Fix-G、実データ episode 93408551 T6 で
    スキーマ確認済み)の**付け先インデックスまで返す版**。2系統ある:

      - 手貼り = ``OptionType.ATTACH``。``area``/``index`` が手札のエネ、
        ``inPlayArea``/``inPlayIndex`` が付け先の場のポケモン。
      - 「みどりのまい」型の特性 = ``OptionType.ABILITY``。``area``/``index`` が特性を使う
        ポケモン自身を指し、エネはそのポケモン自身に付く(付け先 == そのポケモン)。
    """
    try:
        from ptcg_ai.learning import encoder as _enc
        if option.type == OptionType.ATTACH:
            if not is_energy_card(_enc._resolve_card_id(option, state)):
                return None
            area = getattr(option, "inPlayArea", None)
            index = getattr(option, "inPlayIndex", None)
            if area is None or index is None:
                return None
            return (area, int(index))
        if option.type == OptionType.ABILITY:
            card_id = _enc._resolve_card_id(option, state)
            if card_id is None or int(card_id) not in set(ability_card_ids or ()):
                return None
            area = getattr(option, "area", None)
            index = getattr(option, "index", None)
            if area is None or index is None:
                return None
            return (area, int(index))
    except Exception:  # noqa: BLE001 - 判定不能は None(=介入しない、安全側)
        return None
    return None


def _hand_has(state, me: int, card_ids) -> bool:
    try:
        wanted = {int(x) for x in (card_ids or ())}
        for card in (state.players[me].hand or []):
            cid = getattr(card, "id", None)
            if cid is not None and int(cid) in wanted:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


# --- 共通ゲート -------------------------------------------------------------------------

def evaluate(state, me: int, config: dict | None) -> dict | None:
    """壁ルートの共通判定。発火余地が無ければ ``None``。

    Returns(dict):
        wall_ids                 相手の場に見えている壁のカードID集合
        opp_active_wall_id       相手のバトルポケモンが壁ならそのID(でなければ None)
        zero_damage_vs_active    相手のバトルポケモンが壁で、**自分の現バトルポケモンの打点が
                                 0であることが確定**している(``require_zero_damage`` が
                                 False なら「相手アクティブが壁」だけで True)
        zero_damage_vs_any       上記を「場のいずれかの壁」に緩めた版
        active_is_attacker       自分のバトルポケモンが指定アタッカー(ブルル)である
        attacker_bench_index     ベンチに居る指定アタッカーの配列位置(居なければ None)
        attacker_energy          指定アタッカーについているエネルギー数
        attacker_ready           ``attacker_energy >= attacker_attack_cost``
        attacker_ready_with_move Nの筋書き等でバトル場へエネを移せば足りる見込み
        attack_cost              必要エネ数(config そのまま)

    ``None`` を返す条件: config 無効 / 相手の場に壁が見えない / 自分の場に指定アタッカーが
    居ない / 状態が読めない。
    """
    try:
        cfg = resolve(config)
        if not cfg.get("enabled", False):
            return None
        if state is None:
            return None
        wall_ids = {int(x) for x in (cfg.get("wall_card_ids") or ())}
        attacker_ids = {int(x) for x in (cfg.get("attacker_card_ids") or ())}
        if not wall_ids or not attacker_ids:
            return None

        opp = 1 - int(me)
        opp_active = _active_pokemon(state, opp)
        opp_active_id = int(opp_active.id) if opp_active is not None else None
        seen_walls = set()
        for mon in ([opp_active] if opp_active is not None else []) + _bench(state, opp):
            if mon is None:
                continue
            cid = getattr(mon, "id", None)
            if cid is not None and int(cid) in wall_ids:
                seen_walls.add(int(cid))
        if not seen_walls:
            return None

        my_active = _active_pokemon(state, me)
        my_active_id = int(my_active.id) if my_active is not None else None
        active_is_attacker = my_active_id is not None and my_active_id in attacker_ids

        require_zero = bool(cfg.get("require_zero_damage", True))
        if require_zero and my_active_id is not None and not active_is_attacker:
            blocked_walls = {
                w for w in seen_walls if nullifies_damage_from(w, my_active_id) is True
            }
        else:
            # 次のいずれか。どれも「現アタッカーの打点が0か」を問う意味が無いので、
            # 壁が見えていること自体をゲートにする:
            #   * require_zero_damage=False(config で明示的に外している)
            #   * 自分のバトル場が空(KO直後の選び直し。「今の打点」という概念が無い)
            #   * バトル場が既に指定アタッカー本人(=このルートを実行中。ここで
            #     「壁に通るから対象外」と判定すると (a2)/(c) が永久に発火しない)
            blocked_walls = set(seen_walls)
        opp_active_wall_id = (
            opp_active_id if (opp_active_id is not None and opp_active_id in wall_ids) else None
        )

        # 指定アタッカーの所在(バトル場優先。ベンチに複数居ればエネが最も多い個体)。
        attacker_bench_index = None
        if active_is_attacker:
            attacker_energy = _energy_count(my_active)
        else:
            best = None
            for i, mon in enumerate(_bench(state, me)):
                if mon is None:
                    continue
                cid = getattr(mon, "id", None)
                if cid is None or int(cid) not in attacker_ids:
                    continue
                energy = _energy_count(mon)
                if best is None or energy > best[1]:
                    best = (i, energy)
            if best is None:
                return None  # 自分の場に指定アタッカーが居ない
            attacker_bench_index, attacker_energy = best

        cost = int(cfg.get("attacker_attack_cost", 4))
        # Nの筋書き(ベンチ→バトル場に最大2個)で届く見込みか。
        # 供給元は「アタッカー以外の**ベンチ**個体」だけを数える(現バトル場のエネは
        # にげるコストで減るうえ、にげた直後の枚数が読み切れないので数えない=控えめ)。
        move_ids = cfg.get("energy_move_card_ids") or ()
        move_max = int(cfg.get("energy_move_max", 2))
        movable = 0
        for i, mon in enumerate(_bench(state, me)):
            if mon is None or (attacker_bench_index is not None and i == attacker_bench_index):
                continue
            movable += _energy_count(mon)
        movable = min(move_max, movable)
        supporter_played = bool(getattr(state, "supporterPlayed", False))
        ready_with_move = (
            not supporter_played
            and _hand_has(state, me, move_ids)
            and attacker_energy + movable >= cost
        )

        return {
            "wall_ids": seen_walls,
            "opp_active_wall_id": opp_active_wall_id,
            "zero_damage_vs_active": (
                opp_active_wall_id is not None and opp_active_wall_id in blocked_walls
            ),
            "zero_damage_vs_any": bool(blocked_walls),
            "active_is_attacker": active_is_attacker,
            "attacker_bench_index": attacker_bench_index,
            "attacker_energy": int(attacker_energy),
            "attacker_ready": attacker_energy >= cost,
            "attacker_ready_with_move": bool(ready_with_move),
            "attack_cost": cost,
        }
    except Exception:  # noqa: BLE001 - 判定の失敗が意思決定を止めてはならない
        return None


def option_targets_attacker(option, state, ctx: dict, ability_card_ids) -> bool:
    """``option`` が「指定アタッカーにエネルギーを付ける手」か。"""
    target = energy_attach_target(option, state, ability_card_ids)
    if target is None:
        return False
    area, index = target
    if ctx.get("active_is_attacker"):
        return area == AreaType.ACTIVE
    bench_index = ctx.get("attacker_bench_index")
    return area == AreaType.BENCH and bench_index is not None and index == bench_index


def switch_option_index_for_attacker(select, me: int, ctx: dict) -> int | None:
    """交代先を選ぶ select(CARD/SWITCH)で、指定アタッカーを指す選択肢のインデックス。

    **自分のベンチ**を指す選択肢だけを対象にする(``playerIndex == me`` を厳格に要求する。
    ボスの指令の対象選択は相手のベンチ=``playerIndex`` が異なるので巻き込まない。
    このスキーマは `retreat_safety_eval.evaluate` が実エンジンで使っているものと同じ)。
    """
    bench_index = ctx.get("attacker_bench_index")
    if bench_index is None:
        return None
    try:
        for i, opt in enumerate(select.option):
            if (getattr(opt, "area", None) == AreaType.BENCH
                    and getattr(opt, "playerIndex", None) == me
                    and getattr(opt, "index", None) == bench_index):
                return i
    except Exception:  # noqa: BLE001
        return None
    return None
