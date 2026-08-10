"""盤面のポケモンをトークン列として取り出す(Transformer化 T0)。

設計書: ``sample_submission/docs/plans/neural-agent/transformer-tokenized-encoder-design.md`` T0/T1。
調査根拠: ``transformer-tokenized-encoder-investigation.md``。

## このモジュールの役割

現行の ``encoder.py`` は盤面を166〜389次元の**固定長ベクトル**に要約する。本モジュールは
同じ ``State`` から、**可変長のトークン列**(ベンチにいる数だけトークンを作る。空きスロットの
ゼロ埋めをしない)を作る。T0/T1の対象は自分・相手のバトル場とベンチのポケモンのみで、
手札・トラッシュ・山札/サイドのカード単位トークン化はT3で追加する(理由は設計書§2、
「カードIDを持たないゾーン集約トークンは意味を成立させられない」ため)。

## 数値特徴

各ポケモントークンの数値特徴は、``encoder._pokemon_features()`` が返すのと**全く同じ11次元**
(``encoder._POKEMON_FEATURE_NAMES``)をそのまま流用する。二重実装しない(design.mdの
既存踏襲方針と同じ考え方)。

## ポインタ(選択肢→盤面トークン)

``Pokemon.serial`` / ``Card.serial``(``cg/api.py`` で対戦中の個体に一意な番号と明記)を
キーにした ``{serial: token_index}`` の対応表を、トークン列構築と同時に作る。この対応表は
決定点ごとに作り直す前処理であり、学習パラメータではない(Slot Embeddingは使わない)。
selfattentionは位置埋め込みを加えない限り置換同変なので、ベンチの並び順やトークンの
並べ替えに影響されない。

``serial`` そのものはshardに保存しない(この対応表は決定点ごとに使い捨てる中間データで、
保存が必要なのは解決済みの「ローカルトークンindex」だけのため)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cg.api import AreaType, Option, OptionType, Pokemon, SelectData, State

from ptcg_ai.learning.encoder import (
    _POKEMON_FEATURE_NAMES,
    _active_pokemon,
    _pokemon_features,
    _resolve_in_play_pokemon,
    _resolve_pokemon,
)

# 盤面トークンの数値特徴の次元数(ポケモン1体あたり)。encoder._POKEMON_FEATURE_NAMES と同一。
BOARD_TOKEN_NUMERIC_DIM: int = len(_POKEMON_FEATURE_NAMES)

# ゾーンID語彙。T0/T1は0〜3のみ生成する。4以降はT3で使うために番号だけ予約しておき、
# 後から追加してもT0/T1が生成済みのIDが変わらないようにする。
ZONE_SELF_ACTIVE = 0
ZONE_SELF_BENCH = 1
ZONE_OPP_ACTIVE = 2
ZONE_OPP_BENCH = 3
# --- 以下はT3で使う予約ID(T0/T1では生成しない) ---
ZONE_STADIUM = 4
ZONE_SELF_HAND = 5
ZONE_SELF_DISCARD = 6
ZONE_OPP_DISCARD = 7
ZONE_SELF_DECK_PRIZE = 8

ZONE_NAMES: dict[int, str] = {
    ZONE_SELF_ACTIVE: "self_active",
    ZONE_SELF_BENCH: "self_bench",
    ZONE_OPP_ACTIVE: "opp_active",
    ZONE_OPP_BENCH: "opp_bench",
    ZONE_STADIUM: "stadium",
    ZONE_SELF_HAND: "self_hand",
    ZONE_SELF_DISCARD: "self_discard",
    ZONE_OPP_DISCARD: "opp_discard",
    ZONE_SELF_DECK_PRIZE: "self_deck_prize",
}

# owner(0=自分, 1=相手)はzone_idから決定的に導出する(T0.1: 「B. zone_idから
# owner_idを決定的に生成し、その対応をschemaとして固定する」を採用)。owner_id用の
# 別配列は持たない。zone_idさえ分かればownerは常にこの表で一意に決まる。
OWNER_SELF = 0
OWNER_OPP = 1
_ZONE_TO_OWNER: dict[int, int] = {
    ZONE_SELF_ACTIVE: OWNER_SELF, ZONE_SELF_BENCH: OWNER_SELF,
    ZONE_OPP_ACTIVE: OWNER_OPP, ZONE_OPP_BENCH: OWNER_OPP,
    ZONE_STADIUM: OWNER_SELF,  # スタジアムは「出したプレイヤー」を別途持たせる想定(T3で再検討)
    ZONE_SELF_HAND: OWNER_SELF, ZONE_SELF_DISCARD: OWNER_SELF, ZONE_SELF_DECK_PRIZE: OWNER_SELF,
    ZONE_OPP_DISCARD: OWNER_OPP,
}


def owner_of_zone(zone_id: int) -> int:
    """zone_id -> owner_id(0=自分/1=相手)。T1/T2で使うOwner Embeddingのindexに
    そのまま使える(zone_idとowner_idを別配列にしない、というT0.1の決定に対応)。"""
    if zone_id not in _ZONE_TO_OWNER:
        raise ValueError(f"未知のzone_id: {zone_id}")
    return _ZONE_TO_OWNER[zone_id]


# 対象を持たない選択肢(YES/NO/NUMBER/END等)、またはT1がまだ対応していないゾーンを
# 指す選択肢(手札のカード等)に使う番兵値。
NO_TARGET = -1

# pointer解決の理由(デバッグ・監査用)。T1/T2の学習経路では使わない
# (通常収集では計算しない。collect_tokens.py --debug-pointers 時のみ使う)。
RESOLVER_SKILL_SERIAL = "skill_serial"
RESOLVER_ACTIVE_IMPLICIT = "active_implicit"
RESOLVER_AREA_INDEX = "area_index"
RESOLVER_IN_PLAY_INDEX = "in_play_index"
RESOLVER_NO_TARGET_NO_FIELDS = "no_target_no_fields"
RESOLVER_NO_TARGET_UNRESOLVED_ZONE = "no_target_unresolved_zone"


@dataclass
class PointerDebugInfo:
    """1選択肢ぶんのpointer解決の監査情報(通常収集では作らない)。"""
    target_index: int
    target_serial: int | None
    target_card_id: int | None
    resolver: str


@dataclass
class BoardTokens:
    """1決定点ぶんの盤面トークン列。

    ``numeric_features[i]`` / ``card_ids[i]`` / ``zone_ids[i]`` / ``serials[i]`` は
    すべて同じ添字 ``i`` で対応する(トークン ``i`` の情報)。
    """

    numeric_features: list[list[float]] = field(default_factory=list)
    card_ids: list[int] = field(default_factory=list)
    zone_ids: list[int] = field(default_factory=list)
    # 決定点内でのポインタ解決にのみ使う一時データ。shardには保存しない(モジュールdocstring参照)。
    serials: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.numeric_features)


def build_board_tokens(state: State | None) -> BoardTokens:
    """``state`` の自分・相手のバトル場+ベンチをトークン列にする。

    ``state`` が ``None`` の場合は空のトークン列を返す(壊れない。encoder.py の
    ``encode_state_from_state(None)`` がゼロベクトルを返すのと同じ思想)。

    ベンチは実在する数だけトークンを作る(固定スロットのゼロ埋めをしない)。
    トークンの並び順は ``player.bench`` のリスト順をそのまま使うが、この順序に
    self-attention側が依存しないことは前提(モジュールdocstring参照)。並び順を変える
    実装(データ拡張等)を将来追加する場合も、``serials`` を同時に並べ替えれば
    ポインタ解決は壊れない。
    """
    tokens = BoardTokens()
    if state is None:
        return tokens

    your_index = state.yourIndex
    opp_index = 1 - your_index
    me = state.players[your_index]
    opp = state.players[opp_index]
    my_active = _active_pokemon(me)
    opp_active = _active_pokemon(opp)

    def _append(pokemon: Pokemon, defender, hand_size: int, owner_index: int, zone_id: int) -> None:
        tokens.numeric_features.append(
            _pokemon_features(pokemon, defender, hand_size, state, owner_index))
        tokens.card_ids.append(pokemon.id)
        tokens.zone_ids.append(zone_id)
        tokens.serials.append(pokemon.serial)

    if my_active is not None:
        _append(my_active, opp_active, me.handCount, your_index, ZONE_SELF_ACTIVE)
    for pkmn in (me.bench or []):
        if pkmn is not None:
            _append(pkmn, opp_active, me.handCount, your_index, ZONE_SELF_BENCH)

    if opp_active is not None:
        _append(opp_active, my_active, opp.handCount, opp_index, ZONE_OPP_ACTIVE)
    for pkmn in (opp.bench or []):
        if pkmn is not None:
            _append(pkmn, my_active, opp.handCount, opp_index, ZONE_OPP_BENCH)

    return tokens


# ---------------------------------------------------------------------------
# 選択肢 -> 盤面トークンのポインタ解決
# ---------------------------------------------------------------------------

# ゲームルール上、常に「操作しているプレイヤー自身のバトル場ポケモン」を対象にする
# OptionType。cg/api.py のコメントには area/index フィールドの記載が無い(暗黙の対象)が、
# ルール上ワザ・にげるは常に自分のバトルポケモンが対象になる(_option_features 側も
# これらを「対象カードなし」として扱っているのは、area/index が無いからそう見えるだけで、
# 実際に対象が無いわけではない)。
_ACTIVE_IMPLICIT_TYPES = frozenset({OptionType.ATTACK, OptionType.RETREAT})

# SPECIAL_CONDITION も多くの場合バトルポケモンの状態を指すと考えられるが、ATTACK/RETREATほど
# 強い裏付け(「ワザ/にげるは常にバトル場のポケモンしか行えない」というルール上の制約)が
# コード上で確認できていない。確認不能のため、暫定的に同じ扱いにするが要検証としてマークする。
_ACTIVE_IMPLICIT_TYPES_TENTATIVE = frozenset({OptionType.SPECIAL_CONDITION})

# 現時点でT1が対応するゾーン(ACTIVE/BENCH)以外を指す可能性がある area。
# これらを指す選択肢はT1では解決不能(NO_TARGET)。T3でゾーントークンを追加した際に対応する。
_UNRESOLVED_AREAS_T1 = frozenset({
    AreaType.HAND, AreaType.DISCARD, AreaType.PRIZE, AreaType.DECK,
    AreaType.STADIUM, AreaType.PLAYER, AreaType.LOOKING,
})


def _resolve_debug(option: Option, state: State, tokens: BoardTokens) -> PointerDebugInfo:
    """``resolve_option_target_index`` / ``resolve_option_target_index_debug`` の実体。"""
    serial_to_index = {s: i for i, s in enumerate(tokens.serials)}

    def _found(pokemon: Pokemon, resolver: str) -> PointerDebugInfo:
        idx = serial_to_index.get(pokemon.serial, NO_TARGET)
        if idx == NO_TARGET:
            return PointerDebugInfo(NO_TARGET, pokemon.serial, pokemon.id,
                                    RESOLVER_NO_TARGET_UNRESOLVED_ZONE)
        return PointerDebugInfo(idx, pokemon.serial, pokemon.id, resolver)

    # 1. SKILL は option.serial を直接持つ(cg/api.py: 「serial (int):Card serial」)。
    #    area/index を経由せず最も直接的に解決できるケース。
    if option.type == OptionType.SKILL and option.serial is not None:
        idx = serial_to_index.get(option.serial, NO_TARGET)
        card_id = tokens.card_ids[idx] if idx != NO_TARGET else None
        return PointerDebugInfo(idx, option.serial, card_id, RESOLVER_SKILL_SERIAL)

    # 2. ルール上「常に自分のバトルポケモン」が対象になる型。
    if option.type in _ACTIVE_IMPLICIT_TYPES or option.type in _ACTIVE_IMPLICIT_TYPES_TENTATIVE:
        active = _active_pokemon(state.players[state.yourIndex])
        if active is None:
            return PointerDebugInfo(NO_TARGET, None, None, RESOLVER_NO_TARGET_NO_FIELDS)
        return _found(active, RESOLVER_ACTIVE_IMPLICIT)

    # 3. area/index (ACTIVE/BENCH) で直接ポケモンを指す型
    #    (CARD/ABILITY/DISCARD等、area が ACTIVE/BENCH の場合)。
    pokemon = _resolve_pokemon(option, state)
    if pokemon is not None:
        return _found(pokemon, RESOLVER_AREA_INDEX)

    # 4. inPlayArea/inPlayIndex で場のポケモンを指す型(ATTACH/EVOLVE。TOOL_CARD/ENERGY_CARD/
    #    ENERGY は「Area of the attached Pokémon」を area に持つため、実際には3で解決される)。
    pokemon = _resolve_in_play_pokemon(option, state)
    if pokemon is not None:
        return _found(pokemon, RESOLVER_IN_PLAY_INDEX)

    # 5. area が HAND/DISCARD/PRIZE/DECK/STADIUM/PLAYER 等、T1がまだトークン化していない
    #    ゾーンを指している(PLAY・DISCARDの一部・CARDの一部等)。T3で解決対象になる。
    #    NUMBER/YES/NO/END はそもそも area を持たないため、ここに自然に含まれる。
    if option.area in _UNRESOLVED_AREAS_T1 or option.inPlayArea in _UNRESOLVED_AREAS_T1:
        return PointerDebugInfo(NO_TARGET, None, None, RESOLVER_NO_TARGET_UNRESOLVED_ZONE)
    return PointerDebugInfo(NO_TARGET, None, None, RESOLVER_NO_TARGET_NO_FIELDS)


def resolve_option_target_index(option: Option, state: State, tokens: BoardTokens) -> int:
    """``option`` が指す盤面トークンのローカルindexを返す。対象が無い/T1未対応なら ``NO_TARGET``。

    Args:
        option: 対象を調べる選択肢。
        state: ``option`` と同じ決定点の ``State``。
        tokens: 同じ ``state`` から ``build_board_tokens()`` で作ったトークン列
            (``serials`` を検索キーに使う)。
    """
    return _resolve_debug(option, state, tokens).target_index


def resolve_option_target_index_debug(option: Option, state: State, tokens: BoardTokens) -> PointerDebugInfo:
    """``resolve_option_target_index`` と同じ解決結果に加え、target_serial/target_card_id/
    resolver種別を返す(pointer監査・検証専用。通常の学習経路では使わない)。"""
    return _resolve_debug(option, state, tokens)


# OptionType全17種の解決方針一覧(コードとゲーム意味の両方から判定した結果)。
# テスト・レビュー用に、resolve_option_target_index の分岐と対応させて明記する。
OPTION_TYPE_POINTER_TABLE: dict[OptionType, str] = {
    OptionType.NUMBER: "NO_TARGET(対象を持たない純粋な個数選択)",
    OptionType.YES: "NO_TARGET(対象を持たない)",
    OptionType.NO: "NO_TARGET(対象を持たない)",
    OptionType.CARD: "area∈{ACTIVE,BENCH}なら解決可能(3)。HAND/DISCARD/PRIZE等はT1でNO_TARGET、T3で対応",
    OptionType.TOOL_CARD: "area=装着先ポケモンのACTIVE/BENCHで解決可能(3)。対象は装着先ポケモン(装着カード自体ではない)",
    OptionType.ENERGY_CARD: "TOOL_CARDと同じ(3)。対象は装着先ポケモン",
    OptionType.ENERGY: "TOOL_CARDと同じ(3)。対象は装着先ポケモン",
    OptionType.PLAY: "NO_TARGET(手札から出すカード自体が対象。手札はT1未トークン化。T3で対応)",
    OptionType.ATTACH: "inPlayArea/inPlayIndexで解決可能(4)。対象は装着される側のポケモン",
    OptionType.EVOLVE: "inPlayArea/inPlayIndexで解決可能(4)。対象は進化前のポケモン",
    OptionType.ABILITY: "area∈{ACTIVE,BENCH}なら解決可能(3)。特性の発動元がDISCARD等の特殊ケースはNO_TARGET",
    OptionType.DISCARD: "area∈{ACTIVE,BENCH}を指す場合のみ解決可能(3)。それ以外は確認不能につきNO_TARGET",
    OptionType.RETREAT: "ルール上常に自分のバトルポケモン(2)。area/indexを持たないが、にげるはバトル場のポケモンしか行えないため暗黙に解決する",
    OptionType.ATTACK: "ルール上常に自分のバトルポケモン(2)。attackIdのみでarea/indexを持たないが、ワザはバトル場のポケモンしか使えないため暗黙に解決する",
    OptionType.END: "NO_TARGET(ターン終了の宣言そのもの)",
    OptionType.SKILL: "option.serialで直接解決(1)。cardId=0の特殊条件宣言はNO_TARGET",
    OptionType.SPECIAL_CONDITION: "暫定的に自分のバトルポケモン(2、要検証)。ATTACK/RETREATほど強いルール上の裏付けをコードで確認できていない",
}
