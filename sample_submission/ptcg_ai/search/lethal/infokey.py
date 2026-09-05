"""決定ノードの情報集合キー(設計 §2.2 / 規則 R1・R2)。

**このモジュールがキーを作る唯一の場所**である。探索・transposition・
Goal 分析・診断は必ずここを経由し、山札の順列(``Scenario``)を直接見ない。

キーに入れるもの:

- 公開情報(両者の場・HP・付与物・トラッシュ・サイド枚数・スタジアム・各種フラグ)
- 自分の手札の **multiset**(順序・serial は入れない)
- 自分の山札の **multiset と枚数**(``deck_multiset``。順序は入れない)
- 今の選択(``SelectData``)の種類・個数制約・選択肢の**意味**

キーに入れないもの:

- 山札の順序(``Scenario`` / ``your_deck`` の並び)
- カードの serial(物理個体の同一性。合流を妨げるだけで意思決定には不要)
- 相手の非公開手札(そもそも Observation に無い)
- デッキ公開時(``select.deck``)の**並び順**。中身の multiset だけを使う
  (並びは実際の山札順を反映しうるため、条件付けに使わない)

選択肢の扱い(健全性の要):
``Option`` の多くは「手札の index」のような**位置**しか持たない。位置をそのまま
キーに入れると、手札の並びが違うだけの同一局面が別ノードになり、逆に位置を捨てると
「index 3 のカード」が別物である局面同士が衝突する。そこで index は可能な限り
**カードIDへ解決**し、全て解決できた場合のみ選択肢集合をソートして正規化する。
1つでも解決できない選択肢があれば、位置情報を保ったまま(順序も保持)キーに入れる
= 保守的に「合流させない」側へ倒す。

呼び出し側の責務: 解決済みキーは「意味としての行動」で同一視するので、探索は
選んだ意味を**その場のノードで index へ再解決**しなければならない。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from cg.api import AreaType, Observation, SelectData, State

_ME = "me"
_OPP = "opp"


@dataclass(frozen=True)
class InfoKey:
    """情報集合キー。``parts`` は完全な内容、``digest`` は短縮表現。"""

    parts: tuple

    def digest(self) -> str:
        return hashlib.sha1(repr(self.parts).encode("utf-8")).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"InfoKey({self.digest()})"


def info_key(
    obs: Observation,
    me: int,
    *,
    deck_multiset: Mapping[int, int] | None = None,
) -> InfoKey:
    """``obs`` の情報集合キーを作る。

    Args:
        obs: 観測(実戦のものでも探索中のものでもよい)。
        me: 自分のプレイヤーインデックス。
        deck_multiset: 自分の山札の multiset(card id -> 枚数)。
            ``Scenario`` を渡してはいけない。``None`` の場合は ``deckCount``
            だけがキーに入る(合流は安全だが粗くなる)。

    Returns:
        InfoKey: 同じキーなら同じ意思決定をしてよい状態。
    """
    state = obs.current
    if state is None:
        return InfoKey(("no_state",))

    parts: list = [
        ("turn", state.turn),
        ("acting", _rel(state.yourIndex, me)),
        ("first", _rel(state.firstPlayer, me) if state.firstPlayer >= 0 else "undecided"),
        ("flags", state.supporterPlayed, state.stadiumPlayed,
         state.energyAttached, state.retreated),
        ("result", state.result),
        ("stadium", _sorted_ids(card.id for card in state.stadium)),
        ("looking", _looking(state)),
        ("me", _player_parts(state, me)),
        ("opp", _player_parts(state, 1 - me)),
        ("deck_multiset", _multiset_parts(deck_multiset)),
        ("select", _select_parts(obs.select, state, me)),
    ]
    return InfoKey(tuple(parts))


# ---------------------------------------------------------------- helpers


def _rel(player_index: int | None, me: int) -> str:
    if player_index is None:
        return "none"
    return _ME if player_index == me else _OPP


def _sorted_ids(ids) -> tuple:
    return tuple(sorted(int(i) for i in ids))


def _multiset_parts(multiset: Mapping[int, int] | None) -> tuple:
    if multiset is None:
        return ("unknown",)
    return tuple(sorted((int(k), int(v)) for k, v in multiset.items() if v))


def _looking(state: State) -> tuple:
    if state.looking is None:
        return ("none",)
    visible = _sorted_ids(card.id for card in state.looking if card is not None)
    facedown = sum(1 for card in state.looking if card is None)
    return (visible, facedown)


def _pokemon_parts(pokemon) -> tuple:
    if pokemon is None:
        return ("facedown",)
    return (
        pokemon.id,
        pokemon.hp,
        pokemon.maxHp,
        bool(pokemon.appearThisTurn),
        tuple(sorted(int(e) for e in pokemon.energies)),
        _sorted_ids(card.id for card in pokemon.energyCards),
        _sorted_ids(card.id for card in pokemon.tools),
        _sorted_ids(card.id for card in pokemon.preEvolution),
    )


def _player_parts(state: State, player_index: int) -> tuple:
    player = state.players[player_index]
    # active / bench は「順序」がゲーム上の位置そのもの(option の index が指す)なので
    # 並べ替えない。手札・トラッシュは順序に意味が無いので multiset 化する。
    hand = ("hidden",) if player.hand is None else _sorted_ids(card.id for card in player.hand)
    prize_visible = _sorted_ids(card.id for card in player.prize if card is not None)
    prize_facedown = sum(1 for card in player.prize if card is None)
    return (
        tuple(_pokemon_parts(p) for p in player.active),
        tuple(_pokemon_parts(p) for p in player.bench),
        player.benchMax,
        player.deckCount,
        _sorted_ids(card.id for card in player.discard),
        (len(player.prize), prize_visible, prize_facedown),
        player.handCount,
        hand,
        (player.poisoned, player.burned, player.asleep,
         player.paralyzed, player.confused),
    )


def _card_id_at(
    state: State, select: SelectData | None, player_index: int,
    area, index: int | None,
) -> int | None:
    """(area, index) が指すカードの ID。解決できなければ None。"""
    if area is None or index is None or not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    if area == AreaType.ACTIVE or area == AreaType.BENCH:
        pokemon = _pokemon_at(state, player_index, area, index)
        return None if pokemon is None else pokemon.id
    if area == AreaType.HAND:
        sequence = player.hand
    elif area == AreaType.DISCARD:
        sequence = player.discard
    elif area == AreaType.PRIZE:
        sequence = player.prize
    elif area == AreaType.STADIUM:
        sequence = state.stadium
    elif area == AreaType.LOOKING:
        sequence = state.looking
    elif area == AreaType.DECK:
        # デッキが公開されている選択(サーチ)。中身は正当に見えている情報。
        sequence = None if select is None else select.deck
    else:
        return None
    if sequence is None or not (0 <= index < len(sequence)):
        return None
    card = sequence[index]
    return None if card is None else card.id


def _pokemon_at(state: State, player_index: int, area, index: int | None):
    player = state.players[player_index]
    sequence = player.active if area == AreaType.ACTIVE else player.bench
    if index is None or not (0 <= index < len(sequence)):
        return None
    return sequence[index]


def _attached_card_id(
    state: State, player_index: int, area, index: int | None,
    energy_index: int | None, tool_index: int | None,
) -> int | None:
    pokemon = _pokemon_at(state, player_index, area, index)
    if pokemon is None:
        return None
    if energy_index is not None:
        if 0 <= energy_index < len(pokemon.energyCards):
            return pokemon.energyCards[energy_index].id
        return None
    if tool_index is not None:
        if 0 <= tool_index < len(pokemon.tools):
            return pokemon.tools[tool_index].id
        return None
    return None


def _option_parts(option, state: State, select: SelectData, me: int) -> tuple[tuple, bool]:
    """1つの選択肢の意味表現と、位置を完全に解決できたかどうか。"""
    owner = me if option.playerIndex is None else option.playerIndex
    resolved = True
    source: object = None
    if option.area is not None and option.index is not None:
        if option.energyIndex is not None or option.toolIndex is not None:
            source = _attached_card_id(
                state, owner, option.area, option.index,
                option.energyIndex, option.toolIndex,
            )
        else:
            source = _card_id_at(state, select, owner, option.area, option.index)
        if source is None:
            resolved = False

    target: object = None
    if option.inPlayArea is not None and option.inPlayIndex is not None:
        pokemon = _pokemon_at(state, owner, option.inPlayArea, option.inPlayIndex)
        target = None if pokemon is None else pokemon.id
        if target is None:
            resolved = False

    parts = (
        int(option.type),
        _rel(option.playerIndex, me),
        None if option.area is None else int(option.area),
        source,
        None if option.inPlayArea is None else int(option.inPlayArea),
        target,
        option.attackId,
        option.cardId,
        option.number,
        option.count,
        None if option.specialConditionType is None else int(option.specialConditionType),
    )
    if not resolved:
        # 解決できなかったので位置情報を保持する(合流させない側へ倒す)。
        parts = parts + ("positional", option.index, option.inPlayIndex,
                         option.energyIndex, option.toolIndex)
    return parts, resolved


def option_descriptor(option, state: State, select: SelectData, me: int) -> tuple[tuple, bool]:
    """選択肢の**意味表現**と、位置を完全に解決できたかどうかを返す。

    ``action.py`` が「探索が選んだ手」と「実行時の選択肢」を突き合わせるために使う。
    解決できなかった場合(第2要素 False)は位置情報を含む表現になるので、
    別の観測へそのまま適用してはいけない。
    """
    return _option_parts(option, state, select, me)


def _select_parts(select: SelectData | None, state: State, me: int) -> tuple:
    if select is None:
        return ("none",)
    options = [_option_parts(option, state, select, me) for option in select.option]
    all_resolved = all(resolved for _, resolved in options)
    fingerprints = tuple(parts for parts, _ in options)
    if all_resolved:
        # 全て意味へ解決できたので、並びに依存しないよう正規化する。
        fingerprints = tuple(sorted(fingerprints, key=repr))
    deck_listing = ("none",)
    if select.deck is not None:
        # 並び順は使わない(実際の山札順を反映しうる)。multiset だけ。
        deck_listing = _sorted_ids(card.id for card in select.deck if card is not None)
    return (
        int(select.type),
        int(select.context),
        select.minCount,
        select.maxCount,
        select.remainDamageCounter,
        select.remainEnergyCost,
        "resolved" if all_resolved else "positional",
        fingerprints,
        deck_listing,
        None if select.contextCard is None else select.contextCard.id,
        None if select.effect is None else select.effect.id,
    )
