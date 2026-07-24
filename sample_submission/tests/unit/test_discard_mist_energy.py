"""Issue 2 回帰テスト: 改造ハンマー等で相手の特殊エネを壊すとき、ワザの効果を無効化する
ミストエネルギー（card_id 11）を最優先で壊すこと。

discard.choose は option.cardId をそのまま card_id として解決するため（common.resolve_card_id は
cardId があれば state を見ない）、最小限の Option だけで検証できる。
"""

from cg.api import Option, OptionType, SelectContext, SelectData, SelectType

from ptcg_ai.rule_based.card_move import discard

_MIST_ENERGY = 11
_RICH_ENERGY = 13
_TELEPATH_ENERGY = 19


def _energy_option(card_id: int) -> Option:
    return Option(type=OptionType.DISCARD, cardId=card_id)


def _make_select(options: list[Option]) -> SelectData:
    return SelectData(
        type=SelectType.MAIN,
        context=SelectContext.DISCARD_ENERGY,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )


def test_discard_prefers_mist_energy():
    # 相手のポケモンにミスト(11)/リッチ(13)/テレパス(19)が付いている状況を模す。
    options = [_energy_option(_RICH_ENERGY), _energy_option(_MIST_ENERGY), _energy_option(_TELEPATH_ENERGY)]
    result = discard.choose(_make_select(options), state=None)
    assert result == [1]  # ミストエネ(11)のインデックス


def test_discard_without_mist_still_returns_one():
    # ミストエネが無ければ通常どおり1枚選ぶ（クラッシュしない・合法手を返す）。
    options = [_energy_option(_RICH_ENERGY), _energy_option(_TELEPATH_ENERGY)]
    result = discard.choose(_make_select(options), state=None)
    assert len(result) == 1
    assert result[0] in (0, 1)
