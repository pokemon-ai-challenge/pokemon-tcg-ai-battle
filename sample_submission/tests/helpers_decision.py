"""クラスタ⑨ 検証（共通基盤）／担当B

テスト用の Option/Pokemon/PlayerState を組み立てるビルダー関数群。
generic/ と decks/new_deck/ のテスト双方から使う。
"""

from cg.api import Option, OptionType, Pokemon, PlayerState


def make_option(option_type: OptionType, **kwargs) -> Option:
    """任意のフィールドを指定して Option を組み立てる。"""
    raise NotImplementedError


def make_pokemon(card_id: int, hp: int, max_hp: int, **kwargs) -> Pokemon:
    """テスト用の Pokemon インスタンスを組み立てる。"""
    raise NotImplementedError


def make_player_state(**kwargs) -> PlayerState:
    """テスト用の PlayerState インスタンスを組み立てる。"""
    raise NotImplementedError
