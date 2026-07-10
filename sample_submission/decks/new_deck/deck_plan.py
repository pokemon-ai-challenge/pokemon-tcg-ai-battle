"""クラスタ⑥ デッキ方針／担当A

このデッキ固有の方針データ。担当Bの判断ロジック（main_turn_parts/priorities, card_move など）
は、この PLAN を直接 import せず、必ず knowledge.profile_registry.get_deck_plan() 経由で参照する
（境界ルール：担当Bのクラスタは decks/ を直接参照しない）。
"""

from dataclasses import dataclass, field


@dataclass
class DeckPlan:
    """このデッキの方針データ。値は新デッキの60枚が確定し次第、担当Aが埋める。"""

    main_attacker_ids: list[int] = field(default_factory=list)  # 主力アタッカーのカードID
    sub_attacker_ids: list[int] = field(default_factory=list)  # サブアタッカーのカードID
    opening_priority: list[int] = field(default_factory=list)  # 初手・展開で優先したいカードID順
    evolution_priority: list[int] = field(default_factory=list)  # 進化を優先したいカードID順
    energy_priority: list[int] = field(default_factory=list)  # エネルギーを優先して付けたいカードID順
    search_priority: list[int] = field(default_factory=list)  # サーチで最初に探すべきカードID順
    protected_card_ids: set[int] = field(default_factory=set)  # 捨てたくないカードIDの集合
    win_condition_by_prize: dict[int, str] = field(default_factory=dict)  # 残りサイド枚数ごとの勝ち筋メモ


# TODO(担当A): 新デッキの60枚確定後に値を埋める。
PLAN = DeckPlan()
