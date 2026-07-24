"""クラスタ⑤ カード知識アクセス（型定義）／担当B

decks.active（担当Aが埋めるデッキ固有データ）が使う「プロファイル」の型定義。
担当Bの判断ロジックは、カードID/カード名を直接見るのではなく、必ずこれらの型を経由して
「役割」「引き方」を数値・真偽値として受け取る。

この型定義はA/B間の「契約」。フィールドを追加・変更する場合は担当Aと共有すること。

EffectCategory は、グッズ/サポート/どうぐ/スタジアム/特性の効果をひとことで分類するための
共通語彙。バラバラな自由文字列にならないよう、ここに列挙した値の中から選ぶ。
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

EffectCategory = Literal[
    "search",       # デッキ/手札からカードを探す
    "draw",         # ドローによる手札補充
    "heal",         # 回復・ダメカン除去
    "disruption",   # 相手の手札/デッキ/場を妨害する
    "setup",        # 展開・ベンチ強化・エネルギー加速
    "lock",         # 相手の選択肢を制限する（特性ロック、攻撃制限など）
    "other",        # 上記に当てはまらないもの
]


@dataclass
class PokemonProfile:
    """ポケモン1種の役割データ（担当Aが decks/new_deck/pokemon_profiles.py で埋める）。"""

    role: str  # 例: "main_attacker" / "sub_attacker" / "wall" / "support"
    role_score: float  # 主力度（0.0〜1.0 目安）
    bench_value: float  # ベンチに置いておく価値
    combo_with: list[int] = field(default_factory=list)  # 相棒カードID（コンボ関係にあるカード）

    # 特性（Ability）関連。CardData.skills に対応する情報で、Attack とは別物。
    has_ability: bool = False  # 特性を持つか
    ability_category: EffectCategory | None = None  # 特性の効果分類（無ければ None）
    ability_priority: float = 0.0  # 特性を使うべき優先度の目安（0.0なら基本使わない）


@dataclass
class AttackProfile:
    """技1つの追加効果分類データ（担当Aが decks/new_deck/attack_profiles.py で埋める）。"""

    bench_snipe: bool  # ベンチ狙撃効果があるか
    inflicts_special_condition: bool  # 状態異常を付与するか
    draws_cards: bool  # ドロー効果があるか
    disables_next_attack: bool  # 次ターン攻撃不可などのロック効果があるか


@dataclass
class OpponentBenchStatus:
    """相手ベンチ1体ぶんのスナップショット（UsageContext から参照する）。"""

    card_id: int
    hp: int


@dataclass
class UsageContext:
    """グッズ/サポート/スタジアムの使用条件（usage_condition）に渡す、盤面の「今の状態」の
    スナップショット。担当Bが Observation/State から組み立てて渡す
    （ptcg_ai.board_evaluation.usage_context.build_usage_context）。

    担当Aは usage_condition 関数の中で、ここに載っている値だけを見て bool を返す
    （Observation/State を直接扱わない）。載っていない情報が必要になった場合は、
    担当Bにフィールド追加を相談すること（勝手に profile_types.py 以外の経路で
    盤面情報を取得しない）。
    """

    own_hand_ids: list[int] = field(default_factory=list)
    own_active_id: int | None = None
    own_bench_ids: list[int] = field(default_factory=list)
    own_discard_ids: list[int] = field(default_factory=list)
    own_active_energy_count: int = 0
    own_discard_pokemon_count: int = 0
    opponent_active_id: int | None = None
    opponent_active_hp: int | None = None
    opponent_bench: list[OpponentBenchStatus] = field(default_factory=list)
    opponent_active_has_special_energy: bool = False
    stadium_id: int | None = None

    @property
    def own_board_ids(self) -> list[int]:
        """自分の場（バトル場+ベンチ）にいるポケモンの card_id 一覧。"""
        ids = list(self.own_bench_ids)
        if self.own_active_id is not None:
            ids.append(self.own_active_id)
        return ids


# グッズ/サポート/スタジアムを「今使うべきか」判定する条件関数。UsageContext だけを見て bool を返す。
# None なら「常に使ってよい（category/priority だけで判断する）」を意味する。
UsageCondition = Callable[[UsageContext], bool]


@dataclass
class ItemProfile:
    """グッズ1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0  # 同カテゴリ内での使用優先度の目安（tie-break用）
    usage_condition: UsageCondition | None = None  # 「今使うべきか」の判定関数（無ければ常に使用可）


@dataclass
class SupporterProfile:
    """サポート1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0
    usage_condition: UsageCondition | None = None


@dataclass
class ToolProfile:
    """ポケモンのどうぐ1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0


@dataclass
class StadiumProfile:
    """スタジアム1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0
    usage_condition: UsageCondition | None = None


@dataclass
class EnergyProfile:
    """使用する基本/特殊エネルギー1種のデータ。"""

    category: Literal["basic", "special"]


@dataclass
class EnergyCardContext:
    """ENERGY_CARD_PRIORITY_RULES の条件関数に渡す、エネルギー付与1回ぶんの状況。"""

    target_card_id: int  # エネルギーを付ける先のポケモンの card_id
    target_energy_count: int  # 付ける先に現在付いているエネルギー本数


# 条件付きの「どのエネルギーカードを使うか」の条件関数。EnergyCardContext だけを見て bool を返す。
EnergyCardCondition = Callable[[EnergyCardContext], bool]


@dataclass
class EnergyPriorityRule:
    """条件付きの「どのエネルギーカードを使うか」優先順位
    （担当Aが decks/new_deck/deck_plan.py の ENERGY_CARD_PRIORITY_RULES で定義する）。

    condition が True を返す最初のルールの order（card_id を優先度順に並べたもの）を使う。
    """

    condition: EnergyCardCondition
    order: list[int]


@dataclass
class SearchPriorityRule:
    """条件付きの「サーチ/ドローで何を優先して持ってくるか」優先順位
    （担当Aが decks/new_deck/deck_plan.py の SEARCH_PRIORITY_RULES で定義する）。

    condition（UsageContext、盤面のスナップショット）が True を返す最初のルールの
    order（card_id を優先度順に並べたもの）を使う。TO_HAND/LOOK の選択肢のうち、この
    order に載っていないカードは DeckPlan.search_priority（無条件の優先順）にフォールバックする。
    """

    condition: UsageCondition
    order: list[int]


@dataclass
class DeckPlan:
    """デッキ方針データを担当Bの判断ロジックに渡すための正規化された形（クラスタ⑥）。

    担当Aの decks/new_deck/deck_plan.py は「なぜその優先度か」の理由付きメモを持たせた
    独自のデータクラス（AttackerPlan/PriorityEntry など）で書かれることが多く、
    ファイルごとに形が変わり得る。担当Bの判断ロジックが必要とするのは card_id の
    列/集合だけなので、ptcg_ai.shared.profile_registry.get_deck_plan() がこの型に
    変換してから渡す（rule_based/action_selection 配下は decks/ の実際の形を意識しない）。
    """

    main_attacker_ids: list[int] = field(default_factory=list)  # 主力アタッカーのカードID
    sub_attacker_ids: list[int] = field(default_factory=list)  # サブアタッカーのカードID
    opening_priority: list[int] = field(default_factory=list)  # 初手・展開で優先したいカードID順
    evolution_priority: list[int] = field(default_factory=list)  # 進化を優先したいカードID順
    energy_priority: list[int] = field(default_factory=list)  # エネルギーを優先して付けたいカードID順
    search_priority: list[int] = field(default_factory=list)  # サーチで最初に探すべきカードID順
    search_priority_rules: list[SearchPriorityRule] = field(default_factory=list)  # 条件付きサーチ優先順（無条件のsearch_priorityより優先）
    protected_card_ids: set[int] = field(default_factory=set)  # 捨てたくないカードIDの集合
    win_condition_by_prize: dict[int, str] = field(default_factory=dict)  # 残りサイド枚数ごとの勝ち筋メモ

    # エネルギー周回コンボ（例: ACE SPECエネルギーを、山札に戻る特性持ちポケモンに一時的に
    # 付けて再利用する）。該当が無いデッキでは空のままでよい。
    # 受け皿は進化ライン全体（例: ノコッチ／ノココッチ）を集合で持つ。進化前に付けたエネルギーは
    # 進化で引き継がれるため、ラインのどの段階で受け取っても周回コンボは成立する。
    energy_recycle_target_ids: frozenset[int] = field(default_factory=frozenset)  # 周回コンボの受け皿にするポケモンのcard_id集合
    energy_recycle_card_id: int | None = None  # 周回させたいエネルギーカードのcard_id
    energy_recycle_backup_item_id: int | None = None  # 主力への代替エネルギー供給手段（グッズ等）のcard_id
