"""クラスタ⑤ カード知識アクセス（型定義）／担当B

decks.active（担当Aが埋めるデッキ固有データ）が使う「プロファイル」の型定義。
担当Bの判断ロジックは、カードID/カード名を直接見るのではなく、必ずこれらの型を経由して
「役割」「引き方」を数値・真偽値として受け取る。

この型定義はA/B間の「契約」。フィールドを追加・変更する場合は担当Aと共有すること。

EffectCategory は、グッズ/サポート/どうぐ/スタジアム/特性の効果をひとことで分類するための
共通語彙。バラバラな自由文字列にならないよう、ここに列挙した値の中から選ぶ。
"""

from dataclasses import dataclass, field
from typing import Literal

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
class ItemProfile:
    """グッズ1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0  # 同カテゴリ内での使用優先度の目安（tie-break用）


@dataclass
class SupporterProfile:
    """サポート1枚の効果分類データ。"""

    category: EffectCategory
    priority: float = 0.0


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


@dataclass
class EnergyProfile:
    """使用する基本/特殊エネルギー1種のデータ。"""

    category: Literal["basic", "special"]


@dataclass
class MatchupPlan:
    """相手デッキのアーキタイプ（opponent_modeling.rough_predictor の deck_type）ごとの対策データ。

    キーは DeckPlan.matchup_plans の dict キー（deck_type 文字列）側で持つため、
    ここには「そのアーキタイプに対してどう加点するか」だけを持たせる。
    """

    attack_priority_boost: dict[int, float] = field(default_factory=dict)  # attack_id -> 加点
    card_priority_boost: dict[int, float] = field(default_factory=dict)  # card_id -> 加点（board系）
    note: str = ""  # なぜその加点かの理由メモ


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
    protected_card_ids: set[int] = field(default_factory=set)  # 捨てたくないカードIDの集合
    win_condition_by_prize: dict[int, str] = field(default_factory=dict)  # 残りサイド枚数ごとの勝ち筋メモ
    # 相手デッキのアーキタイプ（opponent_modeling.rough_predictor の deck_type）ごとの対策。
    # キーは rough_predictor.json の archetypes キーと一致させること。
    matchup_plans: dict[str, MatchupPlan] = field(default_factory=dict)
