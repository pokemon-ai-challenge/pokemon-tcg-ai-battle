
"""グッズ別プロファイル（⑦、担当A領域）。

usage_condition は、deck_plan.py の ITEM_USAGE_NOTES に書いていた自由記述の使用条件を
実際に判定できる関数に落としたもの（Issue #3 の対応）。UsageContext（担当Bが盤面から
組み立てる）だけを見て bool を返し、Observation/State には触らない。
"""

from __future__ import annotations

from decks.new_deck.deck_plan import BASIC_PSYCHIC_ENERGY_CARD_ID
from ptcg_ai.shared.profile_types import ItemProfile, UsageContext

_CASEY_ID = 741  # ケーシィ
_KADABRA_ID = 742  # ユンゲラー
_ALAKAZAM_ID = 743  # フーディン
_DUNSPARCE_ID = 65  # ノコッチ


def _rare_candy_condition(ctx: UsageContext) -> bool:
    """ふしぎなアメ: バトル場がケーシィ、かつ手札にユンゲラーが無くフーディンがある場合。"""
    return (
        ctx.own_active_id == _CASEY_ID
        and _KADABRA_ID not in ctx.own_hand_ids
        and _ALAKAZAM_ID in ctx.own_hand_ids
    )


def _enhanced_hammer_condition(ctx: UsageContext) -> bool:
    """改造ハンマー: 相手のバトル場のポケモンに特殊エネルギーが付いている場合。"""
    return ctx.opponent_active_has_special_energy


_KADABRA_LINE_IDS = (_CASEY_ID, _KADABRA_ID, _ALAKAZAM_ID)  # フーディン系列（ケーシィ/ユンゲラー/フーディン）
_DUNSPARCE_LINE_IDS = (_DUNSPARCE_ID, 66)  # ノコッチ/ノココッチ


def _buddy_buddy_poffin_condition(ctx: UsageContext) -> bool:
    """なかよしポフィン: まだ確保できていない主要パーツがある時だけ使う。

    以前は「ケーシィ/キチキギスex/ノコッチのいずれかが場にも手札にも無い場合」に発火していたため、
    既にユンゲラー／フーディンまで進化していて『ケーシィ』が手札・場に無いだけの状況でも発火し、
    勝ち筋に絡まないケーシィをわざわざ持ってきてしまっていた。系列単位で「1体でも確保できているか」を
    見るように変更する:
      ・フーディン系列（ケーシィ/ユンゲラー/フーディン）が1体も場にも手札にも無い
      ・またはノコッチ系列（ノコッチ/ノココッチ）が1体も無い
      ・またはキチキギスexが無い
    のいずれかの時だけ、盤面展開が不足しているとみなして使う。
    """
    def _has_any(ids: tuple[int, ...]) -> bool:
        return any(card_id in ctx.own_board_ids or card_id in ctx.own_hand_ids for card_id in ids)

    missing_fudin_line = not _has_any(_KADABRA_LINE_IDS)
    missing_dunsparce_line = not _has_any(_DUNSPARCE_LINE_IDS)
    missing_fezandipiti = _FEZANDIPITI_EX_ID not in ctx.own_board_ids and _FEZANDIPITI_EX_ID not in ctx.own_hand_ids
    return missing_fudin_line or missing_dunsparce_line or missing_fezandipiti



def _night_stretcher_condition(ctx: UsageContext) -> bool:
    """夜のタンカ: トラッシュに回収できるポケモンが1体以上いる場合。"""
    return ctx.own_discard_pokemon_count >= 1


def _sacred_ash_condition(ctx: UsageContext) -> bool:
    """せいなるはい: トラッシュにポケモンが3体以上ある場合。"""
    return ctx.own_discard_pokemon_count >= 3


def _wondrous_patch_condition(ctx: UsageContext) -> bool:
    """ワンダーパッチ: トラッシュに基本【超】エネルギーがある場合。"""
    return BASIC_PSYCHIC_ENERGY_CARD_ID in ctx.own_discard_ids


PROFILES: dict[int, ItemProfile] = {
    1079: ItemProfile(category="setup", priority=0.9, usage_condition=_rare_candy_condition),  # ふしぎなアメ：進化短縮の要
    1081: ItemProfile(
        category="disruption", priority=0.5, usage_condition=_enhanced_hammer_condition
    ),  # 改造ハンマー：相手の特殊エネルギー破壊
    1086: ItemProfile(
        category="setup", priority=0.8, usage_condition=_buddy_buddy_poffin_condition
    ),  # なかよしポフィン：たねポケモン展開
    1097: ItemProfile(
        category="search", priority=0.5, usage_condition=_night_stretcher_condition
    ),  # 夜のタンカ：トラッシュからポケモン/基本エネ回収
    1129: ItemProfile(
        category="other", priority=0.3, usage_condition=_sacred_ash_condition
    ),  # せいなるはい：トラッシュ整理・山札に戻す
    1146: ItemProfile(
        category="setup", priority=0.5, usage_condition=_wondrous_patch_condition
    ),  # ワンダーパッチ：トラッシュの基本超エネを再利用
    1152: ItemProfile(category="search", priority=0.7),  # ポケパッド：手札にあれば基本的に使用する（条件なし）
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うグッズごとの効果分類。knowledge.profile_types.ItemProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙
（search/draw/heal/disruption/setup/lock/other）から選ぶ。priority は同カテゴリ内の
tie-break用（例: どのサーチカードを優先するか）。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> ItemProfile を埋める。
#PROFILES: dict[int, ItemProfile] = {}

