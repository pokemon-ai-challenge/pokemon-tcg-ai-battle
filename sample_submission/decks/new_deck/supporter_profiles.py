
"""サポート別プロファイル（⑦、担当A領域）。

usage_condition は、deck_plan.py の SUPPORTER_USAGE_NOTES に書いていた自由記述の使用条件を
実際に判定できる関数に落としたもの（Issue #3 の対応）。UsageContext（担当Bが盤面から
組み立てる）だけを見て bool を返し、Observation/State には触らない。
"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import SupporterProfile, UsageContext

_CASEY_ID = 741  # ケーシィ
_KADABRA_ID = 742  # ユンゲラー
_ALAKAZAM_ID = 743  # フーディン
# エネルギーカードのcard_id一覧（基本超/テレパス超/リッチ）。deck_plan.pyのENERGY_CARD_PRIORITY_RULES参照。
_ENERGY_CARD_IDS = (5, 13, 19)


def _boss_orders_condition(ctx: UsageContext) -> bool:
    """ボスの指令: 相手のベンチポケモンのHPが、相手のバトルポケモンのHPより低い場合。"""
    if ctx.opponent_active_hp is None:
        return False
    return any(bench.hp < ctx.opponent_active_hp for bench in ctx.opponent_bench)


def _lanas_aid_condition(ctx: UsageContext) -> bool:
    """スイレンのお世話:
    「トラッシュにエネルギーがあり、バトル場のエネルギーが足りず、手札にもエネルギーが無い」
    または「トラッシュにフーディン・ユンゲラー・ケーシィの3体がそろっている」場合。
    """
    discard_has_energy = any(card_id in ctx.own_discard_ids for card_id in _ENERGY_CARD_IDS)
    hand_has_energy = any(card_id in ctx.own_hand_ids for card_id in _ENERGY_CARD_IDS)
    energy_shortage = discard_has_energy and ctx.own_active_energy_count == 0 and not hand_has_energy

    fudin_line_ids = (_CASEY_ID, _KADABRA_ID, _ALAKAZAM_ID)
    fudin_line_in_discard = all(card_id in ctx.own_discard_ids for card_id in fudin_line_ids)

    return energy_shortage or fudin_line_in_discard


PROFILES: dict[int, SupporterProfile] = {
    1182: SupporterProfile(
        category="disruption", priority=0.9, usage_condition=_boss_orders_condition
    ),  # ボスの指令：ベンチ狙撃・詰め
    1184: SupporterProfile(
        category="search", priority=0.4, usage_condition=_lanas_aid_condition
    ),  # スイレンのお世話：トラッシュから回収
    1225: SupporterProfile(category="search", priority=0.8),  # トウコ：手札にあれば基本的に使用する（条件なし）
    1231: SupporterProfile(category="search", priority=0.9),  # ヒカリ：手札にあれば基本的に使用する（条件なし）
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うサポートごとの効果分類。knowledge.profile_types.SupporterProfile の型に沿って
card_id をキーとした辞書を埋める。category は profile_types.EffectCategory の語彙から選ぶ。
1ターン1枚制限の判断は handlers 側で State.supporterPlayed を見て行うため、ここでは
効果分類と priority（tie-break用）のみを持つ。
"""


# TODO(担当A): 新デッキの60枚確定後に card_id -> SupporterProfile を埋める。
#PROFILES: dict[int, SupporterProfile] = {}

