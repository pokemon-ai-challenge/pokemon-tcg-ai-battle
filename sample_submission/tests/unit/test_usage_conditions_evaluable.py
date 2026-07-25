"""回帰テスト: 全プロファイルの usage_condition が例外を出さずに評価できること。

なかよしポフィン(1086)の usage_condition が、デッキから抜いたキチキギスex の定数
``_FEZANDIPITI_EX_ID`` を参照したまま残っており NameError を出していた。この例外は
``usage_gate.is_usable`` → ``priorities/board.py`` の経路を素通りして
``router.route()`` の最終防衛ラインまで飛ぶため、**ポフィンの使用可否だけでなく、その
MAIN判断まるごとが ``fallback.safe_choice()`` に置き換わっていた**（自己対戦20戦で
handler呼び出し1239回中145回＝11.7%）。

条件関数は UsageContext だけを見る契約（profile_types.UsageContext のdocstring参照）なので、
盤面を用意せずに直接呼べる。ここでは「未定義名・属性ミス等で評価自体が失敗しないこと」を
全条件関数について担保する（個々の判定内容の正しさは各カードのテストの担当）。
"""

from decks.new_deck import item_profiles
from ptcg_ai.shared.profile_types import UsageContext

_CASEY_ID = 741  # ケーシィ
_DUNSPARCE_ID = 65  # ノコッチ
_BUDDY_BUDDY_POFFIN_ID = 1086  # なかよしポフィン


def _contexts() -> list[UsageContext]:
    """条件関数の分岐を一通り踏ませる代表的な盤面スナップショット。"""
    return [
        UsageContext(),  # 何も無い（空の場・空の手札）
        UsageContext(own_active_id=_CASEY_ID, own_hand_ids=[_DUNSPARCE_ID]),
        UsageContext(
            own_active_id=_DUNSPARCE_ID,
            own_bench_ids=[_CASEY_ID],
            own_hand_ids=[_BUDDY_BUDDY_POFFIN_ID],
            own_discard_ids=[_CASEY_ID],
            own_discard_pokemon_count=3,
            opponent_active_has_special_energy=True,
        ),
    ]


def test_all_item_usage_conditions_are_evaluable():
    """PROFILES の usage_condition が全て例外なく bool を返す。"""
    for card_id, profile in item_profiles.PROFILES.items():
        if profile.usage_condition is None:
            continue
        for context in _contexts():
            result = profile.usage_condition(context)
            assert isinstance(result, bool), f"card_id={card_id} が bool を返さない: {result!r}"


def test_buddy_buddy_poffin_condition_depends_on_the_two_lines():
    """なかよしポフィン: フーディン系列とノコッチ系列の確保状況だけで決まる。"""
    condition = item_profiles.PROFILES[_BUDDY_BUDDY_POFFIN_ID].usage_condition

    # どちらの系列も確保できていない → 展開不足なので使う。
    assert condition(UsageContext()) is True

    # フーディン系列だけ確保、ノコッチ系列が無い → まだ使う。
    assert condition(UsageContext(own_active_id=_CASEY_ID)) is True

    # 両系列とも1体ずつ確保できている → 使わない（抜けたキチキギスexは条件に含めない）。
    assert condition(UsageContext(own_active_id=_CASEY_ID, own_bench_ids=[_DUNSPARCE_ID])) is False
