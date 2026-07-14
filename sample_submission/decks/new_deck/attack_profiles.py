
"""ワザ別プロファイル（⑦、担当A領域）。

キーは card_id ではなく attackId（cg.api.Attack.attackId）。
本デッキの各ポケモンについて `cg.api.all_card_data()` / `all_attack()` を
実行して確認した実際の attackId を使用している。
"""

from __future__ import annotations

from ptcg_ai.shared.profile_types import AttackProfile

PROFILES: dict[int, AttackProfile] = {
    74: AttackProfile(  # ノコッチ「かじる」
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    75: AttackProfile(  # ノコッチ「あなをほる」
        # コインオモテで次の相手の番、ワザ・効果を受けない（自己防御）。
        # 4分類のどれにも該当しないため全てFalseのまま。
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    76: AttackProfile(  # ノココッチ「ランドクラッシュ」
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    183: AttackProfile(  # キチキギスex「クルーエルアロー」
        # 相手ポケモン1匹を任意選択（ベンチも含む）できる単体除去。
        bench_snipe=True,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    477: AttackProfile(  # シェイミ「けとばす」
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    1070: AttackProfile(  # ケーシィ「テレポートアタック」
        # 自分をベンチと入れ替える効果だが、4分類には該当しない。
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    1071: AttackProfile(  # ユンゲラー「ちょうねんりき」
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
    1072: AttackProfile(  # フーディン「ハンドパワー」
        # ダメージ=手札枚数×2。相手のバトルポケモンのみが対象でベンチ狙撃はない。
        bench_snipe=False,
        inflicts_special_condition=False,
        draws_cards=False,
        disables_next_attack=False,
    ),
}

"""クラスタ⑦ カード別プロファイル／担当A

このデッキで使うワザごとの追加効果分類。knowledge.profile_types.AttackProfile の型に沿って
attack_id をキーとした辞書を埋める（ベンチ狙撃、状態異常、ドロー、次ターン攻撃不可など）。
"""

