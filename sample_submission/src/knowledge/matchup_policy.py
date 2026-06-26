"""Phase 5 Task 5-3: アーキタイプ別戦略オーバーライド。

相手デッキの推定アーキタイプに応じて行動戦略を変化させる MatchupPolicy を定義する。
呼び出し元は get_opponent_model().estimate_arch() の結果を渡す。

戦略根拠は docs/axis_card_analysis.md「アーキタイプ別の戦略的ターゲット」参照。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchupOverride:
    # ベンチ展開を抑える戦略的上限 (ゲーム上限 5 とは別の戦略値)
    max_bench_size: int = 5
    # ボスの指令で積極的にキーポケモンを引き出すか
    boss_rush: bool = False
    # このオーバーライドを適用するデッキ名セット（空 = 全デッキに適用）
    only_for_decks: frozenset[str] = frozenset()


# アーキタイプキー → オーバーライド設定
# デフォルト (未知アーキタイプ) は _DEFAULT を返す
MATCHUP_OVERRIDES: dict[str, MatchupOverride] = {
    # ファントムダイブ (全体 10×6 散布) 対策: ベンチを最大 2 体に絞る
    # マリィのみ: Lucario は Lunatone+Solrock+アタッカー で 3 スロット必要なため bench=2 が逆効果
    "dragapult_ex": MatchupOverride(
        max_bench_size=2,
        only_for_decks=frozenset({"maries_obstagoon_ex"}),
    ),

    # メガニウム (710) のエネ加速を止めることが最優先
    # ボスラッシュ戦略は opponent_model.py の ARCH_BOSS_TARGETS で管理
    "hydrapple_ex": MatchupOverride(boss_rush=True),
    "olivia_ex":    MatchupOverride(boss_rush=True),

    # 格闘攻撃が悪タイプに弱点ダメージ: ベンチを 2 体に絞り被弾面積を減らす
    # マリィのみ: Lucario はエネサイクルに 3 スロット必要
    "mega_lucario_ex": MatchupOverride(
        max_bench_size=2,
        only_for_decks=frozenset({"maries_obstagoon_ex"}),
    ),

    # Crustle (345) は EX 攻撃を無効化するが Dwebble (344, 70HP) は EX で KO 可能
    # ボスの指令でベンチの Dwebble(344) を引き出してEXアタッカーで1撃KO
    # ARCH_BOSS_TARGETS["crustle"] = [344, 756] で既定義済み
    "crustle": MatchupOverride(boss_rush=True),

    # Alakazam (743): Powerful Hand = 我々の手札枚数×20 ダメージ
    # ベンチの Abra(741, 50HP) をボスで引き出して1撃KO → Alakazam 進化ライン破壊
    # ARCH_BOSS_TARGETS["alakazam"] = [741, 742, 65, 66] で既定義済み
    "alakazam": MatchupOverride(boss_rush=True),

    # Ogerpon bullet: マシマシラ(112) がエネルギー転送でアタッカーを次々入れ替える
    # ボスでマシマシラ(112) を引き出してKO → エネ転送エンジン破壊
    # ARCH_BOSS_TARGETS["ogerpon_bullet"] = [112, 184, 108, 96] で既定義済み
    "ogerpon_bullet": MatchupOverride(boss_rush=True),
}

_DEFAULT = MatchupOverride()


def get_matchup_override(arch: str | None) -> MatchupOverride:
    """アーキタイプ名から戦略オーバーライドを返す。不明・未登録は DEFAULT を返す。"""
    if not arch or arch not in MATCHUP_OVERRIDES:
        return _DEFAULT
    override = MATCHUP_OVERRIDES[arch]
    if override.only_for_decks:
        from src.knowledge.deck_plan import get_deck_plan
        if get_deck_plan().name not in override.only_for_decks:
            return _DEFAULT
    return override
