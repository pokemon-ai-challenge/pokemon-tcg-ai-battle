"""
Tier上位デッキのメタデッキDBを生成するビルドスクリプト。
出力: sample_submission/src/knowledge/meta_decks.py

実行方法:
    python cardlist_referenced/build_meta_decks.py

依存: pandas, データはローカルキャッシュ(pdf_card_editor/.cache/tier_ranking/)から読む。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cardlist_referenced" / "pdf_card_editor" / ".cache" / "tier_ranking"
JP_CSV = REPO_ROOT / "data" / "JP_Card_Data.csv"
OUTPUT = REPO_ROOT / "sample_submission" / "src" / "knowledge" / "meta_decks.py"

# ---------------------------------------------------------------------------
# app.py と同じ正規化関数（照合精度を合わせる）
# ---------------------------------------------------------------------------

def _normalize_card_name(name: str) -> str:
    """カード名照合用の正規化。
    app.py の _normalize_card_name に加え、+ も除去する。
    スクレイプ名が「オーガポン+みどりのめんex」でCSVが「オーガポン みどりのめんex」の
    ように区切り文字が異なるケースに対応。
    """
    text = re.sub(r"[（(][^（）()]*[)）]", "", str(name))
    text = re.sub(r"[【】「」『』\[\]（）()・,，、。.+\s]", "", text)
    return text.casefold()


# ---------------------------------------------------------------------------
# 同名別IDのカードに対する明示的な推奨ID（競技文脈ベース）
# ---------------------------------------------------------------------------

CANONICAL_CARD_ID: dict[str, int] = {
    # 正規化後の名前 → 使用するCard ID
    "ケーシィ":       109,   # Ability: Teleporter → フーディン系
    "ユンゲラー":     742,   # Kadabra（フーディンex系の中間進化）
    "フーディン":     743,   # Ability: Psychic Draw → フーディンex本体
    "ノコッチ":       305,   # Trading Places → ドロー要員
    "リオル":         677,   # 80HP Accelerating Stab → メガルカリオex
    "イシズマイ":     344,   # Grass type → イワパレス（草）デッキ
    "イワパレス":     345,   # Grass type, Mysterious Rock Inn → イワパレス本体
    "カジッチュ":     346,   # Grass type, Mini Drain（最新草版）→ カミツオロチex系
    "カミッチュ":      93,   # Ability: Festival Lead → カミツオロチex系
    "チコリータ":     917,   # 新版（Growl）→ メガニウムライン
    "ベイリーフ":     918,   # 新版（Leaf Step）→ メガニウムライン
    "コライドン":     226,   # Fighting type → 競技主流版
    "シェイミ":       343,   # Ability: Flower Curtain → 汎用セットアップ
}

# アーキタイプIDマッピング
ARCH_SLUG: dict[str, str] = {
    "ドラパルトex":      "dragapult_ex",
    "フーディン":         "hatterene",
    "メガルカリオex":     "megarucario_ex",
    "イワパレス":         "crustle",
    "カミツオロチex":     "hydrapple_ex",
    "オリーヴァex":       "olivine_ex",
    "タケルライコex":     "takelraiko_ex",
    "オーガポンバレット":  "orgepon_bullet",
}

# アーキタイプのシグネチャカード（識別用）
ARCHETYPE_KEY_CARDS: dict[str, list[int]] = {
    "dragapult_ex":   [121, 120, 119],
    "hatterene":      [743, 742, 109],
    "megarucario_ex": [678, 677, 674],
    "crustle":        [345, 344, 756],
    "hydrapple_ex":   [150,  93, 346],
    "olivine_ex":     [404, 403],
    "takelraiko_ex":  [ 63,  62],
    "orgepon_bullet": [],
}


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def load_latest_cache() -> dict:
    files = sorted(CACHE_DIR.glob("*.json"), reverse=True)
    for f in files:
        if "v4" in f.name:
            with f.open(encoding="utf-8") as fp:
                return json.load(fp)
    with files[0].open(encoding="utf-8") as fp:
        return json.load(fp)


def build_normalized_catalog(jp_df: pd.DataFrame) -> dict[str, list[int]]:
    """正規化カード名 → ユニークなCard IDリスト（多技で重複行あるので drop_duplicates）"""
    catalog: dict[str, list[int]] = {}
    for _, row in jp_df.drop_duplicates("カード ID").iterrows():
        norm = _normalize_card_name(str(row["カード名"]))
        catalog.setdefault(norm, []).append(int(row["カード ID"]))
    return catalog


def resolve_name_to_id(
    raw_name: str,
    catalog: dict[str, list[int]],
) -> int | None:
    """カード名 → Card ID。複数候補はCANONICAL_CARD_IDで解決、なければ最大ID（最新版）。"""
    norm = _normalize_card_name(raw_name)
    ids = catalog.get(norm)
    if not ids:
        return None  # プール外

    # 正規化前の名前でも、正規化後の名前でも検索
    if raw_name in CANONICAL_CARD_ID:
        return CANONICAL_CARD_ID[raw_name]
    if norm in CANONICAL_CARD_ID:
        return CANONICAL_CARD_ID[norm]

    if len(ids) == 1:
        return ids[0]

    return max(ids)  # フォールバック: 最新版


def build() -> None:
    jp_df = pd.read_csv(JP_CSV, encoding="utf-8-sig")
    catalog = build_normalized_catalog(jp_df)

    cache = load_latest_cache()
    tiers = cache.get("deck_tier", {})
    recipes = cache.get("deck_recipes", {})
    arch_map = cache.get("deck_archetype", {})

    # Tier1/2 のデッキをアーキタイプ別にグループ化（全デッキ）
    arch_decks: dict[str, list[tuple[str, int]]] = defaultdict(list)  # arch → [(deck_id, tier)]
    for deck_id, arch in arch_map.items():
        tier = tiers.get(deck_id)
        if tier in (1, 2):
            arch_decks[arch].append((deck_id, tier))

    entries: list[dict] = []

    for arch_name, deck_list in sorted(arch_decks.items(), key=lambda x: min(t for _, t in x[1])):
        slug = ARCH_SLUG.get(arch_name)
        if slug is None:
            continue

        best_tier = min(t for _, t in deck_list)
        total_decks = len(deck_list)

        # 全デッキにわたってカード出現回数を集計
        # card_deck_count[card_id] = そのカードを採用しているデッキ数
        # card_total_copies[card_id] = 全デッキ合算の総枚数
        card_deck_count: Counter[int] = Counter()
        card_total_copies: Counter[int] = Counter()
        missing_names: set[str] = set()

        for deck_id, _ in deck_list:
            recipe = recipes.get(deck_id, {}).get("cards", {})
            if not recipe:
                continue
            for raw_name, info in recipe.items():
                qty = int(info.get("quantity", 0))
                if qty <= 0:
                    continue
                card_id = resolve_name_to_id(raw_name, catalog)
                if card_id is None:
                    missing_names.add(raw_name)
                else:
                    card_deck_count[card_id] += 1
                    card_total_copies[card_id] += qty

        # 代表デッキ（Tier最上位の最初の1件）のレシピも取得
        rep_deck_id = sorted(deck_list, key=lambda x: x[1])[0][0]
        rep_recipe = recipes.get(rep_deck_id, {}).get("cards", {})
        rep_card_ids: list[int] = []
        for raw_name, info in rep_recipe.items():
            qty = int(info.get("quantity", 0))
            card_id = resolve_name_to_id(raw_name, catalog)
            if card_id is not None:
                rep_card_ids.extend([card_id] * qty)

        decks_with_recipe = sum(
            1 for deck_id, _ in deck_list
            if recipes.get(deck_id, {}).get("cards")
        )
        pool_coverage = len(rep_card_ids) / 60.0

        # 採用率（deck_inclusion_rate）: そのカードを使っているデッキ / 全デッキ数
        inclusion_rate = {
            card_id: round(count / decks_with_recipe, 3)
            for card_id, count in card_deck_count.items()
        }
        # 平均枚数（avg_copies）: 全デッキ合算枚数 / 全デッキ数
        avg_copies = {
            card_id: round(total / decks_with_recipe, 2)
            for card_id, total in card_total_copies.items()
        }

        key_cards = ARCHETYPE_KEY_CARDS.get(slug, [])

        entries.append(dict(
            archetype_id=slug,
            archetype_name_jp=arch_name,
            tier=best_tier,
            source_type="realworld_meta",
            total_decks=total_decks,
            decks_with_recipe=decks_with_recipe,
            rep_card_ids=sorted(rep_card_ids),
            rep_card_counts=dict(Counter(rep_card_ids)),
            missing_card_names=sorted(missing_names),
            pool_coverage=round(pool_coverage, 3),
            key_cards=key_cards,
            card_inclusion_rate=dict(sorted(inclusion_rate.items(), key=lambda x: -x[1])),
            card_avg_copies=dict(sorted(avg_copies.items(), key=lambda x: -x[1])),
        ))

        status = "OK" if pool_coverage >= 0.75 else "LOW"
        print(
            f"[{status}] Tier{best_tier} {arch_name}: "
            f"全{total_decks}デッキ / 代表デッキ{len(rep_card_ids)}/60枚"
            f" coverage={pool_coverage:.1%} 欠落={sorted(missing_names)}"
        )

    _write_output(entries)
    print(f"\n→ {OUTPUT} を生成しました")


def _write_output(entries: list[dict]) -> None:
    lines = [
        '"""',
        'Tier上位メタデッキDB。',
        'このファイルは cardlist_referenced/build_meta_decks.py で自動生成されます。',
        '手動編集する場合はスクリプトの CANONICAL_CARD_ID / ARCH_SLUG を修正してください。',
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class MetaDeckEntry:",
        "    archetype_id: str",
        "    archetype_name_jp: str",
        "    tier: int                           # 1=Tier1, 2=Tier2",
        "    source_type: str                    # 'realworld_meta' | 'kaggle_meta'",
        "    total_decks: int                    # このアーキタイプの総デッキ数（集計元）",
        "    decks_with_recipe: int              # レシピデータが取得できたデッキ数",
        "    rep_card_ids: tuple[int, ...]       # 代表デッキのCard IDリスト（枚数分）",
        "    rep_card_counts: dict[int, int]     # 代表デッキ: Card ID → 枚数",
        "    missing_card_names: tuple[str, ...] # プール外で除外されたカード名",
        "    pool_coverage: float                # 代表デッキの解決率（0.0〜1.0）",
        "    key_cards: tuple[int, ...]          # アーキタイプ識別シグネチャ",
        "    card_inclusion_rate: dict[int, float]  # Card ID → 採用デッキ割合（0.0〜1.0）",
        "    card_avg_copies: dict[int, float]      # Card ID → 平均採用枚数",
        "",
        "    def is_reliable_for_estimation(self) -> bool:",
        '        """代表デッキの75%以上が解決できていれば推定に使用可能"""',
        "        return self.pool_coverage >= 0.75",
        "",
        "    def probable_card_ids(self, min_inclusion: float = 0.5) -> list[int]:",
        '        """採用率が min_inclusion 以上のカードIDリストを返す（デッキ推定のコア）"""',
        "        return [cid for cid, rate in self.card_inclusion_rate.items() if rate >= min_inclusion]",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# 現実世界メタ（pokeka-win-decks.jp / torecamap.co.jp より）",
        "# ---------------------------------------------------------------------------",
        "",
        "REALWORLD_META: dict[str, MetaDeckEntry] = {",
    ]

    for e in entries:
        lines += [
            f'    "{e["archetype_id"]}": MetaDeckEntry(',
            f'        archetype_id="{e["archetype_id"]}",',
            f'        archetype_name_jp="{e["archetype_name_jp"]}",',
            f"        tier={e['tier']},",
            f'        source_type="{e["source_type"]}",',
            f"        total_decks={e['total_decks']},",
            f"        decks_with_recipe={e['decks_with_recipe']},",
            f"        rep_card_ids={repr(tuple(e['rep_card_ids']))},",
            f"        rep_card_counts={repr(e['rep_card_counts'])},",
            f"        missing_card_names={repr(tuple(e['missing_card_names']))},",
            f"        pool_coverage={e['pool_coverage']},",
            f"        key_cards={repr(tuple(e['key_cards']))},",
            f"        card_inclusion_rate={repr(e['card_inclusion_rate'])},",
            f"        card_avg_copies={repr(e['card_avg_copies'])},",
            f"    ),",
        ]

    lines += [
        "}",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Kaggleコンペ固有メタ（将来追加）",
        "# ---------------------------------------------------------------------------",
        "",
        "KAGGLE_META: dict[str, MetaDeckEntry] = {}",
        "",
        "",
        "def get_all_meta_decks() -> dict[str, MetaDeckEntry]:",
        '    """全ソースをマージして返す（KAGGLE_METAが存在すれば優先）"""',
        "    merged = dict(REALWORLD_META)",
        "    merged.update(KAGGLE_META)",
        "    return merged",
        "",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
