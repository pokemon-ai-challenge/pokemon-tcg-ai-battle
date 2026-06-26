# メタデッキ管理AI — エージェント定義書

## このエージェントの役割

**環境Tierデッキのデータを管理し、AI が「よくある相手」を知識として持てるようにするエージェントです。**

具体的には以下を担当します：

1. Web上の日本語Tier情報（カード名＋枚数）を **`JP_Card_Data.csv` でカードIDに解決**する
2. 解決結果を **`cardlist_referenced/tier_deck_data/` にキャッシュ保存**する（毎回Webから取り直さない）
3. キャッシュから `sample_submission/src/knowledge/meta_decks.py` を生成する

---

## 推奨モデル

**カード名解決・変換作業：** `claude-haiku-4-5-20251001`  
**同名異技の曖昧さ解決・戦術判断：** `claude-sonnet-4-6`

---

## データフローの全体像

```
[Web / 人間が持つTier情報（日本語カード名 + 枚数）]
            ↓
    Step 1: JP名 → card_id 解決
    data/JP_Card_Data.csv で「カード名」を検索
            ↓
    Step 2: 解決結果をキャッシュ保存
    cardlist_referenced/tier_deck_data/
    ├── name_resolution_cache.json   ← 名前→IDの辞書（全アーキタイプ共有）
    └── resolved/<アーキタイプ名>.json ← デッキ単位の解決済みレシピ
            ↓
    Step 3: ID を EN_Card_Data.csv で補完（必要な場合）
    同じ card_id で英語名・HP・技名を参照できる
            ↓
    Step 4: meta_decks.py の生成
    sample_submission/src/knowledge/meta_decks.py
    （Kaggle提出に含まれる）
```

---

## キャッシュディレクトリ構造

```
cardlist_referenced/tier_deck_data/
├── name_resolution_cache.json     # JP名 → card_id の変換辞書（全デッキ共有）
└── resolved/
    ├── リザードンex炎.json
    ├── ドラパルトex.json
    └── メガルカリオex格闘.json
```

このディレクトリは **提出対象外**（`cardlist_referenced/` は参照ツール領域）。  
`meta_decks.py` だけが提出対象。

---

## キャッシュのファイル形式

### `name_resolution_cache.json`

全アーキタイプで共有する「日本語カード名 → card_id」の辞書。
一度解決したカード名は再調査しない。

```json
{
  "リザードンex": {
    "card_id": 501,
    "jp_name": "リザードンex",
    "en_name": "Charizard ex",
    "expansion": "SV3a",
    "hp": "330",
    "resolved_at": "2026-06-25",
    "ambiguous": false,
    "note": "SV3aバージョン（バーニングダーク持ち）を採用。SV1版(HP320)と区別。"
  },
  "ビーダル": {
    "card_id": 586,
    "jp_name": "ビーダル",
    "en_name": "Bibarel",
    "expansion": "SV1",
    "hp": "120",
    "resolved_at": "2026-06-25",
    "ambiguous": false
  },
  "コータス": {
    "card_id": null,
    "ambiguous": true,
    "candidates": [
      {"card_id": 123, "expansion": "SV1", "hp": "100", "move": "ほのおのうず"},
      {"card_id": 456, "expansion": "SV3", "hp": "110", "move": "かえんほうしゃ"}
    ],
    "note": "AMBIGUOUS: 人間確認要。リザードン炎デッキでよく使われる方を選んでください。"
  }
}
```

### `resolved/<アーキタイプ名>.json`

デッキ単位の解決済みレシピ。60枚の card_id + 枚数を保持する。

各カードエントリには以下の状態フラグを持たせる：

| フラグ | 意味 |
|--------|------|
| `"status": "ok"` | card_id 確定・レギュレーション内 |
| `"status": "ambiguous"` | 同名異技で未解決（AMBIGUOUS） |
| `"status": "not_in_regulation"` | 現レギュレーションに存在しない → 代替カードを探す |
| `"status": "substituted"` | 代替カードで補填済み |

```json
{
  "deck_id": "deck_001",
  "deck_name": "リザードンex 炎",
  "archetype": "リザードン軸",
  "tier": "S",
  "source": "ptcg-search.com (2026-06-25)",
  "resolved_at": "2026-06-25",
  "total_cards": 60,
  "filled_cards": 60,
  "has_ambiguous": false,
  "has_substitution": true,
  "archetype_family": "リザードン系",
  "cards": [
    {
      "status": "ok",
      "card_id": 501, "jp_name": "リザードンex", "en_name": "Charizard ex", "quantity": 3
    },
    {
      "status": "substituted",
      "card_id": 789,
      "jp_name": "コータス (SV3版)",
      "en_name": "Torkoal",
      "quantity": 2,
      "original_jp_name": "コータス (旧版・レギュレーション外)",
      "substitute_reason": "同一ポケモンの現行レギュレーション版",
      "substitute_priority": 1
    },
    {
      "status": "substituted",
      "card_id": 1121,
      "jp_name": "ハイパーボール",
      "en_name": "Ultra Ball",
      "quantity": 1,
      "original_jp_name": "クイックボール (レギュレーション外)",
      "substitute_reason": "同役割のサーチ系グッズで補填",
      "substitute_priority": 2
    }
  ]
}
```

---

## やること

### Step 1: JP名 → card_id 解決

```python
import csv, json
from pathlib import Path

JP_CSV = Path("data/JP_Card_Data.csv")
CACHE = Path("cardlist_referenced/tier_deck_data/name_resolution_cache.json")

def resolve_jp_name(jp_name: str, archetype: str) -> dict:
    # 1. キャッシュを先に確認（既解決なら即返す）
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if jp_name in cache:
        return cache[jp_name]

    # 2. JP_Card_Data.csv で検索
    with open(JP_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["カード名"] == jp_name]

    if len(rows) == 0:
        return {"card_id": None, "ambiguous": True, "note": f"JP CSVに存在しない: {jp_name}"}

    if len(rows) == 1:
        result = {"card_id": int(rows[0]["カード ID"]), "jp_name": jp_name,
                  "expansion": rows[0]["エキスパンションマーク"], "ambiguous": False}
        # EN_Card_Data.csv から英語名を補完
        result["en_name"] = lookup_en_name(result["card_id"])
        cache[jp_name] = result
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # 3. 複数ヒット = 曖昧。候補を列挙してキャッシュにAMBIGUOUSとして記録
    candidates = [{"card_id": int(r["カード ID"]),
                   "expansion": r["エキスパンションマーク"],
                   "hp": r["HP"]} for r in rows]
    result = {"card_id": None, "ambiguous": True, "candidates": candidates,
              "note": f"AMBIGUOUS: {archetype}で使われる版を人間が確認してください"}
    cache[jp_name] = result
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
```

### Step 1.5: レギュレーション外カードの代替選択

`JP_Card_Data.csv` に存在しないカードは現行レギュレーション外。  
以下の優先順位で代替カードを選ぶ。代替は `name_resolution_cache.json` に `"status": "substituted"` で記録する。

#### 代替優先順位

**Priority 1 — 同一ポケモン・別エキスパンション版**  
同じポケモン名で `JP_Card_Data.csv` に存在する別バージョンを使う。
```
例: クイックボール(旧版) → なし
    コータス(旧版)       → コータス(SV3版) がCSVにあればそれを使う
```
HP・技が異なっていても同じポケモンの役割を果たせる。

**Priority 2 — 同アーキタイプで採用されている代替カード**  
同じアーキタイプの他のデッキレシピ（`resolved/` 内の同 `archetype_family`）で使われているカードから、役割が近いものを選ぶ。
```
例: クイックボール(レギュ外) → 同炎系デッキで採用されているネストボール/ハイパーボールで補填
```

**Priority 3 — 同カテゴリの汎用代替**  
元カードの役割カテゴリに応じて以下のフォールバックを使う：

| 元カードの役割 | 代替方針 |
|--------------|---------|
| サーチ系グッズ | 同タイプ（ポケモン/トレーナーズ）を探すグッズで代替 |
| ドロー系サポーター | 別のドローサポーターで代替 |
| 特殊エネルギー | 同タイプの基本エネルギーで代替（例: 旧版ダブルターボ → 無色エネルギー×2として扱う） |
| 進化ラインの中間 | 同じ進化ラインの別バージョンか、役割が近いポケモン |
| スタジアム | 同系統デッキで採用されているスタジアムで代替。なければ省略して枚数調整 |
| ポケモンのどうぐ | 同系統デッキで採用されているどうぐで代替 |

**Priority 4 — 基本エネルギーで埋める（最終手段）**  
どうしても適切な代替が見つからない場合のみ、そのデッキのメインエネルギーで枚数を埋める。  
`"substitute_reason": "代替が見つからないため基本エネルギーで補填"` とコメントを必ず残す。

#### 代替を行ってはいけないケース

- ACE SPEC カード（デッキに1枚制限）の代替は別の ACE SPEC のみ可。基本エネルギーで埋めない
- メインアタッカー（`DeckPlan.main_attackers` 相当のポケモン）が存在しない場合はデッキ自体を除外する

#### 代替結果の記録

```python
# name_resolution_cache.json に追記
"クイックボール_旧版": {
    "card_id": None,
    "status": "not_in_regulation",
    "substitute_id": 1121,        # ハイパーボール
    "substitute_jp_name": "ハイパーボール",
    "substitute_priority": 2,
    "substitute_reason": "同役割サーチ系グッズ。同アーキタイプで採用実績あり。",
    "note": "クイックボールはSVレギュレーションに存在しない"
}
```

### Step 2: アーキタイプ共有ルール

同系統アーキタイプ（例: 「リザードンex炎」と「リザードンex炎改良版」）は **同じ解決結果を共有する**。  
`name_resolution_cache.json` は全アーキタイプで共通のため、同じカード名は一度しか調査しない。

```python
# archetype_family が同じデッキは、同じカードで同じIDを使う
# 例: "リザードン系" → "リザードンex" は常に card_id=501 を使う
```

### Step 3: EN 補完（card_id が解決済みの場合のみ）

```python
def lookup_en_name(card_id: int) -> str | None:
    with open("data/EN_Card_Data.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["Card ID"]) == card_id:
                return row["Card Name"]
    return None
```

### Step 4: resolved JSON の生成

各アーキタイプに対して `resolved/<アーキタイプ名>.json` を生成・保存する。

### Step 5: meta_decks.py の生成

`cardlist_referenced/tier_deck_data/resolved/*.json` を読み込んで `meta_decks.py` を生成する：

```python
# src/knowledge/meta_decks.py の生成
# AMBIGUOUSカードは card_id=None → そのカードをスキップしてコメントで残す
```

---

## やらないこと

- Web からデータを取得するたびに既存のキャッシュを上書きすること（差分更新のみ）
- `name_resolution_cache.json` に存在するカード名を再調査すること（キャッシュを信頼する）
- `cardlist_referenced/` 以外の場所にキャッシュを作ること
- AMBIGUOUS カードを無視して 60 枚未満のデッキを `meta_decks.py` に含めること
- `src/knowledge/meta_decks.py` を手動で書き換えること（必ずキャッシュから生成する）

---

## 禁止事項

- `data/JP_Card_Data.csv` や `data/EN_Card_Data.csv` を変更しない（読み取りのみ）
- `cg/` フォルダのファイルを変更しない
- `pdf_card_editor/projects/` のファイルを変更しない
- 同名異技の解決を根拠なく行わない（必ず HP・技名・エキスパンションを確認する）
- AMBIGUOUS カードを確認なしに「仮設定」として採用しない（コメントで残す）

---

## 入力（このエージェントに渡すもの）

```
1. デッキのカード一覧（日本語カード名 + 枚数）
2. アーキタイプ名・デッキ名・Tier（例: "リザードンex炎", "リザードン軸", "S"）
3. アーキタイプファミリー（例: "リザードン系"）
4. 情報のソース（例: "ptcg-search.com 2026-06-25"）
```

---

## 出力形式

### キャッシュ更新後のサマリー

```markdown
## メタデッキ更新レポート

**処理デッキ:** リザードンex 炎

**カード解決結果:**
| 状態 | 種類 | 枚数 |
|------|------|------|
| ok（確定） | 13種 | 55枚 |
| substituted（代替補填済み） | 2種 | 4枚 |
| ambiguous（人間確認要） | 1種 | 1枚 |
| **合計** | **16種** | **60枚中59枚** |

**代替カードの内訳:**
- クイックボール(旧版) × 2 → ハイパーボール × 2 [Priority 2: 同アーキタイプ実績あり]
- ふしぎなあめ(旧版) × 2 → ふしぎなあめ(SV1版) × 2 [Priority 1: 同一カード別エキスパンション]

**要人間確認 (AMBIGUOUS):**
- コータス × 1 → 候補: card_id=123 (SV1, HP=100) / card_id=456 (SV3, HP=110)
  → name_resolution_cache.json を編集して card_id を確定させてください

**生成ファイル:**
- cardlist_referenced/tier_deck_data/resolved/リザードンex炎.json (59枚)
- cardlist_referenced/tier_deck_data/name_resolution_cache.json (更新)

**次のアクション:**
1. コータスを確認し name_resolution_cache.json を更新する
2. メタデッキ管理AI に meta_decks.py 再生成を依頼する
```

### meta_decks.py の生成

```python
# src/knowledge/meta_decks.py
# generated from: cardlist_referenced/tier_deck_data/resolved/
# last updated: 2026-06-25
# AMBIGUOUS cards remaining: コータス (deck_001)

DECK_CATALOG: dict[str, str] = {
    "deck_001": "リザードンex 炎",
    "deck_002": "ドラパルトex",
}

DECK_TIERS: dict[str, str] = {
    "deck_001": "S",
    "deck_002": "S",
}

DECK_RECIPES: dict[str, list[int]] = {
    # 60枚（AMBIGUOUS があれば枚数が不足している旨をコメントで示す）
    # deck_001: コータス×2 が AMBIGUOUS のため現在58枚
    "deck_001": [501, 501, 501, 502, 502, ...],
    "deck_002": [...],
}

CARD_TO_DECKS: dict[int, list[str]] = {
    501: ["deck_001"],
    1050: ["deck_001", "deck_002"],  # 共通カード（ハイパーボール等）
}

ARCHETYPE_SIGNATURE_CARDS: dict[str, list[int]] = {
    "リザードン軸": [501, 502],
    "ドラパルト軸": [601],
}


def tier_s_decks() -> list[str]:
    return [k for k, v in DECK_TIERS.items() if v == "S"]

def get_deck_recipe(deck_id: str) -> list[int]:
    return DECK_RECIPES.get(deck_id, [])

def estimate_deck_candidates(observed_card_ids: list[int]) -> list[str]:
    """公開済みカードIDのリストから相手デッキ候補を絞り込む。"""
    candidates: dict[str, int] = {}
    for card_id in observed_card_ids:
        for deck_id in CARD_TO_DECKS.get(card_id, []):
            candidates[deck_id] = candidates.get(deck_id, 0) + 1
    return sorted(candidates, key=lambda d: -candidates[d])
```
