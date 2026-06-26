# 軸カード分析（Axis Card Analysis）

1626デッキのキャッシュデータから抽出した、各アーキタイプの「軸カード」（高頻度採用カード）の記録。

**読む目的：**
- 次の AI セッションがどのカードを「軸カード」として扱うかを把握する
- キャッシュが更新されたとき、再抽出→`meta_decks.py` 更新の手順を確認する
- 相手デッキ推定（`OpponentModel`）の根拠となる `key_pokemon_ids` の妥当性を検証する

---

## 1. データソース

| 項目 | 内容 |
|-----|------|
| キャッシュファイル | `cardlist_referenced/pdf_card_editor/.cache/tier_ranking/` 以下の最新 `.json` |
| 抽出日 | 2026-06-26（`2026-06-26-v4.json`、4.3 MB） |
| デッキ数 | 合計 1626 デッキ（pokeka-win-decks.jp + torecamap.co.jp） |
| Tier 分布 | Tier1(S): ドラパルトex(102)・カミツオロチex(95)・テラスタルバレット(3) ほか |
| キャッシュ更新方法 | `cd cardlist_referenced/pdf_card_editor && streamlit run app.py` → `🏆 Tier上位デッキの採用カードを取り込む` をクリック |

---

## 2. 軸カードとは

**同じアーキタイプの複数デッキで高頻度（≥60%）に採用されているカード。**

軸カードの役割：
1. **アーキタイプ検出の根拠** — 盤面でこのカードが見えたら即座にデッキを絞り込める
2. **戦略ターゲットの決定** — 軸カードを潰すことで相手の戦術を崩せる
3. **汎化対応** — 「ドラパルト軸なら同系の別デッキにも同じ対策が通用する」

---

## 3. 現在の軸ポケモンリスト（キャッシュ 2026-06-26 時点）

「頻度」は全デッキ中の採用率（60% 以上のカードのみ掲載）。  
ID は `data/JP_Card_Data.csv` + `src/knowledge/meta_decks.py` で解決済み。

### Tier S

#### dragapult_ex（ドラパルトex） — 102 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| ドラメシヤ | Dreepy | 119 | 4 | ~99% |
| ドロンチ | Drakloak | 120 | 4 | ~99% |
| ドラパルトex | Dragapult ex | 121 | 2-3 | ~99% |
| マシマシラ | Munkidori | 112 | 2 | ~85% |
| ニャースex | Meowth ex | 1071 | 2 | ~80% |
| バシャーモex | Blaziken ex | 326 | 2 | ~70% |

`key_pokemon_ids = [121, 119, 120, 326]` — ドラパルト専用カードが多く、ドラメシヤ1枚で即確定。

#### hydrapple_ex（カミツオロチex） — 95 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| オーガポン+みどりのめんex | Teal Mask Ogerpon ex | 96 | 4 | ~98% |
| チコリータ | Chikorita | 708 | 2 | ~95% |
| ベイリーフ | Bayleef | 709 | 2 | ~95% |
| メガニウム | Meganium | 710 | 2 | ~95% |
| カミツオロチex | Hydrapple ex | 150 | 2 | ~92% |
| ドロバンコン | Dipplin | 921 | 2 | ~85% |

`key_pokemon_ids = [150, 921, 149, 96, 710]` — Hydrapple ex(150) か Meganium(710) が見えたら確定。

### Tier A

#### mega_lucario_ex（メガルカリオex） — 98 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| リオル | Riolu | 677 | 4 | ~99% |
| メガルカリオex | Mega Lucario ex | 678 | 3 | ~99% |
| ソルロック | Solrock | 676 | 2-3 | ~95% |
| ハリテヤマ | Hariyama | 674 | 2 | ~85% |
| ルナトーン | Lunatone | 675 | 2 | ~85% |
| マクノシタ | Makuhita | 673 | 2 | ~80% |

`key_pokemon_ids = [678, 677, 675, 676, 674]` — リオル(677)は最初に出る Basic ポケモン。

#### raging_bolt_ex（タケルライコex） — 98 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| オーガポン+みどりのめんex | Teal Mask Ogerpon ex | 96 | 4 | ~98% |
| タケルライコex | Raging Bolt ex | 63 | 3 | ~95% |
| テツノイサハex | Iron Hands ex | 75 | 2 | ~85% |
| コライドン | Koraidon | 62 | 1 | ~70% |

`key_pokemon_ids = [63, 75, 96, 62]` — タケルライコex(63) 専用。オーガポン単体では他デッキと区別不可。

#### ogerpon_bullet（オーガポンバレット） — 99 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| オーガポン+みどりのめんex | Teal Mask Ogerpon ex | 96 | 2 | ~99% |
| メガガルーラex | MegaKangaskhan ex | 756 | 2 | ~90% |
| マシマシラ | Munkidori | 112 | 2 | ~85% |
| ラティアスex | Latias ex | 184 | 2 | ~80% |
| オーガポン+いどのめんex | Well Spring Mask Ogerpon ex | 108 | 1-2 | ~70% |

`key_pokemon_ids = [96, 756, 112, 184, 108]` — terasta_bullet との区別は `272` (リーリエのピッピex) の有無。

#### olivia_ex（オリーヴァex） — 94 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| オーガポン+みどりのめんex | Teal Mask Ogerpon ex | 96 | 4 | ~95% |
| メガニウム | Meganium | 710 | 2 | ~92% |
| オリーヴァex | Dolliv ex | 404 | 2 | ~90% |
| チコリータ | Chikorita | 708 | 2 | ~85% |

`key_pokemon_ids = [404, 402, 403, 710, 96]` — オリーヴァex(404) と Meganium(710) の共存が確認ポイント。

#### maries_obstagoon_ex（マリィのオーロンゲex） — 5 デッキ（サンプル少）

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| マリィのベロバー | Marnie's Zigzagoon | 646 | 4 | ~100% |
| マシマシラ | Munkidori | 112 | 4 | ~100% |
| マリィのオーロンゲex | Marnie's Obstagoon ex | 648 | 3 | ~100% |
| マリィのギモー | Marnie's Linoone | 647 | 2 | ~100% |
| ユキワラシ | Snover | 103 | 2 | ~80% |

`key_pokemon_ids = [648, 646, 647, 112, 103]` — サンプル 5 デッキと少ないため暫定。マリィのベロバー(646) は極めて識別性が高い。

#### crustle（イワパレス） — 4 デッキ（サンプル少）

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| イシズマイ | Dwebble | 344 | 4 | ~100% |
| メガガルーラex | MegaKangaskhan ex | 756 | 4 | ~100% |
| イワパレス | Crustle | 345 | 3 | ~100% |
| オーガポンいしずえのめんex | Cornerstone Mask Ogerpon ex | 117 | 1 | ~75% |

`key_pokemon_ids = [345, 344, 756, 117]` — イシズマイ(344) + メガガルーラex(756) の組み合わせが特徴。

### Tier B

#### alakazam（フーディン） — 97 デッキ

| JP名 | 英語名 | Card ID | 枚数 | 頻度 |
|-----|--------|---------|------|------|
| ユンゲラー | Kadabra | 742 | 4 | ~99% |
| ケーシィ | Abra | 741 | 4 | ~99% |
| フーディン | Alakazam | 743 | 3-4 | ~99% |
| ノコッチ | Dunsparce | 65 | 3 | ~85% |
| ノコッチス | Dudunsparce | 66 | 3 | ~80% |

`key_pokemon_ids = [743, 742, 741, 66, 65]` — ケーシィ(741) は Buddy-Buddy Poffin でサーチされるため序盤から見える。

---

## 4. 軸カードとアーキタイプ検出精度

現在の `estimate_deck_candidates()` は `key_pokemon_ids` を**直接使っていない**（`CARD_TO_DECKS` の重み付きスコアで推定）。  
`key_pokemon_ids` は現在 `setup_policy.py` の Active 選択など DeckPlan 内での利用。

### 検出が速いアーキタイプ（専用カードが序盤に出る）
- dragapult_ex: ドラメシヤ(119) が Buddy-Buddy Poffin でターン1に出る
- mega_lucario_ex: リオル(677) が初手バトル場に出る
- alakazam: ケーシィ(741) が Buddy-Buddy Poffin でターン1に出る
- crustle: イシズマイ(344) が初手に出る

### 検出が遅いアーキタイプ（共通カードが多い）
- ogerpon_bullet vs terasta_bullet: 両方ともオーガポン+みどりのめんex(96)を採用、区別に専用カードが必要
- hydrapple_ex vs olivia_ex: どちらも Meganium(710) + オーガポン(96) を採用

---

## 5. キャッシュが更新されたときの再抽出手順

### Step 1: キャッシュを更新する
```bash
cd cardlist_referenced/pdf_card_editor
streamlit run app.py
# ブラウザで '🏆 Tier上位デッキの採用カードを取り込む' をクリック
```

### Step 2: 軸カードを再抽出する

```python
# このスクリプトは cardlist_referenced/tier_deck_data/ から実行する
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from load_tier_cache import load_latest_cache, get_tier_archetypes, get_consensus_recipe

cache = load_latest_cache()

# ポケモンタイプのカード名を特定するためにカードDBを読む
# JP_Card_Data.csv で "ポケモン" タイプのカードを抽出
jp_card_csv = Path(__file__).parent.parent.parent / "data" / "JP_Card_Data.csv"
pokemon_names: set[str] = set()
with open(jp_card_csv, encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("cardType", "") in ("POKEMON", "ポケモン", "Pokemon"):
            name = row.get("name", "").strip()
            if name:
                pokemon_names.add(name)

# 正規化関数（括弧や特殊文字を除去してマッチ精度を上げる）
import re
def normalize(name: str) -> str:
    name = re.sub(r"[（(【\[].*?[）)\]】]", "", name)  # 括弧内を除去
    return name.strip()

normalized_pokemon = {normalize(n): n for n in pokemon_names}

# 各 Tier・アーキタイプの軸カード（≥60% 採用のポケモン）を出力
for tier_num in [1, 2, 3]:
    tier_label = {1: "S", 2: "A", 3: "B"}[tier_num]
    for arch, deck_count in get_tier_archetypes(cache, tier_num, min_count=3):
        recipe = get_consensus_recipe(cache, arch, tier_num, min_frequency=0.6)
        # ポケモンカードのみ抽出
        pokemon_axis = {}
        for jp_name, count in recipe.items():
            norm = normalize(jp_name)
            if norm in normalized_pokemon:
                pokemon_axis[jp_name] = count
        
        if pokemon_axis:
            print(f"\n[Tier {tier_label}] {arch} ({deck_count} decks):")
            for name, cnt in sorted(pokemon_axis.items(), key=lambda x: -x[1]):
                print(f"  {name} × {cnt}")
```

### Step 3: JP名 → Card ID の解決

JP_Card_Data.csv の `name` 列と照合。名前が一致しない場合：
1. 括弧内を除去して部分マッチを試みる（例: `オーガポン+みどりのめんex` → `オーガポン` でグループ化）
2. 複数バージョンがある場合は拡張セット（`cardSet`）で手動絞り込み
3. 解決した ID を `meta_decks.py` の `key_pokemon_ids` と `recipe` に反映

**既知の課題**: キャッシュの JP 名（`オーガポン+みどりのめんex`）が CSV の名前（`オーガポン+みどりのめんex` だが括弧表記の差異あり）と微妙に異なる場合がある。手動照合が必要。

### Step 4: meta_decks.py を更新

- 新アーキタイプが出た場合: `MetaDeck` エントリを追加
- 既存アーキタイプのレシピが変化した場合: `recipe` と `key_pokemon_ids` を更新
- `DECK_CATALOG` 更新後は `CARD_TO_DECKS` が自動再生成される

---

## 6. アーキタイプ別の戦略的ターゲット（まとめ）

軸カードをどう「崩すか」の方針。詳細は `docs/opponent_model.md` の `ARCH_BOSS_TARGETS` を参照。

| アーキタイプ | 崩し方 |
|------------|-------|
| dragapult_ex | 進化前ドラメシヤ(119)をボスで引き出してKO（ドラパルト来る前に潰す） |
| hydrapple_ex / olivia_ex | メガニウム(710)をボスで先にKO（エネ加速が止まると自己回復が機能しない） |
| raging_bolt_ex | 弱点（格闘）を利用してOHKO狙い。ベンチのオーガポン(96)をボスで引き出す |
| mega_lucario_ex | リオル(677)を引き出して進化ライン阻害。ルナトーン(675)ドロー妨害 |
| crustle | ボスでイシズマイ(344)を引き出してEXで倒す。Crustle本体には非EXアタッカー |
| alakazam | ケーシィ(741)ラインを早期に潰して Alakazam 展開を阻害 |
| ogerpon/terasta bullet | マシマシラ(112)・リーリエのピッピex(272)のドロー役を先処理 |

---

## 7. 関連ファイル

| ファイル | 役割 |
|---------|------|
| `src/knowledge/meta_decks.py` | 10デッキのレシピ定義、`estimate_deck_candidates()` の元データ |
| `src/knowledge/opponent_model.py` | `OpponentModel` クラス、`ARCH_BOSS_TARGETS` |
| `cardlist_referenced/tier_deck_data/load_tier_cache.py` | キャッシュ読み込み・コンセンサスレシピ抽出ユーティリティ |
| `docs/opponent_model.md` | 推定ロジックの全体説明 |
| `docs/strategy-knowledge.md` | [META-004] カード名解決の注意点、[META-005] torecamap 統合詳細 |
