# Battle Review Viewer 改善方針

このドキュメントは、`battle_review_viewer` をどう改善していくかの設計メモです。
実装はまだ着手せず、**何を・なぜ・どの順で** やるかを先に固める目的でまとめています。

対象読者は、このリポジトリで AI エージェントを開発しているメンバー全員です。
`sample_submission/main.py` を直接いじっていない人でも、リプレイを開けば
「エージェントが何をして、なぜそうしたか」を追えるようにすることを最終ゴールにします。

---

## 0. 現状サマリ（2026-06-29 時点）

| 要素 | ファイル | 現状 |
|------|----------|------|
| リプレイ生成 | [`export_replay.py`](export_replay.py) | `agent` vs `self`/`random` を1試合回し、`visualize_data()` の各 decision point + 選択肢 + 実際の action を JSON 化 |
| 配信サーバ | [`serve_viewer.py`](serve_viewer.py) | 依存ゼロの軽量 HTTP。`/api/replays`（一覧）と `/api/replays/<name>`（中身）、`web/` を静的配信 |
| 画面 | [`web/`](web/) | サッカー場風レイアウト。**常に Player 0 が下**、カードはテキストチップ、判断理由は無し |

### 現状の課題（このドキュメントで解決したいこと）

1. **盤面レイアウトが本家と違う** … 実際のポケカ／Kaggle リプレイ画面のような配置にしたい（要望画像）。
2. **カードがテキストだけ** … 実際のカード画像を出したい。
3. **「なぜその行動をしたか」が見えない** … 判断理由（できれば日本語）を出したい。
4. **エージェント非依存で最低限が見たい** … `main.py` の中身を知らない人でも動きを追える土台が欲しい。
5. **将来: 相手推定の可視化** … `src/inference` が決定経路に組み込まれたら、推定の進み方も追いたい。

---

## 1. 目標とする盤面レイアウト

要望画像（実機リプレイ風）を領域に分解すると以下になります。**レビュー視点は固定**（レビュー対象エージェント = Player 0 を常に下）にし、手番は色やバッジで示すのが追いやすい、という方針を採ります。
※ 現行は手番が変わるたびに `yourIndex` 基準で上下が入れ替わるため混乱する。これを「視点固定・手番はハイライト」に変更する。

```
┌──────────────────────────────────────────────────────────────┐
│  相手の手札（裏面 ×N、レビューでは表向き表示も可）               │ 上端
├──────────────────────────────────────────────────────────────┤
│  相手ベンチ [カード]×最大5            相手 山札/トラッシュ枚数    │
│              [Loss] MORIOKA TsuoI 456 (-9)   ← 相手名・勝敗(赤)  │
│ [Second]                                                       │
│ Time 600                ┌───────────┐                          │
│ Discard 4    相手トラッシュ│  ⊙ 相手active│  スタジアム            │
│ Deck 36       (重ね表示)  │           │                         │
│                          │  自分active │            [First]      │
│                          └───────────┘            Time 600     │
│              [Win] RickArko 508 (+4)   ← 自分名・勝敗(青)  Discard 1│
│  サイド（裏面 ×6 のスロット）              自分トラッシュ  Deck 44  │
│  自分ベンチ [カード]×最大5                                       │
├──────────────────────────────────────────────────────────────┤
│  自分の手札（表向き ×N）                                         │ 下端
└──────────────────────────────────────────────────────────────┘
```

各領域がバインドするデータ（`frame.visual.current.players[*]` 由来）:

| 領域 | データ源 |
|------|----------|
| active / bench / hand / prize | `players[i].active / bench / hand / prize` |
| 山札枚数 / トラッシュ | `players[i].deckCount` / `players[i].discard`（配列長） |
| 手番 | `current.yourIndex`（= いま動いているプレイヤー）→ そのレーンをハイライト |
| First / Second | `current.firstPlayer`（先攻プレイヤー index）から導出 |
| Turn / Context | `frame.turn` / `frame.context` |
| 勝敗ラベル | `metadata.result` + 終局フレームから導出（Win/Loss/スコア） |
| HP 表示 | `card.hp` / `card.maxHp` |
| エネ・どうぐ | `card.energies` / `card.tools`（個数 → 将来はアイコン） |
| スタジアム | `current.stadium` |
| 状態異常 | `players[i].poisoned / burned / asleep / paralyzed / confused` |

> **Time 600 について**: 現在のリプレイ JSON には残り時間が無い。実機の「持ち時間」は本ツールでは未取得なので、当面は省略 or 固定表示にする（取得経路ができたら追加）。

---

## 2. カード画像の表示（`pdf_card_editor` の抽出を再利用）

### 2.1 再利用できる既存 API

[`cardlist_referenced/pdf_card_editor/pdf_card_tool.py`](../cardlist_referenced/pdf_card_editor/pdf_card_tool.py) に、そのまま使える抽出関数があります。

- `load_pdf_catalog(pdf_path)` … `data/Card_ID List_JP.pdf` を解析し `CardRow(card_id, …, target_page)` を返す → **`card_id → target_page` の対応が取れる**。
- `get_card_asset_map(pdf_path, target_pages) -> dict[page, CardAsset]` … 対象ページのカード画像を **PNG/JPG でキャッシュ抽出**。`CardAsset.card_id` / `.image_path` を持つ。

つまり「`card_id` → カード画像ファイル」が既存コードだけで作れます。

### 2.2 方針: 静的プリダンプ方式（推奨）

`serve_viewer.py` は現在**依存ゼロ**で動くのが利点です。これを壊さないため、
画像抽出は **別スクリプトで事前に静的ファイル化** し、サーバは静的配信するだけにします。

```
新規: battle_review_viewer/build_card_assets.py   （pdf_card_editor の venv 配下で実行）
  └─ 入力: data/Card_ID List_JP.pdf（必要なら deck.csv / リプレイ内 card_id に絞る）
  └─ 処理: load_pdf_catalog → get_card_asset_map で抽出
  └─ 出力: battle_review_viewer/web/card_images/<card_id>.png|jpg
```

- `serve_viewer.py` 側は **追加依存なし**で `web/card_images/` を静的配信できる。
- フロントは `card.id` から `./card_images/<id>.png` を `<img>` で参照。画像欠落時は現行のテキストチップにフォールバック。
- 抽出は pdfplumber / Pillow に依存するが、それは `pdf_tool_requirements.txt` の範囲。`serve` には波及しない。

**代替案（やらない）**: serve 時に `pdf_card_tool` を import して都度抽出。→ サーバに重い依存が乗る・初回が遅い・cg 環境と venv が混ざるので非推奨。

### 2.3 注意点

- `Card_ID List_JP.pdf` は Kaggle 配布物。リポジトリに無い環境では画像が出ない → **画像はあくまで任意レイヤ**にし、無くてもビューアは動く設計を維持する。
- 裏面（相手手札・サイド）は共通の「カード裏」プレースホルダ画像を1枚用意して使い回す。

---

## 3. 「なぜその行動をしたか」（判断理由の可視化）

### 3.1 結論: 日本語の理由出しは「難しくない、ただし配線が要る」

要望「日本語ベースでなぜその行動を行ったかを知れると便利。難しいか?」への回答:

- **メインフェーズ (MAIN) は容易**。[`main_turn.py`](../sample_submission/src/decision/handlers/main_turn.py) が既に
  `MainActionProposal(action, score, label)` を **8カテゴリ分**生成し、最大スコアを採用している
  （[`proposals.py`](../sample_submission/src/decision/main_turn_parts/proposals.py)）。
  `label` は日本語前提のラベルなので、**採用候補の label + score、不採用候補の label + score** を出せば
  「ドロー(score 12) を、攻撃(score 8)・エネ付け(score 5) より優先した」と理由が即可視化できる。
- **難所は2つだけ**:
  1. **配線**: エージェントの戻り値は提出仕様で `list[int]` 固定。理由はこの戻り値に混ぜられない。
  2. **MAIN 以外のハンドラ**（`card_move` / `attack` / `switch` など）は今は `list[int]` を直接返し、
     `label`/`score` 構造を持たないものがある。

### 3.2 推奨アーキテクチャ: 「決定トレースのプロセス内シンク」

`src/inference/runtime_state.py` が既に使っている**モジュール内シングルトン**パターンをそのまま流用します。
提出時の挙動・性能に影響を与えず、レビュー時だけ理由を吸い出せるのが利点です。

```
新規: sample_submission/src/decision/trace.py
  - record_decision(context, chosen_label, chosen_score, alternatives, notes) を提供
  - 既定では「収集オフ」。export_replay.py が enable_trace() した時だけ蓄積。
  - main_turn.choose_main_action 等が「採用/不採用 proposal」をここへ流す。
  - agent() の戻り値は list[int] のまま（提出物への影響ゼロ）。
```

- **Kaggle 提出時**: `trace` は誰も enable しない → 記録は no-op、性能影響なし。
- **リプレイ生成時**: `export_replay.py` が `enable_trace()` し、`agent()` 呼び出し直後に
  `trace.pop()` して、そのフレームの理由として JSON に格納する。
- MAIN 以外のハンドラは**段階的に対応**。未対応の文脈は 3.3 のエンジン真実レイヤだけ出る。

### 3.3 二層構造（重要: エージェント非依存で「最低限」を保証）

ビューアの理由表示は**2層**に分け、下の層は常に出るようにします。

| 層 | 内容 | 出る条件 | 実装状況 |
|----|------|----------|----------|
| **A. エンジン真実層**（常時） | context、合法だった選択肢の人間可読ラベル、実際に選んだ option | 任意のエージェントで常に出る | `export_replay.describe_option()` で**既に実装済み** |
| **B. エージェント理由層**（任意） | 採用/不採用 proposal の label・score、補足メモ | trace を出すエージェントの時だけ | 3.2 を実装すると出る |

これにより要望4「`main.py` を使っていない人でも最低限どう動くか見える」を満たせます。
A 層は `random_agent` でも他人のエージェントでも常に表示されます（"何を選んだか" は必ず分かる）。
B 層が無くても画面は壊れず、「理由（詳細）: このエージェントは未提供」と出すだけにします。

### 3.4 リプレイ JSON への追加（後方互換）

```jsonc
// frame に optional で追加（無くても現行フロントは動く）
{
  "trace": {                         // B層: agent が出した時のみ
    "context": "MAIN",
    "chosen":   { "label": "ドローを優先", "score": 12, "actionIndices": [3] },
    "alternatives": [
      { "label": "ベンチ展開", "score": 8 },
      { "label": "エネ加速",   "score": 5 }
    ],
    "notes": "手札が細いのでドローソースを先に切った"
  }
}
```

A 層（`context` / `options` / `action` / `actionLabels`）は現行 JSON に既に入っているので追加不要。

---

## 4. 将来: 相手推定（`src/inference`）の可視化

### 4.1 現状

`src/inference` は**スタンドアロン**で、決定経路からはまだ呼ばれていない（[`__init__.py`](../sample_submission/src/inference/__init__.py) の docstring 参照）。
`get_runtime_inference(recipe).observe(obs)` を毎観測呼ぶだけで、以下が得られる:

- `last_opponent_estimate: OpponentEstimate` … `posterior`（アーキタイプ→事後確率）、`top_archetype`、`confidence`、`ranked()`
- `last_self_estimate: SelfEstimate` … `probability_in_deck(card_id)`、`key_card_reachable`、`prize_known_ratio()`
- `last_ace_estimate` / `last_threat_damage` … エース推定・脅威打点

### 4.2 方針: 出力時スナップショット（エージェントに手を入れない）

`export_replay.py` に `--inference` フラグを足し、**フレーム生成時に `observe(obs)` を呼んで推定値をスナップショット**します。
これは決定ロジックと独立に動くので、エージェント本体や提出物に影響しません。

```jsonc
// frame に optional 追加
{
  "inference": {
    "opponent": {
      "top": "dragapult", "confidence": 0.62,
      "posterior": { "dragapult": 0.62, "charizard": 0.21, "OTHER": 0.17 },
      "seenKeyCards": [677, 1141]
    },
    "self":   { "prizeKnownRatio": 0.33, "keyCards": [ { "id": 123, "pInDeck": 0.81 } ] },
    "ace":    { "...": "..." },
    "threat": { "maxDamage": 280 }
  }
}
```

### 4.3 画面（インスペクタに「推定」タブを追加）

- 相手アーキタイプ事後分布を**横棒バー**で（`ranked()` 上位数件 + OTHER）。
- `confidence` と「根拠カード（`seenKeyCards`）」を併記し、確信度が上がる様子をフレーム送りで追える。
- サイド/山札推定は `prizeKnownRatio` をゲージで、キーカードの `probability_in_deck` を一覧表示。
- 将来 `src/inference` が**決定経路に統合**されたら、3章の trace と並べて
  「この推定（相手はドラパルト濃厚）だから、この行動（突っ張らない）を選んだ」と**因果が並ぶ**のが理想形。

---

## 5. リプレイ JSON スキーマ（追加分まとめ・すべて後方互換）

| パス | 層 | 必須 | 説明 |
|------|----|------|------|
| `frame.trace` | B（理由・詳細） | 任意 | 3.4 参照。trace 対応エージェント時のみ |
| `frame.inference` | 推定 | 任意 | 4.2 参照。`--inference` 指定時のみ |
| `metadata.deckPdfPath` | 画像 | 任意 | カード画像抽出に使った PDF。ビューアの画像有無判定に使用 |
| `metadata.players` | レイアウト | 任意 | プレイヤー名・先攻後攻・最終スコア（Win/Loss ラベル用） |

既存フィールド（`frame.visual` / `options` / `action` / `actionLabels` / `context` / `turn` / `actingPlayer`）は**変更しない**。

---

## 6. 実装ロードマップ（優先順）

> 各フェーズは独立してマージ可能。**A 層（エンジン真実）と後方互換**を常に死守する。

### フェーズ 1: 盤面レイアウト刷新（最優先・要望1）
- `web/index.html` / `styles.css` / `app.js` を実機風レイアウトへ。
- **視点固定**（Player 0 を常に下）+ 手番ハイライトに変更。
- First/Second・Turn・Context・山札/トラッシュ枚数・サイド枠・スタジアム・状態異常を配置。
- 既存バグも同時に修正:
  - 手札ハイライト（`renderCard` が `.highlight` クラスを付けていない / CSS だけ存在）。
  - サイドの空スロット表示（6枠ぶん出ていない）。
- **データ追加不要**（現行 JSON で完結）。

### フェーズ 2: カード画像（要望2・要望6前半）
- `build_card_assets.py` を追加（`get_card_asset_map` 再利用）→ `web/card_images/<card_id>.png`。
- フロントを `<img>` 化、欠落時はテキストチップにフォールバック。
- カード裏プレースホルダを用意。

### フェーズ 3: 判断理由（要望3・要望4・要望5）
- **3a（先行・低コスト）**: A 層の表示を強化。context の日本語説明辞書を持ち、
  「いまエージェントは〈メインフェーズ〉で〈ドローを選んだ〉」を**どのエージェントでも**出す。
- **3b**: `src/decision/trace.py`（シンク）を追加し、`main_turn` を皮切りに B 層 trace を出す。
- **3c**: 他ハンドラ（attack / card_move / switch …）へ trace を順次拡張。

### フェーズ 4: 推定の可視化（要望・将来）
- `export_replay.py --inference` で `frame.inference` を出力。
- インスペクタに「推定」タブ（相手事後分布バー / サイド推定ゲージ）。
- `src/inference` が決定経路へ統合された後、理由（trace）と推定を並べて因果を見せる。

---

## 7. 論点・リスク（着手前に合意したい点）

1. **レビュー視点**: 「Player 0 固定で下」で良いか？（self 対戦だと両者がエージェント。どちらを"主"にするか）
   → 暫定: **`export` で `agent` を渡した側 = Player 0 を主**にする。
2. **相手手札/サイドの開示**: レビューツールなので**全公開**で良いか（実機は非公開）。
   → 暫定: レビュー優先で**全公開**。裏面トグルを付けて切替可能にする案も。
3. **カード画像の配布**: `Card_ID List_JP.pdf` 非保持の環境では画像が出ない。
   → 画像は**任意レイヤ**として、無くても完全に動くことを保証する。
4. **trace の保守コスト**: ハンドラ追加時に trace 出力を書き続ける必要がある。
   → A 層が常に出るので、B 層は"出せるところから"で運用。未対応でも破綻しない設計にする。
5. **Time（持ち時間）**: 現状データに無い。当面は非表示。取得経路ができたら追加。

---

## 8. 影響範囲（このリポジトリのルールとの整合）

- `cg/` ・ `data/` は**変更しない**（参照のみ）。
- `sample_submission/main.py` の `agent()` シグネチャは**変更しない**。理由トレースは
  プロセス内シンク経由で、提出物の戻り値・挙動に影響を与えない。
- `cardlist_referenced/` は提出に無関係なので自由に再利用してよい（`pdf_card_tool` を import / 流用）。
- `battle_review_viewer/` は提出コードから独立。ここで完結させる。
