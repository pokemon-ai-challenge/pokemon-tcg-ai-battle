# Battle Review Viewer 実装チェックリスト

[`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) の方針を、**実際に作る順番つきの作業チェックリスト**へ落とし込んだものです。
このファイルは「次に何を実装するか」を1項目ずつ潰していくための作業台帳として使います。

- チェックは GitHub 互換の `- [ ]` 形式。完了したら `- [x]` に更新する。
- 各フェーズ末尾に **完了条件（Definition of Done）** を置く。そこが満たせたら次へ。
- **後方互換**を常に死守する: 既存リプレイ JSON のフィールドは変更しない（追加は optional のみ）。

---

## 0. 実装の大原則（全フェーズ共通）

- [ ] 既存 JSON の必須フィールド（`frame.visual` / `options` / `action` / `actionLabels` / `context` / `turn` / `actingPlayer`）は**変更しない**
- [ ] 追加するデータ（`trace` / `inference` / 画像 / `metadata.players` 等）は**すべて optional**。欠けても画面が壊れないこと
- [ ] `serve_viewer.py` は**依存ゼロ**を維持（重い依存は抽出スクリプト側へ隔離）
- [ ] `cg/` `data/` は変更しない。`sample_submission/main.py` の `agent()` シグネチャも変更しない
- [ ] フェーズ間は独立してマージ可能にする（UI だけ先に入れても動く、画像が無くても動く）

### 作る順番（このドキュメントの並び）

```
フェーズ1: UI / 盤面レイアウト刷新        ← 最初
フェーズ2: カード画像取得・表示           ← フェーズ1と並行可
フェーズ3: 「なぜその行動をしたか」        ← チーム差分に強い設計が肝
フェーズ4: 相手推定の可視化（将来）        ← inference 統合後
```

---

## フェーズ1: UI / 盤面レイアウト刷新（最優先）

実機リプレイ風レイアウトへ。**視点固定（Player 0 を常に下）+ 手番ハイライト**に変更する。
データ追加は不要（現行 JSON で完結）。

### 1-1. 設計・準備
- [x] レビュー視点の確定: `export` で `agent` を渡した側 = **Player 0 を常に下**に固定する方針で実装
- [x] 領域マッピングを確認（[`IMPROVEMENT_PLAN.md` §1](IMPROVEMENT_PLAN.md) の表）: active / bench / hand / prize / 山札・トラッシュ枚数 / stadium / 状態異常
- [x] 既存 `web/` をバックアップ or 別ブランチで作業 ※`develop/nagata` 上で直接刷新

### 1-2. レイアウト実装（`web/index.html` + `styles.css`）
- [x] 上端: 相手の手札行、下端: 自分の手札行
- [x] 相手レーン（上）: ベンチ行 / active / 山札・トラッシュ枚数 / サイド枠
- [x] 自分レーン（下）: ベンチ行 / active / 山札・トラッシュ枚数 / サイド枠
- [x] 中央: ポケボール風サークル + 相手active（上）/ 自分active（下）+ スタジアム枠
- [x] サイド枠を **6スロット**ぶん常に表示（現行バグ: 6枠出ていない を修正）
- [x] First / Second バッジ（`current.firstPlayer` から導出）
- [x] Turn / Context バッジ（`frame.turn` / `frame.context`）
- [x] 各プレイヤーの名前・勝敗・スコアラベル（`metadata.players` があれば表示、無ければ "Player 0 / 1"）

### 1-3. 描画ロジック（`web/app.js`）
- [x] **視点固定**: `render()` を「常に players[0] が下・players[1] が上」へ変更（現行の `yourIndex` 依存をやめる）
- [x] **手番ハイライト**: `current.yourIndex` のレーンを枠線・色で強調
- [x] HP 表示（`card.hp` / `card.maxHp`）
- [x] エネルギー・どうぐの個数表示（`card.energies` / `card.tools`）※画像化はフェーズ2/将来
- [x] 状態異常バッジ（`poisoned` / `burned` / `asleep` / `paralyzed` / `confused`）
- [x] **手札ハイライト修正**: 選んだ手札に `.highlight` クラスを付与（位置キー `area:player:index` 方式で実装）
- [x] Prev / Next / Autoplay / スライダーが新レイアウトでも動作

### 1-4. 完了条件（DoD）
- [x] `latest-self.json` / `latest-random.json` を開いて、要望画像に近い配置で全フレーム閲覧できる
- [x] 手番が変わっても上下が入れ替わらず、手番側がハイライトされる
- [x] サイドが6枠、手札ハイライトが機能する
- [x] データを足していないのに既存リプレイがそのまま見られる（後方互換 OK）

### 1-5. 追加実装（要望反映）
- [x] 相手の手札を公開/非公開トグル（デフォルト公開）
- [x] サイドの中身を公開/非公開トグル（デフォルト公開・OFF で裏面）

---

## フェーズ2: カード画像取得・表示（フェーズ1と並行可）

`pdf_card_editor` の抽出を再利用し、**静的プリダンプ方式**で画像を出す。
画像はあくまで**任意レイヤ**: 無くてもテキストチップにフォールバックする。

### 2-1. 抽出スクリプト（新規 `build_card_assets.py`）
- [x] `cardlist_referenced/pdf_card_editor` を `sys.path` に追加し `pdf_card_tool` を import
- [x] `load_pdf_catalog(data/Card_ID List_JP.pdf)` で `card_id → target_page` を構築
- [x] 対象 `card_id` の決め方: ①デッキ + リプレイ内 card_id に限定（軽い） or ②全カード（重いが汎用）→ **①を採用**（`--all` で②も可）
- [x] `get_card_asset_map(pdf, target_pages)` で抽出 → `web/card_images/<card_id>.<ext>` へ保存
- [x] PDF が無い環境では**エラーで落ちず警告して終了**（画像任意レイヤの担保）
- [x] ~~カード裏プレースホルダ画像 `_back.png`~~ → CSS（`.card-back`）で実装済みのため画像不要
- [x] 実行は `pdf_tool_requirements.txt`（pdfplumber / Pillow）の venv 前提。README に明記（→ 次タスク）

### 2-2. サーバ（`serve_viewer.py`）
- [x] `web/card_images/` を静的配信できることを確認（追加依存なし）
- [x] `/api/card-image/<card_id>` の薄いルートは不要（静的配信 + manifest.json で十分）

### 2-3. フロント（`web/app.js` + `styles.css`）
- [x] `renderCard()` を `<img src="./card_images/<file>">` 主体に変更（manifest にある card_id のみ）
- [x] 画像ロード失敗（`onerror`）時は現行のテキストチップへフォールバック（`.img-failed`）
- [x] 相手手札・サイドなど非公開枠は CSS のカード裏（`.card-back`）を表示
- [x] HP / エネ / どうぐのオーバーレイを画像の上に重ねて表示
- [x] `manifest.json` を boot で読み込み。0件なら画像トグルを無効化
- [x] 画像表示のオン/オフ トグル（デフォルト ON、画像が無ければ自動 OFF）

### 2-4. 完了条件（DoD）
- [x] `build_card_assets.py` 実行後、盤面に実カード画像が出る（frame 60 で26枚中25枚ロード確認、active=Mega Lucario ex）
- [x] PDF / 画像が無い環境でも、テキストチップで全フレーム閲覧できる（トグルOFF→画像0/テキスト26で確認）
- [x] `serve_viewer.py` の依存がゼロのまま
- [x] HP/エネのオーバーレイ表示を確認（例 `340/340⚡1`）
- [x] `?frame=N` でのフレーム直リンク（レビュー共有用）を追加

---

## フェーズ2.5: UI 追加改善（要望反映・`battle_review_viewer` 内で完結）

ユーザー要望の2点。確定済みの設計でロック。他メンバーのファイルには触れない。

### 2.5-1. 相手の場のカードを180°回転（向かい合わせ表示）
- [x] 対象は**相手の場のカードのみ**（バトル場＋ベンチ）。サイド（裏面）・デッキ/トラッシュ（数字）は対象外
- [x] 回転は **`.card-img` だけ**に適用（`transform: rotate(180deg)`）。チップ枠・オーバーレイは回さない
- [x] HP/エネのオーバーレイは**正立のまま**（読みやすさ優先）
- [x] 画像なしのテキストチップは回転しない（逆さ文字回避）。`▽ 相手` の向き目印を付与

### 2.5-2. ホバーでカード拡大表示（カーソル付近＋クリック固定）
- [x] 単一のフローティング要素 `#cardPreview` を追加（既定は非表示）
- [x] `.card-chip` への mouseover/mousemove で、大きな画像（常に正立）＋情報を表示
- [x] 情報: 名前 / id / HP(現在/最大) / エネ（カード名） / どうぐ / 進化元
- [x] カーソル付近に出し、画面端ではみ出さないようクランプ
- [x] **クリックで固定（ピン留め）**、同じカード再クリック or 別領域クリックで解除
- [x] ホバー時にカードデータを引けるよう、描画時に軽量レジストリへ登録（`data-card-key` 参照）

### 2.5-3. 完了条件（DoD）
- [x] 相手の場のカードだけ画像が180°回り（`matrix(-1,0,0,-1)`）、自分の場・サイド・カウントは正立のまま
- [x] 相手カードでも HP/エネのバッジは正立で読める（overlay transform=none を確認）
- [x] 任意のカードにホバーで拡大表示、クリックで固定/解除できる（hover/pin/unpin を検証）
- [x] 画像が無い環境でもテキスト情報の拡大表示が出る（フォールバック）

---

## フェーズ2.7: UI 日英切り替え（i18n）

ユーザー要望: **選択肢・選んだ行動・コンテキスト等が英語のみで読みにくい → 日本語化し、日英トグルボタンを上部に追加する。**
ログ（エンジン生ログ）は翻訳対象外。

- [x] `index.html` にヒーロー部へ `EN` / `JP` トグルボタン（`#langToggleButton`）を追加
- [x] 静的テキスト（ボタン・ラベル・見出し等）に `id` を付与し、JS 側で `applyLang()` が差し替える
- [x] `app.js` に言語状態 `lang`（`'ja'` | `'en'`）と `UI` 翻訳辞書を追加
- [x] `t(key)` 関数で翻訳を引く
- [x] `SelectContext` の整数値 0〜48 を `CONTEXT_MAP` で日英翻訳（`ctxName(ctx)` 関数）
  - [x] **バグ修正**: `frame.context` が `0`（MAIN）の時 `0 || "Terminal"` で `"Terminal"` と誤表示されていた問題を `!= null` チェックに修正
- [x] `describeOptionLabel(option, frame)` 関数で選択肢ラベルを日英生成
  - [x] `raw.type` から再構築（Yes/No/Number/Attack/Play/Card/Ability/Attach/Evolve/Energy/EnergyCard/ToolCard/Retreat/End）
  - [x] JP モード時はカード名（ゲームエンジン由来の日本語）はそのまま使い、英語 prefix だけ置換
  - [x] 選択肢・選んだ行動パネル両方に適用
- [x] サマリーのキー名（replay/opponent/…）を日英で切り替え
- [x] `applyLang()` が言語切り替え時に DOM 全体を一括更新 → `render()` 再描画
- [x] ログ（`renderLogs`）の空文字列メッセージも日英対応
- [x] `styles.css` に `.lang-toggle` スタイルを追加

**完了条件（DoD）**:
- [x] 既定 JP で全 UI・選択肢・コンテキスト名が日本語表示される
- [x] `EN` ボタンで英語に切り替わり、`JP` ボタンで戻る
- [x] ログ（エンジン生ログ）は翻訳せず生データのまま表示
- [x] 既存リプレイ JSON を一切変更しない（フロント完結）

---

## フェーズ2.6: カード移動アニメーション（要望反映・`battle_review_viewer` 内で完結）

ユーザー要望: **「カードをどこから取ったのか / どこに追加されたのか / 捨てたのか / どこへ行ったのか」がアニメーションで分かるようにする。**
現状はフレームが切り替わるだけで移動の経路が見えず、何が起きたか追えない。

**設計の柱（後方互換・フロント完結）**:
- データ追加は**不要**。連続する2フレームの盤面スナップショット差分から移動を導出する（`frame.visual.current.players[*]`）。
- `frame.visual.logs`（エンジンのイベント履歴）があれば**優先利用**し、無ければフレーム差分にフォールバック。
- 演出は**任意レイヤ**: アニメ OFF / `prefers-reduced-motion` でも、従来どおり即時切替で全フレーム閲覧できること。

### 2.6-1. 移動の検出（差分エンジン）
- [x] `serial` キーで前フレーム→現フレームのカード対応を取る（`collectVisibleCards` + `diffFrames`）
- [x] 検出する移動種別を列挙:
  - [x] ドロー（山札から hand へ新規出現）
  - [x] プレイ / 展開（`hand` → `active` / `bench` / `stadium`）
  - [x] トラッシュ（`hand` / `active` / `bench` → `discard`）
  - [x] エネ・どうぐ付与（`hand` → `active_energy` / `bench_energy` / `active_tool` / `bench_tool`）
  - [x] 進化（`hand` → `active`/`bench` として play_active/play_bench で検出）
  - [x] きぜつ（`active` / `bench` → `discard`）
  - [x] にげる / 入れ替え（`active` ⇄ `bench`）
  - [x] サーチ / 回収（`discard` / `prize` → `hand`）
  - [x] サイドを取る（`prize` → `hand`）
  - [x] 山札枚数変化（deckCount 差分）
- [x] 1フレームで複数移動が起きるケースを配列で返す（順序付き）
- [x] 検出できない/曖昧な差分は**握りつぶして無演出**（誤演出より無演出を優先）

### 2.6-2. 演出（`web/app.js` + `styles.css`）
- [x] 見えるゾーン間（hand/active/bench/stadium）の移動は **FLIP 方式**（座標差を `transform` で補間、CSS transition で飛ぶ）
- [x] 枚数だけのゾーン（`deck` / `discard`）は**カウントのパルスアニメ**で表現
- [x] 移動種別ごとに**色コード**（ドロー=青 / 展開=緑 / エネ=橙 / トラッシュ=灰 / きぜつ=赤 / にげる=黄 / サイド=橙赤）
- [x] 新規に現れたカードは**出現アニメ**（フェードイン＋スケール `card-arrive`）
- [ ] 消えるカードの退場アニメ（ハイライト残光）※未実装・任意
- [x] アニメ中も HP/エネ/状態異常オーバーレイの整合が崩れないこと

### 2.6-2b. ゾーン視覚化と fly ゴースト演出（追加要望）
- [x] **山札ゾーン**を裏向きカードパイル（青の斜めストライプ＋スタック影）で視覚化（`#selfDeckZone` / `#opponentDeckZone`）
- [x] **トラッシュゾーン**を点線パイル枠で視覚化（着地点が分かる）
- [x] **トラッシュ行きアニメ**: カードが盤面位置 → トラッシュゾーンへ表向きで吸い込まれフェード（`spawnFlyGhost`、フレーム5で7枚同時確認）
- [x] **山札戻しアニメ**: 消えたカード（`vanish`）+ deckCount 増を突き合わせ、裏向きで山札ゾーンへ飛ばす（コードパス実装済み）
- [x] **シャッフルアニメ**: `frame.visual.logs` の `Shuffle` を検出し、該当プレイヤーの山札ゾーンを振動（`deck-shuffle`、フレーム8で opponentDeckZone 確認）
- [x] **エネ付与アニメ**: エネカードを手札 → 対象ポケモンへ飛ばし（`fly-energy` 丸型グロー）、対象ポケモンを `attach-glow` で発光（どのポケモンか明示、フレーム20で確認）
- [x] どうぐ付与も同様（`fly-tool` + `attach-glow-tool`）
- [x] fly ゴーストは `position: fixed` の一時オーバーレイ、`ANIM_MS + 150ms` 後に自動削除

### 2.6-2c. ドロー飛行・シャッフルリフル（追加要望）
- [x] **ドローアニメ**: 山札ゾーン → 手札のカード位置へ、小さく（`fromScale 0.34`）出発し実寸で着地するゴーストを飛ばす。飛行中は実カードを一時 `visibility:hidden`、着地時に復帰＋`card-arrive`（フレーム1で14枚のドロー飛行を確認、着地後の残留非表示=0）
- [x] 相手の手札が非公開（トグルOFF）のドローは裏向きで飛ばし中身を明かさない
- [x] `spawnFlyGhost` を拡張: `fromScale` / `sizeFromTo`（移動先サイズで着地）/ `duration` / `onArrive`
- [x] **シャッフルリフル演出**: 山札ゾーンの振動に加え、6枚の裏向き小カードが左右に分割→交互に戻る（`shuffle-fx` / `shuffle-card` sc-left/sc-right, 0.32s×2）＋「🔀 シャッフル」バッジをポップ（フレーム8で6枚のリフル確認）

### 2.6-2d. ライブモード版 app.js へアニメ再統合（マージ復旧）
- [x] `live_match` 系マージで working-tree の `app.js` からアニメ JS が欠落していた（index.html/styles.css の足場は残存）。HEAD 版から差分エンジン＋fly 演出一式を移植し、ライブモードの `stepFrame` / `render` へ再配線
- [x] `hpColor` 等の既存シンボルと重複しないよう調整、`renderMoveSummary` をリッチ版へ差し替え、`toggleAnimation` を `animEnabled` フラグへ接続

### 2.6-3. 再生コントロールとの統合
- [x] **Next**: 差分を順再生（FLIP アニメ付き）
- [x] **Prev / スライダー**: 即時描画（混乱しない）
- [x] スライダーで大きく飛んだ時はアニメをスキップして即時描画
- [x] アニメ ON/OFF トグルを追加（ツールバー「アニメーション」チェックボックス）
- [ ] 速度（1x / 2x / 即時）の選択 ※未実装・低優先度
- [ ] `prefers-reduced-motion: reduce` の尊重 ※未実装

### 2.6-4. 補助表示（任意・あると親切）
- [x] そのフレームで起きた移動を**テキストで併記**（インスペクタ「このフレームの動き」セクション）
  - 形式: 移動種別バッジ / カード名（JP）/ プレイヤー（自分・相手）
- [ ] インスペクタの「選んだ手」と移動アニメを対応付け ※未実装

### 2.6-5. 完了条件（DoD）
- [x] ドロー / プレイ / トラッシュ / エネ付け / 進化 / きぜつ / にげる / サイド取得が、移動の種類としてテキスト＋FLIP アニメで判別できる（フレーム 17/21/25/40/44/45/46 で動作確認済み）
- [x] アニメ OFF でも全フレームが従来どおり閲覧でき、画面が壊れない
- [x] スライダーで飛んでも誤ったアニメが暴発しない
- [x] データ（JSON）を一切足していない（既存リプレイがそのまま動く＝後方互換 OK）

---

## フェーズ3: 「なぜその行動をしたか」（チーム差分に強い設計が肝）

### 3-0. 設計方針（最重要・チーム開発前提）

`sample_submission/main.py` 配下の関数構成は**メンバーごとに違う可能性がある**。
そこで表示を**2層**に分け、**B層が無くても A層だけで成立**させる。これが「他の人の実装に影響を与えない / 場合分けする」要件への答え。

| 層 | 内容 | 出る条件 | 他人の実装への影響 |
|----|------|----------|--------------------|
| **A. エンジン真実層**（常時） | context の日本語説明 + 合法選択肢 + 実際に選んだ option | **どのエージェントでも常時** | ゼロ（`describe_option` は engine 出力のみ依存） |
| **B. エージェント理由層**（任意） | 採用/不採用 proposal の label・score・補足 | **trace を出した時だけ** | ゼロ（出さなければ自動的に非表示） |

- B層は **プロセス内シンク**（`src/decision/trace.py`）に「書いてくれたら拾う」方式。
  - 書かないエージェント（別メンバー実装・他人の `main.py`・`random_agent`）→ **B層は "理由(詳細): このエージェントは未提供" と出すだけ**。画面は壊れない。
  - 提出時は誰も enable しない → no-op、性能・挙動への影響なし。
- 「使ってもいいが、構成が違えば行動原理までは出さない」という要件は、この**A層は必ず出る / B層は出せる人だけ**で自然に満たせる。

### 3-1. A層の強化（まず低コストで全エージェント対応）
- [x] `context` → 日本語説明の辞書を用意（i18n の `CONTEXT_MAP` / `ctxName`。"Main"→「メイン」等）
- [x] フロントのインスペクタに「いまの局面（context 日本語）」（`contextBadge`）+「選んだ手（option ラベル）」（`renderAction`）を常時表示
- [x] これだけで「**main.py を知らない人でも最低限どう動くか**」が見える（フェーズ2.7 i18n で達成済み）

### 3-2. B層: 決定トレースのシンク（新規 `src/decision/trace.py`）
- [x] 既定オフのモジュール内シングルトン（`runtime_state.py` と同じ流儀）
- [x] API: `enable_trace()` / `disable_trace()` / `record_decision(context, chosen_label, chosen_score, alternatives, notes)` / `pop()`
- [x] 探索ロールアウトの再帰記録を防ぐ `pause()` / `resume()`（ネスト可）を追加
- [x] **import 失敗・未使用でもエージェントが動く**（既定オフで完全 no-op、独立性を担保）
- [x] `record_decision` は不正入力・str化例外も握りつぶす
- [x] 単体テスト `test_trace.py`（no-op / 記録 / pop クリア / 各種 alternatives 形状 / 例外握りつぶし / pause-resume）計9件 green

### 3-3. B層: 既存ロジックからの配線（出せる所から）
- [x] `main_turn.choose_main_action` で、採用 `MainActionProposal`（label/score）と不採用候補（label/score 一覧）を `record_decision` へ流す
- [x] proposal が空の fallback 局面も記録（`合法手フォールバック` + notes）
- [x] **探索(ISMCTS)呼び出しを `trace.pause()`/`resume()` で囲む**（ロールアウトが `choose_main_action` を再帰呼び出しし、シミュレーション上の決定まで記録される爆発を防止）
- [ ] 他ハンドラ（attack / card_move / switch …）は**段階的に**対応（未対応でも A層が出るので破綻しない）※本フェーズは MAIN のみ
- [x] `_record_main_decision` は例外を握りつぶす（理由収集が原因でエージェントを止めない）

### 3-4. 出力側（`export_replay.py`）
- [x] `enable_trace()` してから `run_match`（=`agent()`）を実行
- [x] `agent()` 直後に `trace.pop()` し、そのフレームの `frame.trace` として付与
- [x] trace が空なら `frame.trace` を付けない（optional 維持）

### 3-5. フロント表示（`web/app.js` + `styles.css`）
- [x] A層は常時表示（contextBadge + chosenAction）
- [x] `frame.trace` があれば B層を新設「行動の理由」カードに表示: 採用手（`採用` バッジで強調）+ 不採用候補を score 順、`notes` も
- [x] proposal 内部ラベルを日本語化（`PROPOSAL_LABELS` / `reasonLabel`、未知は生ラベルへフォールバック）
- [x] `frame.trace` が無ければ「理由(詳細): このエージェントは未提供です。」とだけ出す（場合分け）
- [x] 日英トグル対応（見出し・バッジ・ラベル）

### 3-6. 完了条件（DoD）
- [x] `main.py`（標準エージェント）のリプレイで、MAIN フェーズの採用/不採用理由が日本語で出る（`replay-trace-test.json` フレーム4: 採用「グッズでサーチ」55 / 不採用「エネルギーをつける」44・「ターンを終了」5 を確認）
- [x] trace 非対応リプレイ（既存の random 対戦）でも、A層（局面 + 選んだ手）が出て画面が壊れない（B層は「未提供」表示）
- [x] `trace.py` を使わない決定コードでも、エージェントが正常動作（既定オフ no-op、全 sample_submission テスト green）
- [x] 提出パス（trace を enable しない）で挙動・性能に影響がない（`enable_trace()` は export のみ、通常は完全 no-op）

---

## フェーズ4: 相手推定の可視化（将来 / inference 統合後）

エージェント本体に手を入れず、**出力時スナップショット**で推定値を残す。

### 4-1. 出力（`export_replay.py`）
- [ ] `--inference` フラグを追加
- [ ] フレーム生成時に `get_runtime_inference(recipe).observe(obs)` を呼ぶ
- [ ] `last_opponent_estimate` / `last_self_estimate` / `last_ace_estimate` / `last_threat_damage` を `frame.inference` へ格納（optional）

### 4-2. フロント（インスペクタに「推定」タブ）
- [ ] 相手アーキタイプ事後分布を横棒バー（`ranked()` 上位 + OTHER）
- [ ] `confidence` と根拠カード（`seenKeyCards`）を併記
- [ ] サイド/山札推定: `prizeKnownRatio` ゲージ + キーカードの `probability_in_deck`
- [ ] フレーム送りで推定が更新されていく様子が追える

### 4-3. 完了条件（DoD）
- [ ] `--inference` 付き出力で推定タブが表示される
- [ ] `--inference` 無しの既存リプレイは従来どおり見られる（後方互換）
- [ ] （統合後）理由(trace) と推定を並べ、「この推定だからこの行動」と因果が読める

---

## 着手前に合意したい論点（再掲・[`IMPROVEMENT_PLAN.md` §7](IMPROVEMENT_PLAN.md)）

- [ ] レビュー視点は「Player 0 固定で下」で良いか
- [ ] 相手の手札・サイドはレビューなので全公開で良いか（裏面トグルを付けるか）
- [ ] カード画像 PDF 非保持環境の扱い（= 画像は任意レイヤで確定）
- [ ] Time（持ち時間）は現状データに無いので当面非表示で良いか
