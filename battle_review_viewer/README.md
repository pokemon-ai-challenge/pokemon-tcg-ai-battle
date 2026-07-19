# Battle Review Viewer

`sample_submission/main.py` のローカル対戦を、あとから人間が見返しやすくするための replay viewer です。

このフォルダは提出コードとは独立です。  
`sample_submission/` はそのままにして、レビュー用の JSON と Web viewer を repo 直下で管理します。

## できること

- ローカル対戦 1 試合を replay JSON として保存し、あとから 1 手ずつレビュー
- ダークな盤面 UI（バトル場が前・ベンチが後ろ、相手＝赤／自分＝青のグロー、サイドはモンスターボール表示）
- **replay の生成をコマンドではなく画面のボタンから実行できる**（`⚙ Settings → Generate replay`）
- 盤面以外の情報（サマリー / 選択肢 / ログ / Opponent Knowledge など）は
  上部タブから「必要な時だけ」1 パネルずつ大きく開くタブ式ドロワー（盤面と重ならない）

## クイックスタート（クリックで起動）

Windows なら `.cmd` を **ダブルクリック**するだけで、サーバー起動＋ブラウザが自動で開きます。

- `Launch Replay Viewer.cmd` … リプレイ閲覧モードで開く
- `Launch Human vs CPU.cmd` … 人間 vs CPU の対戦モードで開く

止めるときは、別ウィンドウで開いているサーバーの console を閉じてください。
（`py` ランチャか `python` が PATH に必要です。詳細は [`CLICK_TO_START.md`](CLICK_TO_START.md)）

コマンドで起動したい場合も、起動すると自動でブラウザが開きます:

```powershell
python .\battle_review_viewer\serve_viewer.py   # → http://127.0.0.1:8765 が自動で開く
```

> 自動でブラウザを開きたくない場合は環境変数 `VIEWER_NO_OPEN=1` を設定してください。

## フォルダ構成

```text
battle_review_viewer/
├─ Launch Replay Viewer.cmd     # クリック起動（リプレイ）
├─ Launch Human vs CPU.cmd      # クリック起動（対戦）
├─ export_replay.py             # replay 生成（CLI）
├─ serve_viewer.py              # ローカルサーバー（UI からの生成 API を含む）
├─ ml_prediction_debug.py       # 相手デッキ予測器(ML)の推論結果を各フレームに埋め込む
├─ build_archetype_names.py     # rough_predictor.json → web/archetype_display_names.json(日本語表示名)を生成
├─ replays/                     # 生成された replay JSON 置き場
└─ web/                         # index.html / app.js / styles.css
```

## replay を作る

### A. 画面から作る（コマンド不要・おすすめ）

1. viewer を起動する（上記クイックスタート）
2. 上部タブの `⚙`（Settings）を開く
3. `Generate replay` で相手（Random / main.py CPU）とシードを選び、`Generate & open` を押す
4. 1 試合ぶんを裏で実行し、完了すると新しい replay が一覧に追加されて自動で開きます
   （1 試合の生成には数十秒かかることがあります）

内部的には viewer サーバーが `export_replay.py` を別プロセスで実行します
（`POST /api/replays/generate`）。使用デッキは `sample_submission/deck.csv` です。

### B. コマンドから作る

```powershell
python .\battle_review_viewer\export_replay.py --opponent random
```

出力先を固定したい場合:

```powershell
python .\battle_review_viewer\export_replay.py --opponent self --output .\battle_review_viewer\replays\latest-self.json
```

## 見方

- 上部バー: モード / リプレイ選択 / 再生コントロール（前へ・自動再生・次へ・速度・スライダー）
- 右上タブ: `Info` `Action` `Options` `Logs` `Debug` `⚙` — 押すと右からドロワーが開き、
  盤面が左に縮んで重ならないように並びます。もう一度押すと閉じます。
- **初回起動時**はガイドツアーが始まります（各ボタンをスポットライトして「これは何をするか」を
  1つずつ説明し、`次へ` で進める形）。右上の `?` ボタンでいつでも再実行できます
  （既読は `localStorage` に記録され、次回以降は自動では出ません）。
- 特定フレームを直接開きたいときは URL に `?frame=N` を付けます（レビュー共有用）:
  `http://127.0.0.1:8765/?frame=60`

## カード画像を用意する（任意）

盤面に実際のカード画像を出したい場合は、`data/Card_ID List_JP.pdf` から画像を抽出して
`web/card_images/` に書き出します。抽出には pdfplumber / Pillow が必要なので、
`cardlist_referenced/pdf_card_editor` の venv で実行します。

```powershell
& .\cardlist_referenced\pdf_card_editor\.venv\Scripts\python.exe `
    .\battle_review_viewer\build_card_assets.py --deck
```

- 既定では `replays/*.json` とデッキ（`--deck`）に登場する card_id だけを抽出します。
- 全カードを抽出したいときは `--all` を付けます。
- PDF が無い環境では何もせず終了します（画像は任意レイヤ。無くてもテキスト表示で動作）。
- `⚙ Settings` の「Card Images」トグルで画像/テキストを切り替えられます。

> **カード画像は card_id で汎用的に表示されます**（特定デッキ専用ではありません）。
> 表示ロジックは `manifest.json`（`{card_id: ファイル名}`）を引き、`card_images/<id>.jpg` があれば
> その画像を、無ければ**カード名＋HP のテキストにフォールバック**します。つまり
> **PDF に載っているカードなら、抽出済みでありさえすればどのデッキのカードでも正しく画像が出ます**。
> 現状は登場済みデッキ分（約 40 枚）のみ抽出済み。全カードをカバーしたいときは `--all` で
> 一括抽出してください（未抽出のカードはテキスト表示になるだけで、壊れることはありません）。

## Opponent Knowledge デバッグ機能

> Optional: this debug layer is enabled only when
> `sample_submission/ptcg_ai/opponent_modeling` exists. On a viewer-only branch,
> replay export and live match still work; `opponentKnowledgeDebug` remains `null`.

`export_replay.py` は player0（`sample_submission/main.py` のエージェント）視点で
`sample_submission/ptcg_ai/opponent_modeling/opponent_knowledge.py` の `OpponentKnowledge` を動かし、
各フレームに以下を埋め込みます（`opponentKnowledgeDebug` キー。player1 の手番のフレームは `null`）。

- `features`: `get_prediction_features()` のスナップショット
- `diff`: 同じ瞬間の神視点(`visualize_data()`)から独立に再集計した「相手の公開ゾーン」との
  自動突き合わせ結果（`missing`/`extra`/`mismatched`。3つとも空なら完全一致）

突き合わせロジックは `opponent_knowledge_diff.py` にあります。

非公開ゾーン（山札・手札・サイド）版の検証ツールは `hidden_info_diff.py`（手動実行のレポートツール、
自動テスト対象外）。`sample_submission/ptcg_ai/hidden_information/` の `OpponentHiddenState.marginals()`
が出す確率と、`cg.game.visualize_data()` の神視点（未取得のサイドカードの中身まで含む）を突き合わせて
reliability/ECE を測る。

```powershell
python .\battle_review_viewer\hidden_info_diff.py --matches 5
```

出力: `battle_review_viewer/output/hidden_info_diff_report.md`。

画面上部のタブ `Debug` を押すと右からドロワーが開き、観測特徴量と diff 結果が表示されます
（盤面以外の情報は「必要な時だけ」1パネルずつ大きく見せるタブ式ドロワー方式）。

### Debug のサブビュー（拡張可能）

`Debug` パネル内は**サブタブで複数のデバッグビューを切り替えられます**。現状は
`Opponent Knowledge`（観測情報）と `Deck Predictor`（相手デッキ予測器・ML版、実装済み）。

`Deck Predictor` サブタブは `sample_submission/ptcg_ai/opponent_modeling/hybrid_predictor.py`
（LR×NBハイブリッド）の推論結果を `prediction_summary.summarize_prediction()` 経由で表示する。
確信度が閾値（既定0.6）未満のときは「未確定」バッジと候補一覧、閾値以上のときは1位デッキを
断定表示し、いずれもカード別の根拠（evidence）を展開できる。生成は `export_replay.py` /
`live_match.py` が `ml_prediction_debug.py`（`opponentKnowledgeDebug` と同様に各フレームへ
`mlPredictionDebug` キーを埋め込む）を経由して行う。日本語の表示名は
`web/archetype_display_names.json`（生成: `build_archetype_names.py`、`rough_predictor.json` の
アーキタイプキーから変換）を使う。モデルの学習・評価は `kaggle_replays/deck_predictor/`
（パイプラインの実行方法は同ディレクトリの README.md を参照）。

`Hidden Information` サブタブは `sample_submission/ptcg_ai/hidden_information/`（`OwnHiddenState` /
`OpponentHiddenState`）の推定結果を表示する。自分側は「サイド落ち候補 top N」（超幾何分布による
山札/サイドの周辺確率）、相手側は「手札候補 top N」（`HybridDeckPredictor` のアーキタイプ事後分布
×ゾーン配分の混合モデルによる山札/手札/サイドの周辺確率）をテーブル表示する。相手側の代表リスト
（`archetype_card_pool.json`）が未配置の場合は「プール未配置」と表示する。生成は `export_replay.py` /
`live_match.py` が `hidden_info_debug.py`（`opponentKnowledgeDebug` の中に `hidden_info` キーとして
`ml_prediction` と同居させる）を経由して行う。`match_context.py`（提出エージェント本番用の
プロセス全体シングルトン）はここでは使わず、`OpponentKnowledge` と同様に replay/live セッション内で
独立したインスタンスを構築する（`--opponent self` では両プレイヤーとも `main.agent` を使うため、
シングルトンを流用すると player0/player1 が交互に上書きし合ってしまうため）。

デバッグ項目を増やすときは 2 箇所を足すだけです:

1. `web/app.js` の `DEBUG_VIEWS` 配列に `{ id, label }` を 1 行追加
2. `web/index.html` の `.debug-views` 内に対応する `<div class="debug-view" data-debug-view="<id>">…</div>` を追加

（描画が必要なビューは `render()` からそのビューの描画関数を呼ぶ。静的な内容ならそのままでよい）

この仕組みは「実際の cg エンジンで正しく観測できているか」を、目視ではなく自動 diff で
検証するためのものです。手組みフィクスチャの単体テスト（`sample_submission/tests/unit/`）だけでは
拾えない実データ特有の想定漏れ（ログの意味の取り違え、呼び出し順序に依存する不具合など）を
ここで発見・修正済みです。詳細は
[opponent-knowledge-plan.md §7.5](../sample_submission/docs/plans/opponent-deck-predictor/opponent-knowledge-plan.md)
を参照してください。

- **Human vs CPU（live）モードには未対応**: `live_match.py` では CPU が player1（player0 が人間）
  という逆の座席配置になっており、`OpponentKnowledge` 側の「player0 視点」前提と合わないため、
  今回は replay モードのみに接続しています。

## メモ

- replay 生成時は `sample_submission/deck.csv` を使います
- `visualize_data()` の末尾フレームを読み、各 decision point を保存しています
- 最初は「レビューしやすさ」優先なので、Kaggle 完全再現ではなく盤面と選択の把握を重視しています

## 今後の改善方針

実機リプレイ風レイアウト・カード画像表示・判断理由（日本語）・相手推定の可視化など、
今後の改善計画は [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) にまとめています。

実際に作る順番つきの作業チェックリストは [`IMPLEMENTATION_CHECKLIST.md`](IMPLEMENTATION_CHECKLIST.md) を参照してください。
