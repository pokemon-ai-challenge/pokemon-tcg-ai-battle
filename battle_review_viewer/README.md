# Battle Review Viewer

`sample_submission/main.py` のローカル対戦を、あとから人間が見返しやすくするための replay viewer です。

このフォルダは提出コードとは独立です。  
`sample_submission/` はそのままにして、レビュー用の JSON と Web viewer を repo 直下で管理します。

## できること

- ローカル対戦 1 試合を replay JSON として保存
- 各手番の盤面、手札、ベンチ、捨て札、山札枚数を表示
- `context`、選択肢一覧、実際に選んだ action を人間向けに確認
- `Prev` / `Next` で 1 手ずつレビュー

## フォルダ構成

```text
battle_review_viewer/
├─ export_replay.py
├─ serve_viewer.py
├─ replays/
└─ web/
   ├─ index.html
   ├─ app.js
   └─ styles.css
```

## 1. replay を作る

repo ルートで実行します。

```powershell
python .\battle_review_viewer\export_replay.py --opponent random
```

出力先を固定したい場合:

```powershell
python .\battle_review_viewer\export_replay.py --opponent self --output .\battle_review_viewer\replays\latest-self.json
```

## 2. viewer を起動する

```powershell
python .\battle_review_viewer\serve_viewer.py
```

表示 URL:

```text
http://127.0.0.1:8765
```

## メモ

- replay 生成時は `sample_submission/deck.csv` を使います
- `visualize_data()` の末尾フレームを読み、各 decision point を保存しています
- 最初は「レビューしやすさ」優先なので、Kaggle 完全再現ではなく盤面と選択の把握を重視しています
