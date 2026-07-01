# Battle Review Viewer

ローカル対戦を **replay JSON** として保存し、**芝色の盤面**で1手ずつ見返すためのビューアです。
`sample_submission/` とは独立（提出コードには影響しません）。

特徴（このリビルド版）:
- **全カードを実カード画像で表示**（`data/Card_ID List_JP.pdf` から `card_id` 単位で切り出し・キャッシュ）
- **選択肢一覧は固定サイズ**（数が多くてもパネルは伸縮せず、内部スクロール）
- **芝色の盤面**（相手＝上／自分＝下、手番側をハイライト、視点は Player 0 固定）
- 前へ/次へ/自動再生・フレームスライダー、カード画像/相手の手札/サイド公開トグル、JP⇔EN、ホバーで拡大

## 使い方

### 1. リプレイを作る（repo ルートで）
```powershell
python battle_review_viewer/export_replay.py --opponent random
python battle_review_viewer/export_replay.py --opponent self --seed 3
```
`sample_submission/main.py`（＝mPPO エージェント）が Player 0、相手が random or self。
`battle_review_viewer/replays/replay-<日時>-<相手>.json` に保存されます。

### 2. ビューアを起動
```powershell
python battle_review_viewer/serve_viewer.py        # http://localhost:8010
```
ブラウザで <http://localhost:8010> を開き、右上の「リプレイ」で選択（最新が自動選択）。

## 構成
```text
battle_review_viewer/
├─ export_replay.py   # 対戦を1試合実行し replay JSON を保存（visualize_data のgod-view）
├─ serve_viewer.py    # 静的配信 + /api/replays,/api/replay,/api/cards,/cards/<id>
├─ card_images.py     # card_id → PDF から切り出したカード画像（pdf_card_editor を再利用）
├─ replays/           # 生成された replay JSON
└─ web/               # index.html / styles.css / app.js（ビルド不要）
```

## メモ
- カード画像は初回アクセス時に PDF から抽出してキャッシュします（2回目以降は高速）。
- `data/Card_ID List_JP.pdf` が無い環境ではカードは**テキスト表示**にフォールバックします。
- 盤面データはエンジンの god-view（`cg.game.visualize_data()`）。相手の手札・サイドも公開でき、トグルで隠せます。
