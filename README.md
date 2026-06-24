# Pokemon TCG AI Battle

[Kaggle コンペティション](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview) に向けたチーム開発リポジトリです。

ポケモンカードゲームを自律的にプレイする AI エージェントを実装し、提出します。

---

## フォルダ構成

```
pokemon-tcg-ai-battle/
├── data/                   # コンペ提供データ (カード一覧 PDF・CSV)
├── sample_submission/      # コンペ提出コードのベース
└── cardlist_referenced/    # カードリスト参照・印刷ツール (提出とは独立)
```

### `data/`

コンペが提供するカードデータです。デッキ構築時の参照に使用します。
ファイルサイズが大きいため本リポジトリには含まれていません。

**Kaggle のコンペページからダウンロードして `data/` に配置してください。**
[https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data)

| ファイル | 内容 |
|----------|------|
| `Card_ID List_EN.pdf` | カード一覧 (英語版) |
| `Card_ID List_JP.pdf` | カード一覧 (日本語版) |
| `EN_Card_Data.csv` | カードデータ (英語) |
| `JP_Card_Data.csv` | カードデータ (日本語) |

### `sample_submission/`

コンペが提供するサンプル提出コードです。AI エージェントの実装はここを起点にします。

- `main.py` — エージェントのエントリポイント。`agent(obs_dict)` 関数を実装する
- `deck.csv` — 使用するデッキ (カード ID を 60 枚分記載)
- `cg/` — コンペ提供のゲームエンジン (変更不要)

詳細は [sample_submission/README.md](sample_submission/README.md) を参照してください。シミュレーター API ドキュメントは [cabt API docs](https://matsuoinstitute.github.io/cabt/)、公式ルールとの差分メモは [Kaggle Discussion 708586](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586) です。

### `cardlist_referenced/`

カードリストの参照・印刷を補助するローカルツール群です。提出物とは無関係です。

- カード PDF の閲覧・絞り込み・印刷用 PDF 生成ができる Streamlit アプリを含みます

詳細は [cardlist_referenced/README.md](cardlist_referenced/README.md) を参照してください。

---

## クイックスタート

### 提出エージェントを編集する

```powershell
cd sample_submission
```

`main.py` が提出エージェントの本体です。`agent(obs_dict)` に行動ロジックを書き、`deck.csv` に使用する 60 枚のカード ID を記載してください。

### カードリストツールを起動する

```powershell
cd cardlist_referenced/pdf_card_editor
pip install -r pdf_tool_requirements.txt
streamlit run app.py
```

---

## コンペ概要

| 項目 | 内容 |
|------|------|
| コンペ | [Pokemon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview) |
| 形式 | AI エージェントによるポケモンカードゲームの自律対戦 |
| 提出物 | `agent(obs_dict)` 関数を含む Python ファイル + `deck.csv` |
| デッキ枚数 | 60 枚 |
