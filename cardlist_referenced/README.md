# cardlist_referenced/

カードリストの参照・整理・印刷を補助するツール群です。コンペへの提出とは無関係です。

チームメンバーがデッキ構築時にカードを探したり、実際に印刷する際に使用します。

## フォルダ構成

```
cardlist_referenced/
└── pdf_card_editor/                  # カード PDF 管理ツール (Streamlit アプリ)
```

参照元のカード PDF / CSV は、このフォルダではなくリポジトリ直下の `data/` を使います。

```text
data/
├── Card_ID List_JP.pdf
├── Card_ID List_EN.pdf
├── JP_Card_Data.csv
└── EN_Card_Data.csv
```

## pdf_card_editor

ポケモンカード PDF を閲覧・絞り込み・印刷用に管理する Streamlit アプリです。

既定では `../data/Card_ID List_JP.pdf` を自動で探して開きます。  
過去の `cardlist_referenced/` 直下の PDF も候補として読めますが、GitHub 上では `data/` に置くだけで動く前提にそろえています。

### 主な機能

- カード一覧の検索・絞り込み・ラベル管理
- カードのホバー/クリックプレビュー
- 作業リストの保存と再開
- 印刷用 PDF (A4 面付け、実寸 63×88mm) の生成

### 起動方法

```powershell
cd pdf_card_editor
pip install -r pdf_tool_requirements.txt
streamlit run app.py
```

または起動補助ファイルをダブルクリックします。

- `launch_pdf_card_editor.cmd`
- `launch_pdf_card_editor.ps1`
- `launch_pdf_card_editor.vbs`

詳細は [pdf_card_editor/README-pdf-tool.md](pdf_card_editor/README-pdf-tool.md) を参照してください。
