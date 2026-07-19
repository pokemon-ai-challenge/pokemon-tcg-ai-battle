# Click To Start

Windows では次の `.cmd` を **ダブルクリック**するだけで起動できます。
サーバーが立ち上がり、ブラウザが自動で開きます。

- `Launch Replay Viewer.cmd`
  - viewer server を起動し、browser を replay（閲覧）モードで開きます。
- `Launch Human vs CPU.cmd`
  - viewer server を起動し、browser を Human vs CPU（対戦）モードで開きます。

## 起動後にできること

- **replay の生成もブラウザから**できます（コマンド不要）。
  画面右上の `⚙`（Settings）タブ → `Generate replay` で相手とシードを選び `Generate & open`。
- 盤面以外の情報（サマリー / 選択肢 / ログ / Debug）は上部タブから開きます。

## 補足

- server は別 console window で起動します。**止めるときはその console window を閉じてください。**
- `py` launcher か `python` が PATH に入っている必要があります。
- コマンドで直接起動しても自動でブラウザが開きます:
  `python .\battle_review_viewer\serve_viewer.py`
  （自動で開きたくない場合は環境変数 `VIEWER_NO_OPEN=1` を設定）
