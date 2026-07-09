# tests

`sample_submission/tests/` は、ローカル開発で使う検証コード置き場です。

```text
tests/
├── unit/         # 純粋関数・小さいロジック
├── integration/  # 複数モジュールをまたぐ検証
└── local_sim/    # 手動実行のローカル対戦スクリプト
```

- `unit/` は副作用の少ない関数や小さな判定ロジックを確認します。
- `integration/` は複数モジュールをまたぐ結合動作を確認します。
- `local_sim/` は `cg` を使って実際に対戦を回す手動確認用です。
