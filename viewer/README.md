# ポケカ対戦ビジュアライザ

Pokemon TCG のゲームエンジン（`sample_submission/cg/`）を使って、対戦を **ブラウザで視覚的に**
観戦・プレイするローカル Web アプリです。

- **AI 同士の対戦を観戦**：AI とデッキを 2 つ選んで対戦を実行し、リプレイを再生
- **人間 vs AI で対戦**：自分で選択肢を選んで AI と対戦
- **AI が取れる選択肢を表示**：各意思決定での合法手を日本語で一覧表示し、選んだ手をハイライト
- 盤面は cg のデータ（`visualize_data()` / live observation）をベースにリッチ描画
- UI は日本語、追加依存なし（Python 標準ライブラリのみ。`numpy` は mPPO 用に既存）

## 起動

Python 3.11+（`cg` エンジンが動く環境）で:

```bash
python3 viewer/server.py            # 既定ポート 8000
python3 viewer/server.py --port 9000
```

起動後、ブラウザで <http://localhost:8000> を開きます。

> Windows で日本語が文字化けする場合はサーバ側ではなくブラウザ表示なので問題ありません
> （配信は UTF-8）。CLI から直接実行して確認する場合は `PYTHONIOENCODING=utf-8` を付けてください。

## スモークテスト（UI なし）

```bash
python3 viewer/engine.py --smoke --ai0 mppo --ai1 heuristic
```

リプレイ長と勝敗が表示されれば OK。

## 構成

| ファイル | 役割 |
|---------|------|
| `server.py` | HTTP API・静的配信（標準ライブラリ `http.server`） |
| `engine.py` | エンジン呼び出しの直列化・`run_match`（AI vs AI）・`HumanSession`（人間 vs AI） |
| `agents.py` | **AI レジストリ**（追加はここに 1 行） |
| `decks.py` | デッキ探索・ロード |
| `describe.py` | 日本語化・カード DB・盤面/選択肢の view 正規化 |
| `decks/` | **デッキ追加用フォルダ**（CSV を置くだけ） |
| `static/` | フロントエンド（HTML / CSS / JS、ビルド不要） |

## AI を追加する

### 方法A（おすすめ）: `viewer/ai/` に置くだけ（コード編集不要）
`viewer/ai/` に **`main.py` を含むフォルダ**（提出物と同じ形）か、**`agent()` を定義した単体 `.py`**
を置くと、サーバ再起動で自動的に一覧へ出ます。各AIは**別プロセスで実行**されるため、
別々のAIが `main`/`cg`/`tcg_rl` で衝突しません。詳細は [`ai/README.md`](ai/README.md)。

```
viewer/ai/my_ai/main.py     # agent(obs_dict)->list[int]（policy.npz等も同梱可）
viewer/ai/my_ai.py          # 単体ファイルでもOK
```

### 方法B: `agents.py` に手書き登録（同プロセス実行・軽い）
`agents.py` の末尾でレジストリに登録します。エージェントは提出物と同じ
`agent_fn(obs_dict) -> list[int]` のインターフェース。

```python
def build_my_ai():
    def inner(obs):           # obs: cg.api.Observation
        ...                   # return list[int]
    return _safe(inner)       # _safe で合法手を保証（推奨）

register(AgentSpec("my_ai", "私のAI", build_my_ai, note="説明"))
```

登録後はサーバを再起動すると選択肢に出現します。

## デッキを追加する

`viewer/decks/` に 60 枚の CSV を置くだけ。詳細は [`decks/README.md`](decks/README.md)。

## 仕組みのメモ

- エンジンはネイティブ単一インスタンスのため、サーバはロックで**1 対戦ずつ**直列実行します。
- **AI 観戦**は対戦を最後まで実行し、`cg.game.visualize_data()` の god-view スナップショット
  （両者の手札・デッキも表向き、`selected` に実際に選んだ手）をリプレイとして返します。
- **人間対戦**は live observation を逐次進めます（相手の手札は伏せ＝fog of war）。
  人間の番が来るまで AI を自動で進め、その間の AI の選択は「相手AIの行動」に表示されます。
- ワザ名のみエンジン由来の英語表記です（カード名・ログ・選択肢は日本語）。
