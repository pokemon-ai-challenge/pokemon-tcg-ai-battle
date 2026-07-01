# mPPO 実装仕様書（Maskable PPO ドラパルトex エージェント）

> このファイルは**開発者・AI（Claude）向けの作業仕様**です。設計判断・前提・行動空間の定義をすべてここに集約し、
> 実装中に文脈を再導出しないための「単一の真実」とします。初心者向けの解説は `mppo-explained.md` を参照。

最終更新: 2026-06-29 / ブランチ: `mPPO`

---

## 0. 確定した要件（ユーザー合意済み）

| 項目 | 決定 |
|------|------|
| 使用デッキ | `reinforce_learning` ブランチの `deck.csv`（ドラパルトex / ブレイズビートex 軸）をそのまま使用。mPPO に反映済み。 |
| RL の範囲 | **フルエンドツーエンド**。全選択コンテキストを Maskable PPO で学習する。 |
| 学習相手 | カリキュラム: ①ランダム → ②自己対戦(self-play) → ③既存 Flat-MC AI（緩い時間制限）。MC AI の評価関数改善も並行課題。 |
| 提出時推論 | **numpy のみ**で MLP 前向き計算（torch 非依存）。torch があれば優先利用、無ければ numpy にフォールバック。 |
| 学習ライブラリ | `sb3-contrib` の `MaskablePPO` + `gymnasium` + `torch`（CPU）。Python 3.13。 |

---

## 1. ゲームの性質と PPO 化の中心課題

- エージェント = `agent(obs_dict) -> list[int]`。返すのは**選択肢インデックスのリスト**。
- 1 回の選択は `minCount`〜`maxCount` 個を `option`（`list[Option]`）から選ぶ。**場面ごとに選択肢の意味も個数も変わる**。
- PPO は「固定サイズの離散行動を 1 ステップ 1 個」出すアルゴリズム。可変長マルチセレクトと直接かみ合わない。

### 解決策: 「1手1選択 + STOP 行動」への分解（factored action）

マルチセレクトを**逐次の単一選択の列**に分解する。

- 行動空間 = `Discrete(MAX_OPTIONS + 1)`
  - `0 .. MAX_OPTIONS-1` = 「`option[i]` を1つ選ぶ」
  - `MAX_OPTIONS`（最終インデックス）= **STOP**（このマルチセレクトの確定）
- 環境は「現在の選択でこれまでに選んだインデックス」を `buffer` として保持。
  - 行動 i を選ぶ → `buffer.append(i)`
  - STOP を選ぶ（`len(buffer) >= minCount` のときだけ合法）または `len(buffer) == maxCount` に到達 → `battle_select(buffer)` を呼び、次の選択へ。
- これで**全コンテキストを単一の仕組みで**扱える（= フルエンドツーエンド）。

### MAX_OPTIONS の決定
- ほぼ全ての選択は `len(option)` が小さい（MAIN は数〜十数、カード選択でも数十）。
- 上限 `MAX_OPTIONS = 50`（暫定）。`len(option) > MAX_OPTIONS` の稀な選択は**ヒューリスティックで処理**（RL に乗せない）。実測で超過頻度を計測し必要なら拡大。→ 実装で計測ログを残す。

### 行動マスク `action_masks()`（長さ `MAX_OPTIONS+1` の bool 配列）
- `i in [0, len(option))` かつ `i not in buffer` → True
- `STOP` → `len(buffer) >= minCount` のとき True
- それ以外 → False
- これにより PPO は**常に合法手のみ**サンプルする（不正手で例外が出ない）。

---

## 2. 環境 `PokemonTCGEnv`（Gymnasium、学習専用）

ファイル: `tcg_rl/env.py`。`cg.game`（`battle_start/battle_select/battle_finish`）をラップ。

### 重要な実装制約（必読）
- `cg/sim.py` は**プロセス内グローバル** `Battle.battle_ptr` を使う。**1プロセスで同時に1対戦しか走らせられない。**
  - → `DummyVecEnv` で n_envs>1 は不可（グローバルが競合）。並列化は `SubprocVecEnv`（別プロセス）のみ。
  - → まず **n_envs=1** で動かす。安定後に `SubprocVecEnv` を検討。
- ローカル対戦（`battle_start`）はデッキ選択 obs（`select is None`）を**経由しない**。よって env はデッキ選択を扱わない。

### perspective（視点）
- 「学習者(learner)」は player 0 固定（デッキは両者ドラパルト、または learner=ドラパルト vs 任意相手）。
- `yourIndex == learner` の選択 → エージェント（PPO）が決める。
- `yourIndex == opponent` の選択 → env 内部で**相手ポリシー**が決めて適用し、learner の手番まで自動で進める。
- 観測は常に「手番プレイヤー視点」に正規化（learner の手番でしか PPO に obs を返さないので、視点 = learner）。

### `reset()`
1. 相手ポリシーを（カリキュラムに従って）選ぶ。
2. `battle_start(learner_deck, opp_deck)`。
3. 相手手番をスキップして最初の learner 選択まで進める。
4. 最初の SelectData を encode して返す。

### `step(action)`
1. `action` を現在の `buffer` に反映、または STOP 処理。
2. マルチセレクト未確定なら、同じ SelectData の次サブ選択の obs を返す（report 0 reward, not done）。
3. 確定したら `battle_select(buffer)` → 相手手番を自動消化 → 次の learner 選択 or 終局。
4. 終局なら reward（±1 など）+ done=True。

### 報酬 `reward`
- 既定: **終局時のみ** `+1`（learner 勝ち）/ `-1`（負け）/ `0`（引き分け）。
- 任意のシェイピング（フラグで ON/OFF）: サイドを取った +0.1 / 取られた -0.1（ログ `RESULT`/サイド枚数差分から算出）。デフォルト OFF→まず素の報酬で確認。

### 相手ポリシー（`tcg_rl/opponents.py`）
- `RandomOpponent`：合法手から STOP 含めランダム。
- `SelfPlayOpponent`：学習中ポリシーの**凍結スナップショット**（コールバックで定期更新）。
- `MCOpponent`：`tcg_rl/mc_agent.py` の Flat-MC（時間/ロールアウト予算を引数で緩く設定）。
- `ScriptedOpponent`（任意）：簡易ルールベース。
- カリキュラム: 学習ステップ数に応じて相手分布を変える（例: 0–X step は random 100%、以降 self-play と MC を混ぜる）。

---

## 3. 観測エンコード `encode_obs(obs) -> np.ndarray`（固定長 float32）

ファイル: `tcg_rl/features.py`（**numpy + cg.api のみに依存。torch/gym を import しない＝提出に同梱可能**）。

ブロック構成（すべて固定長に詰める。欠損は 0 埋め）:

1. **グローバル**: `turn`（正規化）, 手番フラグ, `turnActionCount`, `supporterPlayed/stadiumPlayed/energyAttached/retreated`, 自分/相手の prize 残数, deckCount, handCount, bench 数, discard 数。
2. **自分のポケモン**（active 1 + bench 上限 5 = 6 スロット）: 各スロット = [存在flag, hp/maxHp, energy数(正規化), tool数, ex/stage one-hot, type one-hot, 特殊状態flag]。card-id 由来の静的属性は `card_db` から引く。
3. **相手のポケモン**（6 スロット）: 公開情報のみで同様にエンコード。
4. **特殊状態**: 自分/相手 active の poison/burn/sleep/paralyze/confuse。
5. **選択コンテキスト**: `context` one-hot（約49種, Enum は将来追加され得るので未知IDは「その他」次元に丸める）, `selectType` one-hot, `minCount/maxCount`, `remainEnergyCost`, `remainDamageCounter`, **現在の `buffer` 長**。
6. **選択肢ごとの特徴**（`MAX_OPTIONS` スロット + STOP）: 各 `option[i]` を [optionType one-hot, 対象が自分/相手, area one-hot, card category(cardId→card_db), attack damage（ATTACK時, 正規化）, energy type, 既にbufferにある flag, 有効flag(mask)] にエンコード。STOP スロットは専用 1-hot。

> **設計鉄則**: `encode_obs` / `action_masks` / 定数（`MAX_OPTIONS`・各 one-hot 次元）は**学習 env と提出 main.py で完全共有**する。ドリフトすると学習済み重みが無意味になる。→ `features.py` を両者から import。

`card_db`: `all_card_data()` / `all_attack()` を起動時に 1 回だけ呼び `cardId->CardData`, `attackId->Attack` の dict を作る（`features.py` 内にキャッシュ）。

---

## 4. ポリシーネットと numpy 推論

- 学習: `MaskablePPO("MlpPolicy", ...)`。net_arch は小さめ（例 `[256, 256]`）。CPU 前提。
- エクスポート `export_policy.py`: 学習済みモデルから policy MLP（共有特徴抽出 + action net）の重みを取り出し `policy.npz`（各層 W,b と活性化, net_arch を保存）。value net は推論不要なので省略可。
- 推論 `tcg_rl/mlp_numpy.py`（**numpy のみ**）: `policy.npz` を読み、`encode_obs` の出力にマスク付き forward → logits。マスク無効手は `-inf`。**argmax**（決定的）で行動を選ぶ。
  - main.py は 1 つの SelectData に対し、内部で **buffer ループ**を回して最終的な `list[int]` を構築（env のサブステップを agent 関数内で完結させる）。

---

## 5. 提出 `main.py`（numpy 推論）

- `agent(obs_dict)`:
  - `obs.select is None`（初回・実機のみ）→ `read_deck_csv()`。
  - それ以外 → `tcg_rl.features.encode_obs` + `tcg_rl.mlp_numpy` で buffer ループし `list[int]` を返す。
- **推論は numpy のみ**（torch を import しない）。numpy が torch 出力を再現するため依存ゼロ。
- **Kaggle 実行対応**: Kaggle は `main.py` を `exec()` で動かすため `__file__` が未定義になり得る。
  `try/except NameError` で `_HERE = "/kaggle_simulations/agent"` にフォールバックして import パス・
  `policy.npz`/`deck.csv` 解決を保証（`_candidate_paths` 経由）。
- 例外時フォールバック: 推論で何か失敗したら**合法な最小手**（先頭から minCount 個 / END 相当）を返し、絶対に落ちない。

### 提出パッケージに必要なファイル（重要）
標準の `main.py + deck.csv + cg/` に加えて、以下を**必ず同梱**:
```
tcg_rl/__init__.py
tcg_rl/features.py     # numpy + cg のみ
tcg_rl/mlp_numpy.py    # numpy のみ
policy.npz             # 学習済み重み
```
`tcg_rl/env.py`, `opponents.py`, `mc_agent.py`, `train_ppo.py` は**学習専用・提出不要**（torch/gym 依存なので入れない）。
→ パッケージ用スクリプト `make_submission.py`（or README 手順）で submission-safe ファイルだけ tar 化。

---

## 6. ファイル構成（mPPO ブランチ）

```
sample_submission/
├─ main.py                 # 提出入口（numpy 推論）        ← 編集
├─ deck.csv                # ドラパルト 60 枚              ← 反映済み
├─ cg/                     # ゲームエンジン（変更禁止）
├─ tcg_rl/
│  ├─ __init__.py
│  ├─ features.py          # encode_obs/action_masks/定数  [提出同梱・numpy+cg]
│  ├─ mlp_numpy.py         # numpy MLP 推論                [提出同梱・numpy]
│  ├─ env.py               # Gymnasium 環境                [学習のみ・gym]
│  ├─ opponents.py         # random/self-play/MC 相手       [学習のみ]
│  └─ mc_agent.py          # Flat-MC + 評価関数（改善対象） [学習のみ・cg]
├─ train_ppo.py            # 学習エントリ                  [学習のみ・torch/sb3]
├─ export_policy.py        # 重み書き出し → policy.npz      [学習のみ・torch]
├─ make_submission.py      # 提出 tar を作る                [任意]
├─ policy.npz              # 学習済み重み（生成物）
└─ docs/
   ├─ mppo-spec.md         # 本書
   └─ mppo-explained.md    # 初心者向け解説 + 課題

（リポジトリ直下に別ツール）
viewer/                    # 対戦ビジュアライザ（ブラウザ観戦・人間vsAI）[提出無関係・標準ライブラリ+numpy]
  ├─ server.py / engine.py / agents.py / decks.py / describe.py
  ├─ decks/                # 観戦用デッキCSVを置く
  └─ static/               # フロント（HTML/CSS/JS、ビルド不要）
```

---

## 7. 実装ステップ（順序）

1. `tcg_rl/features.py`: 定数 + card_db + `encode_obs` + `action_masks` + buffer 状態の表現。
2. `tcg_rl/mlp_numpy.py`: npz ロード + マスク付き forward。
3. `tcg_rl/mc_agent.py`: `reinforce_learning` の Flat-MC を移植（env から呼べる関数 + 改善余地のある `evaluate`）。
4. `tcg_rl/opponents.py`: 相手ポリシー群 + カリキュラム。
5. `tcg_rl/env.py`: Gym 環境（reset/step/buffer/相手消化/報酬）+ `ActionMasker` 連携。
6. `train_ppo.py`: MaskablePPO 学習 + 評価コールバック（vs random 勝率）+ checkpoint。
7. `export_policy.py` → `policy.npz`。
8. `main.py`: numpy 推論 + フォールバック。
9. `make_submission.py` + tar 検証（`local_test_advanced.py` で合法性確認）。
10. `mppo-explained.md`（初心者向け + 課題）。

各段階で `python local_test_advanced.py --opponent random --games N` で**合法手を最後まで返せるか**を検証。

---

## 8. 既知の課題（実装前から想定）

- **行動空間の巨大さ / 報酬の疎さ**: フルエンドツーエンドは状態・行動が膨大で報酬が終局±1のみ。CPU 学習では収束が遅い/不安定。→ カリキュラム・報酬シェイピング・小さめネットで緩和。
- **1プロセス1対戦の制約**: ctypes グローバルにより in-process 並列不可。スループットが低い。→ `SubprocVecEnv` で将来並列化。
- **エンコードのドリフト risk**: env と main.py で `features.py` を共有しないと学習が無意味化。→ 単一モジュール共有を徹底。
- **Enum 拡張**: コンペ中に `SelectContext` 等が増え得る。未知値で落ちないよう「その他」次元で吸収。
- **相手非公開情報（MC 相手）**: MC は `search_begin` に相手の山札・手札の推測が必要で精度が低い。→ MC は「緩い予算の相手」として割り切る。
- **提出環境の不確実性**: torch 有無・メモリ・ステップ時間制限が不明。→ numpy 推論で依存ゼロ化。
- **`MAX_OPTIONS` 超過**: 稀に選択肢が 50 を超え得る。→ 計測し、超過はヒューリスティック処理。

（実装して判明した課題は `mppo-explained.md` の「今後の課題」に追記する。）

---

## 8.5 実装で確定した事実（計測値・既知の落とし穴）

- `OBS_DIM = 2602`、`ACTION_DIM = 51`（`MAX_OPTIONS=50` + STOP）。
- ランダム8戦での**最大選択肢数 = 27**（>50 はゼロ）。`MAX_OPTIONS=50` は十分。
- **numpy 推論は torch と一致**：logits 差は最大 5e-10（`export_policy.py`→`mlp_numpy`）。提出 numpy 化は安全。
- **ライブラリ**：Python 3.13.14 で torch 2.12.1+cpu / gymnasium 1.3.0 / sb3 2.9.0 / sb3-contrib 2.9.0 動作確認。
- **学習スループット**：小ネットで約 300〜800 step/s（CPU、n_envs=1）。相手が self-play/MC だと低下。
- **重大な落とし穴（修正済み）**：`EvalCallback` が2つ目の `PokemonTCGEnv` を作ると `battle_start` が
  **プロセス共有の battle ポインタを奪い**、学習側の対戦が壊れて `battle_select` が例外を投げる。
  → `env.step` で engine 例外を捕捉し**その episode を truncated 終了 → auto-reset で再生成**する耐性を実装。
  （= ctypes グローバル制約の具体的な現れ。並列化時も同根の問題に注意。）
- **MC ヒューリスティック相手の基準値**：ミラー戦でランダム相手に **約68%（27/40）**。
- **学習の伸びの確認**：self-play 8000 step で vs random 勝率 0.42→0.83（短時間でも学習する）。

## 9. 並行課題: Flat-MC 評価関数の改善

現行 `evaluate` = 0.70×サイド差 + 0.15×HP比差 + 0.04×ベンチ差 + 0.02×エネ数。改善候補:
- 進化進捗 / 必要エネ到達度（アタッカー完成度）
- 盤面の「次ターン攻撃継続性」
- ボス・入れ替え等の妨害リソース残量
- ドラパルト特有の勝ち筋（ベンチ狙撃・2-2-2 サイドプラン）
MC 評価改善は PPO の**学習相手の質**を上げる効果もある（強い相手＝良いカリキュラム）。

---

## 10. 現在の状況・更新履歴

### 10.1 ブランチ状態
- ローカル `mPPO` と `origin/mPPO` は同期済み（最新 `4c35739`）。
- 別マシンでの作業をマージで取り込み済み（fast-forward、競合なし）。

### 10.2 マージで入った変更（`dd37bfb` → `4c35739`）
- **`main.py`**: Kaggle の `exec()` 実行で `__file__` 未定義になる問題を修正（§5 参照）。**numpy 推論のまま**。
- **`tcg_rl/env.py`**: `turn_penalty`（1ゲームターン経過ごとの微小マイナス報酬）を追加。
  消極的プレイ（引き分け/長期化狙い）を抑制する。既定 0.0（OFF）。
- **`train_ppo.py`**: CLI/コールバックを拡張。
  - `--mc-budget <秒>`: **MC 相手の1手あたり探索時間**。`>0` で MC が実際に `search` する（強い相手）。
    `0.0` は従来の高速ヒューリスティック相手。
  - `--turn-penalty <float>`: 上記の anti-stall 報酬を env に渡す。
  - `--checkpoint-freq <N>`: `CheckpointExport` コールバックが N step ごとに
    モデル保存＋`policy.npz` エクスポート。**学習を中断しても最新の提出物が残る**。
- **`policy.npz`**: 再学習済みの重みに更新（次元・I/F は不変なので numpy 推論はそのまま動く）。
- **`viewer/`（新規・提出無関係）**: ブラウザで対戦を観戦／人間 vs AI ができるローカル Web ツール。
  - 標準ライブラリ + numpy のみ（追加依存なし）。`python viewer/server.py` で起動。
  - 提出物と同じ `agent_fn(obs_dict)->list[int]` で AI を登録（`agents.py`）。`mppo` も観戦可能。
  - 注意: cg エンジンは**1対戦ずつ直列**（§2 のグローバル制約と同根）。

### 10.3 今後の作業候補（未着手）
- `--mc-budget>0` での MC 探索相手を混ぜた長時間カリキュラム学習。
- `turn_penalty` の効果検証（勝率と平均ターン数のトレードオフ）。
- §9 の `evaluate()` 改良（viewer で挙動を目視確認しながら）。
- 学習の中断再開（`--resume`）と `SubprocVecEnv` 並列化。
