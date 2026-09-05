# climb_lb823 — Kaggle public score **823.5** を記録したモデル一式（凍結アーティファクト）

> **このディレクトリは「823.5 を出した提出物」の同定用パッケージです。**
> 「一番スコアが良かったモデルはどれ？」「823 のモデルの構成は？」「そのとき何のデッキ？」に
> このファイルだけで答えられるようにしてあります。機械可読版は [MANIFEST.json](MANIFEST.json)。

| 項目 | 値 |
|---|---|
| Kaggle 提出 | `55186283` / `submission_climb.tar.gz` |
| **publicScore** | **823.5**（2026-08-02 提出・当時の自己ベスト） |
| 通称 | **climb**（BC 模倣 → field self-play PPO で「登った」方策） |
| 方策の重み | [policy_weights_alakazam_rl_climb.json](../../ptcg_ai/learning/policy_weights_alakazam_rl_climb.json) |
| デッキ | [deck.csv](deck.csv)（Plan A アラカザム／フーディン） |
| 実行 config | [config_abl_5_full.json](config_abl_5_full.json)（= `configs/abl_5_full.json`） |

### 同定用 sha256（これが一致すれば「823.5 のもの」）

| ファイル | sha256 |
|---|---|
| 方策の重み | `395b02486f851be9c668e560449da5624f501b3b3131612689a6c8e2876f5196` |
| deck.csv | `8ae7a618b2655669e45eeb296df2d9271a38cf4977fd4d7dbf75bfaf386a84c2` |
| config abl_5_full | `ca6c37af4b1a4149dcae9a3a7746d2566c2b672a8a2c7516c82668b059a480da` |

> 上記 sha256 は **LF 改行のバイト列**に対する値。本リポジトリは `core.autocrlf=true` なので、
> このディレクトリには `.gitattributes`（`* -text`）を置いて checkout 時の改行変換を止めてある
> ＝ **どの環境で clone しても上の値と一致する**。重み JSON は改行を一切含まないため元から不変。
> 一方 `sample_submission/deck.csv`（リポジトリ本体側）は Windows checkout で CRLF になり得るので、
> ハッシュ照合には**このディレクトリの `deck.csv`** を使うこと（中身は同一）。

> **紛らわしい点（重要）**: 既定の [ptcg_ai/learning/policy_weights.json](../../ptcg_ai/learning/policy_weights.json)
> は **BC（模倣）重みであって climb ではありません**（sha256 `735dd38a…`, `rl_finetuned` なし）。
> 提出 tarball では climb が `policy_weights.json` という名前で同梱されるため混同しやすい。
> ローカルで既定のまま走らせると **climb ではなく BC を測ってしまう**ので、必ず重みパスを明示すること。

---

## 1. 何がスコアを出したのか（3点セット）

823.5 は **方策単独の力ではなく**、以下 3 つの組み合わせです。

```
[デッキ] Plan A アラカザム(フーディン)
   ×
[探索パイプライン] abl_5_full = 確定リーサル探索 → PIMC 前読み → Policy フォールバック
   ×
[方策NN] climb = BC 模倣で事前学習 → field 自己対戦 PPO でファインチューン
```

意思決定 1 手の流れ（`ptcg_ai/ml_policy/ml_policy_agent.py` → `ptcg_ai/search/pipeline.py`）:

```
obs
 ├─(1) 確定リーサル探索 lethal_simple  … 今ターン勝てる手順があれば即採用（発火率 ~1.1%）
 ├─(2) 浅い PIMC 前読み                … Policy top-4 を候補に、相手を推定して決定化 8 本 × 1手先
 └─(3) Policy フォールバック            … 上記が出せないとき Policy argmax
```

- Policy NN は **(2) の候補生成 prior と (3) のフォールバック**で使われる（単独で手を決めてはいない）。
- 実測では **全意思決定の約 40% だけが探索を通る**（CARD 選択等 41.8% は pipeline 対象外、
  対象内でも `top1_shortcut_prob=0.9` で即決される分がある）。

## 2. 方策ネットワークの構成

`kaggle_replays/rl/torch_policy.py::TorchOptionPolicy` 系の **option-scoring MLP**。
各「選択肢」を独立にスコアリングし、合法選択肢の集合上で softmax したものが方策。

```
各 option ごとに:
  x = [ standardize(state_feat) (166)      # 局面（全 option で共有）
        ++ standardize(option_feat) (65)   # その選択肢固有
        ++ card_embedding(option_card_id) (8) ]   = 239 次元
  h     = ReLU(W0·x + b0)   # W0: [32 x 239]  ← 隠れ層 32 ユニット・1 層のみ
  score = W1·h + b1         # W1: [1 x 32]    ← option 1 件につきスカラー 1 個
  π(option) = softmax_over_legal_options(score)
```

| 項目 | 値 |
|---|---|
| 入力 | **239 次元** = 局面 166 + 選択肢 65 + カード埋め込み 8 |
| 隠れ層 | **32 ユニット × 1 層, ReLU** |
| 出力 | `Linear(32→1)` → 合法選択肢で softmax |
| カード埋め込み | 学習可能ルックアップ表 `[1268 x 8]`（範囲外 ID は 0 埋め） |
| 標準化 | state/option は z-score（std==0 の次元は 0 化）、埋め込みは非標準化 |
| パラメータ数 | 約 1.8 万（+ 埋め込み約 1 万）= **極小ネット** |
| 使用情報 | **公開情報のみ**。自分の手札は `handCount / pokemon / trainer / energy` の 4 スカラー、相手手札は枚数のみ |

特徴量の詳細な内訳（166 次元と 65 次元の中身）は
[docs/models/climb-823-model.md](../../docs/models/climb-823-model.md) §3 を参照。

## 3. 学習レシピ

| 段階 | 内容 |
|---|---|
| **① BC（模倣）** | 上位パイロットのリプレイから行動クローン。`n_train=147705 / n_val=17359 / n_test=22626`、test top1 = **0.5806** |
| **② RL（climb）** | `kaggle_replays/rl/train_field.py` の **PPO**。frozen な FIELD 相手に自己対戦してファインチューン |
| PPO ハイパー | clip 0.2 / epochs 4 / GAE(γ=0.999, λ=0.95) / entropy 0.005 / lr policy 3e-4・value 1e-3 / grad clip 1.0 / 報酬 = 勝敗 ±1 |
| スケジュール | iters 60 / games-per-iter 512 / eval-games 400 / `field_frac` 0.45 / champion league（最終 2 体） |
| FIELD 相手 | mega_lucario_ex, archaludon_ex, crustle, dragapult_ex, marnie_grimmsnarl_ex, rocket_mewtwo_ex, shirona_garchomp_ex |
| 収束 | **best_iter 57**、field 加重勝率 **0.615（BC）→ 0.72** |
| Critic | PPO の advantage 推定用（166→64→64→1）。**提出物には含まれない**＝学習時のみ |

## 4. 使用デッキ（Plan A アラカザム／フーディン, 60 枚）

`deck.csv` は **カード ID が 1 行 1 枚で 60 行**という形式（この README の表は可読化したもの）。

| 枚 | ID | カード名 | 種別 |
|---:|---:|---|---|
| 4 | 741 | ケーシィ / Abra | たね |
| 4 | 742 | ユンゲラー / Kadabra | 1 進化 |
| 4 | 743 | フーディン / Alakazam | 2 進化 |
| 3 | 65 | ノコッチ / Dunsparce | たね |
| 3 | 66 | ノココッチ / Dudunsparce | 1 進化 |
| 1 | 343 | シェイミ / Shaymin | たね |
| 4 | 1079 | ふしぎなアメ / Rare Candy | グッズ |
| 4 | 1086 | なかよしポフィン / Buddy-Buddy Poffin | グッズ |
| 4 | 1152 | ポケパッド / Poké Pad | グッズ |
| 2 | 1097 | 夜のタンカ / Night Stretcher | グッズ |
| 1 | 1081 | 改造ハンマー / Enhanced Hammer | グッズ |
| 1 | 1129 | せいなるはい / Sacred Ash | グッズ |
| 1 | 1146 | ワンダーパッチ / Wondrous Patch | グッズ |
| 4 | 1225 | トウコ / Hilda | サポート |
| 3 | 1182 | ボスの指令 / Boss's Orders | サポート |
| 3 | 1231 | ヒカリ / Dawn | サポート |
| 2 | 1197 | クセロシキのたくらみ / Xerosic's Machinations | サポート |
| 1 | 1184 | スイレンのお世話 / Lana's Aid | サポート |
| 3 | 1264 | バトルコロシアム / Battle Cage | スタジアム |
| 4 | 5 | 基本【超】エネルギー / Basic {P} Energy | 基本エネルギー |
| 3 | 19 | テレパス【超】エネルギー / Telepath Psychic Energy | 特殊エネルギー |
| 1 | 13 | リッチエネルギー / Enriching Energy | 特殊エネルギー（**ACE SPEC**） |

合計 **60 枚**（ポケモン 19 / グッズ 17 / サポート 13 / スタジアム 3 / エネルギー 8）。

- Policy 自体は**デッキ非依存**（渡された 60 枚をそのまま操作する）が、
  **RL はこのデッキで自己対戦して最適化されている＝実質デッキ特化**。別デッキで climb 重みを
  使う場合、その +Δ は保証されない。

## 5. 再現・実行方法

### ローカルで climb を評価する

```bash
# FIELD 加重勝率で評価（--rl-weights は learning/ ディレクトリ内のファイル名）
python kaggle_replays/rl/eval_field.py \
  --rl-weights policy_weights_alakazam_rl_climb.json --games 150
```

config から使う場合は `policy_weights_path` を明示する（相対パスは **CWD 基準**で解決される）:

```json
{ "policy_weights_path": "sample_submission/ptcg_ai/learning/policy_weights_alakazam_rl_climb.json" }
```

`PTCG_AI_ML_CONFIG` 環境変数で config 名を差し替えられる（既定は `abl_5_full`）。

### 提出物を作る

提出 tarball では **climb の重みを `ptcg_ai/learning/policy_weights.json` という名前でコピーして同梱**する
（`main.py` は既定名しか読まないため）。同梱後は sha256 `395b0248…` を確認すること。
`decks/` ディレクトリの同梱も必須（欠けると import 不能で即 ERROR になる）。

## 6. 既知の差分（このコミット時点のコード vs 提出時のコード）

`build_ready/submission_climb.tar.gz`（2026-08-02 ビルド）と現在の HEAD を比較すると、
**重み・デッキ・config は完全一致**だが、以下の .py は提出後の開発で変化している:

```
board_evaluation/consequence.py        learning/value_shadow_log.py     search/attack_plan.py
hidden_information/opponent_hidden_state.py  ml_policy/__init__.py       search/leaf_eval.py
learning/policy_model.py               ml_policy/ml_policy_agent.py     search/pimc.py
learning/value_model.py                opponent_modeling/opponent_knowledge.py  search/pipeline.py
                                       opponent_modeling/rough_predictor.py
```

- `main.py` は一致。
- 提出時の全 .py の sha256 は [MANIFEST.json](MANIFEST.json) の `submitted_code_sha256` に記録済み
  （ドリフト検出用）。**823.5 のバイト単位の完全再現には提出 tarball 自体が必要**で、
  その tarball は git 管理外（ローカルの `build_ready/` のみ）。
- 差分の多くは config でゲートされた後続実験のもので、既定 `abl_5_full` の経路は不変の想定だが、
  **検証はしていない**。

## 7. 解釈上の caveat

- **823.5 は「RL 方策単独」のスコアではない**。強いデッキ・探索パイプライン・RL 方策の合わせ技。
- 同じ RL レシピを他アーキ（lucario / オーロンゲ）に適用しても Kaggle では 627.9 / 636.8 に留まった
  ＝ **実ラダーではデッキの寄与が支配的**。
- **ローカル field 勝率 0.72 ≠ Kaggle スコア**。ローカルは固定 400ms 予算で走るのに対し本番は
  ~1350ms/select と条件が違う。
- 隠れ層 32・単層という極小ネットでこの水準に達している＝**表現力より特徴量設計と探索の寄与が大きい**。

---

**詳細なアルゴリズム仕様**: [sample_submission/docs/models/climb-823-model.md](../../docs/models/climb-823-model.md)
