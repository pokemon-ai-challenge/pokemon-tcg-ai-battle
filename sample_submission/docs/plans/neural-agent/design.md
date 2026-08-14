# Neural Agent: 信念状態 × Transformer × デュアルヘッド × 自己対戦RL 設計書

作成日: 2026-07-24
種別: **設計書(方向性の評価と推奨。関数シグネチャ・完了条件付きの実装計画は別ファイルに分ける
—— [[feedback_design_vs_implementation_plan]])**
前提:
- `docs/plans/beyond-bc/beyond-behavior-cloning-design.md`(直接の前身。本書はこれを包含・拡張する)
- `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`(容量abl negative)
- `docs/plans/ml-value-network/`(ValueModel/PolicyModel の設計・オフライン評価)
- `docs/plans/hidden-information/`(信念状態推定レイヤー)
ステータス: **設計のみ。production 未変更。着手順序は本書で決め、各フェーズは別の実装計画に落とす。**

---

## 0. 目的と最大リスク(問題定義)

### 0.1 ユーザー構想(5コンポーネント)

1. **状態表現・信念状態モジュール**: 現盤面 + 「山札/サイド/相手デッキ推論」の確率分布を統合した
   **Belief State Vector** を生成する。
2. **Transformer エンコーディング**: 盤面のカード群 + 信念状態ベクトルを Transformer に入力し、
   カード間シナジー/盤面文脈を捉えた高次元隠れ表現を得る。
3. **方策・価値のデュアルヘッド**: エンコード状態から (a) 自己回帰的にアクション列とその確率を出す
   **方策ヘッド**、(b) その盤面の勝率を出す **価値ヘッド**。まず Kaggle 上位ログの模倣学習で初期化。
4. **OSFP による強化学習**: 初期化モデルで自己対戦。過去の多様なモデルを対戦相手プールに保存し、
   PPO 等で継続更新。ポテンシャルベース報酬整形で確率と期待値のバランスを学習。
5. **推論時タイムマネジメント**: 基本は方策ネットに従い高速プレイ。価値ネットが「クリティカル局面」と
   判定したときだけ、信念状態で盤面を決定化し、方策の確率で枝刈りしながら **ISMCTS** を実行。

### 0.2 総評: 方向性は正しい。ただし自分の negative result と正面衝突する

この構成は不完全情報ゲームを解く現代的な王道(DeepMind **Player of Games / Student of Games**、
**ReBeL**、AlphaStar のリーグ学習)そのものである。信念状態 + 深層方策/価値 + 決定化サーチの三位一体は、
ポーカー・Stratego 系で実証された筋の良い設計。着想自体に異論はない。

**しかし、本リポジトリの実測はこう言っている:**

| 施策 | offline | 実戦(ミラー) | 結論 |
|---|---|---|---|
| Tier1 identity 特徴追加 | 一部微増 | 43–44% | 転移せず |
| Tier3 consequence 特徴 | +0.1〜0.2pt or 悪化 | 35–55%(不安定) | 転移せず |
| 容量増 M64 / M128 | +0.5pt 単調・過学習なし | 49.0% / 49.3%(CI が 50% 跨ぎ) | 転移せず |
| PIMC(研究では head-to-head 勝ち) | — | 本番 ml_policy 上で ml_lethal に有意差なし | 転移せず |
| Beyond BC Phase1 = C(outcome-aware BC) | — | C1 全負け / C2 900試合 50.7% | 転移せず |

→ **これまでのボトルネックは「モデルの表現力・容量」でも「サーチの有無」でもなく、
「offline の改善が本番勝率に乗らないこと」そのもの。**
今回の構想は、その転移しなかった 3 方向(Transformer=容量↑、BC 初期化=offline 学習、ISMCTS=サーチ)を
**同時に一段強くしたもの**であり、根本原因(なぜ転移しないか)には触れていない。
**同じ壁に、桁違いのコストで突っ込むリスクがある。** これが本書の最大の警戒点。

### 0.3 敗因プロファイルとの整合性チェック

敗因タクソノミ([[project_loss_taxonomy]]): **ブローアウト(大差負け)69.2% が最大・広く分散**、
山札切れ 25.9% は構造的([[project_deckout_loss_cause]]、第三者 695 チームも同水準)。

- 大差負けが広く分散 = 「1手のタクティカルな失着」ではなく **試合を通じた価値のじわ負け / デッキ・戦略の劣位**。
- **step5 の『クリティカル局面 ISMCTS』はリーサル・重要トレードのような分岐点で効く道具**であり、
  じわ負けのブローアウトは救いにくい。→ ISMCTS を優先投資すると効く場所を外す可能性が高い(§2.5、§4)。
- 山札切れが構造的なら、方策をいくら磨いても **天井が低い**。デッキ construction 側の問題を
  方策/価値/サーチのどれも解けない(§0.4 の診断で切り分ける)。

### 0.4 本書のゴール

構想を否定するのではなく、**「なぜ転移しないか」を先に切り分けたうえで、
最小コスト・最大可逆性・最小転移リスクの順に、既存資産で実現できる形へ落とす**こと。
特に「転移ギャップの原因診断(Phase 0)」を全 NN 投資の前段に必須の関門として置く。

---

## 1. ユーザー構想 → 既存資産マッピング

**含意を先に言うと: 5 コンポーネントの部品はほぼ全て既にリポジトリにある。** 新規に足りないのは
「信念ベクトルの接続」「Transformer 化」「自己回帰方策」「RL ループ」で、ゼロからの土台構築ではない。

| # | 構想コンポーネント | 既存資産(実体) | 状態 | 新規に要るもの |
|---|---|---|---|---|
| 1 | 信念状態モジュール | `hidden_information/match_context.py`(相手デッキ事後分布・自山札/サイド推定)+ `encoder.encode_state(..., extra_features=)` の**差し込み口が既にある** | 推定は毎ターン計算済みだが**意思決定に未接続**(search factory にしか渡っていない) | 事後分布 → 固定長ベクトル化 → `extra_features` へ接続 |
| 2 | Transformer エンコーダ | `encoder.py`(166 次元 state + 65 次元 option + card 埋め込み)。カード identity 埋め込みは既に学習経路あり | 現状は**手製特徴 + MLP**。set 構造は未活用 | カード集合を token 列にする set-encoder / 学習・推論経路 |
| 3a | 価値ヘッド | `learning/value_model.py` + `value_weights.json`(test AUC 0.746、較正済、pure-Python) | production 稼働中(shadow log のみ、意思決定未使用) | 方策とエンコーダ共有する形への再構成 |
| 3b | 方策ヘッド | `learning/policy_model.py` + `policy_weights.json`(listwise 模倣、pure-Python) | production 稼働中 | 自己回帰(アクション列)化。現状は**選択肢 1 個ずつの独立スコアリング** |
| 3c | BC 初期化 | `kaggle_replays/policy_net/`・`value_net/` の学習パイプライン + 上位ログ | 稼働中 | dual-head 共有学習への統合 |
| 4 | 自己対戦 RL | `cg.api.search_begin/step`(環境ロールアウト)、`league/run_match.py`・`round_robin.py`(対戦harness) | 環境・対戦harnessは有 | RL 学習器(PPO/リーグ)・report・throughput |
| 5 | 価値ゲート ISMCTS | `search/pimc.py`(決定化サンプリング)、`search/attack_plan.py`(shortlist+rollout)、`search/lethal_simple.py` | PIMC は稼働・研究済(本番転移せず) | policy-prior 枝刈り + value leaf 評価の ISMCTS。value ゲート判定器 |

**新規実装で本質的に難しいのは 2(Transformer)・3b(自己回帰方策)・4(RL)** の 3 つ。
1(信念接続)・5(ISMCTS)は既存部品の**接続**が主。

---

## 2. コンポーネント別評価(価値 / コスト / リスク)

評価軸: 価値(効く見込み)・実装コスト・推論レイテンシ・可逆性・転移リスク・既知の弱点。

### 2.1 信念状態の接続(構想 #1)

- **価値: 高**。推定モジュールは既に毎ターン計算しているのに**意思決定に一切効いていない**
  ([[project_ai_architecture_strategy]] の既知ギャップ)。「作ったのに繋いでいない」資産の回収は、
  新規学習なしに検証できる最短経路。
- **コスト: 低**。`encoder` の `extra_features` に事後分布の要約(例: 上位アーキタイプ確率、
  自山札の残りキーカード確率、サイド落ち推定)を連結し、既存 train.py で再学習するだけ。
- **レイテンシ: ほぼ増えない**(推定は既に毎ターン走っている。ベクトル化のコストのみ)。
- **可逆性: 最高**(weights + encoder 特徴長の変更。config で旧重みに戻せる)。
- **転移リスク: 中**。特徴追加は過去に転移していない(§0.2)。ただし過去の特徴は**公開情報の再加工**が主で、
  今回は**非公開情報の確率推定という新規情報**を入れる点が質的に違う。ここが効かなければ、
  上位の Transformer/RL も同じ壁に当たる可能性が高い ⇒ **踏み絵として最優先**。
- **既知の弱点**: [[feedback_conservative_confidence]] —— 推定確率は較正されていないと過信を招く。
  推定器の温度較正(既に相手デッキ予測で実施済)を信念ベクトルにも要求する。

### 2.2 Transformer エンコーダ(構想 #2)

- **価値: 中**。容量増は既に転移しなかった(§0.2)ので「表現力不足」がボトルネックである証拠は薄い。
  ただし現行は**カード集合を手製集約**しており、set-transformer なら「同種カードの枚数依存シナジー」
  「盤面の組み合わせ効果」を原理的に表現できる —— これは MLP+手製特徴では届かない領域。
- **コスト: 中〜高**(学習経路の刷新、set トークン化)。
- **レイテンシ: 要注意(§3.3)**。現行推論は**純 Python(numpy/torch 非依存)**。
  Transformer を純 Python 行列積で回すのは非現実的。**Kaggle 実行環境で numpy(できれば onnxruntime)が
  使えるか**が Phase 2 の可否を決める前提条件。
- **可逆性: 高**(別 weights/別推論経路、opt-in)。
- **転移リスク: 中〜高**。容量増が効かなかった履歴を踏まえ、「表現力が本当にボトルネックか」を
  Phase 1 の結果で先に確認してから着手する(順序の根拠)。

### 2.3 デュアルヘッド + BC 初期化(構想 #3)

- **価値: 中**。方策/価値でエンコーダを共有すると表現学習が相互補強しうる。自己回帰方策は
  「同一ターン内の複数選択(進化 → エネ付け → 攻撃)」を**系列として**扱え、現行の
  「選択肢 1 個ずつ独立スコアリング」の既知の弱点(`_greedy_multi_select` の組み合わせ非最適)を解ける。
- **コスト: 中**(dual-head 学習、系列デコード)。
- **レイテンシ: 中**(系列長ぶんデコード。ただし方策のみなら軽い)。
- **可逆性: 高**。**転移リスク: 中**(BC 初期化は結局 `P(human)` 模倣なので、§0.2 の壁は残る。
  勝率側へ寄せるのは #4 の RL 段階)。

### 2.4 自己対戦 RL / OSFP(構想 #4)

- **価値: 高(理論上)**。目的関数を初めて `P(win)` へ直接向ける唯一の段。Beyond BC の C(outcome-aware BC)が
  弱信号で負けた([[project_beyond_bc_design]])のに対し、RL は環境からの報酬で強い信号を得られる。
- **コスト: 最高**。**レイテンシ(学習): throughput が律速**(§3.2)。
- **可逆性: 高**(別 weights)。**転移リスク: 高**(分布ずれ・価値過大評価・報酬整形の設計難)。
- **現実性の警告(§3.2 に詳述)**: Player of Games / AlphaStar は**数千アクター規模**。
  こちらは**単一 GPU デスクトップ、しかも torch 未導入**。cg エンジンは ctypes 経由の DLL で
  self-play throughput が支配的になる。**BC を超えるところまで PPO self-play を回すのは、
  この計算規模では現実的に厳しい**。フルのリーグ学習(過去モデルプール + 継続 PPO)は
  **投資対効果が最悪の部分**。→ **最後、あるいはやらない**。現実解は「BC 初期化 + 軽い self-play fine-tune」まで。
- ポテンシャルベース報酬整形は方策不変性を保つ良い選択だが、TCG で良いポテンシャル関数を
  設計すること自体が研究テーマ(ValueModel を potential に使う案が自然だが Value ノイズが入る)。

### 2.5 価値ゲート ISMCTS(構想 #5)

- **価値: 中(ただし効く場所に疑問)**。PIMC は本番で転移しなかった([[project_pimc_prod_validation]])。
  policy-prior 枝刈り + value leaf 評価という**質の違う**サーチだが、**敗因が大差負けの分散(§0.3)なら、
  クリティカル局面サーチが救う「タクティカルな分岐点」は負けの主因ではない**可能性が高い。
- **コスト: 中〜高**(ISMCTS 新規、value ゲート判定、予算管理)。
- **レイテンシ: 高**(ノード数比例。ただし「クリティカル時のみ」で平均は抑えられる。
  PIMC 実績で Kaggle 予算比 2–4%)。
- **可逆性: 高**(opt-in)。**転移リスク: 中〜高**(Value ノイズ + 探索分散)。
- **採用条件**: PIMC negative を踏まえ、**「負けプロファイルにサーチで救えるタクティカル分岐点が
  実在する」ことを Phase 0/1 の診断で先に示す**こと。示せないなら投資しない。

---

## 3. 計算資源とデプロイ制約(現実チェック)

### 3.1 GPU デスクトップの現状(2026-07-24 実測)

- SSH 疎通 OK(`shogo@100.99.53.12`、鍵 `show1_ed25519`)。**Windows / PowerShell** マシン。
- **torch 未導入**(`ModuleNotFoundError: No module named 'torch'`)。numpy は導入済([[project_gpu_desktop_tailscale]])。
- GPU は RTX 5060 Ti(Blackwell 世代 = sm_120)。**Blackwell は新しめの CUDA(cu128 系)torch が必要**で、
  古い cu121/cu124 wheel では動かない可能性が高い。torch 導入は Phase 2 以降の前提作業(§7 のプレップ)。

### 3.2 自己対戦 throughput が RL の律速(#4 の現実性)

- RL の学習速度は「単位時間あたりの self-play 手数」で決まる。cg エンジンは ctypes/DLL・シングルプロセス。
- 1 台の GPU では**アクター並列がほぼ効かない**(GPU は方策推論に使えても、環境シミュレーションが CPU 律速)。
- **結論**: from-scratch RL で BC を超えるのは非現実的。#4 は「軽い fine-tune」に限定し、
  プロジェクトの成否を賭けない。まず #1/#2/#3 の supervised 側で天井を上げる。

### 3.3 推論の純 Python 制約 vs Transformer(#2/#3 のデプロイ問題)

- **現行の設計思想は「Kaggle 提出コードは numpy/torch 非依存、pure-Python の math だけ」**
  (`value_model.py`/`policy_model.py` の docstring)。提出物の依存を最小化し、ロード時間と互換性を守るため。
- Transformer / 大きな NN を **純 Python 行列積**で 600s/agent 予算内に回すのは非現実的。
- したがって Phase 2 の**前提条件として**、Kaggle 実行サンドボックスで
  **(a) numpy が使えるか、(b) onnxruntime 等の軽量推論ランタイムが持ち込めるか**を検証する必要がある。
  - 使える → numpy/onnx で小さめ Transformer を回す設計にする。
  - 使えない → Transformer は**蒸留して純 Python MLP に落とす**(表現は Transformer で学習、
    推論は軽量 student)等の回避策が要る。**この検証を Phase 2 の Gate 0 とする**。

### 3.4 提出サイズ・その他

- 提出 tarball は `decks/` 必須([[project_submission_tarball_requires_decks]])。重み JSON の肥大化に注意。
- Enum 追加リスク(CLAUDE.md): エンコーダは未知値を other バケットへ落とす既存方針を維持。

---

## 4. 推奨フェーズ順序(risk-ordered)

> **原則(前身の踏襲): 学習時変更(推論コスト 0・可逆性最高)から。決定時変更(レイテンシ増・Value 依存)は次。
> 表現刷新・RL は最後。安いものから、offline で採らず実戦(CI 下限 > 50%)で決める。**

ユーザー構想の 5 コンポーネントに、前段の**診断(Phase 0)**を必須の関門として挿入した順序:

### Phase 0 —— 転移ギャップ診断(全 NN 投資の前段、必須)

**目的**: 「なぜ offline 改善が本番勝率に乗らないか」を切り分ける。これを飛ばすと全フェーズが同じ壁に当たる。
仮説と切り分け:
1. **学習分布のミスマッチ**: 上位ログは Kaggle フィールドと相手/デッキ分布が違う。
   → 上位 vs フィールドの相手分布比較、模倣一致率の相手別分解。
2. **デッキ劣位(構造)**: 山札切れ構造コストが天井を作っている。→ 既に半分検証済([[project_deckout_loss_cause]])。deck.csv 変更の感度分析。
3. **評価ノイズ**: 真の小改善が CI に埋もれる。→ 必要試合数・分散の見積り(過去の 300 試合スクリーニングの検出力)。

**成果物**: 「Phase 1 で信念ベクトルに何を入れるべきか(相手デッキ分布 or 自山札/サイド)」を
データで指し示す。**この結論がユーザー判断の代替になる**(推測でなくデータで決める)。
コスト低・production 変更なし・read-only 分析中心。→ **最初に着手(本セッションで開始)**。

### Phase 1 —— 信念状態の接続(構想 #1)

Phase 0 が指した情報を `encoder.extra_features` に接続し、ValueModel/PolicyModel を再学習して head-to-head。
**推論経路ほぼ不変・可逆性最高**。ここが**踏み絵**: 非公開情報の確率推定という新規情報が効かなければ、
上位フェーズの前提が崩れる。効けば #2 へ進む強い根拠になる。

### Phase 2 —— Transformer エンコーダ + デュアルヘッド(構想 #2/#3)

Phase 1 で「表現/情報が効く」兆候が出てから。**Gate 0 = §3.3 のデプロイ検証**(numpy/onnx 可否)を先に通す。
set-transformer で盤面カード集合をエンコードし、方策/価値のヘッドを共有。自己回帰方策で複数選択を系列化。

### Phase 3 —— 価値ゲート ISMCTS(構想 #5)

Phase 0/1 の診断で「サーチで救えるタクティカル分岐点が負けに実在する」と示せた場合のみ。
policy-prior で枝刈り、value で leaf 評価、value ゲートでクリティカル時のみ発火。**既定 OFF の opt-in**。

### Phase 4 —— 自己対戦 RL(構想 #4)—— Deferred / 条件付き

#1–#3 の supervised 側で天井が見えてから。**フルのリーグ学習はやらない**(§3.2)。
やるとしても「BC 初期化 + 軽い self-play fine-tune」に限定。**プロジェクトの成否を賭けない研究枠**。

---

## 5. 評価ゲート(全フェーズ共通)

**offline 改善だけで採用しない。** 容量abl・Tier3・Beyond BC と同じ 3 段階 + 採用条件を踏襲:

- **Stage 1 offline**: 破綻/過学習の足切りのみ(模倣一致率が下がっても可 —— 勝率が目的)。
- **Stage 2 診断**: 既知局面(ATTACK 一致率・山札切れ関連・リーサル検出)で意思決定が壊れていないか。
- **Stage 3 head-to-head**: **M32 control(=production)vs 候補**、`ml_lethal_attackplan_v0only`、
  先後半々、300 試合スクリーニング → 段階判定(45% 未満で打切 / 55% 以上で増試合)。
- **採用条件: 95% CI 下限 > 50%。** 満たさなければ production 据え置き。
- ハーネス: `league/_diag_tier1abc_head_to_head.py`(`--baseline/--candidate-weights` 汎用、Wilson CI)を流用。
- 対フィールド検証(Kaggle 提出)は必要に応じ。ミラー ≠ フィールドの分散に注意
  (M128 提出でも 27 試合 55.6% CI[37,72] と判別不能だった)。

---

## 6. 非目的・境界

- **production 据え置き**: `policy_weights.json`/`value_weights.json`/既定 config/推論経路は、
  各フェーズが採用条件(§5)を満たすまで変更しない。実験は別 weights/別 config/別ブランチ。
- **担当領域の尊重**([[feedback_respect_ownership_boundaries]]): 実装は `ml_policy/` `search/` `learning/`
  `hidden_information/` `kaggle_replays/` 内で完結させる。`rule_based/` `action_selection/` の
  ゲーム判断ロジックには手を入れない(必要になれば独断せず AskUserQuestion)。
- **前身との関係**: 本書は `beyond-bc` を否定せず包含する。Beyond BC の Phase1=C(outcome-aware BC)は
  negative 確定だが、本書の Phase 4(RL)が「勝率目的の学習」を弱信号 BC でなく環境報酬でやり直す位置づけ。
- **推測でコードを語らない**: 本書の資産評価は実ファイル(`encoder.py`/`policy_model.py`/`value_model.py`/
  `match_context.py`/`ml_policy_agent.py`)と実測(SSH での torch 未導入確認)に基づく。

---

## 7. 決定ポイント(ユーザー判断を仰ぐ点)と本セッションの自律作業

### 7.0 Phase 0 速報(2026-07-24 夜、read-only 診断の結果 —— 詳細は `NIGHT-SESSION-2026-07-24.md`)

本設計の前提だった「転移ギャップ」について、着手前に 3 件の read-only 診断で強い証拠が出た:

- **H3(評価分解能)**: 現行の 300 試合ゲートは**真勝率 55.67% 以上でないと通過しない**。真 50.5–52% の
  検出力は 0.04–0.11。過去の negative(m64/m128/c2)は全て 49–52% 帯・CI が 50% 跨ぎで、
  **9件中ゲート通過 0件**。⇒ 「転移しなかった」の相当部分は**「測れていなかった」**。
  ただし outcome 重み付け BC(c1)だけは真に有害(CI が 50% 未満)と切り分けられた。
- **H1a(分布ミスマッチ)**: 学習データ(上位ログ)と配備先(フィールド)の相手アーキタイプ分布が
  **正規化 L1 距離 0.997 / KL 4.56 でほぼ半分ズレ**。policy は Alakazam ミラー(top 60.8%)特化なのに、
  フィールド 2・3 番手の **mega_lucario_ex(top 0% / field 17.5%)・archaludon_ex(top 0% / field 15.3%)は
  未学習**で、かつ自分の敗北相手の上位。⇒ **転移ギャップの主因は「特徴/容量/目的関数」ではなく
  学習データのカバレッジの可能性が高い。**

**設計への含意**: Phase 1 の第一手は「信念接続」単独よりも、**学習データのカバレッジ拡大**
([[project_multi_archetype_imitation]] を本線化)× **信念条件付け**の二本立てが有力。
ただしどちらも H3 の評価分解能を先に是正しないと効果を測れない。§7.1 の判断に反映。

### 7.1 ユーザー判断を仰ぐ点(朝に確認)

1. **順序の承認**: Phase 0(診断)→ 1(信念接続)→ 2(Transformer+dual-head)→ 3(ISMCTS)→ 4(RL deferred)で確定してよいか。
2. **Phase 4(RL)の位置づけ**: §3.2 の現実性評価(単一 GPU では非現実的)を受け入れ、
   「軽い fine-tune 止まり / 研究枠」に格下げしてよいか。それとも RL を本命として計算資源(クラウド等)を用意するか。
3. **Phase 2 のデプロイ前提**: Kaggle で numpy/onnx が使えなかった場合、「Transformer を純 Python MLP に蒸留」で
   妥協してよいか(それとも Transformer 推論のためにデプロイ制約側を再検討するか)。

### 7.2 本セッション(夜間)の自律作業 —— 判断不要な範囲で「順番に」進める

- **[完了] 設計書(本書)**。
- **Phase 0 診断の実装計画**を別ファイルに作成([[feedback_design_vs_implementation_plan]] に従う)。
- **GPU デスクトップのプレップ**: Blackwell 対応 torch(cu128 系)の導入・CUDA 動作確認。
  Phase 2/4 の前提作業で、どの順序が承認されても無駄にならない。
- Phase 0 診断のうち **read-only で安全に走るもの**(相手分布比較・相手別模倣一致率など)を可能な範囲で実行し、
  結果を results に残す。
- **朝用サマリ**(`docs/plans/neural-agent/NIGHT-SESSION-2026-07-24.md`)に、やったこと・結果・次の判断待ちを記録。

**やらないこと**: production 重み/ config の変更、承認前の Transformer/RL 学習の本走、
未検証の長時間ジョブの無人投入。
