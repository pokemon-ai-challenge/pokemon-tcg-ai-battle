# ML Step1: 状態・行動エンコーダ + 勝率予測器(バリュー関数) 設計書

作成日: 2026-07-20 / 対象イシュー: [#72](https://github.com/pokemon-ai-challenge/pokemon-tcg-ai-battle/issues/72) / 状態: **完了(オフライン評価 PASS)**

関連ドキュメント:
- 親ドキュメント(全体方針): [`../individual/shogo/ml-agent-plan.md`](../individual/shogo/ml-agent-plan.md)
- アルゴリズム選定理由(なぜこの手法か・却下した代替案): [`./algorithm-selection.md`](./algorithm-selection.md)
- オフライン評価レポート(最終判断): [`./step1-offline-evaluation.md`](./step1-offline-evaluation.md)
- 後続(Step2: 模倣ポリシー): [`./step2-design.md`](./step2-design.md)

---

## 1. 目的とスコープ

Kaggle 上位リプレイから教師あり学習で**勝率予測器(バリュー関数)**を作り、あわせて
**状態エンコーダ**(Observation → 固定長特徴ベクトル)を全 ML パイプラインの共通土台として
確立する。エンコーダは本 Step1 の副産物ではなく主目的の一つであり、
`ml-agent-plan.md` のロードマップで「1. エンコーダ + 勝率予測器」として最重要に位置づけられている
足りない部品を埋める。

イシュー #72 の対応スコープであり、以下は明確にスコープ外:
- 模倣ポリシー(選択肢スコアリング。`ml-agent-plan.md` ロードマップの Step2)
- determinization 探索・`search_begin()` との結合(同 Step3)
- 自己対戦によるファインチューニング

既存コード(`ptcg_ai/rule_based/` 等の現行ルールベースエージェント)は変更しない。新規実装は
`ptcg_ai/learning/`(現状 `__init__.py` のみの空パッケージ)配下と、`ptcg_ai/core/config.py` の
仕組みに乗る新規 config ファイルに閉じ、既存の実行パスとは config 切り替えで共存させる
(`ml-agent-plan.md` の「既存コードと混ぜない」方針を踏襲)。

---

## 2. データ

### 2.1 現状

- `kaggle_replays/replays/` に **4,698 件**のリプレイ(`episode-<id>-replay.json`)が取得済み。
- 既存の `kaggle_replays/training_data/pairs.jsonl` は **1,956 行**あるが、
  `kaggle_replays/extract_training_data.py` による古い部分抽出(勝敗ラベルなし、
  かつ全リプレイを対象にしていない可能性がある)であり、本 Step1 用には
  **勝敗ラベル付きで再抽出**が必要。

### 2.2 リプレイ形式(勝敗ラベルの取り方)

`extract_training_data.py` の docstring どおり、各リプレイ JSON の `steps[i][player]["observation"]`
がそのプレイヤーに提示された観測、`steps[i+1][player]["action"]` がその観測に対して実際に選んだ
行動、`steps[i][player]["status"]` が `"ACTIVE"` のときだけそのプレイヤーが実際に意思決定を
行っている(`INACTIVE` はプレースホルダ)。

勝敗ラベルはリプレイ JSON の**トップレベル `"rewards"`** フィールド(player_index 順のリスト。
例: `[1, -1]` は player_index 0 の勝ち・1 の負けを表す)から取得する。これは
`extract_training_data.py` が現状参照していないフィールドであり、本 Step1 の抽出スクリプトで
新たに読む。局面(observation)ごとに、その局面を観測したプレイヤー自身の `rewards[player_index]`
の符号から「勝ち=1 / 負け=0」のラベルを一意に付与する(同一ゲーム内の全局面に同じラベルが付く)。
勝敗が付かないエピソード(引き分け・エラー終了などで `rewards` が正負に分かれないもの)は
原則除外とし、該当件数はデータ監査(§6 タスク1)で確認して記録する。

### 2.3 抽出スクリプト

`extract_training_data.py` を直接拡張するか、勝敗ラベル用の新規スクリプト
`kaggle_replays/extract_value_dataset.py` を作るかは実装フェーズで判断するが、いずれにせよ
既存の `extract_training_data.py` が持つ以下の要素を踏襲する:
- `load_master_index()` による `kaggle_replays/index/episodes_master.jsonl` との結合
- `rank_at_fetch` / `leaderboard_score_at_fetch` をサンプル重みとして各行に付与
  (取得時点の順位が高いプレイヤーの局面を学習時に重視する用途)
- マスターインデックス未登録エピソードの件数を実行時に警告表示する

`kaggle_replays/deck_predictor/episode_window.py` の `EpisodeIndex` /
`in_recent_top_rank_window()` / `rank_bucket()` は、エピソード作成時刻・相手ランクによる
ウィンドウ絞り込みの既存実装であり、値ネットの学習データ絞り込み(例: 直近・上位ランク帯のみで
再学習)にもそのまま再利用できる候補として扱う。

### 2.4 train/val/test 分割

**エピソード単位**で分割する。同一ゲームの複数局面が train と val にまたがると、
終盤の局面から見れば勝敗がほぼ確定しているため、同一ゲーム由来の局面がリークして
オフライン精度を過大評価してしまう。`kaggle_replays/deck_predictor/` 系パイプラインが
`evaluate.py` 等で採用している「エピソード単位の train/valid 分割」と同じ考え方を踏襲する。

---

## 3. エンコーダ設計

配置: `ptcg_ai/learning/encoder.py`(新規)。

### 3.1 入力方針(方式a: 公開情報のみ)

入力は**公開情報のみ**。自分の手札(`PlayerState.hand`、`cg/api.py` の docstring で
「自分視点では常に非 None、相手は None」と定義されている)は使うが、相手の非公開情報
(手札の中身・山札の並び)は使わない。`ptcg_ai/hidden_information/` の推定値(`marginals()` の
出力など)を特徴量に足すのは将来の拡張とし、エンコーダのインターフェースに
「hidden_information 由来の特徴を追加で受け取れる差し込み口」だけを用意しておく
(本 Step1 では未実装・未使用)。

この差し込み口の**最初の利用予定はマッチアップ(相性)特徴**である:

- 自分のアーキタイプ: 自デッキリストから確定的に導出(推定不要・リスクなし)
- 相手のアーキタイプ: `ptcg_ai/opponent_modeling/` の deck predictor の事後確率ベクトル(較正済み)

相性は盤面にアーキタイプが現れる中盤以降は盤面特徴から暗黙的に学習されるが、
序盤は公開情報だけでは持てない情報のため、この特徴が主に序盤の予測を改善すると期待される。
ベースライン(公開情報のみ)確定後に追加し、増分(特に序盤ターン帯の AUC)を計測して
採否を判断する(§5.1 の層別評価を使う)。

### 3.2 視点正規化

常に「自分/相手」の相対表現に正規化する。`State.yourIndex`(0 or 1)を基準に
`state.players[yourIndex]` を自分、`state.players[1 - yourIndex]` を相手として並べ替えてから
特徴量化する。先攻/後攻自体は `State.firstPlayer` との比較で別途フラグ化する(3.4 参照)。

### 3.3 固定長ベクトルの内訳

- **バトル場・ベンチのポケモン毎**の特徴(自分・相手それぞれ、`Pokemon` 単位):
  - 残 HP・最大 HP に対する比率(`Pokemon.hp` / `Pokemon.maxHp`)、ダメカン相当、
    付いているエネルギー数・種類(`Pokemon.energies`)
  - 技の打点と必要エネルギー充足度: `board_evaluation.attack_features.resolve_damage()` /
    `can_ko()`(弱点・抵抗力・可変ダメージ推定込みの実ダメージ計算)、
    `board_evaluation.energy_requirements.energy_shortfall()` /
    `is_energy_sufficient()`(技のエネルギー充足判定)をそのまま流用して数値化する
  - `board_evaluation.board_features.attacker_score()`(HP比率とエネルギー量からの
    「主力アタッカーらしさ」)、`is_likely_ko_next_turn()`(次の相手ターンで倒されやすいか)も
    流用候補
- **カウント系**:
  - 手札枚数(自分は `PlayerState.handCount` に加え内訳も使えるが、相手は
    `PlayerState.handCount` の枚数のみ。`hand` は相手側 None のため使えない)
  - 山札残枚数(`PlayerState.deckCount`)
  - サイド残枚数(`len(PlayerState.prize)`。中身が見えるかは別として枚数は常に分かる)
  - トラッシュ枚数(`len(PlayerState.discard)`)
- **ゲーム進行**:
  - ターン数(`State.turn`)、先攻/後攻(`State.firstPlayer` と `State.yourIndex` の比較)
  - 今ターンの制限フラグ: `State.supporterPlayed` / `State.energyAttached` /
    `State.retreated`(`State.stadiumPlayed` も同枠で追加候補)

`board_evaluation.board_features.prize_diff()`(サイド差)のような既存の集約特徴も、
エンコーダの1次元としてそのまま採用できる。

### 3.4 状態特徴と選択肢特徴の分離

`ml-agent-plan.md` が Step2(模倣ポリシー)で想定する「各選択肢をスコアリングして選ぶ」形式に
将来拡張できるよう、**状態特徴(state features)と選択肢特徴(option features)を最初から別関数に
分離**する。本 Step1 で実装するのは状態特徴のみ(`encode_state(state: State) -> list[float]` 相当)。
選択肢特徴(`SelectData.option` の各 `Option` → 特徴量)は Step2 のスコープであり、
本 Step1 ではインターフェース上の空き(関数シグネチャ・モジュール分割)だけ確保する。

---

## 4. モデルと学習

### 4.1 モデル選定

- ベースライン: ロジスティック回帰
- 本命: 小型 MLP(隠れ1〜2層)
- 参考(オフライン専用): LightGBM 等の GBDT。特徴量の性能上限を確認する用途にのみ使い、
  提出物には含めない。

### 4.2 学習と提出物の分離

学習はローカルで行い(sklearn / PyTorch いずれも可)、**提出物は JSON 重み +
numpy(または純Python)推論**とする。これは `ptcg_ai/opponent_modeling/ml_predictor.py` の
`MLDeckPredictor` が既に確立しているパターン(オフラインで `deck_predictor_weights.json` を
学習・出力し、ランタイムはそれを読み込んで行列積+softmax だけで推論する)をそのまま踏襲する。
`MLDeckPredictor` 同様、重みファイルが存在しない場合は例外にせず「未ロード状態」を持ち、
`is_ready` 相当のプロパティで判定できるようにする(欠落時もエージェントがクラッシュしない
安全側フォールバック)。

配置:
- 推論: `ptcg_ai/learning/value_model.py`(新規) + 重み JSON(同ディレクトリ、
  `MLDeckPredictor` が `deck_predictor_weights.json` を自身の隣に置くのと同じ配置)
- 学習スクリプト: `kaggle_replays/value_net/` 配下(新規、ローカル専用。
  `kaggle_replays/deck_predictor/` の `extract_decks.py` → `build_dataset.py` → `train.py` →
  `calibrate.py` → `evaluate.py` という分業構成を参考にする)

### 4.3 キャリブレーション

温度スケーリングを適用し、**T >= 1.0 の制約を必須とする**(生の見積もりより確率を強気に
出さない。プロジェクト方針「確信度は慎重側に倒す」に合わせる)。実装形は
`ml_predictor.py` の `meta.calibration.buckets`(エビデンス量に応じたバケットごとの温度)と
同じ構造を踏襲できるが、値ネットの場合「エビデンス量」に相当する軸(例: ターン数。
序盤ほど勝敗との相関が弱くキャリブレーションが崩れやすい、後述 §7)を用いる想定。
`meta.calibration` が無い(旧)重み JSON を読んだ場合は常に T=1.0 として後方互換を保つ、
という `ml_predictor.py` の設計もそのまま流用する。

---

## 5. 評価

### 5.1 オフライン評価

エピソード単位 holdout(§2.4)で以下を測る:
- AUC
- logloss
- キャリブレーションカーブ(予測確率 vs 実際の勝率)

層別評価を必須とする:
- **ターン帯別**(序盤の予測が弱いことを既知特性として可視化する。§7)
- **マッチアップ別**(アーキタイプ対ごと)。リプレイへのアーキタイプラベル付けは
  deck predictor を利用する。21 アーキタイプの総当たり(441 ペア)は 4,698 リプレイに対して
  薄すぎるため、主要アーキタイプ + Other への集約やティア帯での束ねを行う

判断基準: **自明なベースラインを明確に上回ること**。ベースラインは2つ併走させる:
1. サイド枚数差のみ: `board_evaluation.board_features.prize_diff()` 1変数のロジスティック回帰
2. **マッチアップ事前勝率表**: リプレイから集計した「アーキタイプ A vs B の実勝率」を
   引くだけのモデル。値ネットが序盤ターン帯でこれに勝てない場合、序盤の盤面特徴は
   情報を足せていない(= §3.1 のマッチアップ特徴追加が必要)と切り分けられる

### 5.2 オンライン評価(スコープ外に訂正。2026-07-20)

当初この節は「`configs/ml_value.json` を新設し、値ネットで現行ルールベースの盤面評価を
置き換え/補強した構成を現行 config と対戦させる」としていたが、これは前提が誤っていた。

値ネット(本 Step1 の成果物)は **「今の盤面」を1つ評価するだけ**であり、
「ベンチのどのポケモンに交代すべきか」のような**候補間の比較**に使うには、候補ごとに
「その選択をした後の盤面」を作ってスコアリングする必要がある。これは値ネット単体では
できず、Step2(選択肢特徴・模倣ポリシー)か Step3(determinization 探索、`search_begin`/
`search_step`)の仕事である。

加えて、盤面評価が実際に呼ばれている場所(候補の比較・選択)は
`ptcg_ai/action_selection/router.py` とその `handlers/`(コード内コメントに
「クラスタ①選択振り分け／担当B」とある、他メンバー担当領域)であり、`rule_based/` と
同格の「変更しない」対象である。`selector.py` の lethal_search 枠は「確定した手順を返す
完全探索」用であり、状態評価器である値ネットとは役割が異なるため間借りは適切でない。

したがって、**config 接続・対戦リーグは Step2/3 で選択肢スコアリングの仕組みが揃ってから
改めて行う**。Step1 の完了条件は §5.1 のオフライン評価 PASS までとする
(詳細判断: [`./step1-offline-evaluation.md`](./step1-offline-evaluation.md))。

---

## 6. タスク分解(イシューのチェックボックスと対応)

1. [x] データ量・質の監査(局面数、アーキタイプ分布、マッチアップペアの件数分布、勝敗バランス)
   → `kaggle_replays/value_net/audit_report.md`
2. [x] 勝敗ラベル付き抽出スクリプト(§2)
   → `kaggle_replays/extract_value_dataset.py`、`kaggle_replays/training_data/value_positions.jsonl.gz`
3. [x] エンコーダ実装 + 単体テスト(§3、`ptcg_ai/learning/encoder.py`)
4. [x] 学習パイプライン(ロジスティック回帰 → MLP)+ 重みエクスポート(§4、`kaggle_replays/value_net/`)
5. [x] numpy(実際は純Python)推論 + キャリブレーション(§4.2〜4.3、`ptcg_ai/learning/value_model.py`)
6. [x] オフライン評価(§5.1)→ **PASS**(`step1-offline-evaluation.md`)
7. [ ] ~~config 接続 → 対戦リーグ~~ → **Step2/3 へ繰越**(§5.2 参照。理由: 値ネット単体では
   候補間比較ができず、比較には選択肢スコアリング/探索が要る)

**Step1(イシュー #72)はタスク1〜6の完了(オフライン評価 PASS)をもって完了とする。**

---

## 7. リスク

- **上位エージェントの行動分布への偏り**: 学習データが上位プレイヤーの対戦に限られるため、
  特定アーキタイプ・特定戦術に対して過学習するリスクがある。§2.3 のランク帯・時期による
  ウィンドウ絞り込み(`episode_window.py` 相当)や、アーキタイプ別の層別評価で監視する。
- **局面ラベルのノイズ**: 序盤の局面は最終勝敗との相関が弱く、「勝ったゲームの1手目」を
  そのまま正例として学習すると序盤の評価が過信気味になりうる。ターン数による重み付けや、
  ターン数帯ごとの層別評価(オフライン AUC・キャリブレーションカーブをターン帯別に見る)を
  検討する。
- **Kaggle 提出環境の制約**: numpy 推論であれば時間・メモリの問題は起きにくいはずだが、
  実測して確認する必要がある(`ml-agent-plan.md` が指摘する「提出環境で使えるライブラリ・
  時間内に収まるかの確認」を Step1 の完了条件にも含める)。

---

## 8. 人間の判断による確定事項(2026-07-20)

- **使用デッキ**: 現行 `deck.csv` のフーディン(Alakazam)デッキ
  (フーディンライン 4-4-4、ノココッチ 3-3、超エネルギー軸)を中心に据える。
  値ネット(Step1)の学習は全アーキタイプのリプレイで行う(デッキ非依存の状態評価のため)が、
  層別評価はフーディン関連マッチアップを重点的に見る。Step2 の模倣学習は
  フーディン使用プレイヤーのリプレイに絞る想定。
- **接続点**: `ptcg_ai/rule_based/` のコードは変更しない。接続は新規 config
  (`configs/ml_value.json`)と `ptcg_ai/learning/` 配下の新規コードで完結させる。
  どうしても他メンバーの担当領域(rule_based 等)の変更が必要になった場合は、
  実装せずにまず調整する。
- **成功・撤退基準**: 事前の厳密な数値ラインは設けない。オフライン指標(§5.1)で
  ベースラインを上回ることを確認しながら反復し、改善が頭打ちになったら一旦撤退して
  設計を見直してから再挑戦する(反復前提の運用)。
