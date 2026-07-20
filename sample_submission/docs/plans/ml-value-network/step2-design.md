# ML Step2: 模倣ポリシー(選択肢スコアリング) 設計書

作成日: 2026-07-20 / 対象イシュー: 未起票(Step1 = [#72](https://github.com/pokemon-ai-challenge/pokemon-tcg-ai-battle/issues/72) の後続) / 状態: **方針確定・実装未着手**

関連ドキュメント:
- 親ドキュメント(全体方針・ロードマップ): [`../individual/shogo/ml-agent-plan.md`](../individual/shogo/ml-agent-plan.md)
- Step1 設計書(エンコーダ・状態バリューの土台): [`./design.md`](./design.md)
- Step1 アルゴリズム選定理由: [`./algorithm-selection.md`](./algorithm-selection.md)
- Step1 オフライン評価レポート: [`./step1-offline-evaluation.md`](./step1-offline-evaluation.md)
- アルゴリズム選定理由(本 Step2): [`./step2-algorithm-selection.md`](./step2-algorithm-selection.md)
- オフライン評価レポート(最終判断): [`./step2-offline-evaluation.md`](./step2-offline-evaluation.md)
- 後続(確定リーサル探索とのハイブリッド化): [`./step2-lethal-hybrid.md`](./step2-lethal-hybrid.md)

ブランチ運用: 本 Step2 は `feature/ml-value-network`(Step1, PR #73, 未マージ)から分岐した
`feature/ml-imitation-policy` で実装する。PR の base は `feature/ml-value-network` に設定し
(PR #71 → #73 と同じスタック方式)、Step1 が integration にマージされ次第 base を
retarget する。

---

## 1. 目的とスコープ

Kaggle 上位リプレイ(フーディン使用プレイヤー)から**模倣学習(Behavior Cloning)**で
「与えられた選択肢(`SelectData.option`)の中からどれを選ぶか」をスコアリングするポリシーを作る。
`ml-agent-plan.md` のロードマップ「2. 模倣ポリシー」に対応する。

design.md §3.4 が Step1 の段階で確保しておいた「状態特徴/選択肢特徴の分離」を使い、
`ptcg_ai/learning/encoder.py` の `encode_options()`(現状 `NotImplementedError`)を実装する
ことが中心作業になる。状態特徴(`encode_state`)は Step1 のものをそのまま再利用し、二重実装
しない。

明確にスコープ外:
- determinization 探索・`search_begin()` との結合(Step3)。模倣ポリシーは Step3 で
  「探索中の相手の行動モデル」「探索の事前分布」としても使う想定だが、その結合自体は Step3。
- 自己対戦によるファインチューニング(Step4)。
- 複数選択(`maxCount > 1`)の最適化(§2.4 参照。初期スコープでは単純な貪欲フォールバック)。

既存コード(`ptcg_ai/rule_based/`、`ptcg_ai/action_selection/` とその `handlers/`)は変更しない。
これらは他メンバー担当領域であり、Step1 の design.md §5.2 で確認済みの制約をそのまま踏襲する。
新規実装は `ptcg_ai/learning/` 配下(既存の `encoder.py`/`value_model.py` と同居)と、
新規 `ptcg_ai/ml_policy/`(実行時エージェント本体)、新規 config
(`configs/ml_policy.json`)に閉じる。

---

## 2. データ

### 2.1 対象リプレイの絞り込み

design.md §8 で確定済みの方針を踏襲し、**フーディン(alakazam)使用プレイヤーのリプレイに
絞る**。`kaggle_replays/deck_predictor/output/deck_labels.jsonl`
((episode_id, player_index) → archetype、9,396 行)で `archetype == "alakazam"` の
(episode_id, player_index) に該当する局面のみを学習対象にする。

理由: Step2 の模倣対象は「現行 `deck.csv` のフーディンデッキで何を選ぶべきか」であり、
デッキ非依存の状態評価だった Step1(バリュー関数)とは性質が異なる。他アーキタイプの
プレイングを混ぜると、フーディンでは選べない/意味のない選択肢のパターンを学習しかねない。

### 2.2 抽出元

`kaggle_replays/extract_training_data.py` が出力する `(observation, action)` ペアには、
`observation.select.option`(選択肢一覧)と実際に選ばれた `action`(インデックス列)が
既に含まれている。Step1 の `extract_value_dataset.py` のように**勝敗ラベル結合のための
新規抽出は不要**で、既存パイプラインに以下を追加する形で足りる:

- `deck_labels.jsonl` を結合し、フーディン使用者の行のみへ絞り込む
- `rank_at_fetch` によるサンプル重み付け(Step1 §6 と同じ思想)は踏襲する

既存の `kaggle_replays/training_data/pairs.jsonl`(1,956 行、`extract_training_data.py` の
旧実行結果)は 4,698 件のリプレイ全体を対象にしていない可能性がある(design.md §2.1 で
Step1 用には不十分と判定済み)。Step2 でも同様に、**全リプレイに対して再実行**してから
フーディン絞り込みを行う。

### 2.3 監査(タスク1相当、実施前に必須)

Step1 の audit_report.md(タスク1)に相当する監査を、Step2 でも最初に行う:

- フーディン使用プレイヤーの局面数・エピソード数
- `SelectType` / `SelectContext` / `maxCount` の分布(§2.4 のスコープ判断の根拠)
- `minCount == maxCount == 1` の単純選択が全体の何割か

参考値(旧 `pairs.jsonl`、1,956 行、**全アーキタイプ混在**での粗い実測。Step2 開始前に
フーディン限定・全量データで取り直す):

| 指標 | 値 |
|---|---|
| `maxCount == 1` | 1,830 / 1,956 (93.6%) |
| `SelectType == MAIN` | 1,078 (55.1%) |
| `SelectType == CARD` | 681 (34.8%) |
| `SelectType == YES_NO` | 81 (4.1%) |
| その他(ENERGY/DISCARD系/SKILL/EVOLVE/ATTACHED_CARD) | 116 (5.9%) |

この粗い実測から、`MAIN` + `CARD` の単純選択(`maxCount == 1`)だけで大半の意思決定点を
カバーできる見込みが立つ。§2.4 のスコープ限定はこの実測に基づく。

### 2.4 初期スコープ: `maxCount == 1` の単純選択のみ

複数選択(手札を N 枚捨てる、ベンチに複数体並べる、等)は、選択肢間の**組み合わせ**を
評価する必要があり、単純な「各選択肢を独立にスコアリングして上位を選ぶ」では最適とは
限らない(捨て札の組み合わせ最適化など)。§2.3 の実測が示す通り比率は小さいため、
初期スコープでは `maxCount > 1` を模倣ポリシーの対象から外し、**フォールバック
(貪欲法: 個別スコア上位から `minCount`〜`maxCount` 個を独立に選ぶ、または既存
`rule_based_agent` に委譲)** で埋める。§6 タスク7 で「独立スコアの貪欲法でどこまで
劣化するか」を計測し、必要なら次イテレーションで拡張する。

### 2.5 train/val/test 分割

Step1 と同じく**エピソード単位**で分割する(design.md §2.4 と同じリーク回避の理由)。

---

## 3. エンコーダ設計(選択肢特徴)

配置: `ptcg_ai/learning/encoder.py` の `encode_options()` を実装(既存の
`NotImplementedError` を置き換える)。状態特徴 `encode_state()` は変更しない。

### 3.1 状態特徴との結合方式

各選択肢 `option_i` のモデル入力は `state_features ++ option_features(option_i)` の連結
とする(状態は選択肢間で共通、選択肢部分だけが変わる)。同一 `select` 内の全選択肢を1バッチ
として扱い、モデルはスカラースコアを1つ出す(§4.1)。

### 3.2 選択肢特徴の内訳

`cg/api.py` の `Option` は `type`(`OptionType`)ごとに使えるフィールドが異なる
(`cardId`/`attackId`/`index`/`area`/`playerIndex` 等)。`SelectType`/`SelectContext` を
またいで共通に埋められる特徴と、`OptionType` 別の特徴を分ける:

- **共通(全 Option 型)**:
  - `SelectType`/`SelectContext` の one-hot(または整数値、モデル選定時に決定)
  - 選択肢の出現順インデックス(正規化)、同一 `select` 内の選択肢総数
- **`CARD`(`cardId` を持つ選択肢)**:
  - `card_cache` からカード種別(ポケモン/トレーナーズ/エネルギー)、`CardType` の内訳
  - 対象がポケモンなら Step1 の `_pokemon_features` 相当(HP比率・攻撃打点・KO可否等)を
    再利用して数値化。手札のカードなら盤面へ出した場合の簡易効果(進化/エネルギー付与等)
    の分類ラベル
- **`ATTACK`/`SKILL`(`attackId` を持つ選択肢)**:
  - `attack_features.resolve_damage()` / `can_ko()`、
    `energy_requirements.energy_shortfall()` / `is_energy_sufficient()`(Step1 の
    `_pokemon_features` と同じ流用元)
- **`ATTACHED_CARD`/`ENERGY`(`toolIndex`/`energyIndex` を持つ選択肢)**:
  - 対象ポケモンの `attacker_score`、付与後の充足度変化(可能なら差分で表現)
- **`YES_NO`**: 追加特徴なし(共通特徴のみ、`type` の one-hot が実質的な特徴になる)
- **`COUNT`(`number` を持つ選択肢)**: `number` 自体を正規化して特徴化

Step1 の `_pokemon_features` / `attack_features` / `energy_requirements` /
`board_features` を**そのまま関数として再利用**し、Step2 で新規に盤面評価ロジックを
作らないことを徹底する(design.md の「既存特徴量の流用」方針の継続)。

### 3.3 差し込み口

Step1 の `encode_state` 同様、`extra_features` 引数で hidden_information 由来の特徴
(相手の非公開情報推定)を後から足せる空きを `encode_options` にも用意する。本 Step2 では
未使用(design.md §3.1 と同じ理由: まず公開情報のみでベースラインを確定させる)。

---

## 4. モデルと学習

### 4.1 モデル構造: リストワイズ・スコアリング

固定行動空間の分類ではなく、**可変長の選択肢集合に対するリストワイズ学習**とする:

1. 同一 `select` 内の各選択肢について `score(state_features, option_features(option_i))`
   をスカラーで出す(共有パラメータの MLP または線形モデル。選択肢ごとに独立に前向き計算)
2. 全選択肢のスコアに `softmax` を適用し、実際に選ばれたインデックスに対する
   **交差エントロピー損失**で学習する

Step1 のロジスティック回帰(状態1つ→勝率1つ)とは損失関数の形が異なる(1対Nの
softmax-CE)ため、学習パイプラインは Step1 の `kaggle_replays/value_net/train.py` を
そのまま流用せず、**新規 `kaggle_replays/policy_net/` 配下に用意する**
(`deck_predictor/` や `value_net/` と同じ「抽出→構築→学習→評価」の分業構成を踏襲)。

- ベースライン: 選択肢特徴のみ・状態特徴なしの線形モデル(状態を見ずに「良さそうな選択肢」を
  当てられるかの下限)
- 本命: 状態特徴 + 選択肢特徴を入力にした小型 MLP(共有重み、選択肢ごとに独立適用)

### 4.2 学習と提出物の分離

Step1(`ptcg_ai/learning/value_model.py`)と同じパターン: ローカル学習
(sklearn/PyTorch)→ JSON 重み(`ptcg_ai/learning/policy_weights.json`)→ 純 Python 推論
(`ptcg_ai/learning/policy_model.py`、新規)。重み欠損時は `is_ready = False` で
安全側フォールバックする設計も踏襲する。

### 4.3 キャリブレーション

模倣ポリシーの出力は「勝率」ではなく「選択肢間の相対的な良さ(softmax 確率)」なので、
Step1 の温度較正(T >= 1.0、確率を強気に出さない)とは目的が異なる。ここでの確率は
argmax 選択にしか使わない想定のため、**Step2 では較正を必須にしない**。ただし
将来 Step3 で「探索中の相手の行動モデル」として複数候補の分布を使う場合は較正が要る
可能性があり、そのときに改めて検討する(先送りであり却下ではない)。

---

## 5. 評価

### 5.1 オフライン評価(必須)

エピソード単位 holdout(§2.5)で、**選択タイミングごとの Top-1 一致率**
(モデルが最高スコアを付けた選択肢が実際にプロが選んだ選択肢と一致する割合)を測る。

判断基準は Step1 と同じ「自明なベースラインを明確に上回ること」。ベースラインは2つ:

1. **一様ランダム選択**: 期待一致率 = 選択肢内での `1/選択肢数` の平均。多くの決定点で
   選択肢が少ない(2〜5個)ため、この基準は緩め。
2. **現行 `rule_based_agent` との一致率**(Step1 にはなかった発想。design.md 執筆時点では
   接続対象外だった `ptcg_ai/rule_based/` を、ここでは**書き換えずに参照のみ**する):
   記録済み `observation` に対して `rule_based_agent(obs)` を再実行し、実際にプロが選んだ
   選択肢とどれだけ一致するかを測る。模倣ポリシーがこのベースラインを明確に上回るかが
   Step2 の主要な合否判定になる。`rule_based_agent` は observation のみに依存する
   決定的関数である前提(内部状態やタイマー依存の分岐があれば§7のリスクとして扱う)。

層別評価は必須:
- **`SelectType`/`SelectContext` 別**(決定の種類ごとに難易度が大きく異なる。§3.2 で
  ATTACK/CARD/YES_NO 等を別特徴として扱うため、精度も種類別に大きく変わりうる)
- **ターン帯別**(Step1 と同様、序盤ほど教師信号がノイジーな可能性がある)

### 5.2 オンライン評価(対戦リーグ)は別タスク

`ml-agent-plan.md` が「足りない部品4」として挙げる「新エージェント vs 現行ルールベースの
自動対戦リーグ(500試合以上、信頼区間付き)」は、本ブランチ系統には未整備。
Step2 の完了条件は §5.1 のオフライン評価 PASS までとし、対戦リーグの構築(または既存資産の
発掘・移植)は Step2 完了後の別タスクとして扱う。過去に別系統ブランチ
(`feature/ml-architecture-implementation`、旧 `src/decision/` アーキテクチャ)で
自己対戦データ収集・A/B ハーネスを作った形跡があるが、モジュール構成が現行の `ptcg_ai/`
とは別物のため直接の移植はできない。「予算分割は退化しやすい」「do-no-harm 構造が
安全」といった知見のみ参考にする。

---

## 6. タスク分解

1. [x] データ監査(§2.3): フーディン限定・全量での局面数、`SelectType`/`Context`/
   `maxCount` 分布の確定 → `kaggle_replays/policy_net/audit_report.md`
2. [x] 抽出パイプライン拡張(§2.2): `extract_training_data.py` 相当の全量再実行 +
   `deck_labels.jsonl` 結合によるフーディン絞り込み → `kaggle_replays/extract_policy_dataset.py`、
   `kaggle_replays/training_data/policy_positions.jsonl.gz`(187,690件)
3. [x] `encode_options()` 実装 + 単体テスト(§3、`ptcg_ai/learning/encoder.py`)。実リプレイ
   187,690件で例外・NaNなしを確認済み
4. [x] 学習パイプライン(線形ベースライン → MLP)+ 重みエクスポート
   (§4、`kaggle_replays/policy_net/`)
5. [x] 純 Python 推論(§4.2、`ptcg_ai/learning/policy_model.py`)。独立実装(numpy)との
   フォワードパス誤差 8.9e-16 を確認(ゴールデンテスト)
6. [x] オフライン評価(§5.1、ベースライン1・2、層別評価)→ **PASS**
   (詳細判断: [`./step2-offline-evaluation.md`](./step2-offline-evaluation.md))
7. [ ] `maxCount > 1` の貪欲フォールバックの劣化計測(§2.4、未計測)
8. [x] 新規 `ptcg_ai/ml_policy/ml_policy_agent.py` で単体エージェントとして動く状態にした
   (`ptcg_ai/core/agent.py` の `AGENT_TYPE` 分岐に1行追加、既存 `rule_based` 分岐は変更なし)。
   `configs/ml_policy.json` は現時点で切り替える設定値が無いため見送り(§8 参照)
9. [x] (完了条件外だったが実施)対戦リーグ基盤の構築(§5.2)→ `league/`(新規)。
   `ml_policy` vs `rule_based` 500試合(同一デッキミラー戦、先手/後手交互)で
   **`ml_policy` 勝率86.4%(95%CI [83.1%,89.1%]相当、`rule_based`視点13.6% [10.9%,16.9%])**
   を確認。詳細: [`./step2-offline-evaluation.md`](./step2-offline-evaluation.md) §8

**Step2 はタスク1〜6の完了(オフライン評価 PASS)をもって完了とする。** タスク7・8は
模倣ポリシーを実際にエージェントとして動かすための後続作業として同 Issue 内で扱うが、
Step1 が「オフライン評価 PASS」を完了条件としたのと同じ考え方で、まずオフライン精度の
検証を優先する。

---

## 7. リスク

- **`rule_based_agent` 再実行コスト**: §5.1 のベースライン2は全 test 局面で
  `rule_based_agent(obs)` を再実行する。局面数が多い場合、実行時間が評価のボトルネックに
  なりうる。サンプリングでの近似も検討する。
- **選択肢特徴の型ごとの表現力不足**: `OptionType` ごとに使える情報が大きく異なる
  (§3.2)。特に `CARD` 型は「手札のこのカードを出す/使う」という多様な意味を持ちうるため、
  単純な集約特徴では表現力が不足する可能性がある。層別評価(`SelectContext` 別)で
  弱い種類を特定し、必要なら個別に特徴を追加する。
- **フーディン限定データによるサンプル数減少**: Step1 の alakazam 局面数は 23,840
  (全体の一部)。Step2 はこれをさらにエピソード単位・意思決定点単位で分割するため、
  `SelectContext` によっては学習サンプルが薄くなる種類が出る見込み。§2.3 の監査で
  事前に把握し、薄い種類は初期スコープから除外する判断材料にする。
- **模倣の上限は模倣対象の強さ**: `ml-agent-plan.md` が既に明記している通り、模倣学習は
  「上位プレイヤーの真似」が上限。Step2 の目的は「現行ルールベースを模倣ポリシーが
  上回るか」の検証であり、最終的な強さの追求は Step3(探索)以降。

---

## 8. 人間の判断による確定事項(2026-07-20)

- **対象デッキ・データ**: design.md §8 の確定に従い、フーディン使用プレイヤーのリプレイに
  限定する。
- **初期スコープ**: `maxCount == 1` の単純選択のみを模倣ポリシーの対象とする(§2.4)。
  複数選択は貪欲フォールバックで初期対応し、必要性が確認できてから拡張する。
- **接続点**: `ptcg_ai/rule_based/`・`ptcg_ai/action_selection/` は変更しない。接続は
  `ptcg_ai/learning/` / `ptcg_ai/ml_policy/` 配下の新規コードで完結させる。
  `ptcg_ai/core/agent.py` への1行追加のみ許容する。新規 config(`configs/ml_policy.json`)は
  当初想定していたが、`AGENT_TYPE` 自体が `core/agent.py` のハードコードされた定数で
  切り替わる仕組み(config 駆動ではない)であり、Step2 時点で config にトグルすべき値が
  無かったため実装時に見送った(実装時判断、2026-07-20)。デッキ選択は `rule_based_agent.py`
  の `read_deck_csv()`(deck.csv を読むだけの共有ユーティリティ)をそのまま再利用する。
- **評価基盤**: 対戦リーグ(オンライン評価)は Step2 の完了条件に含めない。オフライン
  Top-1 一致率(現行 `rule_based_agent` 比較込み)で PASS/FAIL を判断する。
