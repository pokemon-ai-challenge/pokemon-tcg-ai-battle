# PolicyModel 特徴量拡張 Tier3: 選択肢の1手先結果(consequence)特徴 — 設計・実装方針書

作成: 2026-07-22 / 改訂: 2026-07-22(ユーザーレビューを反映し全面改訂) /
対象: `AGENT_TYPE="ml_policy"` の PolicyModel(模倣学習)
状態: 設計中(実装未着手。既存productionへの変更は無し)

関連ドキュメント:
- [`design-and-implementation-plan.md`](./design-and-implementation-plan.md) — Tier1/2(識別情報埋め込み)。
  §3.6・§0にて「option を1手仮実行した結果の特徴」を**明示的にスコープ外**とし、
  「Tier1/2完了後に独立実験として扱う」としていた。**本文書がその独立実験**にあたる。
- [`../attack-enabling-search/design-and-implementation-plan.md`](../attack-enabling-search/design-and-implementation-plan.md) —
  「1手仮実行して結果を読む」手法の直接の先行実装(`ptcg_ai/search/attack_plan.py`)。
  本Tierは**v1の transaction 解決ロジック(§3.3/§3.6)をそのまま再利用**する(§4)。
- [`../../architecture/current-algorithm-overview.md`](../../architecture/current-algorithm-overview.md) §2 — 現行特徴量の全量。
- [`../../../results/2026-07-22_attackplan_kaggle_submission.md`](../../../results/2026-07-22_attackplan_kaggle_submission.md) —
  実戦357攻撃のうち0ダメージが50.7%、真に無意味なもの15.4%という実測値(本Tierの動機の裏付け)。
- [`../../../results/2026-07-22_attackplan_rock_fighting_energy_reproduction.md`](../../../results/2026-07-22_attackplan_rock_fighting_energy_reproduction.md) —
  本Tierの代表受け入れテスト(§6.1)と同一の盤面(ロック闘エネルギー+メガルカリオex vs フーディン+改造ハンマー)。
  attack_plan v0はこの盤面で「実行後に検証」して動作確認済み。本Tierは同じ盤面で
  「実行**前**の選択肢スコアリング」に効くことを確認する。
- 前例(不採用): [`../../../results/2026-07-22_policy_feature_expansion_tier1abc.md`](../../../results/2026-07-22_policy_feature_expansion_tier1abc.md) —
  識別情報埋め込みはオフライン微増でも実戦転移せず不採用。本Tierでも同じ転移リスクを警戒する(§8)。

---

## 0. TL;DR

PolicyModelは現状「選択肢を選んだ結果どう変わるか」を一切見ずにスコアリングしている。
これが「0ダメージ攻撃を選び続ける」問題の根本原因であることは、attack_plan v0の実戦検証
(357攻撃中55件=15.4%が真に無意味)で裏付け済み。

本Tierは選択肢ごとに「実行したら何が変わるか」を1手仮実行して読む特徴を追加するが、
**単に「HPが減ったか」だけでは不十分**というのが今回の改訂の核心。目指すのは以下のような
推論をPolicyModelが模倣学習だけで再現できること:

> 自分のフーディンは本来120ダメージ出せるが、相手のロック闘エネルギーにより実効0。
> 手札の改造ハンマーを使うと特殊エネルギーが外れ、実効ダメージが0→120に戻る。
> 上位プレイヤーがこの局面で改造ハンマーを選んでいるなら、その理由を学習できる。

これを実現するため、特徴を **(A) 即時結果** と **(B) 将来の行動可能性への影響** の
2グループに明確に分け、(B)、特に `delta_best_effective_attack_damage`(§3.2)を
本Tierの主役に据える。ATTACKだけを対象にしない(attack_planと機能が重複するため)。
学習・実行で同一の情報制約を使い、8特徴を一括投入せず少数ずつablationする。

---

## 1. 動機・背景

### 1.1 attack_plan v0が示した実戦データ

実戦357回のATTACK選択のうち、直接ダメージ0が181回(50.7%)、そのうち有益な副作用も
無い「真に無意味な攻撃」が55回(全体の15.4%)存在した
([results/2026-07-22_attackplan_kaggle_submission.md](../../../results/2026-07-22_attackplan_kaggle_submission.md))。
attack_plan v0はこれを**事後的に検出・veto**することで一部救済しているが、構造的な限界がある:

- **デッキに触れない1手の範囲でしか救済できない**(design-and-implementation-plan.md §3.2)。
- **事後vetoなので「そもそも選ばない/そもそも準備する」学習にならない**。無意味な攻撃を
  PolicyModelが選び、attack_planがそれを差し替える、という2段構えが対戦のたびに繰り返される。
- ATTACK選択肢にしか効かない。**「今この技を打つために何を用意すべきか」という準備行動の
  巧拙はそもそも対象外**(改造ハンマーを"先に"選ぶ、という判断そのものはattack_planの
  スコープに無い。attack_planはPolicyModelが無意味な攻撃を選んでしまった"後"にしか動けない)。

### 1.2 本Tierが埋めるべき穴: 「今すぐ何をすべきか」の予見性

Tier1abc(不採用)は「このカードが何か」を埋め込みで教えようとして失敗した
(識別embeddingだけでは学習データの薄さでノイズが乗り、実戦で悪化)。

本Tierは逆に、**「カードが何であるか」を直接学習させるのではなく、「この行動を取ると
自分の攻撃力・行動可能性がどう変わるか」という一般化しやすい関係を学習させる**方向を取る。
「改造ハンマーというカードIDだから選ぶ」のではなく「この行動をすると実効攻撃力が
+120されるから選ぶ」という、**カードを跨いで一般化できる**シグナルを与えることが狙い。
将来カードプールが増えても、この種の特徴は个別カード知識に依存せずに機能する。

---

## 2. 中核的な設計課題: 「仮実行」を学習・実行の両方でどう賄うか

### 2.1 Tier1/2との質的な違い

Tier1/2(識別情報埋め込み)は現在の`Observation`から直接読み取れる**静的特徴**であり、
encoderの契約(公開情報のみ・`obs.logs`/`obs.select`非依存・決定的)を破らずに実装できた。

本Tierの一部特徴は「この選択肢を実行したら」という仮想実行の結果であり、これを得るには
`cg.api.search_step`でゲームエンジンを1手進める必要がある。ただし**全特徴が仮実行を
要求するわけではない**(§5で静的計算を優先する範囲を切り分ける)。

### 2.2 学習時(オフライン)の情報経路と、なぜ(c)だけを使うか

`kaggle_replays/extract_policy_dataset.py` → `policy_net/build_features.py` の現行パイプラインは、
各意思決定点について**その時点の`observation`と実際に選ばれた`action`だけ**をリプレイJSONの
`steps[i][player]`から抜き出しており、`logs`は書き出し前に削除される。つまり
**「選ばれなかった選択肢を実行したら何が起きたか」は現行の中間データには存在しない。**

考えられる情報源:

| 経路 | 対象 | 精度 | 実行時に再現可能か |
|---|---|---|---|
| (a) 実際に選ばれた選択肢の**真の結果** | 選ばれた1件のみ | 完全に正確 | **不可**(実行時は選ぶ前に特徴が要る。事後の結果は使えない) |
| (b) 選ばれなかった選択肢の**反実仮想結果(replayの両陣営視点から相手の真の手札/山札順を復元)** | 全選択肢 | 高い(未来情報を含む) | **不可**(実行時は相手の非公開情報を知らない) |
| (c) dummy/estimated相当の非公開情報で仮実行した結果 | 全候補選択肢 | 中(本番相当の制約) | **可**(実行時と全く同じ手続き) |

**(a)/(b)は学習時にしか使えない「未来情報」であり、これを学習に使うと、実行時には
存在しない情報に依存したモデルができてしまう。** これはTier1abcで見た「オフライン改善が
実戦へ転移しない」問題の**別バリエーション**ではなく、**より直接的な原因**になりうる
(Tier1abcは単なる次元増のノイズだったが、(a)/(b)を使うと学習信号そのものが実行時に
再現不可能な情報を含む)。

### 2.3 方針(確定): 学習時も実行時と同一の生成手続きを使う

**`training consequence generation == runtime consequence generation`** を設計原則とする
(代替案は採用しない。これは推奨ではなく必須要件とする)。

具体的には、学習時のconsequence特徴計算も**(c)の経路のみ**を使う。dummy/estimated
どちらの隠れ情報スタブを採用するかはlethal_search/attack_planの既存実装
(`hidden_state_source`設定)との整合を見て決めるが、**学習側だけがreplayの正解情報
(相手の真の手札・山札順)を特徴生成に使うことは禁止**する。

この制約により、正解データ(a)/(b)より学習信号は不正確になるが、**train/runtime parityを
精度より優先する**(§8のablationで、この制約が実際にどの程度の精度低下を招くかは計測するが、
制約自体は緩めない)。

### 2.4 レイテンシ・予算

attack_planは「既存チェーンが選んだ1つのATTACK選択肢だけ」を検証するため軽量
(実戦平均0.79ms/回、p99 1.66ms)。本Tierは**候補となりうる複数選択肢**に対して特徴を
計算するため、対象範囲・計算方式(静的 or 仮実行)を絞らないとコストが膨らむ(§5・§7)。

### 2.5 非決定性(コイン技)

`attack_plan._has_coin`と同じ問題。コインが絡む選択肢/効果解決は当該特徴を「不明」を
表す中立値(0)にする。攻撃部分探索(attack_plan)がv0/v1でコイン技を「判定不能として
fall through」しているのと同じ考え方。

---

## 3. 特徴量の定義(2グループに再構成)

### 3.1 グループA: Immediate consequence(即時結果)

「その行動自体が直接何を変えたか」。実行前後の状態を1回比較すれば求まる、比較的
安価な特徴。

| 特徴 | 意味 | 計算 |
|---|---|---|
| `opp_hp_loss` | 相手アクティブのHP減少量 | 実行前後のHP差(`attack_plan._loss`と同一ロジック) |
| `self_hp_gain` | 自分の対象ポケモンのHP回復量 | 同上、自分視点 |
| `opp_energy_removed` | 相手のエネルギーが1枚以上外れたか(0/1) | `attack_plan._energy_removed`と同一 |
| `opp_special_energy_removed` | 上記のうちcardType==SPECIAL_ENERGYのものが外れたか(0/1) | `_energy_removed`をcardType条件で拡張 |
| `self_energy_added` | 自分の対象ポケモンにエネルギーが1枚以上付いたか(0/1) | ATTACHの直接効果。静的計算可(§5) |
| `cards_drawn` | 実行後に自分の手札枚数が増えた枚数 | `handCount`の差 |
| `pokemon_evolved` | 進化が発生したか(0/1) | 対象スロットの`Pokemon.id`が変化 |
| `stadium_changed` | スタジアムが変わった/新設されたか(0/1) | `state.stadium`の差 |

**注**: 旧案にあった`delta_effective_damage`は**廃止**する(§3.3で理由を説明)。

### 3.2 グループB: Future capability delta(将来の行動可能性への影響)— 本Tierの主役

「その行動によって、その後に取れる行動の価値がどう変化したか」。単発の状態比較ではなく、
**実行後の状態で改めて「自分が今取れる最善の行動は何か」を再評価**する必要があるため、
グループAより計算コストが高い。

| 特徴 | 意味 | 計算 |
|---|---|---|
| **`delta_best_effective_attack_damage`** | 実行前後で、自分が出せる**最大実効ダメージ**がどう変わるか(本Tierの最重要特徴、§3.2.1) | `best_effective_attack_damage_after - before` |
| `delta_can_ko` | 実行前後で、相手アクティブをKOできる技が使えるようになったか(0→1) | `best_effective_attack_damage_after >= 相手アクティブ残りHP` の真偽反転 |
| `delta_attack_ready` | 実行前後で`has_ready_attack`(何らかの技が撃てる状態)が0→1に変わったか | 既存状態特徴(encoder.py `_POKEMON_FEATURE_NAMES`)の差分 |
| `delta_energy_shortfall` | 実行前後の`min_energy_shortfall`の減少量 | 同上 |
| (将来検討) `delta_retreat_ready` | 逃げエネが足りるようになったか等 | 優先度低、初期実装では見送り |

#### 3.2.1 `delta_best_effective_attack_damage` の定義(受け入れ基準の中心)

```text
best_effective_attack_damage(state) =
    state の MAIN 選択肢のうち type==ATTACK であるものそれぞれについて、
    その技を仮実行した際の相手アクティブへの直接ダメージ(attack_plan._resolve_attack_damage
    と同一ロジック、resistance/lock等エンジン解決込みの実効値)を求め、その最大値。
    ATTACK選択肢が無い場合は 0。

best_effective_attack_damage_before = best_effective_attack_damage(現在の state)
候補option Xを仮実行(transaction解決、§4)→ 結果 state' を得る
best_effective_attack_damage_after  = best_effective_attack_damage(state')

delta_best_effective_attack_damage = after - before
```

`attack_plan._v0`が「MAIN選択肢のうちtype==ATTACKのものを列挙し、それぞれ
`_evaluate_attack`する」ループを既に持っており(§4参照)、`best_effective_attack_damage`は
その最大値を取るだけの薄いラッパーで実装できる。

**代表例(受け入れテスト、§6.1と同一盤面)**:

```text
before(ロック闘エネルギー装備中):
  best_effective_attack_damage_before = 0

改造ハンマーを仮実行(transaction解決、相手の特殊エネルギーを対象に選択):
  best_effective_attack_damage_after = 120

delta_best_effective_attack_damage = +120
opp_special_energy_removed = 1
```

### 3.3 `delta_effective_damage`(旧案)を廃止する理由

旧案の`delta_effective_damage`は「仮実行による相手Active HP減少量」であり、実質的に
グループAの`opp_hp_loss`と同じ意味だった。この定義では、改造ハンマー・エネルギー貼付・
進化・スタジアムのようにHPを直接減らさない準備行動の価値を一切表現できず、
**本Tierが最も学習させたい「改造ハンマーを打つ理由」を表現できていなかった**。
`opp_hp_loss`(即時)と`delta_best_effective_attack_damage`(将来の行動可能性)に
明確に分離することで、この欠落を埋める。

---

## 4. Multi-step select(複数選択を要する行動)の扱い

「1 option = search_step 1回」を前提にしない。改造ハンマー(PLAY→対象の特殊エネルギー選択
→効果解決)、ハイパーボール(PLAY→捨てるカード選択→サーチ対象選択→効果解決)のように、
1つのMAIN選択の後に追加のSelectが発生するカードは多い。

**対応: `attack_plan.py`のtransaction解決ロジック(§3.3/§3.6、`_transaction`/`_start_transaction`)
をそのまま再利用する。** 既に「MAIN選択→派生コールバックを解決→MAIN復帰 or 終局まで
進める」実装があり、以下の性質を満たす:

- 同一ターン・同一`yourIndex`でMAINへ戻ってきたら成立。
- ターン/`yourIndex`が変わったら失格。
- `COIN`ログが出た線は判定不能として除外(§2.5)。
- deckに触れる効果(DRAW/SHUFFLE/デッキ間移動)は既存の`_is_deck_touching`で検出可能
  (ただし本Tierでは「deckに触れるかどうか」で候補を絞る必要は無い。attack_plan v0/v1の
  用途とは違い、本Tierは**手札にある任意の行動の価値を学習データとして記録するだけ**なので、
  サーチ札のような「デッキに触れる」効果も特徴生成の対象にしてよい。ただし山札の中身は
  非公開情報のため、サーチ先の選択は§2.3のdummy/estimated制約の範囲でしか解決できない点に注意)。

### 4.1 追加選択の分岐数が多い場合の扱い

改造ハンマーのように「相手の特殊エネルギーを1枚選ぶ」程度の分岐は小さいが、
選択肢が多い効果(複数枚から選ぶサーチ等)は組み合わせ数が爆発しうる。

- **v1スコープ**: 追加選択が発生する行動のうち、`attack_plan`の
  `max_combinations_per_select`と同じ上限内で全解決を試し、
  `delta_best_effective_attack_damage`が最大になる解決を代表値として採用する
  (「上手く使えばどれだけの価値があるか」を学習データとして与える設計判断。
  上位プレイヤーの模倣データは基本的に「良い対象を選んだ結果」であるため、この
  楽観的な代表値の取り方はデータの傾向とも整合する)。
- 上限を超える、または解決に失敗する行動は**「unknown」**とし、本Tier特徴は全て0
  (通常特徴のみで判断)にフォールバックする。**「スコープ外として0」と「本当に効果が
  無くて0」を区別するため、`consequence_unresolved`フラグ(1特徴)を追加することを検討**
  (§9未決事項)。
- 2段階以上のネストした追加選択(選択の結果さらに選択が発生するケースが2階層以上続く)は
  v1では対象外とし、将来的に短いaction sequenceとして解決することを検討する(§10非目標)。

---

## 5. 静的計算 vs 仮実行(search_step)の使い分け

「仮実行できるから全部仮実行する」のではなく、正確性・速度・train/runtime parityの
バランスを優先する。**カード効果テキストのハードコードは引き続き禁止**(CLAUDE.mdの
方針、attack_planと同じ)だが、**ゲームルールとして決定的に計算できるもの**は静的計算を
優先してよい。

| 分類 | 例 | 計算方式 |
|---|---|---|
| ルールが自明で決定的な行動 | 基本エネルギーのATTACH(対象・効果が1通りに定まる) | **静的計算を優先**。既存の`board_evaluation`(`min_energy_shortfall`/`has_ready_attack`計算)を仮想的なエネルギー追加後の状態に適用できるなら、search_stepを使わない |
| 進化(EVOLVE) | ステージ変化によるHP/技セットの変化 | 静的計算できるか調査(`all_card_data()`から進化後カードの`attacks`/`hp`が判る場合は静的、不明な副次効果があれば仮実行にフォールバック) |
| カード効果テキスト依存の行動 | 改造ハンマー・ハイパーボール等のITEM/SUPPORTER | **仮実行(search_step)必須**。効果をハードコードしないという設計原則(attack_planと同じ)を守るため |
| ATTACKの実効ダメージ | resistance/lock等エンジン解決込みの値 | **仮実行必須**(静的な`Attack.damage`は可変ダメージ技で0を返す等、信頼できない。attack_plan既知の知見) |

この切り分けにより、レイテンシの大部分を占めうる「多数のATTACH選択肢」を静的計算で
安価に処理し、仮実行はITEM/SUPPORTER/ATTACK等の**カード効果解決が本質的に必要なものだけ**
に絞る。

---

## 6. 段階的な実装計画

Tier1abcの教訓(「まとめて全部入れない」)を踏襲するが、**ATTACKを本命にしない**
(§1.2)。ATTACKはattack_planと機能が重複するため、Stage3aは基盤検証に留め、
Stage3bで本命(準備行動)を評価する。

### Stage3a: 仮実行基盤の検証(対象: ATTACKのみ)

**目的は勝率改善の証明ではなく、基盤の正しさの検証**(attack_planと機能領域が
重なるため、勝率改善を必須条件としない)。

- `best_effective_attack_damage(state)`ヘルパーの実装(§3.2.1)。`attack_plan.py`の
  `_evaluate_attack`/`_resolve_attack_damage`を**探索(search)から独立した共有
  ユーティリティ**として切り出す(attack_plan.py自体は変更しない。担当領域の尊重と、
  既にKaggle提出済みの動作を壊さないため)。
- グループAの`opp_hp_loss`もこの段階で実装(ATTACK選択肢に対して)。
- DoD:
  - [ ] search_stepによる特徴生成が正しく動く(単体テストで固定盤面を突合)。
  - [ ] train/runtime parity(§2.3)を満たす経路で計算されている。
  - [ ] レイテンシ実測(p50/p95/p99、attack_planの計測項目を踏襲)。
  - [ ] 特徴値の妥当性確認(既知の攻撃力を持つ固定盤面で期待値と一致)。

### Stage3b: 本命(対象: ATTACH / EVOLVE / 相手エネルギー除去系ITEM / 対象を安全に解決できるITEM)

**ここが本Tierの価値の中心。** 主に評価する特徴: `delta_best_effective_attack_damage` /
`delta_can_ko` / `delta_attack_ready` / `delta_energy_shortfall` /
`opp_energy_removed` / `opp_special_energy_removed`。

- §4のtransaction解決ロジックをITEM系(対象選択を伴う)に適用。
- §5の静的計算をATTACH/EVOLVEに適用(コスト削減)。
- **必須の固定盤面テスト(受け入れテスト、§6.1)**:
  「ロック闘エネルギー → 改造ハンマー → 攻撃が通る」の盤面で、
  `best_effective_attack_damage_before=0` → 改造ハンマー実行後
  `best_effective_attack_damage_after=120`(`delta=+120`)、`opp_special_energy_removed=1`
  が正しく得られることを確認する(§3.2.1の代表例と同一)。
- DoD:
  - [ ] §6.1の受け入れテストが通る。
  - [ ] transaction解決(§4)が改造ハンマー・ハイパーボール等の代表的な複数選択カードで
    正しく動く(単体テスト)。
  - [ ] レイテンシが予算内(§7)。

### Stage3c: encoder/build_features/policy_model への配線 + オフライン評価

- `encoder.py`に新規関数を追加(既存`encode_*`系との違いをdocstringに明記。
  §2.1参照)。
- `build_features.py`: §2.3の制約(dummy/estimated相当のみ)で新特徴を計算。
  計算失敗/コイン検出/unresolvedは0埋め(fail-softとする理由: 仮実行はエンジンの
  非決定的失敗を含みうるため、既存のfail-fast方針の例外とする)。
- `policy_model.py`: 新特徴を選択肢特徴ベクトルの末尾に連結。標準化mean/stdは再学習必須。
- **モデル容量(hidden=32)は変更しない**(§9)。
- DoD: 旧`policy_weights.json`を新コードで読んでも挙動不変(Tier1/2と同じ後方互換要件)。

### Stage3d: Ablationとミラー対戦(§7・§8)

---

## 7. Ablation計画(clean ablation、一括投入しない)

Tier1abcで一括投入後に悪化原因の切り分けが難しかった教訓を踏襲し、意味の近い特徴を
少数ずつ追加して比較する。

| 実験 | 追加特徴 | 目的 |
|---|---|---|
| **Experiment A(control)** | なし(現行239次元のまま) | 対照群 |
| **Experiment B** | `opp_hp_loss` / `opp_energy_removed` / `opp_special_energy_removed` | グループAの即時結果だけでどれだけ効くか |
| **Experiment C** | `delta_best_effective_attack_damage` / `delta_can_ko` | グループBの本命だけでどれだけ効くか(**本Tierの中心仮説**) |
| **Experiment D** | `delta_attack_ready` / `delta_energy_shortfall` | 準備行動の細かい前進をどれだけ拾えるか |

各実験は前段の実験に追加する形ではなく、**controlに対してそれぞれ独立に追加**して
効果を切り分ける(Tier1abcの「Tier1a単独」検証と同じ考え方)。Experiment Cが最重要
(§0の狙いに直接対応)。B/C/Dすべてが効いた場合のみ、最後に統合構成を1つ作って
再検証する。

---

## 8. 評価プロトコル(Top-1だけで採用判定しない)

Tier1abcでは「オフラインtop1が微増 → 実戦では43%で有意に負け」という非転移が起きた。
本Tierでは以下を**分けて**測り、Top-1は判断材料の1つに留める。

1. **test Top-1一致率**(現行58.06%が対照)。参考値として見るが単独では採用判断しない。
2. **重要局面での一致率**: 「相手アクティブに実効ダメージを妨げる要因がある局面」
   (ロック闘エネルギー等の特殊エネルギー装備、resistance等)だけを抽出した診断セットでの
   一致率。全体Top-1が横ばいでも、この診断セットでの一致率が上がっていれば
   狙った効果が出ている根拠になる。
3. **固定盤面診断**: §6.1の受け入れテストのような手動構成盤面で、モデルが改造ハンマー等の
   準備行動を選ぶか(スコアが高いか)を直接確認。
4. **train/runtime parity計測**: §2.3の制約下で学習した重みの、実際のオフライン一致率。
   (a)/(b)相当のリッチな情報で学習した参考モデルとの差を計測してもよい(採用はしない、
   parity制約による精度低下がどの程度かを把握する目的のみ)。
5. **ミラー対戦(採用の最終ゲート)**: 現行既定重み vs 拡張重みを`league/_diag_tier1abc_head_to_head.py`
   と同型のハーネスで300試合、Wilson 95% CIで有意勝ち越しを確認。
6. **特定マッチアップ**: `league/_diag_attackplan_rock_lucario_head_to_head.py`のロック闘
   エネルギー系デッキ相手の勝率。ここが伸びなければ、attack_plan単体からの追加価値が
   薄いという判断材料になる。

**最終production採用条件は、従来通りミラー対戦(5)で統計的に有意に勝ち越すこと。**
Top-1・診断セット・固定盤面はいずれも採用の必要条件ではあるが十分条件ではない。

---

## 9. モデル容量は今回変更しない

現状hidden=32が小さい可能性はあるが、consequence特徴追加とモデル容量変更を同時に
実施すると寄与を分離できない。**まずは現行モデル容量のままconsequence特徴の純粋な
効果を測る。** 改善が確認できた後、hidden=32/64/128を**別のablationとして**検証する
(本方針書のスコープ外、将来の別方針書で扱う)。

---

## 10. 可逆性(本番影響ゼロの担保)

Tier1abcで確立した後方互換・検証ハーネスの考え方を再利用する。

- **既存production(`policy_weights.json`・既定config・既存`features.npz`・`attack_plan`)は
  無変更**。attack_plan.py自体も変更しない(§6 Stage3aで計算ロジックを"切り出す"際は
  複製または共通モジュール化とし、attack_plan.pyの動作に影響を与えない)。
- 新特徴はconfig/metaで明示的に有効化し、`meta.feature_tiers`(または同等のフラグ)が
  無い旧重みは拡張が無効な状態にフォールバックし、現行と完全に同一動作。
- 追加ファイル(新重みJSON・診断スクリプト・config)は現行に不干渉。

---

## 11. 完了条件(全体 DoD、再掲・統合)

- [ ] §6.1の受け入れテスト(ロック闘エネルギー+改造ハンマー、`delta_best_effective_attack_damage`
  が0→120で正しく検出される)が通る。
- [ ] §4のtransaction解決が代表的な複数選択カードで正しく動く。
- [ ] §2.5の非決定性(コイン)を安全に「不明」扱いできる。
- [ ] §7のablationで各Experimentの寄与が切り分けられている。
- [ ] レイテンシが既存探索群と合わせて予算内(600秒/試合、PIMCの2〜4%を基準線に)。
- [ ] 後方互換: 旧`policy_weights.json`を新コードで読んで挙動不変。
- [ ] §8のミラー対戦で統計的に有意に勝ち越す構成のみ採用。
- [ ] 提出健全性: 空`decks/`展開+1ゲーム実行で欠落なし。

---

## 12. 未決事項(実装着手前に確認)

- `consequence_unresolved`フラグ(§4.1)を追加するか、単に0埋めに留めるか。
- Stage3bの対象ITEM範囲(「対象を安全に解決できるITEM」の具体的な線引き。改造ハンマー・
  ハイパーボールは含むとして、対象選択が3段階以上のカードをどこで打ち切るか)。
- §5の静的計算をEVOLVEにどこまで適用できるか(進化後カードの副次効果の有無による)。
- Stage3aでATTACKにも`delta_best_effective_attack_damage`を適用するか(ATTACK自身を
  仮実行してもattack_planと同じ計算になるため、Stage3aでは`opp_hp_loss`計測に留め、
  `delta_best_effective_attack_damage`はStage3bの準備行動評価でのみ使う案が有力)。
- attack_planとの計算ロジック共有方法(切り出し先モジュール、担当領域の境界)。

---

## 13. 非目標(本Tierでやらないこと)

- 2手以上先の仮実行(attack-enabling-searchのv2相当の深さ2〜3探索とは別物。本Tierは
  常に**1手先**の結果のみを特徴化する)。
- 2階層以上ネストした追加選択の完全解決(§4.1、将来の短いaction sequenceとして再検討)。
- 相手の非公開情報の高精度推定そのものの改善(hidden_information側の課題。本Tierは
  既存の`hidden_state_source`設定をそのまま使う)。
- モデル容量(hidden層サイズ)の変更(§9、将来の別ablation)。
- Tier1/2(識別情報embedding)との組み合わせ効果の検証(まず本Tier単体で判断してから)。
