# T1.1補修 + D1(正式教師選定)+ D2(蒸留trainer準備)実装報告

作成日: 2026-08-08
対象: [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂4
前提: [transformer-t1.1-implementation-report.md](./transformer-t1.1-implementation-report.md)(初回T1.1)の続き
ステータス: **T1.1補修完了。D1(正式教師選定)完了。D2(蒸留trainer)実装・単体テスト完了、
本格蒸留の実行はしていない。T2(pointer gather)・PPO・NumPy推論・Kaggle worker組み込みは
指示通り未着手。**

cg/・data/・既存MLP・既存PPOは一切変更していない。

---

## 1. 修正文書の変更箇所

`transformer-tokenized-encoder-design.md`(改訂4版、内容を訂正):

- 「本改訂3版が正」→「本改訂4版が正」(title/statusは既に改訂4だったが、
  「正しいのはどの版か」の宣言文だけ改訂3のまま残っていた不整合を修正)。
- §0の実装状況を「T0.1+T1」→「T1.1(正規化統計・vocabulary統一・checkpoint契約の
  必須検査・serial単位pointer監査・fuudin_v4実データend-to-end)」に更新。
- §3.2末尾を「上位互換」という断定から、「入力識別情報は細粒度化されるが、
  性能向上は蒸留評価で確認する」という留保付きの表現に修正。
- §3.3に「§6で確定した通り、T3が新規にトークン化するのは相手トラッシュのみ」
  「手札・スタジアム・自分の山札/サイドのカード単位トークン化は本計画の範囲外」を明記。
- §3.4を、`card_id_max`/raw card idベースの説明から、`card_vocab.py`の
  append-only固定vocabularyベースの説明に更新。
- §4.3を、production shardではserial不要・debug shardでは監査用に任意保存、
  という説明に更新(T1.1で`board_token_serial`を実際に追加したため)。
- §9.1.1(新設)に、seed再現性調査の結果と、評価設計を「同じseedのpaired比較」から
  「独立標本比較」に変更した経緯を明記(§4参照)。
- §9.3の「固定相手8体への勝率」の判定基準を、47.59%(直接対戦の非劣性境界)から
  「student勝率−教師勝率の95%CI下限が-5%を上回るか」に修正(§8参照)。
- §0.2に、checkpoint検査の必須化・正規化2方式の追加を反映。
- 報告書側(`transformer-t0.1-t1-implementation-report.md`・
  `transformer-t1.1-implementation-report.md`)の「既存ファイルで変更したものはゼロ」を
  「既存production推論・PPO経路は無変更」に修正(T0で作った`token_shard.py`等
  自分たちの新規ファイルはT0.1/T1/T1.1を通じて反復的に拡張しており、
  「ゼロ」という表現は既存production側のみを指すことを明確化)。

---

## 2. checkpoint検査の必須化方法

`token_policy_t1.load_checkpoint()` の signature を全面変更した。

**旧API(初回T1.1):**
```python
def load_checkpoint(path, *, expect_token_schema_version=None, expect_vocabulary_hash=None,
                    expect_global_feature_manifest_hash=None, expect_feature_profile=Ellipsis, ...):
```
`expect_*`が全て既定値`None`/`Ellipsis`(=検査スキップ)を持つ、**呼び出し側が
渡し忘れると静かに安全性検査が無効化される**設計だった。

**新API(T1.1補修):**
```python
def load_checkpoint(path, *, token_schema_version: str, vocabulary_version: int,
                    vocabulary_hash: str, card_vocab_size: int, feature_profile,
                    global_feature_manifest_hash: str, allow_unsafe_mismatch: bool = False, ...):
```
6項目すべてが**既定値を持たない必須キーワード引数**になり、1つでも省略すると
Python自体が`TypeError`を出す(`test_checkpoint_load_requires_all_env_args`で
全6項目それぞれの省略パターンを確認済み)。

さらに、checkpoint自身の内容として存在必須のフィールド
(`model_config`・`use_board_card_id`・`normalization_mode`・`normalization_stats`)は
欠けていれば`CheckpointMismatchError`、`model_config.board_vocab_size`と
`card_vocab_size`の自己矛盾も検出する。

意図的に不一致を許すには`allow_unsafe_mismatch=True`を明示する
(`test_checkpoint_allow_unsafe_mismatch_requires_explicit_flag`で、
明示しない限り拒否されること・明示すれば通ることの両方を確認)。

**テスト**: `test_checkpoint_contract_round_trip`・`test_checkpoint_mismatch_raises`
(6項目それぞれの単独不一致)・`test_checkpoint_load_requires_all_env_args`・
`test_checkpoint_allow_unsafe_mismatch_requires_explicit_flag`・
`test_checkpoint_missing_normalization_mode_raises`(計5件、全pass)。

---

## 3. 正規化2方式の比較結果

`token_batch.build_normalization_stats(mode, ...)` が唯一の窓口。

| mode | board_numeric | legacy_global | legacy_option |
|---|---|---|---|
| `teacher_stats` | train splitから新規計算(教師に相当する統計が無い) | 教師の`state_mean`/`state_std`をmanifestでスライス(再利用) | 教師の`option_mean`/`option_std`をそのまま再利用 |
| `student_train_stats` | train splitから新規計算 | **train splitから新規計算**(教師を一切参照しない) | **train splitから新規計算** |

validationは統計計算に含めない(`train_val_decision_indices`で分けたtrain側のみを
`compute_normalization_stats`/`build_normalization_stats`に渡す)。

**実測比較**(`test_fuudin_v4_e2e.py`、pool_v1 v47教師、557決定点、8試合、
train/val=試合単位75%/25%、30 step、lr=1e-3):

```
teacher_stats:        train 0.9352 -> 0.6665, val 0.6835
student_train_stats:  train 0.9441 -> 0.6632, val 0.6939
best_mode = teacher_stats(val KL 0.6835 < 0.6939)
```

**この比較は8試合・557決定点という非常に小規模なデータでの1回きりの実行結果であり、
「teacher_statsの方が良い」という一般的な結論を出すには不十分**(データ量が
桁違いに少ない、seedを変えた再現も取っていない)。あくまで「両方式を同条件で
比較し、validation KLが良い方をcheckpointに使う」という**仕組みが動作すること**の
確認であり、本格蒸留(D2実行時)では改めてこの比較をより大きなデータで行う必要がある。

---

## 4. seed再現性調査結果

**結論: 完全な再現はできない。原因はcgエンジン(cg.dll)にPythonから呼べる
seed設定APIが無いこと。マルチプロセスのタスク割当順序が原因という仮説は
単一プロセスでの再現テストにより排除した。**

### 4.1 確認した5項目

| # | 確認内容 | 結果 |
|---|---|---|
| 1 | 1プロセス・1試合を同じseedで複数回実行 | `test_same_seed_does_not_reproduce_same_game_single_process`で実施。同じ`(learner_index, seed)=(0, 555)`を単一プロセス・逐次で3回投げ、決定点数が88→69→124と毎回異なることを確認 |
| 2 | Python random、NumPy、cg側の乱数源 | `collect_tokens.py`は`random.seed(seed)`(Python標準)のみ設定。NumPyの乱数は収集経路では使っていない。cgエンジン(ctypes経由でロードするネイティブライブラリ)は`cg.api.lib`にseed関連の関数が一切無いことを確認(`test_cg_engine_exposes_no_seed_function`) |
| 3 | 試合IDからseedを決定的に生成できているか | できている(`common.seed_base()`が世代・worker番号から決定的にseed範囲を計算)。**問題はseedの生成ではなく、そのseedをcgエンジンが使わないこと** |
| 4 | マルチプロセスのタスク完了順が結果へ影響しているか | **影響していない。** 単一プロセス・マルチプロセス無しでも再現しないため、この仮説は排除される |
| 5 | 先攻・後攻、デッキ、対戦相手の割当が固定されているか | 先攻/後攻(`learner_index`)は`(g % 2, seed0 + g)`のタスク生成で試合ごとに固定されている(=どちらが先攻かはタスクの時点で決まっている)。デッキ・対戦相手も呼び出し引数として固定。**固定されていないのは盤面の中身(デッキシャッフル・サイド配置)であり、これはcgエンジン内部の問題** |

### 4.2 評価設計への反映

「完全再現できる場合は再現テストを追加」については、完全再現はできないと
確定したため、代わりに**「再現しないこと自体」を固定するテスト**
(`test_seed_reproducibility.py`、2件)を追加した。

評価設計は「同じseedによるpaired比較」を前提にしないよう修正した
(design.md §9.1.1)。既存の`evaluate_pool`/`parallel_collect`が、相手ごとの
試合数を`share`で揃え、先攻/後攻を試合indexの偶奇で揃える設計に既になっているため、
「先後・相手・試合数を均等化した独立標本」という代替方針は**追加実装なしで
既に満たされている**。D1(§6)もこの前提で実施した。

---

## 5. パフォーマンス再測定結果

`measure_t1_perf.py`を、warm-up 10回・測定30回・`torch.cuda.synchronize()`を
計測区間の前後に配置・mean/median/p95記録・forward-only/forward+backward/
+optimizer.step分離・peak_allocated/peak_reserved分離、に書き直して再測定した。

RTX 3060 Laptop、fuudin_v4実データ(639決定点、board_max=12、option_max=31)。

| batch | forward_only(mean/median/p95, ms) | forward+backward(ms) | +step(ms) | peak_alloc(GB) | peak_reserved(GB) |
|---|---|---|---|---|---|
| 256 | 2.05 / 2.00 / 2.47 | 9.76 / 9.54 / 12.85 | 9.28 / 9.08 / 12.40 | 0.076 | 0.109 |
| 512 | 2.72 / 2.66 / 3.11 | 11.39 / 10.46 / 15.18 | 11.16 / 10.34 / 13.23 | 0.131 | 0.203 |
| 1,024 | 4.01 / 3.97 / 4.18 | 17.51 / 17.30 / 18.89 | 17.57 / 17.41 / 18.85 | 0.240 | 0.348 |
| 2,048 | 7.07 / 7.01 / 7.35 | 31.37 / 31.28 / 32.85 | 31.53 / 31.33 / 32.59 | 0.462 | 0.705 |
| 4,096 | 13.91 / 13.81 / 14.42 | 60.15 / 60.01 / 61.01 | 60.00 / 59.95 / 60.67 | 0.895 | 1.409 |
| 8,192 | 27.64 / 27.71 / 28.50 | 117.61 / 117.33 / 121.23 | 118.91 / 117.85 / 125.18 | 1.766 | 2.810 |
| 16,384 | 57.03 / 56.75 / 58.37 | 239.38 / 237.90 / 252.49 | 240.39 / 239.54 / 252.66 | 3.508 | 5.610 |

**旧計測(warm-up・同期不足)の不自然な値は再現しなかった。** batch=256で
推論のみが学習stepより遅い現象、batch=8192→16384で学習時間が17倍になる現象、
どちらも解消し、batch sizeに対してほぼ線形にスケールしている。

**[D2.1で訂正]** 上表の通りbatch sizeに対してほぼ線形にスケールしており、
1決定点あたりのスループットは4,096でも8,192でもほぼ同じ(4,096は60.0ms/4,096≈
14.6μs/決定点、8,192は118.9ms/8,192≈14.5μs/決定点)。**「VRAM予算に収まる
最大値だから」という理由だけでbatch sizeを決めるべきではない**(蒸留の学習品質は
batch sizeが大きいほど良いとは限らないため)。

**安全な既定値としてbatch=4,096を採用する。** 実際の採用値は{2,048, 4,096, 8,192}の
中からvalidation KL等の学習品質とGPU余裕を見て選ぶ(§本格蒸留の4条件比較で
正規化方式・温度と合わせて確認する)。16,384はpeak_reserved 5.61GBで6GB予算に
対して余裕が少なく(他プロセスの分を考慮すると危険域)、候補から外す。

---

## 6. v24/v32/v40/v47の正式評価結果

`select_teacher.py`(既存`learner.evaluate_pool()`を再利用、新規実装なし)。
pool_v1の固定相手8体、各checkpoint 1,000試合、near-greedy(`temperature=0.01`の
softmaxサンプリング。厳密なargmaxではない、詳細は下記)、
`collect_winrate`は不使用。結果はJSON保存(`kaggle_replays/rl/distributed/results/teacher_selection.json`)。

| checkpoint | 全体勝率 | 95%CI下限(Wilson) | 所要時間 |
|---|---|---|---|
| v24 | 0.7370(737/1000) | 0.7088 | 254s |
| v32 | 0.7430(743/1000) | 0.7150 | 253s |
| **v40** | **0.7910(791/1000)** | **0.7647** | 249s |
| v47 | 0.7420(742/1000) | 0.7140 | 250s |

**[D2.1で訂正]** 各checkpointの試合数は`games_requested=1000`に対し`overall_valid=1000`
(4件全て、エラー0・引き分け0)であることを`results/teacher_selection.json`から確認した。
「1,000試合要求した」ではなく「1,000試合が有効に成立した」ことを明記する。

**[D2.1で訂正]** 当初「v40のCI下限が他候補のCI上限と重ならない」としていたが誤り。
Wilson区間はそれぞれ独立に計算した区間であり、区間同士は実際には一部重なる
(例: v32は`[0.7150, 0.768]`程度、v40の下限0.7647はこの範囲内)。**個別のCI下限を
比較するだけでは優位性の根拠として不十分**なため、v40と各候補の**勝率差**を
直接計算し直した(`select_teacher.winrate_diff_ci95`、正規近似による2標本比率差の
95%CI、既存関数をそのまま利用・新規実装なし)。

| 比較 | 勝率差(v40 − 候補) | 95%CI |
|---|---|---|
| v40 − v24 | +0.0540 | [0.0169, 0.0911] |
| v40 − v32 | +0.0480 | [0.0110, 0.0850] |
| v40 − v47 | +0.0490 | [0.0120, 0.0860] |

**3件とも差のCI下限が0を上回っており、v40が3候補全てに対して統計的に有意に
勝率が高いと言える(個別のCI下限比較ではなく、直接差のCIによる結論)。**
選定ルールは「全体勝率の差のCI下限が0を上回るか」に訂正する。**選定: v40。**

**[D2.1で追加: 先攻/後攻の内訳]** `parallel_collect`は`learner_index = 試合index % 2`
で割り当てる(`collect_parallel.py`)。§9.1.1で確定した通り`learner_index`は
先攻/後攻を確定的に決める(0→学習対象が常に先攻、1→常に後攻)ため、相手1体
あたり125試合では学習対象が先攻63試合・後攻62試合と、構成上すでに均等に
近い形で割り振られている(新たに試合を追加で回して確認する必要はない)。

**[D2.1で追加・訂正: near-greedy評価の温度について]** `evaluate_pool()`は内部で
`temperature=0.01`のsoftmaxサンプリングを使っており、**argmax(厳密なgreedy)
ではない**ため、以後このD1評価は「greedy」ではなく「near-greedy」と呼ぶ
(既存`learner.py`は変更しないため、この挙動はそのまま)。`temperature=0.01`は
各局面のスコアがfloat32で厳密に同値でない限りargmaxと高確率で一致すると
考えられるが(softmax(logits/0.01)は最大値以外の項が指数的に無視できるほど
小さくなるため)、**今回のD1評価を明示的argmaxで再実行しておらず、
argmaxとの同等性を数値的に確認していないため断定しない。** 厳密なargmax
評価による再検証は、本格蒸留(§10)を妨げない範囲でのフォローアップ課題として残す。

### 相手ごとの内訳(各125試合)

| 相手 | v24 | v32 | v40 | v47 |
|---|---|---|---|---|
| alakazam | 0.696 | 0.776 | **0.792** | 0.712 |
| archaludon_ex | 0.840 | 0.792 | **0.848** | 0.800 |
| crustle | 0.648 | 0.632 | **0.704** | 0.680 |
| marnie_grimmsnarl_ex | 0.528 | 0.552 | **0.568** | 0.504 |
| mega_lucario_ex | 0.744 | 0.736 | **0.768** | 0.728 |
| rocket_mewtwo_ex | 0.704 | 0.720 | **0.824** | 0.752 |
| shirona_garchomp_ex | 0.848 | 0.872 | **0.904** | 0.816 |
| dragapult_ex | 0.888 | 0.864 | 0.920 | **0.944** |

**[D2.1で訂正]** v40は8相手中7相手で**点推定が最高**(相手ごとに125試合しか
無く、個々の相手について有意差を検定していないため「明確に最良」とは言わない、
§9.3の主判定/補助指標の切り分け参照)。dragapult_exのみv47がわずかに上回る
(0.944 vs 0.920)。**「v47が最新だから」という理由ではなく、実測データに
基づいて選定した。** むしろv47は`collect_winrate`(学習中の副産物指標、
温度サンプリング)では最終世代らしく高めの値を示していたが、今回の正式な
near-greedy評価ではv24・v32とほぼ同水準(CI区間が重なる)で、**全体勝率の
直接差CI(v40−v47 = +0.049, 95%CI[0.012, 0.086])で見るとv40が統計的に
有意に上回っていた。**

---

## 7. 選定した教師checkpoint

**`kaggle_replays/rl/runs/pool_v1/models/model_v40.json` を正式教師として選定した。**

理由:
- **[D2.1で訂正]** v40と各候補(v24/v32/v47)の勝率差の95%CI(正規近似、
  `winrate_diff_ci95`)が全て下限>0(§6参照)であり、個別のWilson区間の
  下限比較ではなく、直接の差の検定でv40の優位性を確認した。
- 相手ごとの内訳でも8相手中7相手で最高勝率であり、特定の相手にだけ強い
  「じゃんけん構造」による見かけ上の平均の高さではないことを確認済み
  (`results-and-concerns.md` §3.1で過去に観測された懸念パターンに該当しない)。
- v47(最新世代)は正式評価で明確に劣っていたため不採用。「最新だから」を
  選定理由にしない、という当初方針(§0)の通りの結果になった。

`extended_features_profile=fuudin_v4`であることも確認済み(既存のfuudin_v4 E2E
テスト・蒸留trainerがそのまま使える)。

---

## 8. 蒸留trainerの構造とテスト結果

`kaggle_replays/rl/train_distill_t1.py`(新規)。

### 8.1 構造

```
複数shard読み込み(token_shard.concatenate_shards)
  -> metadata整合性検査(token_shard_format・extended_features_profile・
     teacher_model_sha256が全shardで一致しなければShardMetaMismatchError)
  -> 試合単位train/val分割(token_batch.train_val_decision_indices)
     既存split.jsonがあればそれを再利用(分割結果を固定・保存)
  -> 正規化統計(--normalization-mode teacher_stats|student_train_stats)
  -> 学習ループ:
       token_batch.slice_arrays_by_decisions + build_batch でminibatch化
       (可変長board/optionはpaddingしてtensor化)
       masked_kl_loss(paddingを除外したKL、temperature=--temperature)
       gradient clipping(torch.nn.utils.clip_grad_norm_)
       --eval-every step ごとに:
         compute_distill_metrics(top-1一致率・教師確率>=0.8での一致率・
           教師entropy三分位別一致率・student選択への教師確率)
         validation KLがこれまでの最良を更新したらtoken_policy_t1.save_checkpointで保存
         trainer_state.pt(model/optimizer state・step・best_val_kl)を毎回保存(resume用)
  -> train_log.jsonl に評価ごとの全指標を追記
```

temperature T=1とT=2の比較は、`--temperature`を変えて`--out-dir`を分けて
2回実行し、両方の`train_log.jsonl`のvalidation KLを比較する運用
(本格蒸留実行時に行う。今回は仕組みの検証のみ)。

### 8.2 テスト結果

`test_train_distill_t1.py`(**[D2.1で7件に拡張]**、全pass、production既定教師で
実データ収集して検証):

- `test_trainer_runs_end_to_end`: 2shard・6 step学習が完走し、`split.json`・
  `best.pt`・`trainer_state.pt`・`train_log.jsonl`(全指標キー含む)が生成される
- `test_split_is_fixed_across_reruns`: 異なる`--out-dir`2つが同じ`--split-dir`を
  共有していれば同一の`split.json`(train/val分割)を再利用する
- `test_split_rejected_if_dataset_manifest_changed`(新規): shard構成が変わった
  (=dataset manifest hashが変わった)状態で古い`split.json`を読もうとすると
  `ValueError`
- `test_resume_continues_from_saved_step`: 4 stepで一度終了 → `--resume`で
  続きから8 stepまで学習が進む(step数が正しく引き継がれる)
- `test_resume_rejects_changed_hyperparameters`(新規): resume時に温度・
  正規化モード等のハイパーパラメータが変わっていると拒否される
- `test_resume_bitwise_matches_straight_through_on_cpu`(新規、最重要):
  「4 step連続実行」と「4 step実行→resume→4 step実行」で、CPU上の
  `model_state_dict`の全tensorが`torch.equal`でビット一致する
- `test_mismatched_shard_metadata_rejected`: `_MERGE_CONSISTENCY_KEYS`の
  13フィールドのいずれかが異なるshardを混ぜようとすると`ShardMetaMismatchError`

`kaggle_replays/rl/distributed/test_token_shard.py`(**[D2.1で拡張、22件、
**訂正**: 旧報告の「7フィールド」は誤りで、実際は`_MERGE_CONSISTENCY_KEYS`13項目中
8項目**]**): `token_shard_format`(read_token_shard自体が別途検査するため除外)と
4つの次元フィールド(専用の次元不一致テストで別途確認)を除いた、ラベル値としての
8フィールド(`extended_features_profile`・`teacher_model_sha256`・
`vocabulary_version`・`vocabulary_hash`・`card_vocab_size`・
`global_feature_manifest_hash`・`temperature`・`action_selection_mode`)
それぞれについて単体で不一致を起こすparametrizedテスト、次元不一致テスト、
フィールド欠損テスト(`_MISSING`センチネルで「無い」と「一致してNone」を
区別することの確認)を追加。

`test_masked_kl_loss.py`(**新規、4件**): T²スケーリングの手計算一致、
決定点重み付けがpadding数・合法手数に依存しないこと(paddingにゴミ値を
入れてもKLが変わらないことで間接確認)、paddingでNaNが出ないこと、
`compute_distill_metrics`が常にT=1で計算されること。

`test_first_player_mapping.py`(**新規**): 実対戦6試合で`learner_index`と
実際の先攻/後攻の対応を確認(§9.1.1参照)。

`kaggle_replays/rl/distributed/test_select_teacher.py`(5件、全pass、統計計算の
単体テスト、cgエンジン不要): 勝率差の95%CI計算・マージン判定
(点推定だけでなくCI下限で判定していること)を確認。

**回帰テスト全体: 84件全pass**(既存T1.1テストも含む、79.35s)。

### 8.3 D2.1で追加したshard/dataset/split契約

- **shard結合時の整合性検査を13項目に拡張**(`token_shard._MERGE_CONSISTENCY_KEYS`):
  `token_shard_format`・`extended_features_profile`・`teacher_model_sha256`・
  `vocabulary_version`・`vocabulary_hash`・`card_vocab_size`・
  `global_feature_manifest_hash`・`legacy_state_dim`・`legacy_global_dim`・
  `legacy_option_dim`・`board_token_numeric_dim`・`temperature`・
  `action_selection_mode`(温度とは別に、行動選択方式そのものの不一致も検出する。
  現状は常に`"softmax_temperature"`だが、将来argmax等を追加したときに
  temperatureの値だけでは区別できない収集方針の違いを検出するため)。
  いずれか1項目でも不一致なら`ShardMetaMismatchError`。フィールド自体が
  無いshard(古い形式)は`_MISSING`センチネルで検出し、無条件で拒否する
  (「無い」を「一致」と誤判定しない)。
- **`dataset_manifest.json`(新規、`token_shard.build_dataset_manifest`)**:
  結合対象shardのパス順・各shardのSHA-256・試合数/決定点数・
  全体のハッシュ(`manifest_hash`)を保存。
- **`split.json`を`--out-dir`から切り離した**: `--split-dir`(既定は
  `out_dir.parent/"splits"`)に`split_{manifest_hash先頭16桁}_seed{seed}_val{val_fraction}.json`
  という名前で保存し、複数の`--out-dir`(正規化モード違い・温度違いの実行)が
  同一splitを共有できる。splitには`dataset_manifest_hash`・分割schema
  version・分割seed・**試合ID単位**のtrain/val集合を保存し、
  データセット構成が変わっていたら例外を送出する(決定点単位ではなく試合
  単位で保存するのは情報漏洩防止のため)。

### 8.4 KL損失の数式とreduction

`train_distill_t1.masked_kl_loss(scores, teacher_logits, mask, temperature)`:

1. `scores`・`teacher_logits`の両方を、paddingを`masked_fill(~mask, -inf)`で
   埋めてから`softmax`/`log_softmax`を計算する(**T1.1時点はcontributionだけを
   0にしていたため、paddingに大きな値が入るとsoftmaxの正規化定数自体が
   汚染されるバグがあった。D2.1で発見・修正、詳細は§下記の不具合修正参照**)。
2. 温度`T`で`teacher_probs = softmax(teacher_logits/T)`、
   `log_probs = log_softmax(scores/T)`を計算し、決定点ごとに合法手についてのみ
   `sum(teacher_probs * (log(teacher_probs) - log_probs))`を取る(KL(teacher||student))。
3. バッチ内の決定点で**平均**を取り(合法手数の多寡が重みに影響しない、
   `test_reduction_weighting_independent_of_option_count`で確認)、最後に`T²`を
   掛ける(Hinton et al. 2015の慣行、`test_t_squared_scaling_applied`で確認)。
4. 学習時に使う`train_objective_kl`(学習温度T・T²込み)と、モデル選択・
   T=1/T=2比較に使う`deployment_kl_t1`(常にT=1で計算)を分離してログする。
   `compute_distill_metrics`(top-1一致率・教師確率>=0.8での一致率・教師entropy
   三分位別一致率・student選択への教師確率)も常にT=1固定。

### 8.5 resume契約とbit-exact resumeテスト

`trainer_state.pt`に`run_contract`(dataset manifest hash・split hash・
model config・token schema version・vocabulary hash・card vocab size・
feature profile・global feature manifest hash・normalization mode・
temperature・lr・grad clip・training seed)を保存し、`--resume`時に現在の
実行設定と突き合わせて1項目でも異なれば`ValueError`で拒否する。
Python/NumPy/Torch/CUDAのRNG状態も保存・復元する。

学習バッチの生成を、stepごとに再シャッフルする方式から
`training_batch_stream`(seedをepochごとにのみ変える無限generator)に
書き換え、resume時は同じgeneratorを`start_step`ぶんfast-forwardしてから
再開する。これにより
`test_resume_bitwise_matches_straight_through_on_cpu`が、CPU上で
「中断無し8 step学習」と「4 step→resume→4 step」の`model_state_dict`が
`torch.equal`で完全一致することを確認している。

### 8.6 D2.1で見つけた不具合(3件、いずれも修正・テストで固定済み)

1. **`masked_kl_loss`のpadding汚染バグ**: T1.1時点の実装はpaddingの寄与を
   softmax計算後に0にしていたが、softmax自体はpadding込みの全要素に対して
   計算していたため、paddingに大きな値が入ると正規化定数が汚染され、
   実際の合法手のKLまで狂う(検証: paddingに`999.0`を入れるとKLが0.15→499.45に
   変化)。scores/teacher_logits両方をsoftmax計算**前**に`-inf`で埋めるよう修正。
2. **resume時のRNG状態デバイス不一致**: `torch.load(..., map_location=args.device)`で
   `--device cuda`のときCPU保存のRNG ByteTensorがCUDAに移動し、
   `torch.set_rng_state()`が型エラーで失敗。`trainer_state.pt`の読み込みは
   常に`map_location="cpu"`に固定して修正(モデル/optimizerのstate_dictは
   ロード後にどのデバイスへも正しく移せるため影響なし)。
3. **resume後に学習結果が非決定的にずれる**: 8.5節参照。stepごとの
   再シャッフルseedが`step`依存だったため、resumeで`step`が0以外から
   始まると別のシャッフル順になっていた。

---

## 9. 本格蒸留を開始するために残っている課題

**[D2.1で更新]** 1〜3は本格データ収集・4条件比較・3 seed確認の実施(§10)に
よって解消した。残っているのは以下:

1. **推論速度(pure-Python/NumPy)は未計測のまま。** §5はtorch/GPUでの
   学習側の計測であり、提出時のCPU pure-Python推論速度とは別の話。
2. **DISCARD/ABILITY/CARD/SPECIAL_CONDITIONのpointer解決の一部が未検証**
   (継続課題)。
3. **T1単独の対戦評価(design.md §9の評価ゲート)がまだ実施されていない**
   (§10.5)。**ライブ推論ラッパーが未実装**なのがブロッカー。
4. **T2(pointer gather)・PPO(Transformer自身のrollout)・NumPy推論・
   Kaggle worker組み込みはいずれも未着手。** 指示通り、正式教師の選定
   (§7、完了)とT1単独の対戦評価(上記3、未完了)が完了するまでT2には進まない。

---

## 10. 本格蒸留(D2.1 item10)の実施状況と、T2へのgo/no-go判断

**データ収集・学習曲線・4条件比較・3 seed確認は、このセッションで実際に実行した。**
GPU(RTX 3060、CUDA)での学習は1回あたり数分で終わるため、想定より高速に
本格蒸留の大半を進められた。**T1単独の対戦評価(300試合予備評価・1,000試合
以上のhead-to-head)だけは、下記の理由で未実施。**

### 10.1 データ収集(v40教師、fuudin_v4、alakazam_morioka自己対戦)

`collect_tokens.py --teacher-weights runs/pool_v1/models/model_v40.json`で
3本のshardを収集した(`runs/distill_v40/shards/`)。

| shard | 試合数 | 決定点数 | 所要時間 |
|---|---|---|---|
| shard_10k.npz | 150 | 11,183 | 45.7s |
| shard_plus40k.npz | 550 | 40,599 | 153.7s |
| shard_plus50k.npz | 650 | 46,874 | 173.2s |

累積決定点数: 11,183 / 51,782 / 98,656(3本を`dataset_manifest`で束ねた3段階)。

### 10.2 学習曲線(teacher_stats・T=1固定、データ量のみ変えて比較)

`--batch-size 4096 --steps 1500 --eval-every 150`(各実行3〜4分)。

| データ規模(train決定点) | best `deployment_kl_t1` | top-1一致率(最終step) |
|---|---|---|
| 10k(train 9,008) | 0.5527 | 0.709 |
| 50k(train 41,159) | 0.3795 | 0.738 |
| 100k(train 78,672) | 0.3471 | 0.751 |

**10k→50kでval KLが0.173改善、50k→100kでは0.032しか改善しておらず、
データ量に対する改善は明確に飽和傾向にある。** 100k決定点規模はこの傾向を
見るには十分だが、まだ完全に頭打ちとは言い切れない(引き続きデータを
増やせば多少の改善余地は残る)。

### 10.3 4条件比較(teacher_stats/student_train_stats × T=1/T=2、100k・同一splitで比較)

同じ`dataset_manifest_hash`(=100kのshard3本)・同じsplitを共有した状態で
4条件を比較した(全てseed=0)。`deployment_kl_t1`は常にT=1で計算するため、
学習温度が違っても直接比較できる(§8.4)。

| 正規化 | 学習温度 | best `deployment_kl_t1` |
|---|---|---|
| **teacher_stats** | **T=1** | **0.3471(最良)** |
| student_train_stats | T=1 | 0.3500 |
| teacher_stats | T=2 | 0.3978 |
| student_train_stats | T=2 | 0.3973 |

**T=1がT=2より明確に良い(0.35前後 vs 0.40前後)。正規化方式はT=1の中では
ほぼ差が無い(0.3471 vs 0.3500)が、teacher_statsがわずかに良いためこちらを
採用する。** 最良条件: **teacher_stats・T=1**。

### 10.4 3 seed確認(最良条件: teacher_stats・T=1・100k)

| seed | best `deployment_kl_t1` | top-1一致率 |
|---|---|---|
| 0 | 0.3471 | 0.751〜0.757 |
| 1 | 0.3433 | 0.754〜0.759 |
| 2 | 0.3450 | 0.752〜0.754 |

**平均0.3451、標準偏差約0.0019と、seed間のばらつきは非常に小さい
(相対で1%未満)。** trainerの学習結果はseedに対して安定している。

### 10.5 未実施: T1単独の対戦評価(300試合予備評価・1,000試合以上のhead-to-head)

**この項目だけは実行していない。** 理由: 対戦評価には、実際のcgエンジンの
対局ループの中でT1(torch)を1決定点ずつ呼び出して行動を選ぶ「ライブ推論」
経路が要る。これは既存のオフライン蒸留学習(事前収集したshardをバッチ処理
するだけ)とは別物で、**現状どこにも実装されていない**(`token_policy_t1.py`
にはバッチ処理用の`forward`しか無く、単一決定点をcgの`Observation`から
組み立ててT1へ通し、選択を`agent()`相当の形式で返す経路が無い)。

`evaluate_pool()`はpure-Python`PolicyModel`(JSON重み)を前提にしており、
torch checkpointをそのまま渡せる作りではないため、**新規のライブ推論
ラッパーが必要**(既存`learner.py`/`evaluate_pool()`は変更しない)。
このラッパーを持続テストなしで急いで書いて1,300試合流すよりは、次のセッション
で正しく実装・検証してから評価を回す方が安全と判断し、ここでは実施しなかった
(結果を捏造しない方針を優先)。

### 10.6 go/no-go判断

**T2(pointer gather)・PPO(Transformer自身のself-play)・NumPy推論・
Kaggle worker組み込みへは、依然として進まない(no-go)。**
理由は指示通り「正式教師選定とT1単独蒸留評価が完了するまでT2へ進まない」
という条件のうち、**学習側(データ収集・4条件比較・3 seed確認、§10.1〜10.4)は
完了したが、対戦評価側(300試合予備評価・1,000試合以上の最終評価、§10.5)が
まだ実施されていない**ため、条件をまだ満たしていないから。

**次のアクション(残っているのはこれのみ):**
1. **T1のライブ推論ラッパーを実装する**(§10.5)。cgの`Observation`1件から
   その場でboard/optionトークンを組み立て(`board_tokens.build_board_tokens`・
   `token_batch`の1決定点版)、`stage100k_teacher_t1_seed{0,1,2}/best.pt`を
   forwardして選択を返す。既存`learner.py`/`evaluate_pool()`は変更せず、
   並存する新しい評価スクリプトとして書く。
2. 実装したラッパーに対して、まず数試合の単体テストで既存MLP(`PolicyModel`)
   との入出力の対応が壊れていないことを確認する。
3. 300試合の予備評価(§9.1)で明らかな劣化が無いことを確認してから、
   1,000試合以上のhead-to-head/固定相手評価(§9.3の主判定/補助指標の
   切り分けに従う)を実施する。
4. 上記が揃った時点で初めてT2着手の判断をする。

---

## 11. 参照資料

- [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂4
- [transformer-t1.1-implementation-report.md](./transformer-t1.1-implementation-report.md) — 初回T1.1
- [transformer-t0.1-t1-implementation-report.md](./transformer-t0.1-t1-implementation-report.md) — T0.1/T1
- [transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md) — T0
