# T1.1 実装報告

作成日: 2026-08-08
対象: [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂4
前提: [transformer-t0.1-t1-implementation-report.md](./transformer-t0.1-t1-implementation-report.md)(T0.1・T1)の続き
ステータス: **T1.1完了。正式教師checkpointの選定・T1単独の蒸留評価・T2(pointer gather)・
PPO・NumPy推論・Kaggle worker組み込みは未着手。**

cg/・data/・既存MLP・既存PPO(`train_v3.py`のCritic含む)は一切変更していない。

---

## 1. 変更ファイル一覧

| ファイル | 内容 |
|---|---|
| `sample_submission/ptcg_ai/learning/card_vocab.py`(改訂) | append-only化(`extend_vocab()`追加)。hashを並び順込みで計算するよう修正(旧実装は`sorted()`を挟み順序変化を検出できなかった) |
| `sample_submission/ptcg_ai/learning/legacy_feature_manifest.py`(追記) | `manifest_hash()`追加(checkpoint契約用) |
| `kaggle_replays/rl/token_policy_t1.py`(改訂) | Card ID変換をvocab統一(`option_card_embedding`をraw id indexからvocab index indexへ)。`board_numeric_mean`/`std`バッファ追加。`migrate_option_embedding_from_teacher()`・`save_checkpoint()`/`load_checkpoint()`/`CheckpointMismatchError`を追加 |
| `kaggle_replays/rl/token_batch.py`(追記) | `train_val_decision_indices()`(試合単位分割)・`slice_arrays_by_decisions()`・`compute_normalization_stats()`・`global_stats_from_teacher()` |
| `kaggle_replays/rl/collect_tokens.py`(改訂) | `--debug-pointers`時に`board_serials`(盤面トークンのserial)も出力するよう追加 |
| `kaggle_replays/rl/distributed/token_shard.py`(改訂) | `TOKEN_SHARD_FORMAT`を`/1`→`/2`に。`board_token_serial`配列(debug時のみ)を追加 |
| `kaggle_replays/rl/test_pointer_audit.py`(新規) | serial単位のpointer監査テスト(合成・実データ) |
| `kaggle_replays/rl/test_fuudin_v4_e2e.py`(新規) | fuudin_v4実データのend-to-endテスト |
| `kaggle_replays/rl/measure_t1_perf.py`(新規) | GPUメモリ・推論/学習時間の実測スクリプト |
| テスト改訂 | `test_card_vocab.py`(append-only用に3件差し替え)、`test_token_batch.py`(train/val分割・正規化統計6件追加)、`test_token_policy_t1.py`(vocab統一・checkpoint契約4件追加)、`test_token_shard.py`(board_token_serial検証1件追加) |

**既存production推論・PPO経路は無変更。** `torch_policy.py`(`TorchOptionPolicy`)・
`train_v3.py`(`Critic`)・`policy_model.py`・`encoder.py`・`common.py`(既存shard
reader・`SHARD_FORMAT`)・`worker.py`・`collect_parallel.py`は一切変更していない。
T0で新規に作った`token_shard.py`・`collect_tokens.py`・`token_batch.py`・
`token_policy_t1.py`・`card_vocab.py`はT0.1/T1/T1.1を通じて反復的に拡張しており、
これらは「既存ファイル」ではなく本計画自身が管理する新規ファイル。

---

## 2. fuudin_v4実データの決定点数

pool_v1(`kaggle_replays/rl/runs/pool_v1/models/model_v47.json`、
`extended_features_profile=fuudin_v4`)を教師に、デッキ`alakazam_morioka`で
8試合の自己対戦を収集した(`test_fuudin_v4_e2e.py`)。

```
n_decisions = 656 (実行1回の実測値)
n_trajectories = 8
legacy_state_dim = 389
legacy_global_dim = 145(= manifest.global_dim("fuudin_v4"))
```

**同じseed(seed0=123)でも試合ごとの決定点数は実行のたびに変動する**
(364〜656件の範囲で観測)。これはcgエンジン内部の状態がPythonの`random.seed()`だけで
完全に決定されない(マルチプロセスでのタスク割り当て順序が試合展開に影響しうる)ためで、
今回新規に発見した問題ではなく、既存の`collect_parallel.py`にも共通する性質。
蒸留データ収集自体は十分な試合数を集めれば問題にならない。

---

## 3. 正規化統計の計算方法

`token_batch.compute_normalization_stats()`。試合単位で分けたtrain split
(`train_val_decision_indices()`、val_fraction=0.25)の決定点だけから計算する。

| 対象 | 方法 | 教師の既存統計を流用できるか |
|---|---|---|
| `board_token_numeric_features`(11次元) | **train splitから新規計算**(mean/std) | **できない。** 現行モデルの`state_mean`/`state_std`はスロットごと(self_active/self_bench0/…)に別々の統計を持ち、盤面トークン(全スロット共通の1つの統計)と構造が違う。プールして新規計算する必要がある |
| `legacy_global_features` | 教師の`state_mean`/`state_std`を`legacy_feature_manifest.global_columns(profile)`でスライスするだけ | **できる。** legacy_global_featuresは既存state_featuresの一部をそのまま抜き出したものなので、教師の較正済み統計をそのまま使える(`token_batch.global_stats_from_teacher()`) |
| `legacy_option_features`(65次元) | 教師の`option_mean`/`option_std`をそのまま使う | **できる。** 特徴の定義自体を変えていないため |

stdには`token_batch.STD_FLOOR = 1e-3`の下限を設けた。定数特徴(全行で同じ値)を含む
合成データでテストし、NaNが出ないことを確認済み(`test_normalization_stats_std_floor_and_no_nan`)。
paddingは保存データに含まれないため(padding自体がbatch構築時にのみ発生する)、統計には
混ざらない。

---

## 4. checkpoint schema

`token_policy_t1.save_checkpoint()`/`load_checkpoint()`(`torch.save`/`torch.load`、
`weights_only=False`。自前形式の信頼済みファイルのため)。

| キー | 内容 |
|---|---|
| `model_state_dict` | `T1OptionPolicy.state_dict()` |
| `model_config` | コンストラクタ引数の辞書(再構築に使う) |
| `token_schema_version` | `token_shard.TOKEN_SHARD_FORMAT`(現在`ptcg-rl-token-shard/2`) |
| `vocabulary_version` / `vocabulary_hash` | `card_vocab.CardVocab.version`/`.hash` |
| `feature_profile` | `extended_features_profile`(`None`/`"fuudin_v2"`/`"fuudin_v4"`) |
| `global_feature_manifest_hash` | `legacy_feature_manifest.manifest_hash(profile)` |
| `normalization_stats` | §3の統計一式(dict) |
| `use_board_card_id` | ablationフラグ |
| `card_vocab_size` | `vocab.size` |

ロード時、`expect_*`引数に現在の環境の値を渡すと不一致を`CheckpointMismatchError`で
検出する(`test_checkpoint_mismatch_raises`で確認)。**この`expect_*`任意引数方式は
T1.1補修で撤回した**(呼び出し側が渡し忘れると検査が黙って素通りするため)。
最新のAPI(全項目が必須引数、`normalization_mode`検査を含む)は
[transformer-t1.1-patch-d1-d2-report.md](./transformer-t1.1-patch-d1-d2-report.md) §2参照。

---

## 5. vocabulary移行方法

`token_policy_t1.migrate_option_embedding_from_teacher(teacher_card_embedding_table, vocab)`。

教師(既存MLP)の`card_embedding.table`(raw card idインデックス、`card_id_max+1`行)を、
`CardVocab`のindex体系に並べ替えた`(vocab.size, embed_dim)`のtensorを作る。

- PAD行(index0)はゼロ
- UNK行(index1)は教師のindex0(`policy_model.py`の`_UNKNOWN_CARD_EMBEDDING_INDEX`、
  「識別なし/範囲外」用の予約行)をそのまま引き継ぐ
- 既知カードは`vocab.index_of(cid)`の位置に教師の該当行をコピー

`test_migrate_option_embedding_from_teacher`で、合成テーブル(3カード+識別なし行)を
使い、PAD/UNK/各カードが正しい行に移植されることを確認済み。

**`card_vocab.py`自体もappend-only化した。** `extend_vocab()`が既存`card_vocab.json`の
並びを一切変えず、cgエンジンにあって語彙に無いカードだけを末尾に追加する。既存カードの
indexは新規カード追加後も変わらない(`test_extend_vocab_appends_new_cards_without_reordering`)。
hashは並び順を含めて計算するよう修正した(`test_hash_depends_on_order`)。

---

## 6. end-to-endテスト結果

`test_fuudin_v4_e2e.py::test_full_pipeline_state_to_checkpoint`。

```
経路: State -> token化 -> shard保存 -> 再読込 -> token_batch(train/val試合単位分割)
      -> T1OptionPolicy.forward -> KL loss(教師logits) -> backward -> optimizer.step
      -> checkpoint保存 -> 再読込後の出力一致

train KL loss: 1.0801 -> 0.8348(30 step、Adam lr=1e-3)
val   KL loss(30 step後): 0.9136(有限値、NaN/inf無し)
checkpoint round-trip: 出力完全一致(atol=1e-6)
```

**注意:** `torch.nn.functional.kl_div`をそのまま使うと、paddingされた選択肢で
`teacher_probs=0`・`log_probs=-inf`となり`0 × (-inf - (-inf)) = NaN`(0×∞の不定形)が
発生した。`option_mask`で明示的に無視してから合計する手計算のKLに変更して解決した
(将来の蒸留学習スクリプトでも同じ罠を踏む可能性があるため、`test_fuudin_v4_e2e.py`の
`kl_loss()`にコメントで残してある)。

---

## 7. serialによるpointer監査結果

card id一致だけでは同一card idの別個体を誤検出できない問題を修正し、
`board_token_serial[target_index] == target_serial`を直接検証した。

### 7.1 合成データ(同一card id個体を含む)

`test_pointer_serial_audit_duplicate_card_id_synthetic`。ベンチに同じcard id(756)の
個体を2体(serial=2, serial=3)配置し、それぞれを対象にする選択肢を用意。

```
検証した非NO_TARGET選択肢: 3件(BENCH0, BENCH1, RETREAT)
serial不一致: 0件
NO_TARGET: 0件
2つのBENCHトークンはcard id(756)が同じだが、serialで正しく区別されている
```

### 7.2 実データ(自己対戦4試合)

`test_pointer_serial_audit_real_selfplay`。

```
決定点: 364件 / 総選択肢数: 2,641件
解決(非NO_TARGET): 1,382件
serial不一致: 0件
NO_TARGET: 1,259件
```

いずれも再実行可能なpytestテストとして残した(一回限りの手元スクリプトではない)。

---

## 8. 実測GPUメモリ・推論時間

`measure_t1_perf.py`(RTX 3060 Laptop、CUDA、fuudin_v4実データ8試合、
board_max=12・option_max=23〜26。過去の設計メモの手計算見積もりは
board_max=45〜90を想定していたが、実データでは大幅に少ないことが分かった)。

| batch size | forward+backward+step | 推論のみ(no_grad) | ピークVRAM |
|---|---|---|---|
| 256 | 11.5ms | 11.6ms | 0.070GB |
| 512 | 13.7ms | 3.4ms | 0.122GB |
| 1,024 | 17.9ms | 3.6ms | 0.223GB |
| 2,048 | 30.0ms | 6.6ms | 0.420GB |
| 4,096 | 59.4ms | 16.5ms | 0.845GB |
| 8,192 | 111.9ms | 25.2ms | 1.659GB |
| 16,384 | 1,967.0ms | 52.0ms | 3.293GB |

**旧設計メモの手計算見積もり(batch=16,384で13.8〜28.2GB、batch=2,048を推奨)は
大幅な過大評価だった。** 実測では16,384でも3.3GBに収まり、6GBのVRAM予算に対して
余裕がある。batch=16,384(現行`learner.py`の既定`minibatch_size`と同じ)でも
問題なく使える見込み。

**batch=16,384のforward+backward+stepが8,192の約17倍(111.9ms→1,967.0ms、
理論上は2倍程度のはず)になっている点は未解明。** 5回平均のみで計測しており、
メモリアロケータの挙動やウォームアップ不足の可能性がある。本格的な学習ループを
組む前に、より安定した計測(実行回数を増やす、`torch.cuda.synchronize()`のタイミング
見直し)で再確認することを推奨する。

---

## 9. 本格蒸留開始前の残課題

1. **正式教師checkpointが未確定。** v24/v32/v40/v47を同一条件・試合数≥1,000で
   再評価してから確定する(§前回報告の指摘のまま)。今回のfuudin_v4 E2Eテストは
   実装の動作確認のためv47をそのまま使っており、正式教師の選定とは無関係。
2. **T1単独の蒸留評価(§設計メモ§9の評価ゲート)が未実施。** 300試合の予備評価・
   1,000試合の最終確認・固定相手8体への相手別勝率比較はまだ行っていない。
3. **batch=16,384の速度異常(§8)が未解明。** 本格的な学習ループの前に再計測が必要。
4. **蒸留データの収集量が少ない(検証目的の8〜数十試合のみ)。** 本格的な蒸留には
   より大規模なデータ収集(design.md §7.1「最低1万決定点以上」)が必要。
5. **決定点数の再現性が無い(§2)。** 同じseedでも試合展開が変わりうるため、
   「同じ条件で再現する」ことを前提にした評価には注意が要る。
6. **DISCARD/ABILITY/CARD/SPECIAL_CONDITIONのpointer解決の一部が未検証**
   (前回報告からの継続課題)。
7. **T2(pointer gather)・PPO(Transformer自身のrollout)・NumPy推論・Kaggle worker
   組み込みはいずれも未着手。** ユーザー指示通り、正式教師の選定とT1単独の蒸留評価が
   完了するまでT2には進まない。

---

## 10. 参照資料

- [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂4
  — 本報告を反映した設計メモ本体。
- [transformer-t0.1-t1-implementation-report.md](./transformer-t0.1-t1-implementation-report.md)
  — T0.1・T1(改訂3)の実装報告。
- [transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md) — T0の実装報告。
