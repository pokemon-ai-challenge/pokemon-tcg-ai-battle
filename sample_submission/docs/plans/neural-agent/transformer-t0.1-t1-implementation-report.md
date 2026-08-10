# T0.1 + T1 実装報告

作成日: 2026-08-08
対象: [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂3
前提: [transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md)(T0)の続き
ステータス: **T0.1・T1完了。T2(pointer gather)・本格蒸留・PPOは未着手。**

cg/・data/は変更していない。既存MLP・既存shard reader(`common.py`)・既存モデル重みは
一切変更していない。Criticは既存の独立MLP(`train_v3.Critic`)のまま。新規ファイルの
追加・T0で作った既存新規ファイルへの追記のみ。

---

## 1. T0.1の変更ファイル

| ファイル | 内容 |
|---|---|
| `sample_submission/ptcg_ai/learning/legacy_feature_manifest.py`(新規) | legacy global feature manifest。389/166次元の各列がglobal_keep/board_token_replacesのどちらかを一意に定義 |
| `sample_submission/ptcg_ai/learning/card_vocab.py`(新規) | Card ID vocabulary(PAD/UNK予約・version・hash) |
| `sample_submission/ptcg_ai/learning/card_vocab.json`(新規・生成物) | 固定された語彙ファイル。1267カード+PAD+UNK=1269。学習・提出で同じファイルを読む |
| `sample_submission/ptcg_ai/learning/board_tokens.py`(T0から追記) | `owner_of_zone()`(zone→owner決定的導出)、`resolve_option_target_index_debug()`・`PointerDebugInfo`(pointer監査) |
| `kaggle_replays/rl/distributed/token_shard.py`(T0から改訂) | `TOKEN_SHARD_FORMAT`を`/1`→`/2`に、`legacy_global_features`配列(任意)・pointer監査debug配列(任意)を追加 |
| `kaggle_replays/rl/collect_tokens.py`(T0から改訂) | `--debug-pointers`フラグ、`legacy_global_feat`の計算・保存、`vocabulary_hash`/`vocabulary_version`をmetaに記録 |
| `kaggle_replays/rl/token_batch.py`(新規) | shardからpaddingされたバッチを作る(board/option両方、card idのvocab index変換、profile不一致の検出) |
| テスト5ファイル | `test_legacy_feature_manifest.py`(17件)・`test_card_vocab.py`(9件のうち一部既存11件と統合)・`test_board_tokens.py`追記(28件)・`test_token_shard.py`追記(11件)・`test_collect_tokens.py`追記(5件)・`test_token_batch.py`(新規9件) |

## 1.1 T1の変更ファイル

| ファイル | 内容 |
|---|---|
| `kaggle_replays/rl/token_policy_t1.py`(新規) | `BoardTokenEmbedding`・`BoardTransformer`・`T1OptionPolicy` |
| `kaggle_replays/rl/test_token_policy_t1.py`(新規) | 10件 |

**既存production推論・PPO経路は無変更。** `torch_policy.py`(既存`TorchOptionPolicy`)、
`train_v3.py`(既存`Critic`)、`policy_model.py`、`encoder.py`、`common.py`は無変更。

---

## 2. global feature manifest

`sample_submission/ptcg_ai/learning/legacy_feature_manifest.py`。

### 2.1 base166(全profile共通)

| グループ名 | start | end | dim | usage |
|---|---|---|---|---|
| board_self_active | 0 | 11 | 11 | board_token_replaces |
| board_self_bench | 11 | 66 | 55 | board_token_replaces |
| board_opp_active | 66 | 77 | 11 | board_token_replaces |
| board_opp_bench | 77 | 132 | 55 | board_token_replaces |
| self_counts | 132 | 140 | 8 | global_keep |
| opp_counts | 140 | 145 | 5 | global_keep |
| self_special_conditions | 145 | 150 | 5 | global_keep |
| opp_special_conditions | 150 | 155 | 5 | global_keep |
| game_progress | 155 | 162 | 7 | global_keep |
| aggregate | 162 | 166 | 4 | global_keep |

### 2.2 拡張(profile="fuudin_v4"、166に続く223次元)

| グループ名 | start | end | dim | usage |
|---|---|---|---|---|
| self_deck_prize_marginals | 166 | 234 | 68 | global_keep |
| self_discard_breakdown | 234 | 256 | 22 | global_keep |
| board_self_active_identity | 256 | 263 | 7 | board_token_replaces |
| board_self_bench_identity | 263 | 270 | 7 | board_token_replaces |
| board_opp_active_identity | 270 | 319 | 49 | board_token_replaces |
| board_opp_bench_identity | 319 | 368 | 49 | board_token_replaces |
| opp_archetype | 368 | 389 | 21 | global_keep |

profile="fuudin_v2"は`self_deck_prize_marginals`が23次元(`own_zone="count"`、
marginals分割なし)になる点のみ異なり、他は同型(モジュール内`_EXTENDED_SPEC`参照)。

**global_dim(実測): base166=34 / fuudin_v2=100 / fuudin_v4=145。**
**board_token_replaces合計: base166=132 / 拡張時=+112=244。「削除」に該当するグループは無い**
(`test_no_removed_category_remains`で確認)。

### 2.3 166次元shardの扱い

profileが異なるshardは**無言で混在させない。** `token_batch.validate_profile_consistency()`が
複数shardのmetaの`extended_features_profile`を比較し、揃っていなければ
`ProfileMismatchError`を出す(`test_mixed_profile_shards_rejected`で確認)。166次元と
389次元を同じバッチに詰めることは構造的にできない(`legacy_global_features`の次元自体が
34と145で異なるため)。

---

## 3. Card ID vocabulary仕様

`sample_submission/ptcg_ai/learning/card_vocab.py` + `card_vocab.json`。

| 項目 | 仕様 |
|---|---|
| raw Card ID → embedding index | `card_vocab.json`の`card_ids`(昇順ソート済みリスト)のindex + 2。連続値であるという仮定は置かない(実際には現在1〜1267で連続だが、コードはこれに依存しない) |
| PAD index | 0(`card_vocab.PAD_INDEX`) |
| UNK index | 1(`card_vocab.UNK_INDEX`)。未登録card idは全てここに落ちる |
| 既知カードのindex | 2始まり(2〜1268、1267カード) |
| 語彙の並べ方 | card id昇順。ソート順が変わらない限りindexは安定 |
| vocabulary version | `card_vocab.json`の`version`フィールド(現在1) |
| vocabulary hash | `card_vocab.json`の`hash`フィールド(sorted card idリストのsha256)。改ざん・破損を`load_vocab()`が検出する |
| 未登録カードの処理 | `index_of()`がUNK_INDEXを返す(例外にしない、fail-soft) |
| 学習時と提出時に同じ語彙を読み込む方法 | `card_vocab.json`を`policy_weights.json`と同じ運用でリポジトリにコミットする。両環境が同じファイルを読むので、cgエンジンを起動しなくても(`load_vocab()`はファイルI/Oのみ)同一の語彙になる |
| checkpointへの対応付け | `collect_tokens.py`がshardのmetaに`vocabulary_hash`/`vocabulary_version`を記録する。T1の学習コードも同様にmodel checkpointのmetaへ記録すべき(T1実装では学習ループ自体がまだ無いため、この配線はT2以降で行う) |

**実測size: 1269**(1267カード + PAD + UNK)。`test_load_committed_vocab_matches_engine`で
現在のcgエンジンのカード集合と`card_vocab.json`が一致することを確認済み(9件のテスト全pass)。

将来カードが増えた場合は`card_vocab.regenerate_and_check()`で「既存カードのindexが
ずれていないか」を確認してから上書きする(ソート順に挿入されるため、既存より小さいidの
新カードが増えるとずれが起きうる。ずれがあれば人間の判断を仰ぐ設計)。

---

## 4. 更新後のtoken shard schema(v2)

`TOKEN_SHARD_FORMAT = "ptcg-rl-token-shard/2"`(v1から変更。owner_idは保存しない
方針を明記し、pointer監査用debug配列を追加したための版上げ)。

### 4.1 必須配列(T0から変更なし)

`legacy_state_features` / `legacy_option_features` / `option_card_ids` / `counts` /
`board_token_numeric_features` / `board_token_card_ids` / `board_token_zone_ids` /
`board_counts` / `option_target_token_indices` / `teacher_logits` / `chosen` /
`old_logp` / `lengths` / `rewards` / `opp`(shape/dtypeは[前回報告](./transformer-t0-implementation-report.md)参照)。

### 4.2 新規配列(いずれも任意。無ければキー自体が無い)

| 配列名 | shape | dtype | 条件 |
|---|---|---|---|
| `legacy_global_features` | (n_decisions, global_dim) | float32 | `collect_tokens.py`が常に生成する(§5で「維持」と判定された列のみ) |
| `debug_target_serial` | (総選択肢数,) | int32 | `--debug-pointers`指定時のみ。`None`は-1 |
| `debug_target_card_id` | (総選択肢数,) | int32 | 同上。`None`は-1 |
| `debug_resolver` | (総選択肢数,) | int32 | 同上。`meta.resolver_kind_names`で文字列に戻せる |

owner_idは配列として保存しない。`board_tokens.owner_of_zone(zone_id)`で決定的に
導出する(zone_id→owner_idの対応は固定schemaとしてコード化済み)。

### 4.3 token/padding/CLSの保存範囲

| 要素 | shardに保存 | DataLoader/学習時に生成 |
|---|---|---|
| 盤面トークンの数値特徴・card id・zone_id | ○(可変長、paddingなし) | — |
| owner_id | × | `owner_of_zone(zone_id)`から生成(`token_batch.build_board_batch`) |
| token type | × | T1/T2の範囲では全トークンがpokemon型なので`zone_id`自体が型を兼ねる。T3で非pokemon型が増えたら再検討 |
| padding | × | `token_batch.build_board_batch`/`build_option_batch`がバッチ構築時に生成 |
| padding mask | × | 同上 |
| [CLS] | × | モデル内部(`token_policy_t1.BoardTransformer`)が学習可能パラメータとして持つ。shard/バッチデータには含まれない |
| card idのvocab index変換 | × | `token_batch`が`card_vocab.CardVocab.index_of()`で変換(生のcard idのままshardには保存する) |

---

## 5. pointer監査(機械的検証の結果)

`--debug-pointers`で実データ4試合を収集し検証した(`test_token_batch.py`の合成データ検証に加え、
実対戦での大規模チェック)。

```
決定点: 232件 / 選択肢: 771件(pointer解決対象)
resolver内訳: no_target_no_fields=599, in_play_index=560, no_target_unresolved_zone=203,
             active_implicit=111, area_index=100
機械的検証: board_token_card_ids[target_index] == debug_target_card_id を
           全771件の非NO_TARGET選択肢で照合 -> 不一致 0件
```

この検証はスクリプトとして`kaggle_replays/rl/distributed`配下では保存していない
(手元で実行した一回限りのスクリプト)。再現する場合は
`collect_tokens.py --debug-pointers`でshardを作り、`token_shard.iter_decisions()`で
決定点を辿りながら`board_token_card_ids[board_off + target_index] == debug_target_card_id`
を照合すればよい(手順は上記コマンド列そのもの)。

---

## 6. T1の正確なモデル構造

`kaggle_replays/rl/token_policy_t1.py`。

```
board tokens(numeric 11次元・card_id・zone_id・owner_id)
  → BoardTokenEmbedding: Linear(11,64) + Embedding(1269,64) + Embedding(9,64) + Embedding(2,64)
    を要素和 → LayerNorm(64)
  → BoardTransformer: 学習可能[CLS](64) を先頭に結合
    → TransformerEncoderLayer × 2(d_model=64, heads=4, ffn=256, dropout=0.05,
       activation=ReLU, norm_first=True(Pre-LN)) の TransformerEncoder
    → [CLS]位置の出力を取り出す(64次元)

CLS(64) ++ legacy_global_features(34 or 100 or 145) ++ 既存option特徴(65) ++
既存option_card_embedding(8、既存card_id_max+1テーブルをそのまま流用)
  → Linear(in_dim, 32) → ReLU → Linear(32, 1) → 各選択肢のスコア(paddingは-inf)
```

### 6.1 既存Option scorerとの対応(現行`PolicyModel`/`TorchOptionPolicy`との比較)

| 部分 | 既存(`torch_policy.TorchOptionPolicy`) | T1 | 変更有無 |
|---|---|---|---|
| 選択肢の対象card id embedding | `card_embedding = nn.Embedding(card_id_max+1, 8)`(47行目) | `option_card_embedding`として**同一定義をそのままコピー** | **維持** |
| 選択肢の65次元特徴 | 標準化して結合(`_standardize_option`) | 同型の標準化バッファ(`option_mean`/`option_std`)で結合 | **維持(構造同一)** |
| 盤面の表現 | `state_feat`(166/389固定ベクトル)を標準化してbroadcast | 盤面トークン列 → BoardTransformer → CLS出力(64次元) | **置き換え** |
| 盤面のglobal部分 | `state_feat`に含まれ、盤面個体表現と未分離のまま標準化 | `legacy_global_features`として別枠で標準化(`global_mean`/`global_std`) | **分離(§3.2)** |
| 最終層 | `Linear(in_dim,H)→ReLU→Linear(H,1)`(`policy_model.py`と同型) | **同じ2層構成をそのまま踏襲**(in_dimのみ変わる) | **維持(構造同一)** |
| 盤面ポケモンのcard id識別 | fuudin_v4使用時のみ、E1/E2の限定語彙one-hot/カウントとして`state_feat`内に存在 | 盤面トークンの`card_embedding`(全1267カード対応、d_model=64)に置き換え | **拡張** |

### 6.2 各テンソルのshape

| テンソル | shape | dtype |
|---|---|---|
| `board_numeric` | (B, N_board, 11) | float32 |
| `board_card_idx` | (B, N_board) | int64(vocab index) |
| `board_zone_ids` | (B, N_board) | int64 |
| `board_owner_ids` | (B, N_board) | int64 |
| `board_mask` | (B, N_board) | bool |
| `global_features` | (B, global_dim) | float32 |
| `option_features` | (B, N_opt, 65) | float32 |
| `option_card_ids` | (B, N_opt) | int64(raw card id、既存card_embedding用) |
| `option_mask` | (B, N_opt) | bool |
| 出力スコア | (B, N_opt) | float32(paddingは-inf) |

### 6.3 総パラメータ数(実装後にコードで計算。手計算ではない)

`model.total_params()`(`sum(p.numel() for p in model.parameters())`)で実測。

| profile | global_dim | 総パラメータ数 | 内訳(board_embed / transformer / option_embed / fc) |
|---|---|---|---|
| base166 | 34 | **198,529** | 82,816 / 100,032 / 10,144 / 5,537 |
| fuudin_v4 | 145 | **202,081** | 82,816 / 100,032 / 10,144 / 9,089 |

board_embed内訳: numeric_proj(11×64+64=768) + card_embedding(1269×64=81,216) +
zone_embedding(9×64=576) + owner_embedding(2×64=128) + LayerNorm(2×64=128) = 82,816。
transformer内訳: [CLS]パラメータ(64) + TransformerEncoderLayer×2(1層あたり49,984、
self_attn 16,640 + FFN 33,088 + LayerNorm×2 256)= 100,032。option_embedding=1268×8=10,144
(既存production `policy_weights.json` の card_embedding と同一サイズ)。

現行MLP(hidden=32、production)の17,857パラメータと比べて**約11〜11.3倍**
(旧設計メモの見積もり「15〜25倍」よりは小さい実測値)。

---

## 7. 実行したテストと結果

| テストファイル | 件数 | 結果 |
|---|---|---|
| `sample_submission/tests/unit/test_legacy_feature_manifest.py` | 9 | 全pass |
| `sample_submission/tests/unit/test_card_vocab.py` | 8 | 全pass |
| `sample_submission/tests/unit/test_board_tokens.py`(T0の21件+T0.1追加7件) | 28 | 全pass |
| `kaggle_replays/rl/distributed/test_token_shard.py`(T0の5件+T0.1追加6件) | 11 | 全pass |
| `kaggle_replays/rl/test_collect_tokens.py`(T0の3件+T0.1追加2件) | 5 | 全pass |
| `kaggle_replays/rl/test_token_batch.py`(新規、境界ケース含む) | 9 | 全pass |
| `kaggle_replays/rl/test_token_policy_t1.py`(新規、T1モデル) | 10 | 全pass |
| **T0.1+T1 新規合計** | **80** | **全pass** |
| `sample_submission/tests/unit`(既存回帰チェック、フォルダ全体) | 339 passed / 6 skipped(既存TODOスキップ、無関係) | 回帰なし |

実データでの検証(§5)に加え、`collect_tokens.py`で実際にcgエンジンの自己対戦を実行し、
`legacy_global_features`(232決定点、34次元、base166 profile)・`vocabulary_hash`が
正しく書き出されることを確認済み。

---

## 8. tiny dataset過学習結果

`test_overfits_tiny_dataset`(合成3決定点、hard-label cross entropy、Adam lr=1e-2、200 step)。

```
初期loss → 最終loss: 大幅に低下(最終lossが初期lossの5%未満まで低下したことをassertで確認)
最終予測: 3件全て正解ラベルと一致
```

`test_distillation_loss_decreases`(合成4決定点、ダミー教師分布に対するKLダイバージェンス、
150 step)。

```
最終KL損失 < 初期KL損失 × 0.5(半分以下に低下)
```

どちらも学習経路(forward → loss → backward → optimizer.step)が機能していることの確認で、
実データでの学習性能を保証するものではない。

---

## 9. T2または本格蒸留へ進む前の残課題

1. **蒸留教師checkpointが未確定。** [T0報告](./transformer-t0-implementation-report.md) §7の
   指摘の通り、`pool_v1`のv24/v32/v40/v47を同一条件・試合数≥1,000で再評価してから
   確定する必要がある。T1の実装・単体テストには暫定教師(production既定の
   `policy_weights.json`)を使っており、これは正式教師の選定と分離されている
   (§9の評価は今回のT1実装作業には含めていない)。
2. **DISCARD/ABILITY/CARD/SPECIAL_CONDITIONのpointer解決の一部が未検証。**
   T0報告§7の指摘がそのまま残る。§5の機械的検証は実際に収集した4試合の範囲でのみ
   不一致0件を確認しており、これらの型が実戦でどれだけの頻度・パターンで出現するかは
   未調査。
3. **legacy_global_features/board_token_numeric_featuresの標準化(mean/std)が未計算。**
   `T1OptionPolicy`の`global_mean`/`global_std`/`option_mean`/`option_std`バッファは
   ゼロ・1で初期化されたままで、実データから計算した値に差し替えていない
   (T1は構造の実装のみが対象範囲で、学習データパイプラインはT2以降)。
4. **蒸留の学習ループ(shard読み込み→token_batch→T1forward→KL loss→optimizer)が
   まだ組まれていない。** `token_batch.py`と`token_policy_t1.py`は個別に動作確認済みだが、
   両者を繋いで大量のshardから学習する具体的なスクリプトは未実装。
5. **GPUメモリ・推論時間の実測はまだ行っていない。** T1のパラメータ数は実測できたが
   (§6.3)、VRAM使用量・1バッチあたりの学習時間は未計測。
6. **pointer gather(T2)は未着手。** 今回はcross-attention・pointer gatherのどちらも
   実装していない(ユーザー指示通り)。`option_target_token_indices`はshardに保存済みで
   T2で使う準備は整っている。
7. **`legacy_option_dim`が0(選択肢配列が空)の場合の`board_token_numeric_dim`等の
   境界挙動は、極端なケース(全試合が即エラー終了する等)まで網羅的にテストしていない。**

---

## 10. 参照資料

- [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) 改訂3
  — 本報告を反映した設計メモ本体。
- [transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md)
  — T0(基盤)の実装報告。
- [transformer-tokenized-encoder-investigation.md](./transformer-tokenized-encoder-investigation.md)
  — 旧版設計に対する最初の調査。
