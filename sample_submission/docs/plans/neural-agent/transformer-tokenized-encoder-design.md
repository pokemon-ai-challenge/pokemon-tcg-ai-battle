# Transformer化(盤面トークン入力・段階拡張)設計メモ 改訂4版

改題の理由: 旧題「全ゾーン・トークン入力」は範囲を過大に表していた。T1/T2が
実際にトークン化するのは自分・相手のバトル場+ベンチ(盤面ポケモン)のみで、
手札・トラッシュ・山札/サイド等の「全ゾーン」はT3以降で段階的に追加する
(§3.3・§6)。現状の実装範囲に合わせて改題した。

作成日: 2026-08-07 / 改訂: 2026-08-08 / 改訂2〜4: 2026-08-08
ステータス: **T1.1補修(D2.1)まで実装済み(正規化2方式・checkpoint検査の必須化・
append-only vocabulary・seed非再現性の確定・shard/dataset/split契約・
resume契約とbit-exact resumeテスト)。正式教師checkpoint選定(D1)は完了し、
`model_v40.json`を選定した(§7)。v40教師での本格蒸留(データ収集・学習曲線・
4条件比較・3 seed確認)も完了し、最良条件はteacher_stats・T=1
(deployment_kl_t1約0.345、3 seedで安定)。**T1単独の対戦評価
(300試合予備評価・1,000試合以上のhead-to-head、§9の評価ゲート)だけが
未実施**(cg対局ループの中でT1をライブ推論するラッパーが未実装なのが
ブロッカー)。NumPy推論・Kaggle worker組み込み・T2のpointer gather・PPO学習は
未着手。T1単独の対戦評価が完了するまでT2には進まない。詳細は
[transformer-t1.1-patch-d1-d2-report.md](./transformer-t1.1-patch-d1-d2-report.md) §10。**

改訂4の内容: T1.1本体は
[transformer-t1.1-implementation-report.md](./transformer-t1.1-implementation-report.md)、
T1.1補修+D1(正式教師選定)+D2(蒸留trainer)は
[transformer-t1.1-patch-d1-d2-report.md](./transformer-t1.1-patch-d1-d2-report.md)参照。
T0.1/T1(改訂3)の詳細は
[transformer-t0.1-t1-implementation-report.md](./transformer-t0.1-t1-implementation-report.md)。

**本改訂4版が正。改訂3版・改訂2版・改訂版・旧版の内容のうち本文書と矛盾する記述は無効。**
特に「T1では自分の山札/サイド marginals・自分トラッシュ内訳を削除する」(改訂2版§3.2〜3.3、
§6のT1定義)は誤りだったため撤回し、下記§3.2(改訂)に置き換える。

---

## 0. 結論

今の方策モデル(`kaggle_replays/rl/torch_policy.py` の `TorchOptionPolicy`)を、
Transformer(トークン列をself-attentionで処理するモデル)に置き換える設計。
狙いは2つ。

1. 盤面の各要素(ベンチのポケモン等)を固定スロットではなく可変長のトークン列として扱い、
   並び順への依存をなくす。
2. **相手のトラッシュの中身**(現在は枚数しか使っていない、完全な公開情報)を含む、
   今まで方策に届いていなかった情報を追加する。

初期値は蒸留(既存方策の出力を教師にする)、学習資源はPPO更新をRTX 3060 Laptop、
収集をKaggle workerに割り振る、という方針は維持する。

**現時点で実装済みなのはT1.1(T0.1+T1に加え、正規化統計・Card ID vocabularyの
append-only統一・checkpoint契約の必須検査・serial単位のpointer監査・fuudin_v4
実データでのend-to-end検証)。** pointer gather・cross-attention・
option self-attention・Critic共有・PPO学習・pure-Python/NumPy推論・
Kaggle worker組み込みはまだ実装していない。正式教師checkpointの選定と
T1単独の蒸留評価が完了するまでT2には進まない。

---

## 0.1 改訂3の要点(T0.1・T1で直した/加えたもの)

- **T1で非盤面特徴を削除しない。** 自分の山札/サイド marginals(68次元)・自分トラッシュ
  内訳(22次元)は「盤面トークンに変換できない」ことを理由に削除する必要はなく、
  T1でも既存のglobal特徴としてそのまま維持する(§3.2改訂)。旧改訂2版の該当記述は誤り。
- **legacy global feature manifestを実装した**
  (`sample_submission/ptcg_ai/learning/legacy_feature_manifest.py`)。389/166次元の
  どの列がglobal特徴として残り、どの列が盤面トークンに置き換わるかを、ハードコードした
  sliceではなく名前付きグループで一意に定義する。
- **Card ID vocabularyを固定した**
  (`sample_submission/ptcg_ai/learning/card_vocab.py` + `card_vocab.json`)。
  raw card idが連続値である前提を置かず、PAD/UNK予約・バージョン・hash付きの
  変換表として学習時・提出時で同じファイルを読む。
- **zone_idからowner_idを決定的に導出する方式に統一した**(`board_tokens.owner_of_zone`)。
  owner用の別配列は持たない。
- **pointer監査情報(debug配列)を追加した**(`collect_tokens.py --debug-pointers`)。
  実データ4試合・771件のpointer解決を機械的に検査し、不一致0件を確認済み(§実装報告参照)。
- **T1(盤面Transformer本体)を実装した**(`kaggle_replays/rl/token_policy_t1.py`)。
  pointer gather・cross-attention・option self-attention・Critic共有は含まない。

## 0.2 改訂4の要点(T1.1で直した/加えたもの)

- **Card ID変換方式を統一した。** 盤面・選択肢の両方が`card_vocab.py`の固定語彙
  (PAD=0/UNK=1、**append-only**)でindexするようになり、raw card idを直接
  Embedding indexに使う経路(旧T1の`option_card_embedding`)を廃止した。
  `card_vocab.py`自体もappend-only(新規カードは末尾に追加、既存indexは並べ替えない)に
  作り直し、hashは並び順を含めて計算するよう修正した(旧実装は`sorted()`を挟んでいたため
  順序変化を検出できなかった)。
- **正規化統計を実装し、2方式を選べるようにした(T1.1補修)。** 当初は
  「board_numericのみ新規計算、global/optionは教師の既存統計を再利用」という
  1方式(`teacher_stats`)だけを実装していたが、これを`token_batch.NORMALIZATION_MODES`
  の一方に位置づけ、もう一方として`student_train_stats`
  (board_numeric・global・optionの**全て**を試合単位のtrain splitのみから新規計算、
  validationは統計計算に含めない)を追加した。`token_batch.build_normalization_stats(mode, ...)`
  が唯一の窓口。`normalization_mode`をcheckpointに保存し、ロード時に検査する
  (§checkpoint検査)。本格蒸留では両方式を小規模データで比較し、validation KLが
  良い方を採用する(実測比較は
  [transformer-t1.1-patch-d1-d2-report.md](./transformer-t1.1-patch-d1-d2-report.md) §3参照)。
- **checkpoint契約を実装し、検査を必須化した(T1.1補修)。**
  `token_policy_t1.save_checkpoint`/`load_checkpoint`が
  token_schema_version・vocabulary_version/hash・feature_profile・
  global_feature_manifest_hash・normalization_mode・正規化統計・use_board_card_id・
  card_vocab_sizeを一緒に保存する。**当初`load_checkpoint`の検査引数(`expect_*`)は
  省略可能で、渡し忘れると検査がスキップされる設計だった。** これを撤回し、
  現在の環境の値(`token_schema_version`/`vocabulary_version`/`vocabulary_hash`/
  `card_vocab_size`/`feature_profile`/`global_feature_manifest_hash`)を
  **すべて必須引数**にした(1つでも省略すると`TypeError`。黙って安全性を
  回避できない)。意図的に不一致を許す場合のみ`allow_unsafe_mismatch=True`を
  明示する。
- **pointer監査をserial単位に強化した。** card id一致だけでは同一card idの別個体を
  誤検出できない問題を修正し、`board_token_serial`をshardに保存(debug時のみ)、
  `board_token_serial[target_index] == target_serial`を機械的に検証する再実行可能な
  テスト(`kaggle_replays/rl/test_pointer_audit.py`)を追加した。
- **fuudin_v4(389次元)実データでのend-to-end動作を確認した。**
  State→token化→shard保存→再読込→token_batch→forward→KL loss→backward→
  optimizer.step→checkpoint保存→再読込後の出力一致を、pool_v1(v47)を教師にした
  実際のcgエンジン自己対戦データで確認済み(§実装報告参照)。
- **`old_logp`はPPOに使えないことを明記した**(§8.3)。

---

## 1. なぜ「モデルを大きくする」ではダメなのか(既存の実測結果)

`sample_submission/docs/plans/rl-selfplay/improvement-chain.md` の実験
(`fuudin_v3`、52世代・24,960試合)で、中間層を32から128に広げても勝率は
動かなかった(前半0.716±0.018、後半0.715±0.018、差-0.001±0.026)。
「同じ情報を、より大きいモデルに食わせる」だけでは勝率は上がらないことが
既にデータで示されている。この設計は**容量を増やすことを目的にしない。**

---

## 2. 相手のトラッシュが完全に無駄になっている(未使用の公開情報)

`sample_submission/cg/api.py` の `PlayerState` を確認した。相手の `discard: list[Card]`
は中身が全部見える公開情報だが、`sample_submission/ptcg_ai/learning/extended_features.py`
の `_archetype_probs()`(313〜348行目)は相手のトラッシュのカードidを読んだ後、
21次元のアーキタイプ事後確率に潰しており、**個別カードの情報は方策に届かない。**
T3でこれをカード単位のトークンとして追加する。

---

## 3. 現行モデルとの情報量の差分(訂正: 「同一情報」ではない)

### 3.1 現行モデルの構造(比較のため)

```
入力 = [盤面の固定ベクトル(166〜389次元) ++ 選択肢の固定ベクトル(65次元) ++ カードID埋め込み(8次元)]
     → Linear → ReLU → Linear(1個の隠れ層)
     → スカラー1個(その選択肢のスコア)
方策 = 全選択肢のスコアをsoftmax
```

盤面は固定スロット(`encoder.py` の `BENCH_SLOTS = 5`)で表現され、並び順に結果が依存する。

**現行方策の入力情報は、使っているモデルによって異なる。** 以降の比較は、蒸留教師の
候補である `kaggle_replays/rl/runs/pool_v1` を基準にする。このrunのモデルは
`meta.extended_features_profile = "fuudin_v4"`、盤面389次元(基本166次元+拡張223次元)
であることをファイルを読んで確認済み(§7で教師checkpointそのものの選定を再検討する)。

### 3.2 T1は「同一情報」ではない。正確な差分表(改訂3: 非盤面特徴は削除しない)

旧版は「T1(旧S1a)= 入力情報は現行と完全に同じ」としていたが、これは誤り。
**改訂2版はさらに、自分の山札/サイド marginals・自分トラッシュ内訳をT1で削除する
予定にしていたが、これも撤回する。** 盤面トークンに変換できないことは、
削除してよい理由にならない。以下、fuudin_v4(389次元)を基準にした特徴ブロックごとの
対応(改訂3)。

| 特徴ブロック | 次元 | 現行での生成箇所 | T1での扱い | 判定 |
|---|---|---|---|---|
| ポケモン数値特徴(自分/相手 × active/bench、11次元×12) | 132 | `encoder._pokemon_features` | 盤面トークンの数値特徴として移行(固定スロットのゼロ埋めのみ廃止) | **board_token_replaces(構造だけ変更)** |
| 手札内訳・カウント系・状態異常・ゲーム進行・集約(base166のうち非盤面) | 34 | `encoder.encode_state_from_state` | `legacy_global_features`としてそのまま維持 | **global_keep(維持)** |
| 自分の山札/サイド marginals(残り枚数・山札確率・サイド確率、カード単位) | 68 | `extended_features.py`(`own_zone="marginals"`) | **`legacy_global_features`としてそのまま維持**(旧改訂2版の「T3へ延期」を撤回) | **global_keep(維持)** |
| 自分トラッシュのカード単位内訳 | 22 | `extended_features.py`(B) | **`legacy_global_features`としてそのまま維持**(同上) | **global_keep(維持)** |
| 相手アーキタイプ21クラス事後確率 | 21 | `extended_features._archetype_probs` | `legacy_global_features`としてそのまま維持 | **global_keep(維持)** |
| 自分の盤面ポケモン識別(E1、`own_poke_ids`語彙7種の one-hot/カウント) | 14 | `extended_features.py`(E1) | 盤面トークンのcard id embedding(`use_board_card_id=True`時)に置き換え。語彙は7種→全1267カードに拡大 | **board_token_replaces(情報の質が変わる)** |
| 相手の盤面ポケモン識別(E2、`opp_poke_vocab`語彙48種+other) | 98 | `extended_features.py`(E2) | 同上(相手側) | **board_token_replaces** |
| 相手トラッシュのカード単位内訳 | (存在せず) | — | T3で新規追加(§2) | **新規(T1では追加しない)** |

正確な列インデックス(start/end)は
`sample_submission/ptcg_ai/learning/legacy_feature_manifest.py` が唯一の定義元
(`groups(profile)` の戻り値)。実測値: base166のglobal=34次元、fuudin_v4のglobal=145次元
(34+68+22+21)、board_token_replaces=244次元(132+112)。**「本当に削除する」に該当する
特徴ブロックは無い**(旧改訂2版の§3.2に書かれていた「削除」判定は誤りだった)。

**T1の情報量まとめ:**

- `use_board_card_id=True` のT1は、fuudin_v4の情報を全て保持する
  (board_token_replacesの112次元は「7/48種の語彙に限定したone-hot/カウント」から
  「card_vocab.pyの固定語彙全体を対象にした連続埋め込み」に表現が変わる)。
  **これを「上位互換」と断定しない。** 入力が持つ識別情報は語彙制限が外れる分だけ
  細粒度化されるが、細粒度化が実際の判断精度・勝率の向上につながるかどうかは
  蒸留評価(§9)で確認する。offline改善が実戦成績に転移しなかった前例
  (`model-capacity-ablation-implementation-plan.md`、Tier1/Tier3の経緯)がある以上、
  情報量の変化と性能の変化は別に確認すべき事柄として扱う。
- `use_board_card_id=False` のT1は、E1・E2ぶん(112次元相当の識別情報)を失う
  (アブレーションとして意図的に行う場合のみ)。

### 3.3 「カードIDを持たない手札/トラッシュ集約トークン」は成立しない

手札・トラッシュを「カード種類ごとに1トークン」としてトークン化する場合、
カード種類ごとにトークンを分けるという操作自体が「どのカードか」という
識別情報を前提にしており、card idを持たないトークンでは「3枚のネストボール」と
「2枚のハイパーボール」を区別する手段がない。

**手札・トラッシュ・山札/サイドのカード単位トークン化は、card id embeddingが
前提として必須な機能なので、これらのゾーンをトークン化する段階(T3)で
まとめて導入する。T1・T2では盤面ポケモン(active/bench)のみをトークン化し、
手札・トラッシュ・山札/サイドは従来通りglobal特徴(集約スカラー)のまま
据え置く。** これにより「card idを持たないゾーン集約トークン」という
矛盾した構成を作らずに済む。

**§6で確定した通り、T3が新規にトークン化するのは相手トラッシュのみ**
(§2の未使用公開情報)。自分の山札/サイド marginals・自分トラッシュ内訳は
T0.1で既に`legacy_global_features`として維持済みなので、この2つをT3で
「新規追加」することはない。**手札・スタジアム・自分の山札/サイドのカード単位
トークン化は本計画(T0〜T3)の範囲外とする。** 必要になった時点で、
本メモとは別に計画する。

### 3.4 デッキ依存性

card id embeddingは `card_vocab.py` の固定語彙(append-only、デッキ非依存)を使うため、
デッキを変えてもindexは変わらない。**raw card idをそのままEmbedding indexにする
設計は使っていない**(T1.1で撤回。旧`card_id_max+1`テーブル方式は
`kaggle_replays/rl/torch_policy.py`の既存`TorchOptionPolicy`にのみ残る、
T1では使わない設計)。語彙は`sample_submission/ptcg_ai/learning/card_vocab.json`に
固定され、新しいカードが増えても既存カードのindexは並べ替わらない(append-only、
`card_vocab.extend_vocab()`)。

---

## 4. 実装済み: 盤面トークン化とポインタ(T0)

`sample_submission/ptcg_ai/learning/board_tokens.py` に実装した。

### 4.1 トークンの中身

盤面トークンは自分・相手のバトル場+ベンチのポケモンのみ(§3.3の理由によりT1もこの
範囲)。1トークンは次の3つの並行配列の同じ添字で表す。

- `numeric_features`(11次元): `encoder._pokemon_features()` と全く同じ値を再利用。
- `card_ids`: そのポケモンの `Pokemon.id`(card id)。
- `zone_ids`: 4種類(`ZONE_SELF_ACTIVE=0`/`ZONE_SELF_BENCH=1`/`ZONE_OPP_ACTIVE=2`/
  `ZONE_OPP_BENCH=3`)。T3で使うゾーンID(スタジアム・手札・トラッシュ・山札/サイド)は
  4以降に予約済みで、T3導入時に既存IDは変わらない。

固定スロットのゼロ埋めはしない(ベンチが1体なら1トークンしか作らない)。

### 4.2 ポインタ方式(確定)

**標準的なcross-attentionはserialポインタを自動的には利用しない**という指摘の通り、
cross-attentionだけでは「この選択肢が指すのはこの個体」という決定的な対応関係を
表現できない(attentionは学習によって重みを決めるsoft機構であり、正解の対応を
外から強制する仕組みではない)。そこでT2では次の構成にする。

```
board_repr  = board Transformerの出力                    # (N_board, d_model)
global_repr = board_repr[CLSトークンの出力]                # (d_model,)
target_repr = board_repr から pointer で gather            # (d_model,)。対象が無い場合は
                                                            #   学習可能な NO_TARGET 表現を使う
option_repr = option encoderの出力(該当選択肢のトークン)   # (d_model,)

score = MLP(concat(global_repr, option_repr, target_repr))
```

`pointer`(gather用のインデックス)は学習ではなく、決定点ごとの前処理で
決定的に計算する(§4.3)。cross-attentionは「盤面全体のうちどこが選択肢に
関係しそうか」を学習で発見する補助として**併用してよいが、`target_repr`の
gatherに置き換えることはできない**(cross-attentionが対象を正しく見出す保証は
学習結果に依存するが、pointer gatherは常に正しい対象を返す)。設計メモ§6の
指示通り、**T2ではまずpointer gatherのみを実装し、これで不十分だと分かった
場合にのみcross-attentionを追加する。**

対象を持たない選択肢(YES/NO/NUMBER/END等)は、学習可能な `NO_TARGET` 埋め込み
(`nn.Parameter`、全選択肢で共有)を `target_repr` として使う。

### 4.3 pointerの解決: `Pokemon.serial` / `Card.serial`

`cg/api.py` の `Pokemon` dataclass(339〜348行目)・`Card` dataclass(333〜336行目)は
どちらも `serial: int` フィールドを持つ。**Pokemon以外の通常のCardにもserialは
存在する**ことをコードで確認した(`Card` dataclass自体が `id`/`serial`/`playerIndex`
の3フィールドで、Pokemonと共通の設計)。

`serial` は対戦中の個体に対して一意な番号で、既にこのリポジトリの他モジュールで
識別キーとして使われている(`consequence.py` 75行目、`attack_plan.py` 99行目、
`opponent_knowledge.py` 各所)。

**手札の同名カード複数枚を個体として区別する必要があるか:** T1・T2の対象は
盤面ポケモン(active/bench)のみで、手札はカード単位トークン化の対象外(§3.3)。
そのため現時点では手札カードの個体区別は不要。**§6で確定した通り、手札・
スタジアム・山札・サイド等の追加トークン化は本計画の範囲外であり、T3で新規に
追加するのは相手トラッシュのカード単位トークンのみ**(手札のトークン化は
行わない)。相手トラッシュを「カード種類ごとに1トークン(枚数を数値特徴に持つ)」
という集約方式でトークン化する予定で、個体(serial)単位のトークンにはしない設計
にする(トラッシュ内で同名カードのどの1枚かを区別する必要のある選択肢は
無いため)。

**カード種類単位のトークンにまとめる場合、pointerをcard id + zoneで解決できるか:**
できる。「相手トラッシュの中のネストボール」のように、集約トークンの対象は
「card id × zone」の組で一意に定まる(1つのゾーン内で同じcard idのトークンは
1つに集約されるため、同名カード同士を区別する必要がそもそも無い)。
これは個体識別(serial)とは異なる解決方式で、T3(相手トラッシュ)で導入する。

**トラッシュ・スタジアム・プレイヤー・ゾーンを対象とする選択肢の扱い:**
T1・T2では盤面ポケモン以外のゾーンをトークン化しないため、これらを指す選択肢は
`NO_TARGET` になる(§5のOptionType別解決表を参照。`AreaType.HAND`/`DISCARD`/
`PRIZE`/`DECK`/`STADIUM`/`PLAYER`/`LOOKING` を指す選択肢は `_UNRESOLVED_AREAS_T1`
としてまとめて未対応扱いにしている、`board_tokens.py` 参照)。T3で相手トラッシュを
追加した時点で、トラッシュを指す選択肢のみ解決可能になる(手札・スタジアム等は
本計画の範囲外のままT1.1と同じくNO_TARGET)。

**serial自体をshardに保存する必要があるか(T1.1で更新):**
**production shard(通常収集)では不要。** serialは決定点ごとに使い捨てる
中間キーで、トークン列構築とポインタ解決が同じ処理内で完結するため、
学習に使うのは解決済みの「ローカルトークンindex」だけで十分。

**debug shard(`collect_tokens.py --debug-pointers`)では監査目的で任意保存する。**
`board_token_serial`(盤面トークンごとのserial)・`debug_target_serial`(選択肢が
解決した先のserial)を追加し、`board_token_serial[target_index] == debug_target_serial`
を機械的に検証できるようにした(card id一致だけでは同一card idの別個体を
誤って指しても検出できないため、T1.1で追加。
`kaggle_replays/rl/test_pointer_audit.py`)。**通常のtrain/蒸留用shardには
これらのdebug配列を含めない**(容量削減、production経路と監査経路を分ける)。

---

## 5. OptionType別pointer解決表(実装確定)

`sample_submission/ptcg_ai/learning/board_tokens.py` の
`resolve_option_target_index()` / `OPTION_TYPE_POINTER_TABLE` に実装。
cg/api.pyの全17種類のOptionTypeを網羅する(単体テスト
`test_option_type_pointer_table_covers_all_enum_members` で確認済み)。

| OptionType | T1/T2での解決方針 | 根拠 |
|---|---|---|
| NUMBER | NO_TARGET | 対象を持たない純粋な個数選択 |
| YES | NO_TARGET | 対象を持たない |
| NO | NO_TARGET | 対象を持たない |
| CARD | area∈{ACTIVE,BENCH}なら解決可能。HAND/DISCARD/PRIZE等はNO_TARGET(T3で対応) | `_resolve_pokemon`(encoder.py)がarea/indexから解決 |
| TOOL_CARD | area=装着先ポケモンのACTIVE/BENCHで解決可能。対象は**装着先ポケモン**(装着カード自体ではない) | area は「Area of the attached Pokémon」(cg/api.py コメント) |
| ENERGY_CARD | TOOL_CARDと同じ | 同上 |
| ENERGY | TOOL_CARDと同じ | 同上 |
| PLAY | NO_TARGET(手札のカードが対象。手札は未トークン化) | 手札はT3までトークン化しない |
| ATTACH | inPlayArea/inPlayIndexで解決可能。対象は**装着される側のポケモン** | `_resolve_in_play_pokemon` |
| EVOLVE | inPlayArea/inPlayIndexで解決可能。対象は進化前のポケモン | 同上 |
| ABILITY | area∈{ACTIVE,BENCH}なら解決可能。特性の発動元がDISCARD等の特殊ケースはNO_TARGET | `_resolve_pokemon` |
| DISCARD | area∈{ACTIVE,BENCH}を指す場合のみ解決可能。それ以外は確認不能につきNO_TARGET | 資料不足で全ケースは断定不能 |
| **RETREAT** | **ルール上常に自分のバトルポケモン。** area/indexを持たないが、にげるはバトル場のポケモンしか行えないため暗黙に解決する | ゲームルール(コード上の裏付けは `rule_based/main_turn_parts/buckets.py` 50行目で `OptionType.RETREAT` がカテゴリ分類上「バトル場の行動」として扱われていることと整合。ルールブック上「にげるはバトルポケモンのみ」) |
| **ATTACK** | **ルール上常に自分のバトルポケモン。** attackIdのみでarea/indexを持たないが、ワザはバトル場のポケモンしか使えないため暗黙に解決する | 同上(ルール上「ワザはバトルポケモンのみ」) |
| END | NO_TARGET | ターン終了の宣言そのもの |
| SKILL | **`option.serial`で直接解決。** area/indexを経由せず最も直接的なケース | `cg/api.py`: 「serial (int):Card serial」。cardId=0(特殊状態の宣言)かつserial未設定ならNO_TARGET |
| SPECIAL_CONDITION | **暫定的に自分のバトルポケモン(要検証)。** ATTACK/RETREATほど強いコード上の裏付けを確認できていない | 確認不能。実際のゲームプレイでの検証が必要 |

**ATTACK/RETREATについて:** 単にarea/indexが無いことをもって「対象なし」と
判定するのは誤り。ゲームのルール上、ワザ・にげるは常にバトル場のポケモンにしか
行えないため、意味的には常に対象がある。`board_tokens.py` の
`_ACTIVE_IMPLICIT_TYPES`(ATTACK/RETREAT、確度高)と
`_ACTIVE_IMPLICIT_TYPES_TENTATIVE`(SPECIAL_CONDITION、確度低・要検証)として
明示的に分けて実装した。

---

## 6. 段階構成(T0〜T3)

| 段階 | 内容 | ステータス |
|---|---|---|
| **T0** | token schema・盤面トークンエンコーダ・新shard形式・教師logits保存・padding/mask生成用のカウント情報・encoderテスト・既存形式とのdual-write | **実装済み** |
| **T0.1** | 非盤面global特徴を維持する修正(§3.2)、legacy global feature manifest、Card ID vocabulary固定、zone→owner決定的導出、pointer監査情報 | **実装済み** |
| **T1** | 盤面Transformer(self-attention)。盤面card id embedding(`use_board_card_id`でablation可能)。選択肢のスコアリングは現行方式のMLPを維持(§3.2改訂により情報量はfuudin_v4を保持)。Criticは既存の独立MLPのまま | **実装済み(`kaggle_replays/rl/token_policy_t1.py`)** |
| **T2** | 選択肢→対象盤面トークンへのpointer gather(§4.2の方式)。pointer gatherで不十分な場合のみcross-attentionを追加 | 未着手 |
| **T3** | 相手トラッシュのカード単位トークン追加(自分の山札/サイド・トラッシュはT0.1で既にglobal特徴として維持済みのため、T3は相手トラッシュの新規追加のみ) | 未着手 |

**Option self-attentionとTransformer/Criticのパラメータ共有は必須構成から外し、
後続の任意実験とする**(ユーザー指示通り)。T1/T2/T3のいずれのスコープにも含まれない。
実施する場合は別途、任意実験として計画する。

**T3の範囲(確定):** 相手トラッシュのカード単位トークンを追加する
(§2の狙い、未使用の公開情報)。自分の山札/サイド marginals・自分トラッシュ内訳は
T0.1で既に`legacy_global_features`として維持済み(§3.2)なので、T3で改めて
追加する対象ではない。**T3が新規に追加するのは相手トラッシュのみ**であり、
手札・スタジアム等の他ゾーンをトークン化するかどうかは本メモの範囲外
(必要になった時点で別途計画する)。

---

## 7. 蒸留教師の選定(確定: v40)

**D1で正式評価を実施し、`model_v40.json`を正式教師に選定した**
(`kaggle_replays/rl/distributed/select_teacher.py`、既存`evaluate_pool()`を再利用)。
v24=0.7370(CI下限0.7088)・v32=0.7430(0.7150)・**v40=0.7910(0.7647)**・
v47=0.7420(0.7140)、1,000試合ずつ・near-greedy(`temperature=0.01`のsoftmax
サンプリング。厳密なargmaxではない)・8相手中7相手でv40が点推定最高。
詳細は
[transformer-t1.1-patch-d1-d2-report.md](./transformer-t1.1-patch-d1-d2-report.md) §6・§7。
以下は選定に至った調査の経緯(訂正: v47を無条件に選ばない、という判断根拠)。

`kaggle_replays/rl/runs/pool_v1/history.jsonl` を確認した(47世代分の記録)。

**`eval_pool_winrate`(固定評価相手群への、収集とは別の評価専用対戦。
near-greedy、`temperature=0.01`)が記録されているのは世代8/16/24/32/40のみ。**

| 世代(next_generation) | eval_pool_winrate(400試合) |
|---|---|
| 8 | 0.7175 |
| 16 | 0.70 |
| 24 | **0.77(記録された中で最高)** |
| 32 | 0.7625 |
| 40 | 0.76 |

**v41〜v47には`eval_pool_winrate`の記録が無い。** これらの世代には代わりに
`collect_winrate`(温度サンプリングで収集中に測った勝率。near-greedyではなく、
評価専用の固定対戦でもない、学習中の副産物としての勝率)しか記録がなく、
v47の値は0.7495。**`collect_winrate`と`eval_pool_winrate`は測定方法が違う別の指標で
あり、単純に比較できない。**

400試合というサンプルサイズでの95%信頼区間の幅は約±4%(正規近似)なので、
v24(0.77)・v32(0.7625)・v40(0.76)の差は統計的に有意とは言えず、
**「v47が最良」という主張はこの記録からは支持されない。むしろv47は
正式な評価(`eval_pool_winrate`)を一度も受けていない。**

**(この段落はD1着手前の計画メモ、実施済み。以下は歴史的記録として残す。)**
T1以降で本格的に蒸留データを収集する前に、v24/v32/v40/v47の候補に対して
同一条件(同じ固定相手群、同じ試合数≥1000)で`eval_pool_winrate`
相当の再評価を行い、その結果で教師checkpointを確定する方針だった。
**「可能なら同じseed」としていたが、§9.1.1で確定した通りcgエンジンのseedは
Pythonから制御できないため、実際のD1評価では同じseedによるpaired比較ではなく、
先攻/後攻・相手・試合数を均等化した独立標本として実施した。**
これは既存の`kaggle_replays/rl/distributed/learner.py` の `evaluate_pool()` 関数を
そのまま使えた(新規実装は不要)。T0の範囲では教師checkpointを確定しておらず、
`collect_tokens.py` の `--teacher-weights` はCLI引数で明示指定する方式にし、
コード側に既定値(v47等)をハードコードしていない。

---

## 8. 新shard形式(T1.1時点、`ptcg-rl-token-shard/2`)

`kaggle_replays/rl/distributed/token_shard.py`。既存の `common.py`
(`SHARD_FORMAT = "ptcg-rl-shard/2"`、こちらとは無関係の別のバージョン体系)は
**一切変更していない**。新形式は並存する別モジュール
(`TOKEN_SHARD_FORMAT = "ptcg-rl-token-shard/2"`。T0時点の`/1`からT1.1で
`/2`へ上げた。owner_idを保存しない方針を明記し、pointer監査用のdebug配列を
追加したための版上げ)。

### 8.1 必須配列

| 配列名 | shape | dtype | 意味 |
|---|---|---|---|
| `legacy_state_features` | (n_decisions, 166 or 389) | float32 | 既存 `encode_state_features` の出力(dual-write) |
| `legacy_option_features` | (総選択肢数, 65) | float32 | 既存 `encode_options_from_state` の出力(dual-write) |
| `option_card_ids` | (総選択肢数,) | int32 | 既存 `encode_option_card_ids` の出力(dual-write、raw card id) |
| `counts` | (n_decisions,) | int32 | 決定点ごとの選択肢数(既存shardの`counts`と同じ役割) |
| `board_token_numeric_features` | (総盤面トークン数, 11) | float32 | `board_tokens.build_board_tokens` の数値特徴 |
| `board_token_card_ids` | (総盤面トークン数,) | int32 | 盤面トークンのcard id(raw) |
| `board_token_zone_ids` | (総盤面トークン数,) | int32 | 盤面トークンのゾーンID(0〜3、T1/T2範囲) |
| `board_counts` | (n_decisions,) | int32 | 決定点ごとの盤面トークン数(padding/mask用) |
| `option_target_token_indices` | (総選択肢数,) | int32 | 選択肢が指す盤面トークンの**決定点内ローカルindex**。`NO_TARGET=-1` |
| `teacher_logits` | (総選択肢数,) | float32 | 教師MLPの生スコア(softmax前・温度適用前) |
| `chosen` | (n_decisions,) | int32 | 教師が実際に選んだ選択肢のindex |
| `old_logp` | (n_decisions,) | float32 | 収集時の対数確率(温度適用後のsoftmaxから)。**§8.3参照: 蒸留専用** |
| `lengths` | (n_trajectories,) | int32 | 試合ごとの決定点数 |
| `rewards` | (n_trajectories,) | float32 | 試合ごとの報酬(勝ち1/負け0) |
| `opp` | (n_trajectories,) | int16 | 試合ごとの相手番号 |

### 8.1.1 T1.1で追加した任意配列

| 配列名 | shape | dtype | 条件 |
|---|---|---|---|
| `legacy_global_features` | (n_decisions, global_dim) | float32 | `collect_tokens.py`が常に生成(`legacy_feature_manifest`のglobal_keep列のみ) |
| `board_token_serial` | (総盤面トークン数,) | int32 | `--debug-pointers`指定時のみ。盤面トークンごとの`Pokemon.serial` |
| `debug_target_serial` | (総選択肢数,) | int32 | 同上。選択肢の解決先serial(`None`は-1) |
| `debug_target_card_id` | (総選択肢数,) | int32 | 同上 |
| `debug_resolver` | (総選択肢数,) | int32 | 同上。`meta.resolver_kind_names`で文字列に戻せる |

`meta`(JSON文字列として格納)には `token_shard_format`・`n_decisions`・
`n_trajectories`・`legacy_state_dim`・`legacy_option_dim`・`legacy_global_dim`・
`board_token_numeric_dim`・`run_id`・`generation`・`teacher_weights_path`・
`teacher_model_sha256`・`extended_features_profile`・`vocabulary_version`・
`vocabulary_hash`・`games_requested`・`temperature`・`collected_at`・
`wins`/`valid`/`errors`・`written_at`/`host`/`python`・
(debug時のみ)`resolver_kind_names` を含む。

### 8.2 `counts`/`board_counts`と選択肢・トークンの対応

既存 `learner.py` の `counts`/`offsets`/`seg` と同じ考え方。決定点 `i` の選択肢は
`option_card_ids[cumsum(counts)[i]-counts[i] : cumsum(counts)[i]]`、盤面トークンは
`board_token_card_ids[cumsum(board_counts)[i]-board_counts[i] : cumsum(board_counts)[i]]`
で取り出せる(`token_shard.decision_slice()`/`iter_decisions()` に実装済み)。

`option_target_token_indices[j]` は、選択肢 `j` が属する決定点の盤面トークン範囲内での
**ローカルindex**(0始まり)。範囲外を指すことはない(実装は
`{serial: ローカルindex}` の辞書から引くため)。

### 8.3 `old_logp`はPPOには使えない(蒸留専用)

`old_logp` は教師MLP(pure-Python `PolicyModel`)が温度付きsoftmaxで選択したときの
対数確率であり、**教師の方策の下での値**。PPOの重要度比 `π_新(a)/π_収集(a)` は
「収集時の方策」と「更新対象の方策」が**同一のネットワーク**であることを前提にする
(`kaggle_replays/rl/distributed/test_distributed.py` の
`check_onpolicy_consistency` が既存PPOでこの前提を検査している)。

蒸留で作ったTransformer(T1〜)は教師MLPとは別のネットワークなので、
**shardの`old_logp`をTransformerのPPO更新にそのまま使うことはできない。**
Transformerで自己対戦PPOを始める段階(T2以降)では、**Transformer自身で
self-playしてold_logpを新規収集する必要がある**(既存`worker.py`/
`collect_parallel.py`と同じ構造の、Transformer版の収集経路が別途必要になる。
現時点では未実装)。

---

## 9. 評価ゲート(訂正: 300試合を予備評価に格下げ)

### 9.1 300試合と1,000試合の位置づけ

| 段階 | 試合数 | 目的 |
|---|---|---|
| 予備評価 | 300 | 明らかな劣化の早期検出のみ。**最終判断には使わない** |
| 最終確認 | 1,000(基本) | 既存MLPとの非劣性確認 |

固定相手8体それぞれについて同じ試合数で比較する。**「同じseedで揃える」ことは
できない(§9.1.1)。** 先攻/後攻は`parallel_collect`が試合indexの偶奇で
自動的に半々に割り振る仕組みに乗せることで均等化する(新規の仕組みは不要)。

### 9.1.1 「同じseedによるpaired比較」はできない(T1.1補修で調査・確定)

当初の設計は「可能なら同じseedで比較する」としていたが、**cgエンジンには
Pythonから呼べるseed設定APIが無い**ことを確認した
(`cg.api.lib`にseed関連の関数が存在しない、`test_seed_reproducibility.py`
`test_cg_engine_exposes_no_seed_function`)。

単一プロセス・マルチプロセス無しで、同じ`(learner_index, seed)`タプルを
`collect_tokens._play_one`に3回渡した結果、決定点数は88→69→124と毎回異なった
(**[D2.1] 確率的な観察のためpytestではなく`diagnose_seed_reproducibility.py`の
診断スクリプトとして実行・記録**)。**マルチプロセスの
タスク割当順序が原因という仮説はこれで排除できる**(単一プロセス・逐次呼び出しでも
再現しないため)。**Pythonから制御できないネイティブ側(cgエンジン内部)の乱数が
原因である可能性が有力だが、cgのソース自体は確認していないため断定はできない。
いずれにせよpaired比較には利用できない、というのが評価設計上の結論。**

**この結果、評価設計は「同じseedによるpaired比較」を前提にしない。**
先攻/後攻・相手・試合数を均等化した**独立標本**として評価する
(既存`evaluate_pool`/`parallel_collect`が、相手ごとの試合数を`share`で揃え、
先攻/後攻を試合indexの偶奇で揃える設計に既になっているため、追加の仕組みは不要。
D1の教師選定(§7)もこの前提で実施した)。

**なお、先攻/後攻そのものは制御できる。** `test_first_player_mapping.py`で
実際の対戦ログを確認した結果、`battle_start(deck0, deck1)`の**スロット順が
先攻/後攻を決定的に決めており**(`firstPlayer`は常にスロット0側)、これは
Pythonの`random.seed()`に非依存。`learner_index`(どちらのスロットに学習対象を
置くか)によって「学習対象が先攻か後攻か」は**確定的に**決まる(learner_index=0
→常に先攻、learner_index=1→常に後攻)。したがって`evaluate_pool`が
「試合indexの偶奇でlearner_indexを振る」ことは、そのまま「先攻/後攻を均等化する」
ことと同義であり、learner_indexの座席(スロット)と実際の先攻/後攻を
混同しない限り、§9.1で述べた先攻/後攻の均等化は保証されている。

### 9.2 非劣性マージン5%での必要試合数と検出力(計算し直した)

非劣性検定(non-inferiority test): 「studentの勝率が教師比で5%以上悪くない」
(真の勝率 ≥ 45%)を、片側検定(有意水準5%、`z=1.645`)で確認する。

`真の勝率50%(教師と同等)のときの検出力`(この検定で正しく「非劣性」と結論できる確率):

| 試合数 | 帰無仮説棄却の閾値(観測勝率がこれを超えれば非劣性と判定) | 検出力 |
|---|---|---|
| 300 | 49.72% | **53.8%** |
| 600 | 48.34% | 79.2% |
| 1,000 | 47.59% | **93.6%** |
| 1,500 | 47.11% | 98.7% |
| 2,000 | 46.83% | 99.8% |

計算式: 帰無仮説の境界を45%とし、閾値=45%+1.645×√(0.45×0.55/n)。検出力は
真の勝率50%のもとで観測勝率がこの閾値を超える確率(正規近似)。

**300試合では検出力53.8%しかなく、真にstudentが教師と同等でも約半分の確率で
「非劣性を確認できない」という誤った結論になる。これが「300試合を予備評価に
格下げ、最終確認は1,000試合を基本にする」根拠。** 1,000試合で検出力93.6%、
より高い保証が要る場合は1,500試合(98.7%)を検討する。

**「300試合で95%信頼区間下限45%以上」を絶対条件にしない。** 上記の通り
300試合はそもそも検出力が低く、この基準を満たさなかったからといって
本当に劣っているとは限らない(逆に満たしたからといって強く保証されるわけでもない)。
300試合は「明らかな劣化(50%を大きく割り込む)の早期検出」だけに使う。

### 9.3 合格条件(暫定値の明記)

| 指標 | 扱い |
|---|---|
| 既存MLPとの**直接対戦**head-to-head(1,000試合以上) | §9.2の非劣性検定で「非劣性」と判定されること(観測勝率が47.59%を超える)。**これは教師とstudentが同じ試合で直接対戦する場合の基準。** head-to-headはstudentの座席(learner_index)と実際の先攻/後攻を均等化し、引き分け/timeout/エラーの扱いを事前に固定した上で実施する |
| 固定相手8体への勝率(**教師 vs student、相手ごと・全体、主判定**) | **47.59%は使わない。** 教師とstudentは同じ相手プールに対して独立に評価する(§9.1.1、paired比較不可)ため、「studentの勝率 − 教師の勝率」の95%信頼区間を、相手・先攻後攻で層別した上で**全体**について計算し、**下限が-5%を上回るか**を主判定とする(`select_teacher.compare_pool_results`、§評価方法の分離) |
| 固定相手8体への相手ごとの勝率差(**補助指標、単体では非劣性判定に使わない**) | 相手1体あたり125試合程度では非劣性判定(下限-5%)に必要な検出力が不足する(§9.2と同様の理由で、相手ごとに個別の非劣性を要求する場合は事前にサンプルサイズ・検出力を計算すること)。相手ごとの差CIは**単体の相手で明らかな崩壊(大きく負に外れる)が無いかを確認する補助指標**としてのみ使い、個々の相手について-5%下限を満たすことを合格条件にはしない |
| top-1一致率 | **暫定値90%以上。** 参考: production実測の模倣学習自体の一致率が`test_top1=0.5806`なので、蒸留(既存モデル→新モデル)はこれより高い一致率が出るはず |
| validation KL | 実測してから閾値を決める(先行値なし) |
| **教師確率最大値が0.8以上の局面でのtop-1一致率** | 実測してから閾値を決める。「教師が自信を持って選ぶ局面」での一致度を別枠で見る(平均の一致率だけでは、教師が迷っている局面での不一致に埋もれる) |
| **教師entropy別の一致率** | 実測してから閾値を決める。教師のエントロピーが低い(確信度が高い)局面と高い(迷っている)局面を分けて集計する |
| **studentが選んだ行動に教師が割り当てていた確率** | 実測してから閾値を決める。「studentの選択を教師がどれだけ支持するか」という、top-1一致率より緩い指標 |
| 推論速度 | 未計測。ベンチマーク計画は investigation.md §9 参照(実測後に閾値化) |
| 複数seed | 最低3 seedで学習し、head-to-head勝率の分散を確認する |

---

## 10. T0の完了条件と結果

実装後の詳細な結果・テスト一覧は
[transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md)
にまとめた。要約:

| 完了条件 | 結果 |
|---|---|
| 既存MLPの推論結果を変更しない | 満たす(既存ファイル無変更、新規テストで`score_options_from_state`との一致を確認) |
| 同じStateを複数回encodeして完全に同じトークン列になる | 満たす(`test_deterministic_same_input_same_output`) |
| ベンチトークンを並べ替えても、serialから解決したpointerが同じ個体を指す | 満たす(`test_pointer_survives_bench_reorder`) |
| 同一card idのポケモンが複数いてもpointerを一意に解決できる | 満たす(`test_pointer_resolves_uniquely_for_duplicate_card_id`) |
| pointer無し選択肢を安全に表現できる | 満たす(`NO_TARGET=-1`、`test_no_target_options`等) |
| 可変長の盤面・選択肢をshardへ保存し、再読み込みできる | 満たす(`test_round_trip_shapes_and_values`、実データでも確認) |
| 既存389次元・65次元・cidsも同時保存される | 満たす(dual-write) |
| teacher raw logitsがcountsと完全に整合する | 満たす(長さ不一致は`write_token_shard`が例外を出す設計、テスト済み) |
| 旧shard readerを壊さない | 満たす(`common.py`無変更) |
| 新形式の単体テストがすべて通る | 満たす(29件、全てpass) |

---

## 11. 変更していないもの(可逆性)

- 提出用の推論コード(`sample_submission/ptcg_ai/learning/policy_model.py`)は
  一切変更していない。
- 既存の重み(`policy_weights.json`、`kaggle_replays/rl/runs/*/models/model_v*.json`)は
  変更していない。
- 既存shard形式(`kaggle_replays/rl/distributed/common.py`)・既存収集経路
  (`worker.py`/`collect_parallel.py`)は一切変更していない。新形式は並存する
  別モジュール(`token_shard.py`/`collect_tokens.py`)。
- `cg/` フォルダ・`data/` フォルダは対象外(CLAUDE.mdの絶対ルール通り、触っていない)。

---

## 12. 参照資料

- [transformer-tokenized-encoder-investigation.md](./transformer-tokenized-encoder-investigation.md)
  — 旧版の記述に対する調査根拠(ファイル名・行番号つき)。
- [transformer-t0-implementation-report.md](./transformer-t0-implementation-report.md)
  — T0実装の詳細な報告。
- `sample_submission/docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`
  — 現行モデルのパラメータ数・構造の正確な内訳。
- `sample_submission/docs/plans/rl-selfplay/results-and-concerns.md` /
  `improvement-chain.md` — 分散self-playの実測結果と教訓。
- `sample_submission/ptcg_ai/learning/encoder.py` / `extended_features.py` /
  `policy_model.py` / `board_tokens.py`(新規) — 現行+T0の特徴量・推論コード。
- `kaggle_replays/rl/torch_policy.py` / `distributed/{common,worker,learner,token_shard}.py`
  (`token_shard.py`は新規) / `collect_parallel.py` / `collect_tokens.py`(新規) /
  `rollout.py` / `train_v3.py` — 現行+T0のPPO学習パイプライン。
- `cg/api.py` — `OptionType`・`Pokemon`・`Card`・`PlayerState`等のデータ構造
  (変更不可、参照のみ)。
