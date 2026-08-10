# T0実装報告

作成日: 2026-08-08
対象: [transformer-tokenized-encoder-design.md](./transformer-tokenized-encoder-design.md) T0
ステータス: **T0完了。T1(Transformer本体)・NumPy推論・TorchTokenPolicyは未着手。**

cg/・data/は変更していない。既存MLP・既存shard(`common.py`)・既存モデル重みは
一切変更していない。新規ファイルの追加のみ。

---

## 1. 変更ファイル一覧(すべて新規追加。既存ファイルの変更ゼロ)

| ファイル | 役割 |
|---|---|
| `sample_submission/ptcg_ai/learning/board_tokens.py` | 盤面トークンエンコーダ・ポインタ解決(OptionType別解決表を含む) |
| `sample_submission/tests/unit/test_board_tokens.py` | 上記の単体テスト(21件) |
| `kaggle_replays/rl/distributed/token_shard.py` | 新shard形式の読み書き(`write_token_shard`/`read_token_shard`/`decision_slice`) |
| `kaggle_replays/rl/distributed/test_token_shard.py` | 上記の単体テスト(5件) |
| `kaggle_replays/rl/collect_tokens.py` | 既存教師MLP(`PolicyModel`)でself-playしながらトークン形式shardを収集するスクリプト |
| `kaggle_replays/rl/test_collect_tokens.py` | 上記の単体テスト(3件) |
| `sample_submission/docs/plans/neural-agent/transformer-tokenized-encoder-design.md` | 改訂2(T0〜T3構成・情報量差分表・pointer方式・評価ゲート見直し) |
| `sample_submission/docs/plans/neural-agent/transformer-t0-implementation-report.md` | 本ファイル |

既存ファイルで**読んだが変更していない**もの: `encoder.py`(`_pokemon_features`等を
インポートして再利用)、`policy_model.py`(`PolicyModel`をそのまま教師として利用)、
`common.py`(`sha256_file`のみ利用)、`extended_features.py`(`begin_match`/`observe`を
既存の呼び出しパターンのまま踏襲)。

---

## 2. 新shard schema

`kaggle_replays/rl/distributed/token_shard.py`、`TOKEN_SHARD_FORMAT = "ptcg-rl-token-shard/1"`。
既存 `common.py` の `SHARD_FORMAT = "ptcg-rl-shard/2"` とは独立した別形式(置き換えではない)。

## 3. 各配列のshape・dtype・意味

| 配列名 | shape | dtype | 意味 |
|---|---|---|---|
| `legacy_state_features` | (n_decisions, 166 or 389) | float32 | 既存 `encode_state_features` の出力(dual-write) |
| `legacy_option_features` | (総選択肢数, 65) | float32 | 既存 `encode_options_from_state` の出力(dual-write) |
| `option_card_ids` | (総選択肢数,) | int32 | 既存 `encode_option_card_ids` の出力(dual-write) |
| `counts` | (n_decisions,) | int32 | 決定点ごとの選択肢数 |
| `board_token_numeric_features` | (総盤面トークン数, 11) | float32 | ポケモン1体あたりの数値特徴(`encoder._pokemon_features`と同一) |
| `board_token_card_ids` | (総盤面トークン数,) | int32 | 盤面トークンのcard id |
| `board_token_zone_ids` | (総盤面トークン数,) | int32 | ゾーンID(0=自分active/1=自分bench/2=相手active/3=相手bench。T3用に4以降予約) |
| `board_counts` | (n_decisions,) | int32 | 決定点ごとの盤面トークン数 |
| `option_target_token_indices` | (総選択肢数,) | int32 | 選択肢が指す盤面トークンの決定点内ローカルindex。`NO_TARGET=-1` |
| `teacher_logits` | (総選択肢数,) | float32 | 教師MLPの生スコア(softmax前・温度適用前、`PolicyModel._forward`の生値) |
| `chosen` | (n_decisions,) | int32 | 教師が実際に選んだ選択肢のindex |
| `old_logp` | (n_decisions,) | float32 | 収集時の対数確率(温度適用後softmax) |
| `lengths` | (n_trajectories,) | int32 | 試合ごとの決定点数 |
| `rewards` | (n_trajectories,) | float32 | 試合ごとの報酬(勝ち1/負け0) |
| `opp` | (n_trajectories,) | int16 | 試合ごとの相手番号 |

詳細(meta schema・counts/board_countsの復元方法)は design.md §8 を参照。

## 4. OptionType別pointer解決表

design.md §5 に全17種類の一覧を掲載(`board_tokens.OPTION_TYPE_POINTER_TABLE`と対応)。
要点: ATTACK/RETREATはarea/indexを持たないが、ゲームルール上常に自分のバトルポケモンが
対象なので明示的に解決する。SKILLは`option.serial`で直接解決する。SPECIAL_CONDITIONは
同様の暗黙解決を暫定適用しているが、ATTACK/RETREATほどの裏付けがなく要検証。
それ以外でHAND/DISCARD/PRIZE等の未トークン化ゾーンを指すものはNO_TARGET(T3で解決可能になる)。

## 5. T1で維持・削除・追加される現行特徴量

design.md §3.2 に詳細な差分表を掲載。要点:

- **維持**: ポケモン数値特徴132次元(盤面トークンへ構造だけ変更して移行)、
  手札内訳・カウント系34次元、相手アーキタイプ21次元(いずれもglobal特徴のまま)。
- **T1時点では削除**(fuudin_v4比): 自分の山札/サイド marginals(68次元)、
  自分トラッシュのカード単位内訳(22次元)。§3.3の理由(card idを持たない
  ゾーン集約トークンは意味を成立させられない)によりT3にまとめて移す。
- **置き換え**(`use_board_card_id=True`時): 自分/相手の盤面ポケモン識別
  (E1 14次元・E2 98次元、one-hot/カウント、語彙制限あり)を、盤面トークンの
  card id embedding(全1267カード対応、連続埋め込み)に置き換える。情報の質が
  変わるため「同一」ではない。`use_board_card_id=False`ならこの112次元相当も失われる。
- **新規**: 相手トラッシュのカード単位トークンはT3まで追加しない。

## 6. 実行したテストと結果

| テストファイル | 件数 | 結果 | 主な確認内容 |
|---|---|---|---|
| `sample_submission/tests/unit/test_board_tokens.py` | 21 | 全pass | 決定性・ベンチ並べ替えへの頑健性・同一card id複数体の一意解決・NO_TARGET・OptionType全網羅 |
| `kaggle_replays/rl/distributed/test_token_shard.py` | 5 | 全pass | 合成データでの書き込み/読み込み往復・decision_slice復元・teacher_logits長さ不整合の検出・空データの拒否・形式不一致の検出 |
| `kaggle_replays/rl/test_collect_tokens.py` | 3 | 全pass | 教師の生スコアが通常の推論経路(`score_options_from_state`)と完全一致(既存MLP無変更の確認)・legacy特徴が既存encoder関数と一致(dual-write整合性)・pointerの長さ整合 |
| 既存テスト(`test_encoder.py`/`test_policy_model.py`/`test_ml_policy_agent.py`) | 46 | 全pass(回帰なし) | `board_tokens.py`が`encoder.py`の private関数を再利用しているが、既存コードへの影響が無いことを確認 |

**新規テスト合計29件、既存回帰チェック46件、合わせて75件、全てpass。**

さらに、実際のcgエンジンで4試合の自己対戦を実行し(`collect_tokens.py`、
production既定の教師重み使用)、323決定点・2639選択肢・2615盤面トークンを
含む実データのshardを書き出し、以下を目視確認した。

- `counts.sum() == legacy_option_features.shape[0]`(2639 == 2639)
- `board_counts.sum() == board_token_numeric_features.shape[0]`(2615 == 2615)
- 盤面トークン数値特徴・教師ロジットにNaN/infなし
- `option_target_token_indices` の値域が `[-1, board_counts内の最大値]` の範囲内
- 盤面トークンが0件の決定点(対局開始直後)でも例外にならず、全選択肢が正しくNO_TARGET

## 7. T1に進む前に残っている問題

1. **蒸留教師checkpointが未確定。** `pool_v1`のv47は正式な`eval_pool_winrate`評価を
   一度も受けておらず、記録がある中ではv24(0.77、400試合)が最高。v24/v32/v40/v47を
   同一条件・試合数≥1,000で再評価してから教師を確定する必要がある(design.md §7)。
2. **DISCARD/ABILITY/CARD/SPECIAL_CONDITIONのpointer解決に、コード上の裏付けが
   弱い分岐が残っている。** 特にSPECIAL_CONDITIONの「常に自分のバトルポケモン」は
   暫定的な仮定で、実際のゲームプレイ(実リプレイ)での検証が済んでいない。
3. **`option_target_token_indices`の妥当性を実リプレイで大規模検証していない。**
   今回の確認は合成データ(21件)+実データ4試合の目視確認のみ。T1で実際に
   pointer gatherを使い始める前に、より大規模な実対戦データでの整合性確認
   (例: 数千決定点規模で「pointerが指すトークンのcard idが、選択肢が本来
   指すべきカードのidと一致するか」を機械的に検査する)を推奨する。
4. **教師logitsのtemperature別比較は未実装。** design.md §7.1の「T=1、2などを
   比較できるようにする」は、raw logitsを保存する設計(本T0で満たしている)までで、
   実際に複数temperatureで蒸留損失を比較する実験コードはまだ無い。
5. **`collect_tokens.py`は単一世代・固定教師の収集のみを想定している。** 複数世代・
   複数相手からの蒸留データ収集(design.md §7.1の4番目)を行う場合、
   `--teacher-weights`/`--deck-o`を変えて複数回実行しshardを集める運用になるが、
   これを束ねるスクリプト(既存`push_kaggle.py`のような一括実行の仕組み)はまだ無い。
6. **GPUメモリ・推論時間の実測はまだ行っていない。** design.md §9.2(投稿版investigation.md
   §8・§9)の見積もりは手計算のままで、T1でTransformer本体を実装した後に実測が必要。
7. **DISCARD/ABILITYがACTIVE/BENCH以外を指すケースの実例が未確認。** 現状の実装は
   「area∈{ACTIVE,BENCH}以外はNO_TARGET」という安全側のフォールバックだが、
   実際にこれらの型がどのAreaTypeを取りうるかを実リプレイで洗い出せていない。
