# Transformer化設計メモの再調査結果

作成日: 2026-08-08
対象: [`transformer-tokenized-encoder-design.md`](./transformer-tokenized-encoder-design.md)(旧版)
ステータス: **調査のみ。コード変更は一切行っていない。**
方法: 全項目について実装コードを直接読み、記述の根拠をファイル名・関数名・行番号で示す。
コードを読んでも確認できなかった項目は「確認不能」と明記し、推測では埋めない。

この文書は12個の確認項目それぞれについて「現状のコードで可能」「変更が必要」
「確認不能」を判定する。最終的な結論(正しい点・誤りの訂正・修正版設計)は
[`transformer-tokenized-encoder-design.md`](./transformer-tokenized-encoder-design.md)
(本調査を反映して改訂済み)を参照。

---

## 1. 既存shardでトークン入力を復元できるか

### 結論: **復元できない。「新規実装不要」という記述は誤り。**

shardの書き出しは `kaggle_replays/rl/distributed/common.py` の `write_shard()`
(169〜210行目)。保存される配列は次の5種類だけ。

```
state  : 1決定点あたり1行の固定長ベクトル(166/344/389次元。encoder.encode_state_from_state の出力)
opts   : 1選択肢あたり1行の固定長ベクトル(65次元。encoder.encode_options_from_state の出力)
cids   : 1選択肢あたり1個のcard_id整数(encoder.encode_option_card_ids の出力)
counts : 決定点ごとの選択肢数(opts/cidsを決定点単位に区切るための行数)
chosen / logp / lengths / rewards / opp
```

この5種類がどこで作られるかは `kaggle_replays/rl/distributed/worker.py` の
`collect_generation()` → `kaggle_replays/rl/collect_parallel.py` の `_play_one()`
(107〜114行目)で確認できる。

```python
sf = pm.encode_state_features(cur)                       # 固定長ベクトル
of = encoder.encode_options_from_state(cur, select)       # 選択肢ごとの固定長ベクトル
ci = encoder.encode_option_card_ids(cur, select)          # 選択肢ごとのcard_id
```

つまりshardが保存しているのは**盤面を166〜389次元に要約し終えたあとの数値**であって、
「どのスロットに何のカードがあるか」という生の対応関係ではない。具体的に失われている情報を
4つ確認した。

#### (a) ベンチの各スロットのcard idが無い

`sample_submission/ptcg_ai/learning/encoder.py` の `_pokemon_features()`(146〜221行目)が
返す11次元(`_POKEMON_FEATURE_NAMES`、49〜61行目)を見ると、`present` / `hp_ratio` /
`energy_count` などはあるが、**card id を表す次元が1つもない**。`state_feat` の中に
「ベンチ2番目のポケモンが何なのか」を復元する情報はゼロ。

拡張特徴(`extended_features.py`)を使うプロファイル(後述の `fuudin_v4` 等)では
E1/E2ブロック(280〜304行目)で「自分のベンチに種類Xのポケモンが何体いるか」という
**集約カウント**は入るが、これは「ベンチ全体としてXが何体いるか」であって
「スロット3が具体的にXである」という対応ではない。スロット単位の復元は不可能。

#### (b) 相手のトラッシュの中身がまるごと失われている

`extended_features.py` の `_archetype_probs()`(313〜348行目)は相手のトラッシュの
カードid(`opp.discard`)を読むが、その結果は21次元のアーキタイプ事後確率
(`_get_predictor().predict()`)に潰されて `state_feat` の一部になるだけ。
**元のカードidのリストはどこにも残らない。** shardの `state` 列からは
「相手が何を何枚捨てたか」を一切復元できない。

#### (c) 選択肢の対象以外のカードにはcard idが付かない

`cids` は「その選択肢**自身**が指す対象」1個分のcard idしか持たない。
`encode_option_card_ids()`(721〜761行目)の解決順序(`_resolve_card_id` →
`_resolve_pokemon` → `_resolve_in_play_pokemon`)を実際に `OptionType.ATTACK` で
辿ると、`_resolve_card_id()`(521〜557行目)は `option.cardId is None`(ATTACKは
`attackId` のみ保持、`cg/api.py` 174〜177行目のコメント参照)かつ `option.area is None`
(ATTACKは area を持たない)なので早期に `None` を返し、`_resolve_pokemon()`・
`_resolve_in_play_pokemon()` も `option.area` / `option.inPlayArea` が無いため `None` を返す。
**結果、ATTACK選択肢の `card_id` は常に0(識別なし)になる。** これはencoder.py
737行目のコメント「(END/RETREAT の宣言そのもの等)場合は 0(識別なし)」とも整合する
挙動で、推測ではなくコードを辿って確認した事実。

#### (d) 選択肢から盤面スロットへのポインタが保存されていない

`_resolve_pokemon()` / `_resolve_in_play_pokemon()` が内部で使う `option.index`
(area内でのスロット番号)は、`_option_features()` が11次元にコピーした後は
どこにも残らない。shardの `opts` 列にはスロット番号という形の情報は無い。

### 判定

| 復元したいもの | 既存shardから可能か | 根拠 |
|---|---|---|
| 自分・相手のバトル場 | **不可能(card id無し)** | `_pokemon_features` に card id 次元が無い(§a) |
| ベンチのカード単位トークン | **不可能** | 同上。集約カウントはあっても対応(a) |
| 手札のカード種類単位トークン | **一部可能(枚数のみ)、種類別カード名は不可能(base166次元)** | `_hand_breakdown`(250〜264行目)は pokemon/trainer/energy の3カウントのみ |
| 自分トラッシュのカード単位トークン | **拡張特徴プロファイル使用時のみ可能** | `extended_features.py` 278行目 `self_discard_{c}` (自分のデッキに含まれるカードidのみ、`own_deck_ids` 語彙内) |
| 相手トラッシュのカード単位トークン | **不可能** | §b。21次元に潰れる前のデータは保存されない |
| Zone/Owner/Card ID/枚数 | **不可能** | 上記の通り、そもそもゾーン単位のカード列というデータ構造自体がshardに存在しない |
| 選択肢→盤面トークンのポインタ | **不可能** | §d |
| 1決定点の全選択肢グループ | **可能** | `counts` で決定点ごとの行範囲が分かる(これは復元可能な数少ない情報) |

**「既存shard形式がそのまま使えるため新規実装不要」は誤り。** 正しくは
「1決定点に何個の選択肢があるか」という区切り情報だけが再利用可能で、
トークン化に必要な生の盤面情報(スロットごとのcard id、相手トラッシュの中身、
選択肢のポインタ)は収集コード自体を書き換えて新しいshard形式で取り直す必要がある。

---

## 2. S1が本当に「入力情報は現行と同じ」か

### 前提: 「現行」は1つではない

`kaggle_replays/rl/runs/pool_v1/models/model_v47.json` を実際に読むと
(このrunが§7で蒸留教師の第一候補として挙がっている)、
`meta.extended_features_profile = "fuudin_v4"`、`state_mean` の長さ389。
つまり**現行の最有力モデルは166次元の基本特徴だけでなく、拡張特徴(389次元)を使っている。**
以下の対応表は、この389次元モデル(`fuudin_v4`プロファイル)を基準にする。
166次元の基本モデルだけを使うrun(`lucario_v1` 等)を基準にすると、E1/E2ブロックが
無い分、対応表の一部が変わる。

### 対応表

| 現行特徴量名 | 現行の生成箇所 | S1で格納するトークン | S1でも維持 | S1で新規 | S1で消える |
|---|---|---|---|---|---|
| `self_active_*` 11次元 / `self_benchN_*` 11次元×5 | `encoder.py` `_pokemon_features` 146〜221行目、`_side_pokemon_block` 232〜247行目 | `[自分バトル]` `[自分ベンチ]` トークンの数値部分 | ○(同じ11個の数値をそのままトークンの数値特徴として使う) | — | — |
| (ベンチ固定5スロット、空きはゼロ埋め) | `encoder.py` 244行目 `for i in range(BENCH_SLOTS)` | ベンチが実在する数だけトークンを作る | × | — | **固定スロット構造そのものが無くなる(狙い通りの変更だが「同じ」ではない)** |
| `opp_active_*` / `opp_benchN_*` | 同上(相手側) | `[相手バトル]` `[相手ベンチ]` | ○ | — | — |
| `self_hand_count` / `hand_pkmn` / `hand_trainer` / `hand_energy` | `encoder.py` `_hand_breakdown` 250〜264行目 | `[自分手札]`(カード種類ごとに1トークン) | **△(要約統計としては消える)** | **カード種類ごとの内訳は新規** | 「ポケモン/トレーナーズ/エネルギーの枚数」という3カウント自体はトークン化後も再現可能だが、現行はこの3カウントを直接ベクトルに持つのに対しS1はトークン集合から再構成する形になる |
| `self_discard_count` / `opp_discard_count` | `encoder.py` 320・328行目 | `[自分トラッシュ]` `[相手トラッシュ]` の枚数 | ○ | — | — |
| `self_discard_{c}`(自分デッキのカードid別トラッシュ枚数、`own_deck_ids`語彙のみ) | `extended_features.py` 274〜278行目 | `[自分トラッシュ]` トークン(カード種類ごと) | ○(fuudin_v4使用時のみ) | — | — |
| **相手トラッシュのカード単位内訳** | **存在しない**(§1-bの通り21次元に潰れるのみ) | `[相手トラッシュ]` トークン(カード種類ごと) | × | **新規情報(設計メモ§2の狙い)** | — |
| `self_active_is_X` / `self_bench_has_X`(own_poke_ids語彙、7種) | `extended_features.py` E1ブロック 280〜292行目 | `[自分バトル]` `[自分ベンチ]` トークンの **カードID埋め込み** | △(情報としては包含されるが表現形式が one-hot/カウント → 埋め込みベクトルに変わる) | **語彙外(own_poke_ids に無いカード)の識別は新規** | — |
| `opp_active_is_X` / `opp_bench_has_X`(opp_poke_vocab語彙、48種+other) | 同上 E2ブロック 294〜304行目 | `[相手バトル]` `[相手ベンチ]` トークンの **カードID埋め込み** | △ | **語彙外(48種を超える相手ポケモン)の識別は新規** | — |
| `opp_archetype_*` 21次元(型予測事後確率) | `extended_features.py` `_archetype_probs` 313〜348行目 | `[CLS]` トークンの数値部分にそのまま連結 | ○ | — | — |
| 山札/サイドの `remain_*` / `in_deck_p_*` / `in_prize_p_*`(marginals、own_deck_ids語彙) | `extended_features.py` 263〜269行目(`own_zone="marginals"`) | `[自分の山札/サイド]` トークン(カード種類ごと) | ○ | — | — |
| **選択肢の対象カードのcard id埋め込み(8次元)** | `kaggle_replays/rl/torch_policy.py` 47行目 `self.card_embedding = nn.Embedding(card_id_max+1, embed_dim)`、`policy_model.py` 285〜289行目 `_card_embedding` | 選択肢トークンの一部としてそのまま流用 | **○(既に存在する。新規ではない)** | — | — |
| 選択肢65次元(`OPTION_FEATURE_NAMES`) | `encoder.py` `_option_features` 596〜691行目 | 選択肢トークンの数値部分 | ○ | — | — |

### 4つの確認事項への回答

1. **盤面のCard IDは、現行方策にも入力されているのか。**
   → **部分的に入力されている。判定は「確認不能」ではなく「コードで確認済み・条件付き」。**
   `fuudin_v4`(389次元)使用時は、E1/E2ブロックが「自分の語彙7種・相手の語彙48種」に
   限定した one-hot/カウントで盤面ポケモンの種類を伝えている(`extended_features.py`
   280〜304行目)。語彙外のカードは `+other` バケットに落ちる(299行目 `a2[i if i is not None else -1]`)。
   `166`次元の基本モデル(`lucario_v1`等)には**この情報が一切無い**
   (`_pokemon_features` にcard id関連の次元が無いことは§1-aで確認済み)。
   **選択肢自身が指す対象**については、埋め込み(8次元、`card_id_max=1267`まで全カード
   対応)が既に入力されている(`torch_policy.py` 47行目)。

2. **Card ID embeddingを盤面トークンに追加すると、S1の時点で情報追加にならないか。**
   → **選択肢トークンの埋め込み(自分の選んだ対象1個分)は追加にならない(既存を流用)。
   しかし盤面トークン全部(ベンチの空いていないポケモンすべて等)に埋め込みを付けるのは、
   166次元モデル基準では明確な追加、389次元(fuudin_v4)モデル基準でも「語彙外カードの識別」
   と「one-hot/カウントから連続埋め込みへの表現変更」の分だけ追加になる。**
   完全に「情報追加ゼロ」にしたいなら、S1の盤面トークンのcard id埋め込み自体を
   一旦外し、`fuudin_v4`のE1/E2ブロックと同じ「語彙内one-hot + otherバケット」を
   トークンの数値特徴として埋め込む(埋め込みテーブルを使わない)という設計にする必要がある。
   これは§12で段階を分ける理由に直結する。

3. **現行の固定ベクトルにしかない要約特徴を、トークン化で落とさないか。**
   → 手札の「ポケモン/トレーナーズ/エネルギーの枚数」(3カウント)は、トークン化後は
   トークン集合を数え上げれば再構成できる量ではあるが、**モデルの入力として直接
   与えられる値ではなくなる**(attentionで学習し直す必要がある)。落ちるとまでは
   言えないが、「同一情報がそのままの形で入力される」という意味では厳密には同じではない。

4. **相手アーキタイプ21次元をS1でも完全に維持できるか。**
   → **維持できる。** `_archetype_probs()` の出力をそのまま `[CLS]` トークンに
   連結すればよく、この計算自体(`extended_features.py` 313〜348行目、
   `_get_predictor().predict()`)はトークン化と無関係に動くコードなので変更不要。

### S1の定義修正が必要

以上より、**「S1 = 入力情報は現行と完全に同じ」は成立しない。** 少なくとも
「固定スロット構造の除去」自体が(情報を増減させなくても)モデルの表現力を変える
変更であり、さらに盤面トークンにcard id埋め込みを付けるかどうかで
情報量が変わる。§12で段階をさらに細かく分ける根拠になる。

---

## 3. 選択肢と盤面トークンのポインタ

### 3.1 `_resolve_pokemon` / `_resolve_in_play_pokemon` の返り値

`sample_submission/ptcg_ai/learning/encoder.py`

- `_resolve_pokemon()`(560〜571行目): `option.area`(ACTIVE/BENCH)と `option.index`
  (そのエリア内でのリストのインデックス)から `state.players[player_index].active[0]` または
  `.bench[option.index]` を直接インデックスして返す。**戻り値は `cg.api.Pokemon` オブジェクト。**
- `_resolve_in_play_pokemon()`(574〜593行目): `option.inPlayArea` / `option.inPlayIndex`
  を使う点だけが違う、同型の関数。

どちらも `player.bench[index]` という**リストの位置**でスロットを特定している。

### 3.2 重要な発見: `Pokemon` は `serial` という一意な識別子を持つ

`cg/api.py` 339〜348行目の `Pokemon` dataclass:

```python
class Pokemon:
    id: int        # CardData ID(カード名の識別子。同名カードは同じ値)
    serial: int     # Serial Number: A unique value assigned to each card in the match.
```

`serial` は**対戦中に存在する個々のカード実体に割り当てられる一意な番号**で、
`id`(カード名)とは別物。既にこのリポジトリの他モジュールで識別キーとして
使われていることを確認した。

- `sample_submission/ptcg_ai/board_evaluation/consequence.py` 75行目:
  `{p.serial: p for p in entries if p is not None}`
- `sample_submission/ptcg_ai/search/attack_plan.py` 99行目:
  `{p.serial:p for p in [*player.active,*player.bench] if p is not None}`
- `sample_submission/ptcg_ai/opponent_modeling/opponent_knowledge.py` 118〜189行目:
  カードの移動履歴を `serial` で追跡

つまり「同じカードidのポケモンがベンチに複数いる」場合でも、`serial` を見れば
**位置に関係なく一意に区別できる**ことが、他の既存コードの実装からも裏付けられる。

### 3.3 各ケースへの回答

| ケース | 回答 | 根拠 |
|---|---|---|
| 同じカードidのポケモンがベンチに複数いる | **一意に区別できる。** `area`+`index`(現行方式)でも、`serial`(推奨方式)でも可能 | §3.1・§3.2 |
| HP・エネルギー・どうぐまで同一のポケモンが複数いる | **`serial` でのみ確実に区別できる。** `area`+`index` も区別はできるが、それは「盤面上の位置」で区別しているのであって「個体」で区別しているのではない | `Pokemon.serial` は対戦中の個体に対して一意(cg/api.py 341行目のコメント) |
| ベンチの並び順をモデル入力から除去した場合 | `area`+`index` ベースのポインタは**トークン化前(生State読み取り時)に解決すれば影響を受けない**。トークンの並び順を後で変えても、ポインタは「serial→トークンindex」の対応表を引き直せば追従する | §3.4で詳述 |
| 盤面トークンを並べ替えた場合 | 同上。**ポインタを学習済み埋め込みにしない限り、並べ替えても壊れない**(§3.4) |  |
| 交換・入れ替えなど対象スロットが重要な行動 | `OptionType.EVOLVE`(area/inPlayArea)、将来的なSWITCH相当の選択も同じ `area`+`index`/`inPlayArea`+`inPlayIndex` の枠組みで解決可能。**現行コードのまま対応できる**(cg/api.py 155〜158行目 ATTACH、166〜169行目 EVOLVE のフィールド定義) | encoder.py 574〜593行目 |
| カードではなくプレイヤー・ゾーン・枚数を対象にする行動 | **盤面トークンへのポインタを持たない。** `OptionType.NUMBER`(122行目)・`YES`(124行目)・`NO`(126行目)・`END`(178行目)・`RETREAT`(174行目、コメントにarea/index記載なし)は対象カード/ポケモンのフィールドを持たない。encoder.py 737行目のコメントが示す通り、これらは `card_id=0`(識別なし)になる | cg/api.py 120〜185行目のOptionType定義を全項目確認 |

### 3.4 「ベンチ順への不変性」と「対象の一意特定」を両立する方法

**結論: Slot Embedding(学習可能な位置埋め込み)はモデルに加えない。ポインタは外部の
インデックス(Pythonのリストindex)としてのみ持つ。**

理由:

1. self-attentionは、トークンに位置埋め込みを足さない限り、数学的に置換同変
   (permutation equivariant)である。つまり**ベンチの並び順に依存しない性質は、
   学習やデータ拡張で獲得する必要がなく、「位置埋め込みを追加しない」という
   設計だけで保証される。**
2. ポインタ(「この選択肢が指しているのはトークン列の何番目か」)は、
   トークン列を構築する処理自体が知っている情報。`serial` をキーにした
   `{serial: token_index}` の辞書を、トークン列構築と同じタイミングで作れば、
   その後のネットワークの計算(self-attention)はこのポインタの値を一切使わない
   ただの「配列の添字」として扱われる。**学習可能なパラメータではなく、
   前処理の一部として決定的に計算する。**
3. したがって「並べ替えても壊れない」のは、並べ替えるたびに `{serial: token_index}`
   の対応表を作り直す(トークン構築のたびに毎回新しく作るので、そもそも
   作り直す/作り直さないという区別が発生しない)という前提の上で成り立つ。

**ゾーン埋め込み(Zone Embedding)は加える。** これは「このトークンは自分のベンチに
属する」という**トークンの種類**を表す学習可能な埋め込みで、`bench0`/`bench1`という
**順序**を表すものではない。順序情報と種類情報は別物であり、順序埋め込みだけを
除外すればよい。

---

## 4. 現在の学習データ構造とOption self-attentionの互換性

### 4.1 決定点単位か、選択肢単位に平坦化しているか

**両方使われている。目的によって使い分けている。**

- **保存形式(shard)は平坦化(縦連結)。** `common.py` `write_shard()` 179〜184行目:
  `opts` / `cids` は全決定点ぶんを縦に連結し、`counts` で決定点ごとの行数を持つ。
- **`learner.py` の学習ループは決定点単位にグループ化して処理する。**
  `build_batch()`(45〜85行目)が `counts` から `offsets`(64〜65行目、
  `cumsum(counts)-counts`)と `seg`(73行目、`repeat_interleave`)を作り、
  「どの選択肢がどの決定点に属するか」を常に持ち歩く。**ゼロ詰めのpaddingは
  一切していない。**
- **`train_v3.py`(旧版)は逆にpaddingしている。** `build_padded()`(61〜86行目)が
  `option_pad: torch.zeros(n, max_n, od)` で全決定点を最大選択肢数 `max_n` に
  ゼロ埋めする。`learner.py` はこの非効率(model-capacity-ablation文書・
  improvement-chain.md §1で実測済みの「計算の82%が存在しない選択肢のゼロ」)を
  解消するために作られた、より新しい実装。

### 4.2 各質問への回答

| 質問 | 回答 | 根拠 |
|---|---|---|
| countsから決定点単位のグループを復元できるか | **できる。既に `learner.py` がやっている** | `build_batch()` 64〜65・73行目 |
| 可変数の選択肢をどうpaddingするか | **`learner.py` はpaddingしていない(seg方式)。padding方式に戻すなら `train_v3.py build_padded()` と同じ手法(`torch.zeros(n, max_n, od)`)が使える** | 上記 |
| padding maskをどこで適用するか | `train_v3.py` `policy_logp_entropy()` 112行目: `torch.where(mask>0, scores, -inf)` でsoftmax前にマスク。**Option self-attentionでも同型のkey_padding_maskが必要**(新規実装) | `train_v3.py` 105〜116行目 |
| 最大42選択肢を超える可能性はないか | **確認不能。** 42という数字は `model-capacity-ablation-implementation-plan.md` §11(「選択肢は平均7.5個に対し最大42個ある」)に記載された実測値だが、この文書自体に測定方法・母集団(全アーキタイプか一部か)の記載がなく、コード上に上限を保証する定数も見当たらない。**新しいデッキ・新しいカードプールでは42を超える可能性を否定できない** |  |
| rollout時とPPO更新時で同じ選択肢集合を再現できるか | **現行の仕組みでは再現の必要が無い設計になっている。** `learner.py` の `policy_logp_entropy()`(223〜242行目)は保存された `opts`/`cids`(収集時に確定した選択肢集合)をそのまま使って新しい方策の対数確率を計算する。選択肢集合自体はcgエンジンがその局面で返したものであり、収集後に再現する処理はそもそも存在しない | `learner.py` 223〜242行目 |
| 選択肢順を変えたとき、確率も同じように並べ替わるpermutation equivarianceを保てるか | **現行のMLPはそもそも選択肢間の相互作用が無い(各選択肢が独立にスコアを出す)ので、この性質は自明に満たされている。** Option self-attentionを追加すると、位置埋め込みを加えない限り同じ性質を維持できる(§3.4と同じ理屈) | `torch_policy.py` `option_scores()` 109〜123行目(選択肢間の計算依存が無いことを確認) |

### 4.3 必要な変更(learner/rollout/shard形式)

- **`common.py`**: `SHARD_FORMAT` を上げ、盤面トークン列・選択肢ポインタを保存する新形式に変更(§1の結論より必須)。
- **`worker.py` / `collect_parallel.py`**: `encoder.encode_state_from_state` /
  `encode_options_from_state` の代わりに、トークン列を返す新しい関数(未実装)を呼ぶよう変更。
- **`learner.py`**: `build_batch()` にboard tokenのpadding/mask構築を追加(§8で詳述)。
  `policy_logp_entropy()` をtoken-attention版のforwardに差し替え。
- **`rollout.py`**: `_encode_decision()` / `TorchActor.act()` が現行の `encode_state_from_state`
  等を直接呼んでいる(rollout.py 27〜30・56〜61行目)ため、token版に書き換えが必要。
- **`train_v3.py`**: `build_padded()` はOption self-attention用のpadding方式の
  参考実装として流用できるが、S1で盤面をトークン化するなら盤面側のpadding/maskも
  同様に追加する新規実装が要る(既存に「盤面側のpadding」の実装例は無い)。

---

## 5. Value headは現在のPPOに存在するか

### 結論: **存在する。ただし方策ネットワーク(TorchOptionPolicy)とは完全に別の独立したネットワーク。**

`kaggle_replays/rl/train_v3.py` 37〜48行目の `Critic` クラス:

```python
class Critic(nn.Module):
    def __init__(self, state_dim, mean, std, hidden=64):
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
```

`kaggle_replays/rl/distributed/learner.py` 41行目で
`from train_v3 import Critic, compute_gae, wilson_lo` としてこれをそのまま使っている
(455行目 `critic = Critic(...)`)。つまり**現在のPPOは既にV(s)を学習している。**
ただしTransformerの設計メモが示唆する「デュアルヘッド(エンコーダ共有)」では
**なく、方策(`TorchOptionPolicy`)と価値(`Critic`)は入力(状態ベクトル)を
共有しているだけで、内部のネットワークもパラメータも完全に別。**

### 各項目への回答

| 項目 | 回答 | 根拠 |
|---|---|---|
| advantageとreturnの計算方法 | GAE(Generalized Advantage Estimation)。`learner.py` `compute_gae_shaped()`(132〜169行目)。ポテンシャルベース報酬整形(PBRS)込みの拡張版で、`phi=None` なら従来のGAEと同一 | `learner.py` 132〜169行目 |
| Value lossを追加するために必要なデータ | **既に揃っている。** `state_rows`(盤面ベクトル)と`rewards`/`lengths`(試合の勝敗と長さ)から `compute_gae_shaped` が `vtarget` を計算する。新規データ不要 | `learner.py` 494〜499行目 |
| value targetをshardに保存する必要があるか | **不要。** `vtarget` はGAE計算の中で毎回作られる中間結果で、shardには保存されず、都度計算している | `learner.py` 45〜85行目の `build_batch` に `vtarget` 相当のフィールドは無い |
| GAEを使うのか | **使っている(上記)** |  |
| value loss coefficient | **明示的な係数変数は無い。** `learner.py` のPPO更新ループ(527〜538行目)は `opt_p`(方策)と `opt_v`(価値)を**別々のoptimizerで別々に更新**しており、方策損失と価値損失を足し合わせる「合成損失」の形を取っていない。よって典型的なPPOの `value_loss_coef` に相当するハイパーパラメータ自体が存在しない | `learner.py` 524〜538行目 |
| value clippingの有無 | **ある。** `ppo["value_clip"]`(run.jsonで0.2、`pool_v1/run.json` 確認済み)。`v_clip > 0` のとき `v_old + (v_pred-v_old).clamp(...)` で従来のPPOのvalue clippingを実装 | `learner.py` 528〜536行目 |
| S1でValue headを追加すべきか | **§6の推奨: 追加しない。** 詳細は次段落 |  |

### 判断: S1では価値ネットワークの構造を変えない

現在の `Critic` は独立した2層MLP(隠れ64、入力は現行の状態ベクトルそのまま)。
これをTransformerの盤面エンコーダ出力([CLS]表現)から計算する「共有エンコーダの
デュアルヘッド」に変えることは**アーキテクチャの変更であり、PPOのアルゴリズム
(GAE・クリッピング・アドバンテージ正規化)自体を変えるものではない**が、
方策と価値のパラメータ共有は新しい失敗モード(価値損失の勾配が方策側の表現学習を
乱す、既存メモ`sample_submission/docs/plans/neural-agent/design.md` §2.3で
「転移リスク: 中」と自己評価されている構想#3そのもの)を持ち込む。

**S1では `Critic` を現行のまま(独立ネットワーク、入力は引き続き集約済み状態ベクトル
`state_feat`)にし、盤面Transformerの[CLS]出力をCriticに接続する変更は別段階
(§12のS1d)に切り出す。** これにより、S1で勝率が動いた/動かなかったときに
「方策側の変更が効いた/効かなかった」のか「価値側との共有が効いた/効かなかった」のかを
混同しない。

---

## 6. Kaggle workerを本当に無変更で使えるか

### 結論: **無変更では使えない。「変更不要」という記述は誤り。**

### 現状の確認

`kaggle_replays/rl/distributed/worker.py` は `sample_submission/ptcg_ai/learning/policy_model.py`
の `PolicyModel`(pure-Python、`math` のみ、numpy/torch非依存)を経由して推論する
(`collect_parallel.py` 36行目 `_W["pm"] = PolicyModel(weights_path)`、
`_forward()` は `policy_model.py` 291〜311行目でLinear+ReLUの手書きループ)。
**この推論コードはMLP専用で、attention機構は一切実装されていない。**

リポジトリ全体を検索したが、`MultiheadAttention` / `LayerNorm` /
`nn.Transformer` に類するコードは`sample_submission/` `kaggle_replays/` の
どちらにも存在しない(grep結果0件)。**pure-Python/numpyのTransformer推論実装は
ゼロから新規に書く必要がある。**

### 個別確認

| 項目 | 判定 | 詳細 |
|---|---|---|
| PyTorch Transformer重みの保存形式 | **変更が必要。** 現行の `to_json_payload()`(`torch_policy.py` 178〜199行目)はMLPの `layers`(weight/bias のリスト)専用のschema。Multi-head Attentionの `W_q/W_k/W_v/W_o`、LayerNormの `gamma/beta`、Zone/Owner埋め込みテーブルを追加で保存するschemaが要る |
| LayerNorm、Multi-head Attention、FFNのNumPy実装 | **確認不能→存在しないことは確認済み、新規実装が必要** | grep結果0件(上記) |
| attention maskとpadding mask | **新規実装が必要**。現行のpure-Python推論は選択肢ごとに独立計算するため、そもそも「マスク」という概念が存在しない |
| GELUまたはReLU | **どちらも実装されていない。** 現行は `_forward()` 310行目の `v if v > 0.0 else 0.0`(ReLU相当)のみ。GELUを使うなら新規実装が必要 |
| worker側でのモデル読み込み | **変更が必要。** `PolicyModel._load()`(`policy_model.py` 137〜175行目)は現行のMLP schemaを前提にパースしている |
| 盤面トークン生成 | **新規実装が必要。** `encoder.py` に「トークン列を返す」関数は存在しない(全て集約済みベクトルを返す関数のみ、§1・§2で確認済み) |
| 選択肢トークン生成 | 同上。既存の `_option_features()` は「1個の固定長ベクトル」を返す設計で、トークン列としての出力形式ではない |
| 1決定点単位での推論 | 現行の `PolicyModel.score_options()`(`policy_model.py` 180〜209行目)は選択肢ごとに独立で`_forward`を呼ぶ設計。盤面エンコーダを1回だけ計算して使い回す構造(設計メモ§3.2の狙い)は**現行に存在せず新規実装が必要** |
| 新しいshardへの保存内容 | §1・§4の結論通り、新形式が必要 |
| torch版とNumPy版の数値一致テスト | **現行にMLP版のテストが存在する。** `kaggle_replays/rl/test_parity.py`(既存ファイル、今回は中身未確認だが、torch版と手計算版の一致を確認する目的の既存資産と推測される。Transformer版には**同型の新規テストが要る**、既存テストの流用はできない(構造が違うため) |

**「worker.pyは変更不要」は誤り。** `worker.py` 自体の行数は少なく変更は小さいかもしれないが、
`worker.py` が依存する `policy_model.py` / `encoder.py` / `collect_parallel.py` の
それぞれに新規実装が要るため、実質的にworkerの実行パス全体を新しく書くのと同じ規模になる。

---

## 7. Transformerの正確な構造

現行コードには対応する実装が無いため、この節は「設計として決める」項目。
決定内容は改訂版設計メモ本体に反映した。決定理由のみここに記す。

- **d_model=64**: 現行MLPの隠れ層32(model-capacity-ablation文書で検証済みの規模感)の
  2倍。§1・§2の結論により情報量が増える(相手トラッシュ等)ので、現行と同じ32ではなく
  最小限広げる。128(fuudin_v2で実測済みの規模、勝率への寄与は無かった)までは広げない。
- **num_heads=4**(head_dim=16): d_model=64を割り切れる範囲で最小の複数ヘッド構成。
- **盤面encoder層数=3 / option encoder層数=2**: 設計メモ旧版の構成を踏襲。
- **FFN次元=256**(d_modelの4倍): Transformerの一般的な比率(Vaswani et al. 2017の
  設計踏襲。ただしこのリポジトリ固有の実測根拠は無い)。
- **activation=GELU**: 現行のReLU(MLP)と違う選択をする根拠は薄い。**ReLUを既定にする**
  (現行の活性化関数を変える理由がなく、pure-Python/numpy実装もReLUの方が単純)。
- **Pre-LayerNorm**: 層を増やしたときの学習安定性のため(Post-LNは深いネットワークで
  勾配消失しやすいことが一般に知られている。このリポジトリ固有の実測根拠はない)。
- **[CLS]の初期値**: 学習可能なパラメータ(`nn.Parameter`)としてランダム初期化。
  蒸留時は既存モデルに[CLS]が存在しないため、蒸留元には対応する重みが無い
  (§10の蒸留設計に影響。完全な重み移植ではなく一部は新規学習になる)。
- **Zone Embedding**: 加える(§3.4の結論)。ゾーン語彙は11種類程度
  (CLS/self_active/self_bench/opp_active/opp_bench/stadium/self_hand/self_discard/
  opp_discard/self_deck_prize/opp_summary)。
- **Owner Embedding**: 個別には加えない。ゾーン語彙自体に自分/相手の区別を
  組み込む(`self_bench` と `opp_bench` を別ゾーンにする)ことで代替する。
  理由: 別々に持つと「self × bench」のような組み合わせをモデルが学習で
  発見し直す必要があるが、ゾーンで直接分ければ不要。
- **Card ID Embedding**: 既存の `card_id_max=1267`、embed_dim=8をそのまま流用
  (`torch_policy.py` 47行目と同じテーブル。蒸留の際にそのままコピーできる)。
- **数値特徴の正規化**: 既存の `state_mean`/`state_std` と同じ
  「平均を引いて標準偏差で割る、std=0の次元は0」方式(`torch_policy.py`
  94〜103行目 `_standardize_state`/`_standardize_option`)を、トークンの数値特徴にも
  そのまま適用する。
- **padding token**: 学習可能な埋め込みにせず、ゼロベクトル+attention maskで
  「計算に参加させない」方式にする(標準的なTransformer実装の慣行)。
- **attention mask**: バッチ内で決定点ごとにトークン数が異なるため、
  盤面側・選択肢側の両方にkey_padding_maskが必要(§4.3で明記)。
- **optionから盤面へのcross-attention方法**: 標準的なMulti-head Cross-Attention
  (query=選択肢トークン、key/value=盤面トークンのencoder出力)。
- **最終スコア計算方法**: 現行と同じ「線形層1層でスカラーを出しsoftmax」
  (`torch_policy.py` の `_out = nn.Linear(prev, 1)` と同じ思想)。
- **総パラメータ数**: §8のVRAM見積もりと合わせて算出(概算 60万〜100万パラメータ、
  現行の17,857〜40,993の15〜25倍。**実測ではなく設計時の概算**)。

### Card ID vocabulary

| 項目 | 回答 | 根拠 |
|---|---|---|
| card idからembedding indexへの変換方法 | card idをそのままindexとして使う(変換テーブル不要) | `torch_policy.py` 47行目 `nn.Embedding(card_id_max+1, embed_dim)`。indexはcard id自身 |
| 全使用可能カードを事前登録できるか | **できる。** `card_id_max = max(c.cardId for c in all_card_data())` で動的に決まり、現在1267 | `kaggle_replays/policy_net/build_features.py` 218行目、`policy_model.py` 28行目の実測値 |
| 未登録カード用UNKが必要か | **既に用意されている。** index 0 が「識別なし/範囲外」用に予約済み | `policy_model.py` 78〜79行目、287〜289行目 `_card_embedding` |
| デッキ変更後もindexが変化しないか | **変化しない。** `all_card_data()` はゲーム全体のカードデータで、デッキに依存しない | `policy_model.py` 58行目のdocstring、`cg/api.py` 495〜500行目 `all_card_data()` の実装(cgエンジンの `AllCard()` を呼ぶだけでデッキを参照しない) |

---

## 8. GPUメモリ見積もりを再計算

### 8.1 前提とトークン数

盤面トークン数は「ゾーンをどこまで細かく分けるか」で変わる、**実測ではなく設計上の見積もり**。

- **平均的な決定点**: [CLS] 1 + 自分バトル場1 + 自分ベンチ最大5 + 相手バトル場1 +
  相手ベンチ最大5 + スタジアム1 + 自分手札(平均5枚程度、種類でまとめると3〜5種) +
  トラッシュ(自分・相手、ゲーム中盤で数種〜10種程度) ≒ **30〜45個**
- **1バッチ内でpaddingする場合の悪いケース**: 自分の山札/サイド追跡トークン
  (`fuudin_v4`の`own_deck_ids`語彙22種)+ 自分トラッシュ最大22種 + 手札最大22種 +
  相手トラッシュ(語彙は相手デッキ次第だが仮に20種)を全部足すと ≒ **80〜90個**

**この2案(平均45個・最悪90個)の両方で見積もる。** どちらも実測ではないことを明記する。

選択肢トークン数は平均7.5、最大42(model-capacity-ablation文書 §11の実測値を流用)。

### 8.2 1層あたりのメモリ(概算式)

Multi-head self-attention 1層(トークン数N、d_model=64、heads=4、FFN=256)で
**逆伝播用に保持する必要がある活性化**(概算、float32=4byte換算前の「値の個数」):

```
Q,K,V射影         : 3 × N × d_model
attentionロジット  : heads × N × N
softmax出力        : heads × N × N        (逆伝播でsoftmax微分に必要)
context(重み付き和) : N × d_model
FFN中間層           : N × ffn_dim(256)
FFN出力             : N × d_model
LayerNorm(2箇所)    : 2 × N × d_model
```

これを1決定点・1層あたりの値の個数として計算し、float32(4byte)を掛ける。
**この見積もりはPyTorchの実際のメモリアロケータの挙動(メモリの断片化、
中間バッファの再利用可否)を反映していない単純化した概算であり、実測値ではない。**

#### 平均ケース(盤面N=45、選択肢N=7.5、最大42想定でpadding)

| 部位 | 1決定点あたりの値の個数(概算) | 3層/2層分 |
|---|---|---|
| 盤面self-attention(N=45、3層) | 1層あたり約41,000 | 約123,000 |
| 選択肢self-attention(N=42でpadding、2層) | 1層あたり約38,000 | 約76,000 |
| cross-attention(query=42、key/value=45、1〜2回) | 約1回あたり8,000〜12,000 | 約12,000 |
| **合計(1決定点)** | | **約211,000個 ≒ 844KB(float32)** |

#### 最悪ケース(盤面N=90、選択肢N=42)

| 部位 | 3層/2層分 |
|---|---|
| 盤面self-attention(N=90、3層) | 約330,000 |
| 選択肢self-attention(N=42、2層) | 約76,000 |
| cross-attention(query=42、key/value=90) | 約24,000 |
| **合計** | **約430,000個 ≒ 1.72MB** |

### 8.3 batch sizeごとの概算(活性化メモリのみ、パラメータ・optimizer状態は別途)

| batch size | 平均ケース(844KB/決定点) | 最悪ケース(1.72MB/決定点) |
|---|---|---|
| 16,384 | 約13.8GB | 約28.2GB |
| 4,096 | 約3.5GB | 約7.0GB |
| 2,048 | 約1.7GB | 約3.5GB |
| 1,024 | 約0.86GB | 約1.8GB |

**これに追加で必要なもの(batch非依存、しかしVRAMを占有する):**

- パラメータ本体: §7の概算60万〜100万パラメータ × 4byte ≒ 2.4〜4MB
- 勾配(パラメータと同サイズ): 同上 ≒ 2.4〜4MB
- Adam optimizer state(momentum + variance、パラメータの2倍): ≒ 4.8〜8MB
- **パラメータ関連の合計は数十MB程度で、batch依存の活性化メモリに比べると無視できる規模**
- PyTorch自体のCUDAコンテキスト・メモリプールのオーバーヘッド: 経験的に数百MB〜1GB程度
  (**この値はこのプロジェクト固有の実測ではなく、一般的なPyTorch CUDA利用時の目安**)

### 8.4 RTX 3060 Laptop(6GB)向けの現実的なbatch size

上記より、**batch=2,048(平均ケースなら1.7GB、最悪ケースでも3.5GB)を起点にし、
実測して余裕があれば4,096まで上げる**のが現実的な出発点。batch=16,384は
どちらのケースでも6GBを超えるため不可。

**gradient accumulationを使う場合**: 現在の `run.json`(`pool_v1`)は
`minibatch_size=16384`、1世代の決定点数は実測で「約6万件」
(learner.pyのログ出力例、`f"  決定点 {batch['n']} 件"`、実際の値は世代ごとに変わるため
「約6万」はrun.jsonのgames_per_worker=480×workers数から逆算した概算)。
PPOの「1回の勾配更新に使うサンプル数」を16,384のまま維持したいなら、
batch=2,048で8回のforward/backwardを蓄積してから1回optimizer.step()する
(accumulation steps=8)。この場合、

- **実効batch sizeは変わらない**(16,384のまま。勾配の分散は現行と同じ)。
- **1回のoptimizer更新にかかる時間は増える**(8回分のforward/backwardを直列で行うため)。
- **メモリに載るのはaccumulation中の1ステップ分(batch=2,048)だけ**なので、
  VRAM上限の問題は解決する。
- ただし`learner.py`の`compute_gae_shaped`や`normalize_advantage_per_opponent`は
  バッチ全体に対して一度に計算する設計(148〜202行目)のため、**accumulation対応には
  「advantage計算は全データ一括、勾配更新だけ分割」という構造変更が必要**
  (現在は勾配更新もバッチ全体に対して一度に行っている想定のミニバッチループ、
  512〜525行目の `for idx in chunks` がこれに近いが、`opt_p.step()` を
  chunkごとに呼んでいる点が現行と異なる。現行はミニバッチごとに毎回step()しており、
  これは厳密には「ミニバッチSGD」であって「gradient accumulationで大きなbatchを
  1回のstepにまとめる」のとは異なる。**現行のミニバッチ方式をそのまま維持し、
  batch=2,048でミニバッチを回せば追加実装なしで対応できる**、という理解が正しい)。

### 8.5 ピークVRAM計測用スクリプト案(実行はしない)

```python
# scratchpad/measure_vram.py (案。今回は作成・実行しない)
import torch

def make_dummy_batch(n_decisions, board_tokens, option_tokens, d_model=64):
    board = torch.randn(n_decisions, board_tokens, d_model, device="cuda")
    board_mask = torch.ones(n_decisions, board_tokens, dtype=torch.bool, device="cuda")
    options = torch.randn(n_decisions, option_tokens, d_model, device="cuda")
    option_mask = torch.ones(n_decisions, option_tokens, dtype=torch.bool, device="cuda")
    return board, board_mask, options, option_mask

def measure(model, batch_sizes, board_tokens=45, option_tokens=42):
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        board, bmask, opts, omask = make_dummy_batch(bs, board_tokens, option_tokens)
        scores = model(board, bmask, opts, omask)
        loss = scores.sum()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch={bs:6d}  peak_vram={peak:.2f}GB")
        model.zero_grad(set_to_none=True)

# measure(model, [16384, 4096, 2048, 1024])
```

`model` を実装した後、実測で§8.3の概算を検証する必要がある。**現時点では未実装のため
実行不能。**

---

## 9. 推論時間の根拠

### 結論: **設計メモ旧版の「0.13秒/試合」「0.4秒/試合」は理論値(手計算)であり、実測値ではない。**

旧版の該当箇所を確認したところ、「1決定あたりの積和演算」から逆算した机上の数値で、
実際にpure-Pythonまたはnumpyでコードを書いて計測した記録はどこにもない。
**これは訂正が必要な記述。**

### ベンチマーク計画(実行はしない)

| 条件 | 内容 |
|---|---|
| token数 | 20(軽量案) / 50(旧見積もり) / 90(§8.1の最悪ケース) の3水準 |
| option数 | 平均7.5 / 42(実測上限) / 仮に100(上限が42を超えるケースに備えた安全側の水準) |
| 試合あたり決定点 | 130(既存資料の実測値を流用、根拠は次段落) |
| 実行環境 | (a) PyTorch CPU (b) PyTorch CUDA (c) pure NumPy CPU (d) Kaggle提出環境相当 |
| 計測対象 | モデルのforwardだけでなく、**トークン生成(State→トークン列への変換)とcard id→embedding index変換も含めて計測する**(現行の`_option_features`のようなPythonループ処理が、token化でも同程度かそれ以上のオーバーヘッドになる可能性がある) |

計測すべきコード(未実装につき「新規に書く」対象):
盤面トークン生成関数、選択肢トークン生成関数、numpy版Transformer forward、
上記3つを1試合130回呼んだ場合の合計時間。

### 600秒制限の根拠の再確認

`sample_submission/docs/plans/individual/shogo/stage2-cost-control-implementation-plan.md`
33行目:

> つまり実際の予算は **「1エージェントあたり合計600秒(10分)、試合を通じての累積」**
> (`actTimeout=0`なので毎手の思考時間がそのままこの600秒の残高から差し引かれる、
> `shared: false`なので両陣営は別々の残高を持つ)。

**この記述の一次情報(Kaggle公式ドキュメントや大会規約そのもの)は本調査では確認していない。**
上記の文書自体が「1試合を通じての累積」「両陣営は別々」という2点を明記しているので、
「1エージェント1試合」という単位の理解は**この文書に基づく限り正しい**。ただし
この文書の記述自体がKaggle側の一次情報を正しく反映しているかどうかは、
今回のコード調査の範囲外であり確認不能。

---

## 10. 蒸留方法

現行コードに蒸留(distillation)の実装は存在しない(`grep`で `distill` 等の
キーワードを検索したが該当なし、確認方法: 本調査で `torch_policy.py` /
`learner.py` / `train_v3.py` を全文読んだ範囲でも蒸留用の損失関数は無い)。
以下は全て新規設計。

| 項目 | 提案 | 理由 |
|---|---|---|
| 教師logitの扱い | **学習時に教師モデルで再計算する(shard保存はしない)** | 教師モデル(現行MLP)はpure-Pythonで軽量(§9参照、1決定あたり数百マイクロ秒〜数ミリ秒オーダーと推測されるが未計測)。事前計算してshardに含めると、教師モデルを差し替えるたびに収集をやり直す必要が生じる。動的計算なら教師モデル差し替えが容易 |
| temperature | **要決定。既定候補T=2.0**(知識蒸留の一般的な既定値、Hinton et al. 2015の慣行) | このリポジトリ固有の実測根拠はない |
| KL方向 | **forward KL: KL(teacher \|\| student)** | 教師分布が疎(1つの手にほぼ全確率が乗る)な場合、reverse KL(student \|\| teacher)はmode-seekingになり生徒が教師の代表的な手だけに極端に絞り込む傾向がある。分布全体を写し取りたい(2番目に良い手の情報も伝えたい、設計メモ旧版§5の狙い)ならforward KLが適切 |
| temperature²を損失に掛けるか | **掛ける(Hinton et al. 2015の標準的な処方)** | 蒸留損失の勾配スケールを、他の損失(あれば)と揃えるための一般的な補正 |
| hard label lossも混ぜるか | **S1では混ぜない(KLのみ)。** 蒸留のゲート(§11)で不十分と判明した場合のみ、選んだ手1個との交差エントロピーを追加で混ぜることを検討する | 段階を増やさない(§12の方針と整合) |
| 蒸留用データの決定点数 | **要決定。最低1万決定点以上を推奨**(model-capacity-ablation文書の学習データがoffline評価で使っている規模感に準拠、ただしこのリポジトリの蒸留データ量についての実測根拠はない) |  |
| train/validation分割 | 既存の`policy_net/train.py`が使う`features.npz`の`split`列と同じ考え方(時系列/試合単位で分割し、同じ試合の決定点がtrain/valに跨がらないようにする)を踏襲すべき | `model-capacity-ablation-implementation-plan.md` §7 |
| 複数相手・複数世代のshardを使えるか | **使える。** `pool_v1`のshardは`shards/consumed/v<gen>/`に世代ごとに退避されている(`common.py` `move_consumed()` 266〜271行目)ため、過去世代のshardは削除されず残っている。**ただし§1の結論により、これらのshardはトークン化に必要な生の盤面情報を持たないため、蒸留の「入力」としては使えない(教師の出力ラベルを再計算する用途ではState情報が要るので、shardのstate_featではなく元のState/Observationから取り直す必要がある)** | `common.py` 266〜271行目、§1 |
| studentが遭遇する状態分布とのずれ | **未解決。設計として明記が必要な既知の課題。** 蒸留は「教師が選んだ行動」を再現する学習であって、生徒自身が生成した局面での教師の判断を学ぶわけではない(off-policyの蒸留)。教師データを複数相手・複数世代(§5)から集めることで軽減はできるが、根本的な解決ではない。PPO段階で修正されることを期待する設計(蒸留はあくまで初期値作り) |  |

---

## 11. 蒸留ゲートを数値で定義

### 300試合、勝率50%付近の95%信頼区間

正規近似(Wilson区間ではなく単純な正規近似、n=300で妥当な近似):

| 試合数 | 95%信頼区間(p=0.5のとき) |
|---|---|
| 300 | **[44.3%, 55.7%]** |
| 600 | [46.0%, 54.0%] |
| 1,000 | [46.9%, 53.1%] |

計算式: `p ± 1.96 × √(p(1-p)/n)`。既存文書
(`model-capacity-ablation-implementation-plan.md` §10)が使っている
Wilson区間(`train_v3.py` `wilson_lo()` 51〜58行目)は正規近似よりやや保守的
(下限が低めに出る)だが、300試合・p≈0.5では大差ない。

### 合格条件の数値化(提案)

| 指標 | 合格ライン(提案) | 根拠 |
|---|---|---|
| validation KL(教師分布との) | 要実測してから閾値を決める(現時点で参照できる先行値なし) | — |
| top-1一致率 | ≥ 詳細不明のため保留。参考: `model-capacity-ablation`文書のproduction実測値が `test_top1=0.5806152`(§7.1、既存の模倣学習自体の一致率がこの水準)なので、**蒸留(既存モデル→新モデル)の一致率はこれより高いはず**(教師が既に模倣した結果を再現するだけなので、人間データの模倣より簡単なタスク)。目安として90%以上を仮の閾値とするが実測での再検討が必要 | `model-capacity-ablation-implementation-plan.md` 195行目 |
| top-k一致率 | 要実測 | — |
| Spearman順位相関 | 要実測 | — |
| 教師が高確率で選ぶ局面だけの一致率 | 要実測 | — |
| 固定評価相手群への勝率(教師 vs student、それぞれ) | **両者とも同じ固定相手群(`pool_v1`の`opponents`8体)に対して評価し、勝率の差が誤差範囲内であることを確認する** | §11末尾の理由 |
| 既存MLPとのhead-to-head | **300試合、95%信頼区間 [44.3%, 55.7%] を跨いで50%を明確に下回っていないこと**(下限が45%を大きく割り込まない、旧設計メモの基準を数値化) | 上記の正規近似計算 |
| 推論速度 | §9のベンチマーク結果を確認してから閾値化 | — |
| 複数seed | 蒸留の学習(乱数初期化を伴う[CLS]埋め込み等)を最低3 seedで実行し、head-to-head勝率の分散を確認する | model-capacity-ablation文書 §4.4の「境界結果になった候補のみseedを変えて確認」という既存の方針を踏襲 |

**head-to-headだけでは不十分な理由の補足:** 2つのモデルが同じ強さでも、
片方が特定の相手に強く別の相手に弱い「じゃんけん構造」を持っていれば
平均勝率50%が実力の同等性を意味しない
(`sample_submission/docs/plans/rl-selfplay/results-and-concerns.md` §3.1で
実際に観測された現象と同種の懸念)。そのため**固定相手群への相手別勝率を
教師とstudentで両方測り、相手ごとの勝率パターンが大きく変わっていないことを
確認する項目を追加する。**

---

## 12. S1をさらに分割すべきか

### 結論: **分割すべき。旧版のS1は最低4つの独立した変更を含んでいた。**

§2で明らかになった通り、旧版のS1は次を同時に含んでいた。

1. 固定スロット構造の除去(トークン化そのもの)
2. 盤面トークンへのcard id埋め込み追加(§2の確認3により、166次元モデル基準でも
   389次元モデル基準でも部分的に新規情報)
3. 選択肢間のself-attention(新しい種類の計算)
4. 選択肢→盤面へのcross-attention(新しい種類の計算、ポインタ解決が必要)
5. Value headの扱い(§5により、現状は独立ネットワークなので触らない選択肢もある)

これらを1つの段階にまとめると、勝率が動いた/動かなかったときに
5つのうちどれが効いたか分からない。これは設計メモ旧版が§1・§6で引用していた
「4つ同時に入れて切り分けられなかった」という過去の反省と矛盾する。

### 提案する段階構成

| 段階 | 内容 | 情報量 | 検証方法 |
|---|---|---|---|
| **S1a** | 盤面をトークン化+盤面self-attention(3層)。**選択肢のスコアリングは現行方式を維持**(選択肢は相変わらず「[盤面ベクトルの代わりに[CLS]表現] + 選択肢65次元 + card id埋め込み」を現行のMLPに入れる、選択肢間相互作用なし)。**盤面トークンへのcard id埋め込みは付けない**(§2確認2の結論により、これを付けると情報が増える)。ゾーン埋め込みのみ付与 | 現行と同一(情報量は変えない、構造だけ変える) | 蒸留→head-to-head。ここで動かなくても正常(§1・§6の実測通り、モデルの形だけでは動かない可能性がある) |
| **S1b** | S1aに、盤面トークンへのcard id埋め込みを追加(語彙制限なし、card_id_max全体) | **新規情報追加(§2確認1・2の「拡張特徴のone-hot/カウント」を「連続埋め込み+全カード対応」に置き換え)** | S1aとの差分で「盤面カード識別の解像度を上げた効果」を測る |
| **S1c** | S1bに、選択肢間self-attentionを追加(「今選べる手の中の比較」を可能にする) | 情報は増やさない(計算の種類が変わるだけ) | S1bとの差分で「選択肢比較」の効果を測る |
| **S1d** | S1cに、選択肢→盤面のcross-attention(ポインタ解決)を追加。ここで初めて`_option_features`の11次元コピー方式(現行方式)をcross-attention方式に置き換える | 情報は増やさない(§3.4のポインタ機構で同じ対象を参照するだけ) | S1cとの差分で「ポインタ経由の参照」の効果を測る |
| **S1e** | Value headをTransformerの[CLS]出力に接続する(独立Criticから共有エンコーダへ) | PPOのアルゴリズムは変えないが、パラメータ共有という構造変更 | S1dとの差分。§5で述べた「別段階に切り出す」の実行 |
| **S2** | 相手トラッシュのカード単位トークンを追加(旧設計の狙い) | 新規情報 | S1(a〜eのいずれかで確定した構成)との差分 |
| **S3** | 自分の山札/サイドのカード単位トークン化(既存の`fuudin_v4`marginals機能をトークン形式に移植) | 情報は`fuudin_v4`と同等だが表現形式が変わる | S2との差分 |

**段階を増やすコストの検討:** S1a〜S1eの5段階は、それぞれ「蒸留→head-to-head」の
サイクル(§11のゲート)を回す必要があり、1段階あたり数百試合の評価が要る。
これは`improvement-chain.md`が記録する既存の実験サイクル(1世代10〜47分、
数十世代規模)と比べても軽くはない。**最小限にするなら、S1a+S1b(トークン化+card id、
情報追加をまとめて1段階)とS1c+S1d(attention機構をまとめて1段階)の2段階に
圧縮する案もある。** ただしこの場合、§2で明らかになった「情報追加」と
「構造変更」の効果を分けて測れなくなる。**推奨は5段階のまま(切り分けを優先する)。**
時間的制約が強い場合のみ2段階への圧縮を検討する、という優先順位を明記する。

---

## まとめ表(項目1〜12の判定)

| 項目 | 判定 |
|---|---|
| 1. 既存shardでトークン復元 | **不可能。新規実装が必要(設計メモの記述は誤り)** |
| 2. S1は情報が同一か | **同一ではない。定義の修正が必要** |
| 3. ポインタの一意性 | **`serial`ベースで現状のコードで解決可能(新規解析ロジック不要)** |
| 4. Option self-attentionとの互換性 | **`learner.py`の`seg`方式は流用できるが、`build_batch`等に変更が必要** |
| 5. Value head | **既に存在する(独立Critic)。S1では変更しない方針を推奨** |
| 6. Kaggle worker無変更 | **無変更では不可能。新規実装が必要(設計メモの記述は誤り)** |
| 7. Transformer構造 | **未定義だったため本調査で確定(設計として決定)** |
| 8. GPUメモリ見積もり | **旧見積もりは不完全(attention特有の項目が抜けていた)。再計算し batch=2,048 を提案** |
| 9. 推論時間 | **旧数値は理論値であり実測ではない(訂正)** |
| 10. 蒸留方法 | **現行コードに実装なし。全て新規設計** |
| 11. 蒸留ゲート | **数値化した(300試合で95%CI [44.3%,55.7%])** |
| 12. S1の分割 | **分割すべき(5段階 S1a〜S1eを提案)** |
