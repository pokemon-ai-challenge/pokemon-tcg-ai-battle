# opponent_modeling

相手のデッキ・戦略をモデル化するモジュール。

## 相手デッキ予測器（rough_predictor）

相手の公開盤面・観測済みカードから相手デッキのアーキタイプを大まかに判定する
ルールベース予測器を実装予定。

- 実装計画・設計方針: [docs/plans/opponent-deck-predictor/plan.md](../../docs/plans/opponent-deck-predictor/plan.md)
- MVP 方針メモ: [docs/plans/opponent-deck-predictor/mvp-strategy.md](../../docs/plans/opponent-deck-predictor/mvp-strategy.md)
- 観測情報記録器の設計: [docs/plans/opponent-deck-predictor/opponent-knowledge-plan.md](../../docs/plans/opponent-deck-predictor/opponent-knowledge-plan.md)
- 関連 Issue: #25（MVP 本体）, #26（特徴量 config）

### ファイル
- `rough_predictor.py` — `predict(state, opponent_knowledge=None)` 本体
- `rough_predictor.json` — デッキ特徴量・role・しきい値の config
- `opponent_knowledge.py` — 実装済み。相手の公開情報（`observed_cards`）を `OpponentKnowledge` で蓄積し、
  `get_prediction_features()` で `rough_predictor.py` 等に渡せる形にする。
  設計の詳細・未確定事項は [opponent-knowledge-plan.md](../../docs/plans/opponent-deck-predictor/opponent-knowledge-plan.md) を参照。

## config リファレンス（`rough_predictor.json`）

JSON はコメントを書けないため、各キーの意味はここに記載する。

### role の考え方

role は「**そのカードがどれだけ確実にそのデッキを指すか（採用率・特異性）**」で分ける。
combo で強くなるか、ではなく採用率で切るのが基本方針。

### `role_weights` — role ごとの共通スコア

カードごとに個別点は持たず、role で決まる共通スコアを使う。1枚（1グループ）見えるごとに
`role_weights[role]` が加点される（観測ゾーンによる重み付けはしない。理由は後述）。

| role | 重み | 意味・使いどころ | 例 |
|------|-----:|------------------|----|
| `anchor` | 10 | デッキの主軸カード。定義上そのデッキ。最も強い根拠。 | メガルカリオex, フーディン |
| `signature` | 8 | 単体でもかなり強い専用寄りカード。anchorほど確定ではないが、1枚で候補を大きく絞れる。 | テツノイサハex, コライドンex |
| `exclusive_core` | 9.5 | そのデッキ以外ではほぼ採用されない専用カード。単体表示はかなり高くするが、確定判定は根拠数で抑える。 | Nのゼクロム, Nの城 |
| `evolution_line` | 4 | 進化ライン（進化元〜中間を統合）。段ごとに確信度を上げたい場合は別エントリ、同じ根拠として扱いたい場合は `names` でセット化する。 | リオル, ケーシィ/ユンゲラー |
| `core` | 3 | そのデッキにほぼ必ず入る固定ギミック・専用サポート。単体でも十分な根拠。 | ソルロック・ルナトーン, ふしぎなアメ |
| `flex` | 1 | 入りうる（構築による）カード。目安は**採用率おおよそ10%以上**。弱めの根拠。 | 採用が分かれるサブアタッカー等 |
| `energy` | 2 | デッキ色の判断材料。単体では確定材料にせず補助的に使う。 | 闘/超エネルギー |
| `shared_anchor` | 7 | 近い複数デッキにまたがる強い主軸候補。単体で絞れるが確定ではない。 | オーガポン みどりのめんex |
| `shared_line` | 5 | 近い複数デッキにまたがる進化支援ライン。単体ではそこそこ、進化元と合わさると強い。 | メガニウムライン |
| `generic` | 0 | 多くのデッキに入る汎用カード。判定の根拠にしない（加点0）。 | ネストボール, 博士の研究 |

> 数値はあくまで初期値。チューニングで上下してよい。順序の意図は
> 「主軸 > ほぼ専用カード > 専用寄りカード > 進化ライン > core > flex ≧ エネルギー > 汎用」。

### `core` / `flex` / 対象外の判断フロー

1. 「10リスト見たら7+くらいに入ってる？」→ Yes なら `core`
2. 「このデッキ以外の無関係なアーキタイプでもよく見る？」→ Yes なら追加しない（対象外。汎用カードは
   リストに載せなくても自動的に0点なので `generic_cards` に列挙する必要もない）
3. 上記どちらでもない（構築次第で入る/入らない、かつ関係ないデッキではあまり見ない、目安10%以上）
   → `flex`

### 同じカードが複数 archetype に登場する場合（Pokemon カードの共有）

スコアリングは archetype ごとに独立計算されるため、同じ重みで複数 archetype の `core`/`flex` に
入っていても**それらの間の相対順位は歪まない**（全員が同じだけ底上げされるだけ）。
現在は raw score そのものではなく `normalized_score = score / confident_score` と、
`prediction_decision` のしきい値で `unknown` を判定する。

判断基準は「**関係ない戦略・タイプのデッキにまで登場するか**」で見る:
- 同じタイプ・同じ戦略系統のデッキ同士で共有される Pokemon（例: 闘タイプ複数デッキで共有される
  支援ポケモン）→ そのまま `core`/`flex` でよい。無関係なデッキとはちゃんと判別できるので有用
- **3個以上の archetype に同じカードを重複登録することになりそう** →
  フォーマット全体のステープル（汎用）のサインとみなし、全部から剥がして `generic_cards` に
  移す（判定根拠にしない理由も記載する）。ACE SPEC でも同様（例: `アンフェアスタンプ` は
  5 archetype 全部に登場したため除外）

### カードエントリの書き方（単体 / グループ）

`archetypes.<key>.cards` の各要素は、単体カードか OR グループのどちらか。

- **単体**: `name`（必須）＋ `card_id`
  ```json
  { "name": "メガルカリオex", "card_id": [678], "role": "anchor", "reason": "..." }
  ```
- **グループ（OR）**: `names`（配列）＋ `card_ids`（名前ごとに1個ずつのフラットリスト）。
  **いずれか1枚でも見えればそのroleで1回だけ加点**（複数見えても二重計上しない）。
  常にセットで採用され、片方だけ入ることが無いカード群に使う。
  ```json
  { "names": ["ソルロック", "ルナトーン"], "card_ids": [676, 675], "role": "core", "reason": "..." }
  ```

> 同じ進化ラインを1つの根拠として扱いたい場合は、`names` でセット化するか、別エントリに
> 同じ `line_key` を付ける。`line_key` が同じカードは、別roleでも最初の1回だけ加点する。
> 例: `ノコッチ` / `ノココッチ`, `アチャモ` / `ワカシャモ` / `バシャーモex`。

### `card_id` の型（常に `list[int]` または `null`）

`card_id` は `data/JP_Card_Data.csv` の「カード ID」列から解決する。**同名カードが複数回再録され、
異なる ID を持つことがある**ため、型は常に「一致した全 ID のリスト」または `null`（未解決）に統一する。
単一マッチでも `678` ではなく `[678]` と書く（predict.py 側の型分岐を減らすため）。

- 解決できた（1件でも複数件でも）→ 一致した ID を**全部**リストに入れる
  例: `"リオル": [333, 677, 974]`（3つの異なる再録・弱体化違いすべてにマッチさせる）
- CSV に該当カードが見つからない → `null`（名前だけでマッチし続ける。将来カードプールに追加されたら埋める）
- `names` グループ側の `card_ids` は現状「名前ごとに1個の ID」のフラットリストのまま
  （グループ内の1名が複数プリントを持つケースは今のところ未対応。発生したらネスト構造を検討する）

`predict.py` 実装時は「`card_id` があればそのリストに observed cardId が含まれるかで判定、
無ければ `name` で照合」という優先順位にする。

### `role_cards` / `role_reason_templates` — 簡潔形式（reasonがroleの繰り返しになるカード用）

`cards`（詳細形式）は1カード5行前後になり、同じroleのカードが多いデッキでは冗長になる。
`reason` が「roleの意味を言い換えているだけ」のカードは、`archetypes.<key>.role_cards` に
role単位のフラットリストとして書く。

```json
"role_cards": {
  "evolution_line": [{ "name": "リオル", "card_id": [333, 677, 974] }],
  "core": [{ "name": "パワープロテイン", "card_id": [1141] }, { "name": "マクノシタ", "card_id": [673] }],
  "flex": [{ "name": "ファイトゴング", "card_id": [1142] }],
  "energy": [{ "name": "ロック【闘】エネルギー", "card_id": [20] }]
}
```

各要素は `{"name":..., "card_id":...}` のみ（`reason` を持たない）。evidence 表示用の reason は
トップレベルの `role_reason_templates`（role → `"{name}: ..."` のテンプレート文字列）から
`{name}` を置換して自動生成する。

**`cards` と `role_cards` の使い分け**:
- `cards`（詳細形式）: anchor、OR グループ（`names`）、`ace_spec`、または
  **roleだけでは伝わらない固有の注記**（例: 「2進化系デッキで共通、このデッキ限定の証拠としてはやや弱め」）
  があるもの
- `role_cards`（簡潔形式）: 上記に当てはまらない、単純な name＋role＋card_id だけのカード

predict.py 実装時は `cards` と `role_cards`（展開後）をマージして1つの評価対象リストとして扱う。

### ゾーン（active/bench/discard 等）はスコアに使わない

当初は `zone_multipliers`（バトル場=1.2, ベンチ=1.0, トラッシュ=0.9, 一時公開=0.8 のような倍率）を
検討したが、**デッキタイプ推定という目的には使わない**方針にした。

理由: このゲームエンジンの `Observation` に載るカードは常に表向きで確定した情報であり、
バトル場で見えても・ベンチで見えても・トラッシュで見えても「そのカードが60枚の山札に
入っていた」という事実の確からしさは変わらない（100%）。ゾーンによる重み付けは
「今どれだけ盤面上の脅威か」を測る話であって、それは `board_evaluation` モジュールの役目であり、
`rough_predictor`（デッキタイプ推定）のスコアリングには混ぜない。

代わりに気をつけるべきは、**同じ実カード（`serial`）を複数ターンにわたって
何度も観測してスコアを重複加算しないこと**。`opponent_knowledge` 側で `serial` ベースの
重複排除を行う想定（1枚のカードは何度観測されても1回だけ加点する）。

なお `predict()` の `evidence` には、どのゾーンで見えたかを**参考情報として**含めてよい
（スコアには影響しないが、他モジュールが後で参照する可能性があるため）。

> 補足: `board_evaluation`（盤面の現在の脅威度評価）や `hidden_information`（山札残り枚数の
> 超幾何分布推定）は、ゾーン情報が本質的に重要になる別の関心事だが、それぞれ別のconfig／
> 別の仕組み（厳密なカード枚数カウントなど）で扱うべきもので、このJSONの範囲外とする。

### ACE SPEC カードの扱い（`ace_spec` フラグ + `ace_spec_bonus`）

ACE SPEC はデッキ内に1枚しか入らないカード（`CardData.aceSpec`）。役割自体は既存の
`role`（`core`/`flex` など）で表現する。多くのデッキに入りうるため、`ace_spec` は
「ACE SPEC として記録する」ための直交フラグで、原則として追加ボーナスにはしない。
新しい role は作らない。

```json
{ "name": "（採用されているエーススペック名）", "card_id": null,
  "role": "core", "ace_spec": true, "reason": "..." }
```

- 加点式: `role_weights[role]`。`ace_spec: true` の場合も top-level の
  `ace_spec_bonus` は初期値 `1.0` なので、追加倍率は掛からない。
- `ace_spec` は単体カード（`name`）・グループ（`names`）どちらにも付けられる
  （型ごとに違うエーススペックを使う場合は、後述の `variants` 側に個別で持たせる方が自然）。
- ACE SPEC は採用率が高くても最大 `flex` に留める（例: `マキシマムベルト`,
  `リッチエネルギー`, `偉大な大樹`, `ネオアッパーエネルギー`, `シークレットボックス`,
  `ミラクルインカム`）。
- **複数 archetype で共有される ACE SPEC は「複数 archetype に登場する場合」のルールに従い
  `generic_cards` に移す**（例: `アンフェアスタンプ` は5 archetype全部に登場したため除外）。

### `variants` — デッキの型（サブアーキタイプ）の扱い

フーディンやドラパルトのように、**主軸・進化ラインは同じだが構築が複数ある**デッキは、
別 archetype に分けず `archetypes.<key>.variants` に型固有カードをまとめる。

```json
"archetypes": {
  "alakazam": {
    "...": "...",
    "variants": {
      "variant_key": {
        "display_name": "型の表示名",
        "cards": [
          { "name": "型Aで使う専用テックカード", "card_id": null,
            "role": "flex", "reason": "..." }
        ]
      }
    }
  }
}
```

- **型 vs 別archetype の判断基準**: `anchor`（と基本 `evolution_line`）が同じなら型、
  `anchor` が違うなら別 archetype にする。
- 型のスコアは archetype 本体のスコア・`prediction_decision` の判定には**使わない**。
  まず archetype（大まかなデッキ）を確定させ、その後の補助情報として型固有カードの
  一致度から型を示す、という2段構成にする（このIssueの目的が「大まかな判定」のため）。
- 現状は全 archetype とも `variants: {}`（空）。型ごとの専用カードが分かった時点で追記する。

### `energy_types` — エネルギー色（絞り込み用）

各 archetype が使うエネルギー色。**positive（このデッキだと確定させる）には弱く、negative（このデッキではないと絞り込む）に効く**信号として扱う。

- 予測器内（scoring）: MVP では **弱い positive のみ**（`role_weights.energy` を色一致したデッキに加点）。
  テックエネルギー・プリズム等の混入があるため、「色が違うから除外」という hard な減点は
  rough 予測器では誤除外（false exclusion）が危険なので**やらない**。positive だけでも
  色が合う候補が相対的に上がり、絞り込みには効く。
- 消費側（戦略レイヤー）での hard 除外: 「格闘エネが見えた→ドラパルト系ではない→ベンチ狙撃を警戒しなくてよい」
  のような確定的な絞り込みは予測器の外でやる。そのため各デッキの色定義（`energy_types`）を
  config に持たせ、predict() の出力にも観測エネルギー色を含める想定。戦略側が自前で候補を消せる。
- 将来のチューニング候補: 色ミスマッチ時の減点（energy mismatch penalty）を予測器に足す余地はある。
  誤除外リスクを許容できると判断したら導入する。

### `generic_cards` — 汎用カード

スコアリングは `archetypes.<key>.cards` / `role_cards` に載ったカードしか参照しないため、
そこに載せない汎用カードは**そもそも一切加点されない**。よって多くの汎用カード（ネストボール、
博士の研究など）はわざわざ `generic_cards` に列挙する必要はなく、単に何にも載せないだけでよい。

`generic_cards` に**明示的に載せる**のは、「複数 archetype に登場する場合」のルールで
判定根拠から**降格させた**カード（例: `アンフェアスタンプ`, `スボミー`）。理由（`reason`）付きで
記録することで、後から見た人が「なぜこのカードがどの archetype にも入っていないか」を追跡できる。
`role_weights.generic: 0` は「どの archetype にもマッチしなかったカード＝0点」という既定の意味。

### `combo_rules`（各 archetype 内, 原則使わない）

特定カードが**揃って初めて意味を持つ**ときの追加加点。`names` のカードが全部見えていれば
`bonus` を加算する。`any_names` を指定した場合は、その中のどれか1枚が見えていればよい。

> 「常にセットで採用される（片方だけは無い）」カード群は combo ではなく **`names` グループ**で表現する。
> combo_rules は「別々に採用され得るが、揃うと別デッキの証拠になる」ような真の組み合わせだけに限定する。
> 2進化 anchor の `ふしぎなアメ + evolution_line` は top-level の `evolution_candy_combo` で共通生成する。

### `evolution_candy_combo`（top-level）

`ふしぎなアメ` と `evolution_line` が両方定義されている archetype に、自動で
`ふしぎなアメ + (進化元 or 中間進化)` の combo を足す。2進化 anchor では、進化元と
ふしぎなアメが同時に見えた状態を anchor に近い強い根拠として扱う。

### しきい値・判定状態

- 各 archetype の `confident_score` — そのデッキを「十分見えた」とみなす基準スコア。
- `prediction_decision.min_evidence_count` — 最有力候補の根拠数がこれ未満なら `insufficient_evidence`。
- `prediction_decision.min_top_normalized_score` — 最有力候補の `normalized_score` がこれ未満なら `insufficient_evidence`。
- `prediction_decision.min_normalized_margin` — 1位と2位の `normalized_score` 差がこれ未満なら `ambiguous`。

### confident_score 調整進捗

`rough_predictor.json` の archetype 順に、会話で方針を決めて `confident_score` と主要roleを見直したものにチェックを入れる。

- [x] `mega_lucario_ex` — メガルカリオex
- [x] `alakazam` — フーディン
- [x] `dragapult_ex` — ドラパルトex
- [x] `kamitsuorochi_ex` — カミツオロチex
- [x] `oliva_ex` — オリーヴァex
- [x] `takeruraiko_ex` — タケルライコex
- [x] `ogerpon_teal_ex` — オーガポン みどりのめんex
- [x] `n_zoroark_ex` — Nのゾロアークex
- [ ] `omatsuri_ondo` — おまつりおんど
- [ ] `gekkouga_ex` — ゲッコウガex
- [ ] `shirona_garchomp_ex` — シロナのガブリアスex
- [ ] `toxtricity` — ストリンダー
- [ ] `yadoking` — ヤドキング
- [ ] `rocket_honchkrow` — ロケット団のドンカラス

### 返り値の見方

- `normalized_score = score / confident_score`。候補比較と `unknown` 判定に使う。
- `match_rate` は `normalized_score` を `0.0..1.0` に丸めた画面表示用の到達率。
- `status` は `confident` / `insufficient_evidence` / `ambiguous` / `no_candidate`。
- `deck_type` が `unknown` でも `top_candidate` で現在の最有力候補を確認できる。
- `confidence` は後方互換のために残している旧フィールドで、viewer では使わない。
