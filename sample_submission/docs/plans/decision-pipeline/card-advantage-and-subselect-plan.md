# 汎用カードアドバンテージ拡張 と sub-select への将来拡張プラン

模倣学習が真似できない「強いが稀・多段・直感に反する」プレイ(代表例: リッチエネルギーを
ノコッチに付けて+4ドロー→ノココッチの特性にげあしドローで+3ドロー&1枚制限札を山札回収する
反復ドローエンジン)を、**デッキ特化せず**拾えるようにするための設計・実装記録。

## 対象コンボの正体(2026-07-25 調査)

- **リッチエネルギー**(ID 13, ACE SPEC 特殊エネ): 手札からつけたとき**山札を4枚引く**。無1個ぶん。
- **ノココッチ**(ID 66)特性**にげあしドロー**: 1ターン1回、**3枚引いてこのポケモン+付いた全カードを
  山札に戻してシャッフル**。→ リッチを山札回収=1枚制限札を再利用可能にする反復エンジン。

## 実装済み(A+B、commit 6cf4036、既定OFF・config-gated・本番不変・unit 37 pass)

- **A. 汎用カードアドバンテージ項** — `leaf_eval.HandcraftedEvaluator(card_advantage_coeff=…)`。
  末端評価に手札枚数差(self−opp)のロジット項を追加。既定 0.0=無効。ドロー全般を後押しする
  汎用項(特定コンボを名指ししない)。`build_evaluator` が config から受ける。
- **B. 非policy候補** — `pipeline` の `extra_candidate_types`(例 `["ABILITY","ATTACH"]`)。模倣が
  低評価する手を Policy top-k に関係なく探索候補へ含める。`_select_candidate_indices` に切り出し済み。
- 実験config: `configs/abl_5_full_combo.json`(card_advantage_coeff=0.05, extra=["ABILITY","ATTACH"])。

## 判明した構造的な壁(なぜ A+B ではこのコンボに届かないか)

デッキ操作のログ構造調査(2026-07-25)で判明:

1. **MAIN の「ATTACH」は総称オプション**(`OptionType.ATTACH`, cardId 無し)。「どのエネルギーを
   付けるか(=リッチを選ぶか)」は**別の sub-select**(`SelectType != MAIN`)で決まる。
2. **sub-select のカード選択肢は `option.index`(手札インデックス)で識別**され、`option.cardId` は
   使われない。→ 初版 `league/combo_probe.py` の検出(`type==ATTACH and cardId==13`)は**バグ**で、
   `rich_attach=0` は「打っていない証拠」ではなかった。正しくは `option.index -> hand[index].cardId==13`。
3. **`pipeline.search` は `select.type == MAIN and maxCount == 1` でしか動かない**(それ以外は None で
   Policy にフォールバック)。→ energy-choice の sub-select には A も B も一切関与しない。**A+B は
   原理的にこのコンボの決め手へ届かない。**
4. 正しい計測(index→hand で検出, 12試合×2config)では、エージェントはリッチを**時々は選ぶ**
   (17〜24回)。ただし A+B で狙いの列が明確に増えたとは言えず(到達局面が違い比較が濁る/
   トラッシュ・サーチ等の文脈も混ざる)、勝率が動く見込みは弱い。パイプライン全体もまだ本番未超え。

## 将来やるなら(sub-select 拡張の実装プラン)

**目標**: 「どの energy/card を選ぶか」の sub-select でも、card-advantage を意識した先読み選択を
効かせ、リッチ付与(=+4ドロー)のような手を選べるようにする。デッキ非特化を維持。

1. **`pipeline.search` の適用範囲を拡張**: 現状の `MAIN & maxCount==1` ゲートに加え、
   energy/card 選択系の sub-select(`SelectType` が MAIN 以外で、`option` がカード選択)も対象にする。
   各選択肢を `search_step` で1手進めて末端を `leaf_evaluator`(card_advantage 有効)で評価し、最良を選ぶ。
   - 注意: sub-select は「MAINで ATTACH を選んだ後の続き」なので、`search_begin` からの
     determinization を MAIN 起点で張り、sub-select まで `search_step` で辿る必要がある(現状の
     first-move 単発評価では足りない)。多段の rollout 設計を見直す。
2. **精密な検出/計測**: `option.index -> hand[index].cardId==13`(リッチ選択)と、続く
   `ABILITY`(ノココッチ にげあしドロー, 発生元 cardId==66)の**シーケンス**を1試合内で追跡し、
   「コンボ完遂回数/game」を数える正しい probe を用意する(初版 `combo_probe.py` の cardId 検出は要修正)。
3. **採用ゲート**: コンボ発火が増えても、対フィールド勝率(`field_gauntlet.py`/`field_gauntlet_deck.py`)で
   有意に上がって初めて採用。ミラー限定は誤誘導(第4弾の教訓)。

**費用対効果の注意**: 1枚制限コンボのための sub-select 拡張は中規模で、パイプライン全体が本番を
超えていない現状では優先度は高くない。ドロー主体デッキが主戦場になったとき等に再検討する。
