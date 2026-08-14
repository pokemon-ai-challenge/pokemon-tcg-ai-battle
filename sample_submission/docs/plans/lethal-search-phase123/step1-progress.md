# Step 1 進捗記録

各サブステップごとに「変更したファイル / 変更理由 / 追加したテスト / テスト結果 /
残っているブロッカー / Phase 1・2 の確定性への影響 / 次へ進める理由」を残す。

- 基準文書: [`step0-capability-report.md`](./step0-capability-report.md)
- 設計: [`design.md`](./design.md)

---

## Step 1-1: 安全性・情報境界・B1 ガード・determinization テスト

状態: **完了**（2026-08-12）。Phase 1/2 の探索本体には未着手。

### 1. 変更したファイル

すべて**新規追加**。既存の製品コードは 1 行も変更していない
（`lethal_simple.py` / `selector.py` / `router` / `configs/` は無変更）。

| ファイル | 役割 |
|---|---|
| `ptcg_ai/search/lethal/__init__.py` | パッケージ説明（探索本体は未実装であることを明記） |
| `ptcg_ai/search/lethal/types.py` | `Proof` / `EstimateKind` / `ChanceClass` / `StopReason` / `ValueKind` / `ProbabilityInterval` / `ValueEstimate` |
| `ptcg_ai/search/lethal/scenario.py` | 山札順列 `Scenario` の隔離型（repr で順序を出さない・allow-list を定義） |
| `ptcg_ai/search/lethal/infokey.py` | 情報集合キー生成（**唯一の生成箇所**） |
| `ptcg_ai/search/lethal/engine.py` | `cg.api` ラッパ（資源解放・`HiddenState`・事象抽出・再生一致検査 R3） |
| `ptcg_ai/search/lethal/chance.py` | C/M/S の能力ベース分類（R3）、B1 ガード（R4）、未対応効果の検出（B4/B8/B9） |
| `ptcg_ai/search/lethal/diagnostics.py` | 診断ログの検閲（順列・禁止キー・長い int 列を拒否） |
| `ptcg_ai/search/lethal/value.py` | critic 呼び出し契約（視点変換・UNAVAILABLE）と価値種別の分離 |
| `tests/unit/lethal/*` | 上記の単体テスト（合成 Observation） |
| `tests/integration/lethal/test_information_boundary.py` | 実エンジンでの T1〜T5・B1・R3・副作用なしの確認 |

### 2. 変更理由

Step 0 で分かったのは「この 1 デッキ・限定サンプルではこう見えた」であって仕様ではない。
そこで**観測を仕様として使わない**構造を、探索本体より先に固定した。

- 順列を型で隔離し、決定側から物理的に見えなくする（R1/R2）
- クラス分類を**ログ推論ではなく能力**で決め、C の主張には再生一致検査を必須にする（R3）
- B1 を「解決済み」にせず、実行時ガード＋回帰テストで継続的に保証する（R4）
- critic は「安全な呼び方」と「視点」だけ先に固定し、校正（B3）は未解決のまま据え置く

### 3. 追加したテスト

**単体 55 件**（`tests/unit/lethal/`）

| ファイル | 観点 |
|---|---|
| `test_infokey.py` | 山札順・serial・手札並びがキーに影響しない／中身が変わればキーが変わる／デッキ公開時の listing の**並び**を使わない／解決不能な選択肢は位置を保持して合流させない |
| `test_reveal_guard.py` | B1 ガードの状態機械（root 公開・途中公開・1 ステップ内のログ順・再公開でのリセット） |
| `test_chance_classifier.py` | **R3 未検査なら C にしない**／SHUFFLE は S へ倒す方向にだけ使う／デッキ公開 root は S／コインは M／B4・B8・B9 を検出して `UNSUPPORTED_EFFECT` |
| `test_diagnostics_and_scenario.py` | `Scenario`/`HiddenState` の repr に順序が出ない／import の allow-list／**順序を読む権限**の allow-list／診断ログの検閲 |
| `test_value_contract.py` | 自分視点／`yourIndex != me` で `1-p`／NaN・範囲外・例外・未ロードは `UNAVAILABLE`／`ValueEstimate` が壊れた確率を構築時に弾く |

**結合 10 件**（`tests/integration/lethal/`、実エンジン）

| テスト | 内容 |
|---|---|
| `test_info_set_runner_is_invariant_to_hidden_deck_order` | **T1**。同一 multiset・別順列で root action が一致。空振り防止として「乱数事象を含む局面が 3 件以上」も同時に検査 |
| `test_scenario_peeking_runner_is_detected` | **T5**。順序を覗く実装を検出できる |
| `test_engine_lookahead_runner_is_detected` | **T5**。1 シナリオ先読み（determinization の典型形）を検出できる |
| `test_info_key_is_stable_across_deck_permutations` | T2 の実データ版 |
| `test_no_draw_before_shuffle_after_deck_reveal` | **B1 回帰**。デッキ公開経路で SHUFFLE を挟まないドローが無いこと |
| `test_deck_visible_root_is_never_classified_controllable` | B1。実デッキ使用局面を列挙可能と判定しない |
| `test_replay_check_separates_controlled_from_engine_random` | **R3**。供給順ドローは再生一致 → C、シャッフル後は再生結果に依らず S |
| `test_sessions_release_resources_and_reject_nesting` | セッションの入れ子禁止と後始末 |
| `test_router_on_search_observation_does_not_mutate_tracker` | 探索用状態が通常方策・相手予測へ副作用を与えない |
| `test_deck_multiset_is_the_only_belief_exposed` | 決定側へ渡るのは multiset だけ |

### 4. テスト結果

```
tests/unit/lethal + tests/integration/lethal : 65 passed（3 回連続で安定）
tests/integration                            : 15 passed, 5 skipped
tests/unit（value_model を除く）             : 278 passed, 6 skipped
```

**検出力の実測**（16 局面 × 4 順列。うち深さ1に乱数事象を持つ局面は 7）

| runner | 発散した局面 |
|---|---|
| 正しい実装の代理（新情報境界で止まる） | **0 / 16**（＝T1 を満たす） |
| 順序を覗く実装 | 15 / 16（検出） |
| 1 シナリオ先読み実装 | 8 / 16（検出） |

**別件の既存失敗**: `tests/unit/test_value_model.py::test_golden_predictions_match` が
30 件中 9 件で許容誤差超過。原因は作業ツリーの未コミット変更
`ptcg_ai/board_evaluation/attack_features.py`（ダメカン配置に弱点・抵抗を適用しない修正、
= ブロッカー B6 への対処）で、`encoder` の特徴量が変わり golden 予測とずれたもの。
**本 Step の変更とは無関係**。value net の golden を再生成するか、変更を戻すかの判断が必要。

### 5. 残っているブロッカー

| ID | 状態 |
|---|---|
| B1 | **条件付き解決を維持**。実行時ガード（R4）を実装し、回帰テストを追加した。「SHUFFLE を挟まないドローが存在しない」ことは引き続き実測で監視する |
| B2 | 未解決（Kaggle の 1 手上限・プロセス永続性）。本 Step では予算に触れていない |
| B3 | 未解決（critic 校正）。呼び出し契約と視点のみ固定。**Phase 3 は既定 OFF のまま** |
| B4 / B8 / B9 | 未解決。実装せず、**検出して `UNSUPPORTED_EFFECT` に落とす**関数と単体テストのみ追加 |
| B5 | 未着手（粗視化）。既定は恒等写像（厳密列挙）の方針を維持 |
| B6 | 未解決。上記 value_model golden の扱いとして表面化 |
| B7 | 未着手（Transformer value head） |

### 6. Phase 1・2 の確定性への影響

- **今の時点で確定性に影響する変更は無い**。新パッケージはどこからも import されておらず、
  `selector` の探索モジュールは `lethal_simple` のまま。
- 影響するのは Step 1-3 以降。そのときに前提となるのは次の 3 点で、いずれも本 Step で固定済み:
  1. C を主張するには R3（再生一致）の合格が必須
  2. デッキ公開経路で SHUFFLE を挟まないドローが出たら `UNSUPPORTED_EFFECT`（R4）
  3. 決定ノードは `InfoKey` で同一視し、順列は決定側から見えない（R1/R2）

### 7. 次へ進める理由 / 進めない理由

- T1 が通り、かつ**壊した実装 2 種が検出される**ことまで確認できたので、
  ハーネスとして機能している（テストが空振りしていない）。
- ガードと分類器は単体で決定表を網羅しており、実エンジンでも期待通りに分類された。
- したがって Step 1-2（toy state の完全列挙 oracle）へ進める。
- ただし **Phase 1/2 本体には未着手のまま**であり、T1 を本体へ適用するのは Step 1-3 以降。
  本 Step で「安全」と言えるのは、**代理実装に対して**ハーネスが機能することまでである。

### 保証できない範囲（本 Step の限界）

- 1 デッキ（フーディン）・ランダムプレイ由来の局面・浅い深さ（≤2〜5）での検証である。
- 「SHUFFLE 以外の乱数源が無い」ことは依然として未証明（U2）。R3 はそれを
  **前提にしない**設計だが、R3 自体も 2 回再生の一致で判定しているため、
  極めて低確率で同じ結果が再現する事象を C と誤判定しうる。
  → **Step 1-2 で、確定証明の判定軸を R3 から「outcome 集合の証明可否」へ移した**（下記）。
- 多コイン・サイド取得・相手選択は**検出できるだけ**で、正しく扱えるわけではない。

---

## Step 1-2: toy state の完全列挙 oracle

状態: **完了**（2026-08-13）。Phase 1/2 本体・マクロ本体・Phase 3 には未着手。
新パッケージは依然としてどこからも import されていない（既存エージェント無改変）。

### 0. R3 の弱点への対処（判定軸の変更）

「2 回再生して一致したら C」は**反証にしか使えない**（偶然一致しうるし、
そもそも outcome 集合の完全性を何も言っていない）。そこで確定証明の判定軸を
**outcome 集合そのものの証明可否**へ移した（`enumeration.certify_for_proof`）。
ユーザ指示の 5 条件をそのまま検査項目にしている。

| 条件 | 実装上の検査 |
|---|---|
| outcome 集合が確定している | 有限・明示・ラベル一意 |
| 全非 0 確率 outcome を覆う | `coverage_certified` かつ質量合計が厳密に 1（`Fraction`） |
| 各 outcome を構築できる | 各 `Outcome.constructible` |
| 確率質量を正しく扱える | `Fraction` 必須（`float` は型で弾く） |
| 未確認 outcome を排除できる | `unprocessed_mass == 0` |

加えて、由来が `sampled` / `partial` の集合と、クラス S の事象は**無条件で不可**。
1 つでも欠ければ `StopReason.OUTCOMES_NOT_ENUMERABLE` → Phase 2 は `UNKNOWN`。
R3 は「C の必要条件（反証用）」へ格下げした。

### 1. 変更したファイル

| ファイル | 種別 | 役割 |
|---|---|---|
| `ptcg_ai/search/lethal/enumeration.py` | 新規（製品） | `Outcome` / `OutcomeSet` / `certify_for_proof` / 多変量超幾何による厳密列挙 |
| `ptcg_ai/search/lethal/backend.py` | 新規（製品） | 探索が必要とする最小操作の Protocol のみ（ロジック無し）。toy と実エンジンを同じ形にして、本体実装時に同じ oracle で検証できるようにする |
| `tests/toy/toy_game.py` | 新規（テスト） | toy state 定義（Step 1-2a） |
| `tests/toy/oracle.py` | 新規（テスト） | 独立 ground-truth oracle（Step 1-2b） |
| `tests/toy/candidates.py` | 新規（テスト） | 比較対象の探索実装 + 意図的に壊した実装 |
| `tests/toy/macros.py` | 新規（テスト） | 遅延生成マクロと、健全でない枝刈りを入れた版 |
| `tests/unit/lethal/test_toy_oracle.py` | 新規（テスト） | 比較ハーネス（Step 1-2c〜1-2f） |
| `tests/integration/lethal/test_information_boundary.py` | 修正 | 局面採取をフレーキーでなくした（後述） |

### 2. toy state の仕様

自分のターンだけの最小ゲーム。カードは 4 種。

| id | 名前 | 効果 |
|---|---|---|
| 1 | STRIKE | 相手に (3 + ブースト) ダメージ |
| 2 | BOOST | このターンの以後の STRIKE に +3 |
| 3 | DRAW2 | 山札の上から 2 枚引く（**唯一の新情報境界**） |
| 4 | DUD | 何もしない |

- 行動は「手札の 1 枚を使う」か「ターン終了」。行動回数に上限（= primitive depth）。
- 勝利は相手 HP <= 0。勝敗は公開情報だけで決まる。
- 山札は隠れた順序を持ち、`DRAW2` のときだけ結果に影響する。
- 実エンジンの性質のうち **「ドローだけが情報公開」「打点は手札の使い方に依存」
  「行動回数が有限」** を写している。

### 3. oracle の独立性

| | oracle | 候補探索 |
|---|---|---|
| 隠れ情報 | 山札の**具体的な順列を全列挙**し、観測キーでグルーピング | 山札の **multiset** のみ |
| outcome | 順列を数えて分割 | 多変量超幾何分布で解析的に列挙 |
| 確率 | `len(group)/len(belief)` | `Fraction` の閉じた式 |
| 使わないもの | `InfoKey` / マクロ / `enumeration.py` / `engine.py` を**一切使わない** | — |

共有しているのは **toy のゲームルール**（`toy_game.py`）だけで、これは
「同じゲームを解いている」ことの前提なので共有が正しい。
確率の独立性は `test_hypergeometric_matches_counting_by_permutation`
（解析式 vs 順列数え上げの完全一致）でも直接確認している。

### 4. 比較したもの・件数・一致率

シナリオ 8 件（下表）。1 実行あたりの oracle 比較は
Phase 1 相当 8 + Phase 2 相当 8 + 方策木 8 + coverage 3 + 完全性 8 +
マクロ vs プリミティブ 8 + マクロ vs oracle 8 + determinization 18 順列 × 3 = 計 100 超。
**一致率 100%（不一致 0）**。toy テストは 66 件、全て green（実行 0.3 秒）。

| シナリオ | Phase 1 相当 | Phase 2 相当 | P_lethal |
|---|---|---|---|
| `deterministic_two_strikes` | PROVEN_WIN（最短 2 手） | PROVEN_WIN | 1 |
| `boost_then_strike` | PROVEN_WIN（root=BOOST） | PROVEN_WIN | 1 |
| `draw_then_always_win` | PROVEN_NO_WIN | **PROVEN_WIN**（root=DRAW2） | 1 |
| `draw_sometimes_win` | PROVEN_NO_WIN | PROVEN_NO_WIN | 5/6 |
| `draw_rare_miss` | PROVEN_NO_WIN | PROVEN_NO_WIN | 9/10 |
| `no_lethal` | PROVEN_NO_WIN | PROVEN_NO_WIN | 0 |
| `not_enough_actions` | PROVEN_NO_WIN | PROVEN_NO_WIN | 0 |
| `depth_limited`（深さ 1） | **UNKNOWN** | **UNKNOWN** | — |

`5/6` と `9/10` は手計算とも一致する（oracle の妥当性の傍証）。

### 5. 3 値の扱い

- `depth_limited` は Phase 1/2 とも `UNKNOWN`（`StopReason.DEPTH_LIMIT`）。
  **「見つからなかった」を `PROVEN_NO_WIN` にしない**ことをテストで固定した。
- outcome を打ち切った集合（`unprocessed_mass > 0`）を渡すと、勝てる局面でも
  `PROVEN_WIN` にならず `OUTCOMES_NOT_ENUMERABLE` になることを確認した。

### 6. 意図的に壊した実装の検出

| 壊した実装 | 検出されたシナリオ |
|---|---|
| 最小確率 outcome を捨てて正規化 | 3 / 8（`draw_rare_miss` で偽 `PROVEN_WIN`） |
| 1 サンプルで確定判定 | 1 / 8（seed 依存。だから証明の根拠にできない） |
| 深さ打ち切りを `PROVEN_NO_WIN` にする | 1 / 8（`depth_limited`） |
| 健全でないマクロ枝刈り（攻撃札があれば BOOST 不要） | 2 / 8（`boost_then_strike` 等で勝ち筋を喪失） |
| 山札の一番上を覗く（determinization） | determinization 境界テストで検出 |

### 7. determinization 境界

- **ドロー前**: 同一公開情報・同一手札 multiset・同一山札 multiset で
  隠れた順序だけを変えた全順列（6 通り / 12 通り）に対し、
  観測キー・proof・root action が完全一致。
- **ドロー後**: `draw_then_always_win` の方策木で、outcome ごとに
  **後続行動が実際に分岐している**ことを確認（`{BOOST,BOOST}` を引いたときだけ
  BOOST を先に使う）。「公開後は結果に応じて再選択してよい」が成立している。

### 8. マクロ

- 安全側マクロは**合法手を 1 つも落とさず順序だけ変える**ことをテストで固定。
- マクロ探索の結果は、プリミティブ全探索とも oracle とも全シナリオで一致
  （= マクロ化で勝ち筋を落としていない）。
- 健全でない枝刈りを入れた版は検出される。

### 9. 副次的に見つかった問題（修正済み）

Step 1-1 で追加した結合テストがフレーキーだった（8 回中 2〜3 回失敗）。
原因は**不変性違反ではなく**、私が入れたカバレッジ条件
「乱数事象を含む局面が 3 件以上」で、エンジンの初期シャッフルが
こちらから seed できないため、試合内容によっては 2 件しか採れなかったこと。
候補プールを分類してから選ぶ方式に変え、8 回連続で安定を確認した。

### 10. 残っているブロッカー

Step 1-1 から変化なし（B1 条件付き解決 / B2・B3 未解決 / B4・B8・B9 は検出のみ /
B5 未着手 / B6 は下記 / B7 未着手）。

**B6 関連**: `tests/unit/test_value_model.py::test_golden_predictions_match` が
30 件中 9 件で失敗している。原因は既存作業ツリーの `attack_features.py` 変更であり、
**リーサル探索の回帰判定からは除外**する（指示どおり golden 再生成も
`attack_features.py` の変更も行っていない）。

### 11. toy で確認できたこと / 実エンジンでまだ保証できないこと

**toy の範囲で確認できたこと**

- 候補探索の 3 値判定が独立 ground-truth と全件一致する
- outcome 列挙の完全性（質量合計が厳密に 1・全ラベル一意・全構築可能）
- 不完全な列挙・サンプリング・クラス S は証明に使えないことが機構として効く
- ドロー前の順序非依存と、ドロー後の再選択が両立している
- マクロ化で勝ち筋を落としていない
- 上記 5 種の欠陥実装をハーネスが検出できる

**実エンジンではまだ保証できないこと**（Step 1-2 時点。Step 1-3 の結果は後述）

- toy には「相手の選択」「多コイン」「サイド取得」「シャッフル後のドロー」が無い。
  実エンジンではこれらが `UNSUPPORTED_EFFECT` / クラス S になり、
  **Phase 2 が `UNKNOWN` になる頻度は toy より確実に高い**。
- toy の outcome は「山札 multiset から k 枚」しか無い。実エンジンの効果
  （サーチ・公開・条件付きドロー）は未モデル化。
- 実エンジンでの outcome 構築（順列を作って `search_begin` し直す）は
  Step 1-3 以降。toy では順列の再構築が確実にできるが、実エンジンでは
  B1 ガードに引っかかる経路や、プレフィックス再生の失敗がありうる。
- したがって本 Step で言えるのは「**toy state の範囲で、探索の判定ロジックと
  outcome 完全性の機構が正しく働く**」ことまでで、Phase 1/2 の安全性が
  実エンジンで証明された訳ではない。

---

## Step 1-3: 安全基盤と Phase 1/2 の接続（adapter 層）

状態: **完了**（2026-08-13）。
**本番 entry point へは接続していない**（`selector._SEARCH_MODULES` 無改変。
テスト `test_module_is_not_registered_in_the_selector` がそれを固定している）。
Phase 3・critic ゲート・大規模枝刈り・本番予算の引き上げには着手していない。

### 1. 接続位置（実際の Agent action entry との関係）

```text
main.agent → ptcg_ai.core.agent → action_selection.selector.select_action   ← 既存（無改変）
                                        ├─ lethal_simple.search()          ← 既存の簡易リーサル探索
                                        └─ router.route()                  ← 通常方策

今回追加したもの（未接続。config で選べる形にはなっている）
  ptcg_ai.search.lethal.entry.search(state, legal_actions, context)
        ├─ precheck（自分の番・残りサイド ≤ 2・未決着）
        ├─ SearchSession（cg の資源管理）
        ├─ CgBackend（LethalBackend 実装）
        ├─ phase1.search() → PROVEN_WIN なら最初の1手
        ├─ phase2.search() → PROVEN_WIN なら最初の1手
        └─ それ以外は必ず None（呼び出し側は通常方策へ）
```

接続に必要な残作業（Step 1-4 以降で判断）:
1. `selector._SEARCH_MODULES` に `"lethal_phase12": entry` の 1 行
2. `configs/rule_lethal_phase12.json` の追加（既存 `rule_lethal.json` は据え置き）
3. `selector` が組み立てる context に `full_deck` を追加（Phase 2 の山札 multiset 用。
   無くても動くが Phase 2 の outcome 列挙は行わず `UNKNOWN` になる）

### 2. 既存の簡易リーサル探索との境界（指示 12）

| | 既存 | 今回追加 |
|---|---|---|
| モジュール | `ptcg_ai/search/lethal_simple.py` | `ptcg_ai/search/lethal/`（パッケージ） |
| config の `module` | `"lethal_simple"` | `"lethal_phase12"` |
| 3 値 | 無し（見つかる / 見つからない） | `PROVEN_WIN` / `PROVEN_NO_WIN` / `UNKNOWN` |
| ランダム性 | `verify_shuffles` で 1 回引き直して棄却 | クラス分類 + outcome 集合の証明可否 |
| 予算キー | `time_limit_ms` | `phase12_ms`（別名。混ぜない） |

既存ファイルは 1 行も変更していない。フラグ名・関数名の再利用もしていない。

### 3. 変更ファイル

**新規（製品）**: `phase1.py` / `phase2.py` / `transposition.py` / `budget.py` /
`action.py` / `cg_backend.py` / `entry.py`
**書き換え（製品・Step 1-2 で置いた Protocol の具体化）**: `backend.py`
**追記（製品・自作分のみ）**: `types.py`（`LethalResult`）、`infokey.py`（`option_descriptor`）、
`engine.py`（`begin_with`）、`scenario.py`（allow-list に `cg_backend` を追加）
**新規（テスト）**: `tests/fixtures/lethal_positions.jsonl`（保存盤面 26 件・127 KB）、
`tests/toy/toy_lethal_backend.py`、`tests/unit/lethal/test_phase_algorithms_toy.py`、
`tests/unit/lethal/test_engine_action.py`、`tests/integration/lethal/test_entry_harness.py`

**既存コードは無改変**: `selector.py` / `router` / `lethal_simple.py` / value model /
`attack_features.py` / 既存 config。

### 4. 抽象アクション → EngineAction

このエンジンでは行動 = `obs.select.option` のインデックス列なので、変換は恒等。
ただし位置だけでは「同じ番号が別のカード」になりうるため、各インデックスの**意味**
（`Option.type` + 解決したカードID + 対象ポケモンID + `attackId` …）を併せ持つ。

- `action.describe()` : 選択列 → `EngineAction`（意味付き）。違法なら None
- `action.validate()` : 実行直前に合法性と意味の一致を再検査。不一致なら None
- `action.reindex()`  : 別観測へ意味から引き直す。**一意に決まらなければ None**

`entry` は探索の root と実行時の観測が同一なので `validate(same_observation=True)` を通す。

### 5. fallback 経路（すべて None → 通常方策）

`disabled` / `no_observation` / `precheck` / `no_hidden_state` / `not_proven` /
`action_validation_failed` / `exception:<型名>`。診断は `entry.last_diagnostics()` で取れる。

### 6. 保存盤面ハーネスの結果（26 局面）

| 指標 | 結果 |
|---|---|
| 探索を開始した局面 | 14 / 26（残り 12 は precheck で通常方策へ） |
| `PROVEN_WIN` | **3**（すべて Phase 1） |
| `PROVEN_NO_WIN` | 1 |
| `UNKNOWN` | 10 |
| 違法な行動を返した回数 | **0** |
| 例外が外へ出た回数 | **0** |
| 1 局面あたりの探索時間 | 平均 70 ms / 最大 100 ms（予算 100 ms） |

`PROVEN_WIN` の 3 局面は、**1 手ずつ適用して再探索するループを回すと、
実際にエンジンが勝ちを宣言する状態へ到達する**ことを確認した
（`test_proven_win_chains_to_an_actual_engine_win`。対戦へは適用していない）。

拒否理由の内訳（診断）:

| 理由 | 件数 | 中身 |
|---|---|---|
| `opponent_to_act_during_our_turn` | 83 | **相手のバトルポケモンを倒した後、相手が次のバトルポケモンを選ぶ**（`TO_ACTIVE`）。B4 の実例 |
| `reveal_without_draw` | 11 | デッキサーチで山札が公開される選択。Step 1-3 では列挙しない |

### 7. Step 1-3 で見つけて直した欠陥（ハーネスの成果）

1. **ターン終了を「自ターン中の相手選択」と誤判定していた** → 攻撃の枝が全部拒否され、
   勝ち筋を 1 つも証明できなかった。`TURN_END` ログとターン番号の変化で区別するよう修正。
2. **サイド取得を無条件で拒否していた** → KO する枝（＝勝ち筋そのもの）が消えていた。
   決着した局面は全チェックを短絡し、サイド取得は「新情報境界」として扱うよう修正
   （Phase 1 は展開せず、Phase 2 は列挙できないので `UNKNOWN`）。
3. **outcome 構築の無制限化で探索が病的に遅くなる**（1 実行 616 秒を観測）。
   `search_begin` + 再生の回数に上限（既定 200）を入れ、超過は `NODE_LIMIT` → `UNKNOWN`。

### 8. determinization（実エンジン）

同じ信念（multiset）で山札の並びだけを変えた 3〜4 通りで、
**返す手だけでなく展開ノード数まで完全に一致**する（14 局面すべて）。

注意: 予算を壁時計で切ると、打ち切り位置が実行ごとにぶれて結果が変わりうる。
これは情報漏洩ではなく**時間切れの揺らぎ**で、その場合は必ず `UNKNOWN` → 通常方策
となるため安全側である。テストはノード基準の予算で比較している。

### 9. toy oracle 回帰

**製品の `phase1.py` / `phase2.py` そのもの**を toy backend 経由で走らせ、
Step 1-2 の oracle と全 8 シナリオで一致（proof・root action・最短手数）。
加えて、予算切れ・深さ・chance 深さ・列挙拒否・構築失敗・候補不完全の各ケースで
`UNKNOWN` に倒れること、transposition の hash 衝突が結果を変えないことを固定した。

### 10. テスト結果

```
tests/unit/lethal + tests/integration/lethal : 202 passed（8 回連続で安定、約 19 秒）
full suite                                   : 1 failed, 433 passed, 11 skipped
```
唯一の失敗は既存作業ツリーの B6 関連（`attack_features.py`）による
`test_value_model.py::test_golden_predictions_match`。**リーサル探索の回帰判定から除外**。

### 11. 新しく分かったブロッカー / 更新

| ID | 状態 |
|---|---|
| **B4（相手選択）** | **最大の制約であることが判明**。KO 後の `TO_ACTIVE` が典型で、拒否 83 件の全部がこれ。Phase 2 で AND ノード化しない限り「KO してから更に動く」線は証明できない |
| B9（サイド取得） | 拒否ではなく「新情報境界」へ格下げ。勝ち筋は救えるが、KO 後に続く線は `UNKNOWN` |
| B8（コイン） | `manual_coin` を使っていないので、コインが出た枝は `UNSUPPORTED_EFFECT` |
| B1 | 条件付き解決を維持。ガードは `cg_backend` の全ステップに入っている |
| B2 | 未解決。予算は現行と同じ 100 ms のまま（Phase 1 と Phase 2 で分け合う） |
| B3 | Phase 3 自体を接続していないので影響なし |
| **新規: END の先を探索しない** | 相手ターン開始時の山札切れによる勝ちは検出できない（既存 `lethal_simple` と同じ範囲） |
| **新規: `full_deck` が context に必要** | 無い場合 Phase 2 の outcome 列挙を行わない（`UNKNOWN`）。接続時に selector 側で 1 行追加が要る |

### 12. 実エンジンでまだ保証できないこと（指示 5）

- シャッフル後 outcome の完全列挙（クラス S。列挙せず `UNKNOWN`）
- 多コイン（`manual_coin` 未使用のため、コイン自体が `UNSUPPORTED_EFFECT`）
- 相手選択（B4。検出して止めるだけ）
- サイド取得の制御（B9。新情報境界として扱うだけ）
- 複雑なカード効果一般（デッキ公開・looking などは列挙対象外）
- 任意 outcome の状態構築（**ドローのみ**検証付きで構築できる。それ以外は不可）
- 深い探索での情報境界（今回の確認は深さ 8・ノード 300〜4000 の範囲）
- 全カード効果に対する完全性（デッキ 1 種類・保存盤面 26 件での確認）

したがって Step 1-3 で言えるのは「**保存盤面 26 件の範囲で、探索結果を合法な
EngineAction へ安全に変換でき、証明できないときは必ず通常方策へ戻る**」ことまでで、
実戦全体での安全性を証明したものではない。

---

## Step 1-4: B4（相手選択）のモデル化と情報境界の確定

状態: **完了**（2026-08-13）。
**本番 entry point へは接続していない**（`_SEARCH_MODULES` 無改変・テストで固定）。
Phase 3・critic ゲート・大規模枝刈り・本番予算の引き上げには着手していない。

### 1. B4 をどうモデル化したか

まず実測した（`probe_k2_to_active.py`、保存盤面 26 件・深さ 4）。

| 観測 | 件数 |
|---|---|
| ターン終了（`MAIN`、相手の番へ） | 727 |
| **ターン継続中の相手選択 `TO_ACTIVE`** | **50**（選択肢 1 個 26 / **2 個 24**） |
| ターン継続中の相手選択 `DISCARD` | 7（選択肢 7 個） |
| 相手に 2 択以上がある局面 | 3 / 26 |

遷移の順序（実測）:

```text
ATTACK → HP_CHANGE → きぜつしたポケモンが DISCARD へ → サイド取得（MOVE_CARD_REVERSE）
       → 相手の TO_ACTIVE（ターン番号は変わらない）→ 相手が選ぶ → TURN_END
```

エンジンは相手の選択肢も `obs.select.option` として提示するので、こちらから
列挙して 1 つずつ適用できる。よって **AND ノードとして実装した**（原設計 §5.3 通り。
設計からの逸脱は無い。詳細は `design.md` §2.5 に追記）。

- `CgState.opponent_node` を追加し、`is_opponent_node()` で探索へ伝える
- ターン終了（`TURN_END` ログ／ターン番号の変化）と明確に区別する
- 相手選択ノードは**終端ではない**（Step 1-3 では終端かつ未対応扱いだった）

### 2. Phase 1 での相手選択

同じ AND 規則を使う。「1 本の勝ち手順」を相手の都合で選ばせない。
相手の選択が新情報を公開する場合は、飛ばすと AND が成立しないので `UNKNOWN`。

### 3. Phase 2 での相手選択

全選択 `PROVEN_WIN` → `PROVEN_WIN` ／ 1 つでも `PROVEN_NO_WIN` → `PROVEN_NO_WIN` ／
それ以外 → `UNKNOWN`。相手の選択肢を全部列挙できないとき（組合せ上限）は
`PROVEN_WIN` を主張しない。**確率平均も、有利な選択の採用もしない。**

実エンジンでの効果（保存盤面 26 件、予算 100 ms）:

| | Step 1-3 | **Step 1-4** |
|---|---|---|
| `PROVEN_WIN` | 3 | **4** |
| `PROVEN_NO_WIN` | 1 | **2**（＋Phase 1 のみ 1） |
| `UNKNOWN` | 10 | **7** |
| 相手選択ノードの扱い | 拒否 83 件 | **探索 136 件** |

### 4. サイド取得の状態遷移

実測の通り、サイド取得は **KO と同じステップ**で起き（`MOVE_CARD_REVERSE`）、
相手の `TO_ACTIVE` はその**後**に来る。したがって:

- サイド取得は「相手の隠れ情報」ではなく、**自分のサイドの中身が手札に入る**イベント。
  枚数の変化は公開情報で、勝利条件（サイド 0）に直結する。
- 決着した局面（`result != -1`）は全チェックを短絡する。**サイド取得で勝つ線を
  取りこぼさない**（Step 1-3 の修正）。
- 決着しない場合は「新情報境界」として扱う（取ったカードの中身は我々の信念に
  無い＝制御できないため、Phase 2 の列挙対象にしない）。状態遷移自体は飛ばさず、
  そのまま相手選択ノードへ続く。

### 5. `full_deck` の情報境界

`list[int]` を探索へ渡す設計をやめ、型で固定した（`deck_view.py`）。

```text
selector / agent
   ├─ deck.csv の list[int]   ← 順序を持つ表現。境界（entry）で捨てる
   └─ KnownDeckComposition    ← 探索へ渡すのはこれだけ（multiset）
```

- `KnownDeckComposition` は `(card_id, 枚数)` の昇順タプルのみを保持。
  `order` / `to_engine_list` のような順序アクセサを**持たない**
- `CgBackend` は `KnownDeckComposition` 以外を受け取ると `TypeError`
- `repr` に中身を出さない。診断ログは長い int 列を拒否（既存ルール）
- Phase 1/2 の経路は critic を一切呼ばない（テストで固定）

検証: デッキリストの並びを変えても、**返す手もノード数も一致**（26 局面）。

### 6. fixture の機能別分類

局面数ではなく「何を検証する局面か」で管理するテストを追加した。
現在のカバレッジ（自動出力）:

```
kind:gate 14 / kind:main 8 / kind:deck_open 4
reveal 13 / opponent_choice 7 / opponent_choice_multi 2 / draw 1
prizes_1 1 / prizes_2 13 / prizes_6 12
```

必須タグ（gate・main・deck_open・opponent_choice・opponent_choice_multi・draw・reveal）
が 1 件以上あることをテストで要求している。
**`draw` が 1 件しかない**のは薄い。Step 1-5 で fixture を足す必要がある。

### 7. 追加したテストと結果

| 追加 | 件数 |
|---|---|
| toy の B4 ケース A/B/C（Phase 1/2 × oracle 一致・平均しない・都合よく選ばない） | 8 |
| `deck_view` の情報境界 | 6 |
| 実エンジンの B4（ノード存在・2 択以上・AND 意味論） | 2 |
| 情報境界（デッキ並び不変・critic 不使用） | 2 |
| fixture 分類 | 1 |
| 資源解放・実行時間 | 2 |

```
tests/unit/lethal + tests/integration/lethal : 223 passed（3 回連続で安定、約 19 秒）
full suite                                   : 1 failed, 453 passed, 12 skipped
```
唯一の失敗は既存作業ツリーの B6 関連（`attack_features.py`）による
`test_value_model.py::test_golden_predictions_match`。**リーサル探索の判定から分離**。

### 8. `PROVEN_WIN` の再生テスト

Step 1-3 で入れた「1 手ずつ適用して再探索し、実際に `result == me` へ到達するか」を
そのまま自動テストとして維持している（`test_proven_win_chains_to_an_actual_engine_win`）。
途中で `UNKNOWN` になった場合は**失敗**として扱う（勝てたから OK にはしない）。
現在 4 局面で成立。

### 9. 資源とランタイム

- `SearchSession` に確保/解放カウンタを入れ、**全 ID の解放**をテストで確認（漏れ 0）
- outcome 構築（`search_begin` + 再生）に上限 200 回。超過は `NODE_LIMIT` → `UNKNOWN`
- 1 局面あたりの `entry.search` 実行時間: 平均 35 ms / p95 103 ms / 最大 104 ms（予算 100 ms）
- lethal テスト全体: 約 19 秒（Step 1-3 で観測した 616 秒の病的ケースは再現しない）

### 10. 本番 entry へ登録できるか

**まだ登録しない。** 判断条件（Step 1-4 指示 13）に対する現状:

| 条件 | 状態 |
|---|---|
| B4 の扱いが明確 | ✅ AND ノードとして実装・テスト済み |
| `full_deck` の順序が decision 側へ漏れない | ✅ 型で固定 |
| fixture が機能別に整理されている | ⚠️ 分類は入れたが **`draw` が 1 件**と薄い |
| `PROVEN_WIN` 再生テストの自動化 | ✅ |
| disabled 時に既存 Agent と完全一致 | ✅ |
| `UNKNOWN` 時に既存 Agent と完全一致 | ✅ |
| unsupported effect で誤 `PROVEN_WIN` が出ない | ✅（toy・実エンジンとも） |
| toy oracle 回帰が全件 green | ✅ |
| 既存 full suite で新規 failure 0 | ✅（B6 由来の 1 件のみ、独立） |
| resource leak が無い | ✅ |

→ **fixture の拡充（特にドロー局面）が残っているため、登録は Step 1-5 以降**。

### 11. 残るブロッカー

| ID | 状態 |
|---|---|
| **B4** | **解決（AND ノード化）**。ただし相手選択が新情報を公開する場合は `UNKNOWN` |
| B8（多コイン） | 未解決。`manual_coin` 未使用のため、コインが出た枝は `UNSUPPORTED_EFFECT` |
| B9（サイド取得の制御） | 未解決。新情報境界として扱う（勝ち筋は救う） |
| B1 | 条件付き解決を維持 |
| B2（Kaggle の 1 手上限） | 未解決。予算は 100 ms のまま |
| B3（critic 校正） | Phase 3 未接続なので影響なし |
| B6 | 既存作業ツリーの件。分離して扱う |
| 新規: END の先（山札切れ勝ち） | 未対応のまま |
| 新規: fixture の `draw` 局面が 1 件 | Step 1-5 で追加 |

### 12. 実エンジンでまだ保証できない範囲

- シャッフル後 outcome の完全列挙（クラス S → `UNKNOWN`）
- 多コイン（コイン自体が未対応）
- 相手選択が**新情報を伴う**場合（`DISCARD` 等で中身が公開される場合）
- サイド取得で得たカードの制御
- 複雑なカード効果一般（デッキ公開・looking は列挙対象外）
- 任意 outcome の状態構築（**ドローのみ**検証付きで可能）
- 深い探索（今回の確認は深さ 8・ノード 300〜4000）
- 全カード効果に対する完全性（デッキ 1 種類・保存盤面 26 件）

**安全に実装できたこと**: B4 の AND 化、デッキ情報の型による境界、資源解放、
`PROVEN_WIN` の再生検証、3 値の厳密な区別。
**探索能力として不足していること**: ドロー系 fixture の薄さ、コイン未対応、
シャッフル後の列挙、深い探索。
**まだ推測に留まっていること**: 他デッキ・他アーキタイプでの `TO_ACTIVE` 以外の
相手選択の種類と頻度（今回のデッキで観測できたのは `TO_ACTIVE` と `DISCARD` のみ）。

---

## Step 1-5: 機能別 fixture と実エンジンでの確定性検証

状態: **完了**（2026-08-14）。**本番 entry へは未登録**（`_SEARCH_MODULES` 無改変）。
Phase 3・critic ゲート・大規模枝刈り・本番時間予算の確定には着手していない。

### 1〜6. fixture の機能別カバレッジ

fixture は 26 → **36 件**（189 KB）。各行に `tags`（required_capability）と
`golden`（expected result）を持たせた。

| capability | 件数 | 備考 |
|---|---|---|
| main（サイド 3 枚以上） | 22 | precheck で探索しない側 |
| reveal（新情報境界） | 20 | |
| gate（サイド ≤ 2） | 14 | 探索が起動する側 |
| **draw** | **9** | Step 1-4 の 1 件から増強（最優先項目） |
| draw_multi（2 枚以上） | 9 | 実測では 3〜4 枚ドロー |
| opponent_choice | 8 | |
| prize_take | 8 | |
| deck_open | 6 | |
| immediate_win | 4 | |
| opponent_choice_multi（2 択以上） | 3 | |

新情報境界ケース: `reveal` 20 + `draw` 9（重複あり）。
必要数はテスト `test_capability_coverage` で機械的に要求している。

### 7. 各 fixture の expected result（golden）

`golden` に `started` / `phase1_proof` / `phase2_proof` / `stop_reasons` /
`first_action` / `fallback_reason` / `nodes` / `refusals` を保存し、回帰テストで比較する。
現在の分布:

| golden (phase1, phase2) | 件数 |
|---|---|
| (None, None) — precheck で起動せず | 22 |
| (UNKNOWN, UNKNOWN) | 6 |
| (PROVEN_WIN, —) | 4 |
| (PROVEN_NO_WIN, PROVEN_NO_WIN) | 2 |
| (PROVEN_NO_WIN, UNKNOWN) | 1 |
| (UNKNOWN, PROVEN_NO_WIN) | 1 |

`PROVEN_WIN` だけでなく `UNKNOWN` の停止理由も golden 化してあるので、
後の最適化で安全側の判定が崩れたら検出できる。

### 実エンジンでの outcome 完全列挙（最重要の実測）

**既定設定では、実デッキのドローは 1 件も列挙できない。**

| 実測 | 値 |
|---|---|
| ドローの chance ノード | 34 件（全て拒否） |
| 内訳 | `OUTCOMES_NOT_ENUMERABLE` 29 / `UNSUPPORTED_EFFECT` 5 |
| 実際のドロー枚数 | **3〜4 枚** |
| outcome クラス数 | **36 〜 516**（中央値 66）。既定の上限は 24 |
| そのときの山札枚数 | 8〜18 枚（終盤） |

これは欠陥ではなく組合せの現実で、設計 §2.4 が予告していた粗視化（B5）が
未実装であることの帰結。**列挙できないときに確定証明へ進まない**動作は正しく働いている。

一方、上限を 80 に上げると**実エンジンでも C クラスのパイプライン全体が通る**ことを
確認した（`test_real_engine_outcome_enumeration_pipeline_when_it_fits`、2 ノード）:

1. 全 outcome を列挙 → 2. 質量合計が厳密に 1 → 3. 証明可否の判定に合格 →
4. **全 outcome を構築** → 5. 各構築状態を再生し、**実際に意図通り引けたことを検証**

つまり能力自体は実エンジンで確認できた。既定を保守的なままにしているだけである。

### B1 / B4 / B8 / B9 の現在状態

| ID | 状態 | 根拠 |
|---|---|---|
| **B1** | 条件付き解決を維持。指示された 2 系列を明示的に回帰テスト化 | `DECK_REVEAL→MOVE_CARD→SHUFFLE→DRAW` は通す／`DECK_REVEAL→MOVE_CARD→DRAW` は `DRAW_BEFORE_SHUFFLE_AFTER_REVEAL` で必ず拒否（人工系列で発火確認） |
| **B4** | AND ノードとして解決済み | 相手選択ノード 25 件・その後の遷移 68 件を実エンジンで探索 |
| B4' | **新情報を伴う相手選択は未観測** | 検出は実装済み（`after_opponent_choice_revealed` を計測）。今回の corpus では **0 件**。安全側では Phase 1 → `UNKNOWN`、Phase 2 → 列挙拒否 |
| B8 | 未対応のまま | コインが出た枝は `UNSUPPORTED_EFFECT` |
| **B9** | ケース分離した | A: 取得で即勝利 → terminal WIN／B: 取得したが勝利せず → 相手選択ノードへ正しく遷移（実測 2 件）／C: 取得カードの利用 → 新情報境界／D: 取得のランダム性 → 列挙対象にしない |

### 8. PROVEN_WIN 再生

`PROVEN_WIN` を返した **4 件すべて**で、1 手ずつ再探索 → 適用を繰り返して
実際に `result == me` へ到達（成功率 4/4）。途中で `UNKNOWN` になったら失敗扱い。
capability 内訳: gate 4 / immediate_win 2。
まだ収集できていない再生ケース: gust+attack / evolve+attack / draw→attack /
KO→opponent choice→continue / special win。

### 9〜11. 安全指標

| 指標 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0**（再生テストで全件が実際に勝ちへ到達） |
| illegal action | **0**（36 件） |
| unsupported effect による誤証明 | **0** |
| fallback safety | **100%**（証明できなければ必ず `None`） |
| resource leak | **0**（確保 = 解放をテストで確認） |

### 12. 機能別レイテンシ（予算 100 ms、36 件）

| capability | n | p50 | p95 | max | nodes_max |
|---|---|---|---|---|---|
| ALL | 36 | 0.0 | 102.6 | **p99 102.6 / max 102.6** | 430 |
| gate | 14 | 101.4 | 102.6 | 102.6 | 430 |
| draw | 9 | 0.0 | 101.8 | 101.8 | 365 |
| opponent_choice | 8 | 38.0 | 101.4 | 101.4 | 359 |
| prize_take | 8 | 39.5 | 101.4 | 101.4 | 377 |
| deck_open | 6 | 0.0 | 0.0 | 0.0 | 0 |
| immediate_win | 4 | 1.5 | 7.5 | 7.5 | 28 |
| main | 22 | 0.0 | 0.0 | 0.0 | 0 |

p50 が 0 ms の群が多いのは、precheck（サイド ≤ 2）で探索を始めないため。
**本番用の時間予算はここでは確定しない**（B2 未解決）。

### 13. 本番接続条件（更新版）に対する現状

| 条件 | 状態 |
|---|---|
| toy oracle regression green | ✅ |
| disabled 時に既存 Agent と完全一致 | ✅ |
| `UNKNOWN` 時に既存 Agent と完全一致 | ✅ |
| false `PROVEN_WIN` = 0 | ✅ |
| illegal action = 0 | ✅ |
| `PROVEN_WIN` 再生テスト green | ✅（4/4） |
| resource leak = 0 | ✅ |
| B1 guard regression green | ✅（2 系列を明示化） |
| B4 AND regression green | ✅ |
| draw fixture が複数存在 | ✅（9 件） |
| deck reveal fixture が複数存在 | ✅（6 件） |
| 新情報境界テスト green | ✅ |
| 機能別 fixture が十分に存在 | ✅（機械的な要求として固定） |
| p95/p99 runtime を把握 | ✅（p95 102.6 / p99 102.6 ms） |
| full suite で**リーサル探索由来の新規 failure 0** | ✅ |

`tests/unit/test_value_model.py::test_golden_predictions_match` は既存作業ツリーの
`attack_features.py` 変更（B6）由来で、**リーサル探索由来ではない**。
「既存 failure を除外したから green」ではなく、**リーサル探索由来の新規 failure が 0 件**である。

### テスト結果

```
tests/unit/lethal + tests/integration/lethal : 237 passed（2 回連続、約 27 秒）
full suite                                   : 1 failed(既存 B6), 467 passed, 12 skipped
```

### 14〜15. 検証できたこと / 安全側に倒していること / 未検証

**検証できた**
- 機能別 fixture のカバレッジと golden 回帰
- 実エンジンでの C クラス完全列挙パイプライン（列挙→構築→再生→一致→質量 1）
- B1 の 2 系列（人工系列での発火を含む）
- B4 の AND 意味論（実エンジン・toy 両方）
- B9 のケース A/B（即勝利・取得後継続）
- `PROVEN_WIN` 4 件の再生成功、違法手 0、資源漏れ 0

**安全側に UNKNOWN へ落としている**
- 実デッキのドロー（outcome クラス 36〜516 > 上限 24）
- シャッフル後のドロー（S クラス）
- コインを含む枝（B8）
- デッキ公開局面からのドロー（B1）
- 取得したサイドの中身を使う線（B9 ケース C/D）
- 新情報を伴う相手選択（検出のみ、今回 0 件）

**まだ未検証**
- 粗視化（B5）を入れた場合の Phase 2 の実効性 ── **現状 Phase 2 は実エンジンで
  ほぼ発火しない**。今 `PROVEN_WIN` を出しているのは全て Phase 1
- gust/evolve/draw→attack などの再生ケース
- 他デッキ・他アーキタイプ
- Kaggle 実行環境での 1 手上限（B2）

---

## Step 1-6: B5（outcome 粗視化）の必要性測定 — **実装せず、不可能性を確認**

状態: **測定完了・実装中止**（2026-08-14）。本番 entry へは未登録のまま。

### 1. OutcomeEquivalenceKey の仕様（設計したもの）

「今後の探索に対して完全に等価」であるためにキーへ含める必要があるもの
（実リポジトリ・実エンジンで探索が実際に参照する項目から導出）:

| 項目 | 参照している場所 |
|---|---|
| 公開状態（両者の場・HP・付与物・トラッシュ・サイド枚数・各種フラグ） | `infokey._player_parts` / `is_win` / `is_terminal` |
| 自分の手札 multiset | `infokey._player_parts`（打点・合法手の両方に効く） |
| **残りの既知山札 multiset** | `infokey.info_key(deck_multiset=...)`、`draw_outcomes()` の入力 |
| 今の選択（種類・個数制約・選択肢の意味） | `infokey._select_parts` |
| ターン内フェーズ・相手選択待ちか | `CgState.opponent_node` / `is_terminal` |
| 継続効果・解決中のカード | `select.effect` / `contextCard` |

このキーが一致する 2 outcome は、後続の合法行動集合・状態遷移・後続の確率分布・
勝利可能性・公開情報のすべてが一致する。

### 2. 測定結果（実装前に必ず測る、という指示に従った）

実エンジンの draw chance ノード 5 個・**outcome 合計 827 個**について、
強さの違う 3 種類のキーで class 数を数えた。

| ノード | outcomes | K0（完全同値） | K1（合法手集合＋山札＋手札枚数） | K2（山札＋手札枚数のみ・**意図的に不健全**） |
|---|---|---|---|---|
| A | 167 | 構築 0（全 outcome の構築に失敗 → 安全側で `UNKNOWN`） | — | — |
| B | 72 | **72** | 72 | 72 |
| C | 36 | **36** | 36 | 36 |
| D | 36 | **36** | 36 | 36 |
| E | **516** | **516** | 516 | 516 |

**圧縮率はすべて 1.00×**。意図的に弱くした K2 ですら 1 つも統合できない。

### 3. なぜ圧縮できないのか（構造的な理由）

ドローした multiset と、**残りの山札 multiset は 1 対 1 対応**する
（引いた分だけ山札から減るため）。したがって、山札構成をキーに含める限り、
異なる outcome が同じクラスに入ることは**原理的にあり得ない**。

そして山札構成をキーから外すことは、後続のドロー確率を変えるので健全ではない
（唯一の例外は「以後その山札から二度と引かない」と証明できる場合だが、
その場合でも手札 multiset が outcome ごとに異なるので、やはり統合できない）。

→ **B5（outcome の同値粗視化）は、ドローに対しては原理的に効果が無い。**
ユーザ指示 11 の判断基準（「516 → 480 程度なら導入価値が低い」）に照らすと、
実測は 516 → 516 なので、**実装しないのが正しい判断**である。

したがって Step 1-6 では、toy への粗視化実装・bad grouping テスト・実エンジン適用
（指示 4〜10）へは進まなかった。同値関係が空である以上、実装しても
「1 つも統合されないコード」が増えるだけで、確定証明の完全性を損なうリスクだけが残る。

### 4. 代わりに有望な方向（未実装・次の検討候補）

outcome を統合するのではなく、**証明を共有する**方向なら健全にできる可能性がある。

```text
「引いたカードを一切使わない勝ち手順 L が存在する」
   かつ「L の打点が手札の枚数にしか依存しない（枚数は全 outcome で同じ）」
   ⇒ L は全 outcome で合法かつ勝ち
   ⇒ 516 outcome を 1 回の証明で覆える
```

これは近似ではなく、L が参照するカードを検証すれば確定的に言える。
ただし「L が引いたカードを使っていない」ことの検証機構が必要で、
現在の実装には無い。**Step 1-7 の候補**として記録する（実装は未着手）。

### 5. 併せて観測された問題

167 outcome のノードで、**全 outcome の構築に失敗**した（`apply_outcome` が
すべて拒否）。原因は未特定だが、動作としては安全側（`UNKNOWN`）に倒れている。
false `PROVEN_WIN` は発生していない。原因調査は Step 1-7 の課題。

### 6. 本 Step での不変条件

コードの変更は**していない**（測定のみ）。したがって Step 1-5 で確認した
false `PROVEN_WIN` = 0 / illegal action = 0 / resource leak = 0 /
fallback safety = 100% はそのまま維持されている。

---

## Step 1-7a: 167 outcome 構築失敗の原因特定と修正

状態: **完了**（2026-08-14）。本番 entry へは未登録のまま。
Step 1-7b 以降（ドロー非依存リーサルの証明共有）は未着手。

### A. failure reason の分類

| failure_reason | count | first_failed_operation | representative_case |
|---|---|---|---|
| `SHUFFLE_ENCOUNTERED` | **167 / 167** | `CgBackend._materialize`（プレフィックス再生中） | `((1225,2),(1264,1))` mass=1/272 |

分類は 1 種類だけだった（複数原因ではない）。
入力状態: root（path 長 0・draws 空・shuffled False）、山札 18 枚 / 10 種、
`deckCount` 18、対象は 3 枚ドローのアクション。

### 原因

`_materialize()` の判定が**1 ステップ内のログ順を見ていなかった**。

```python
# 修正前（誤り）
if events.shuffled and len(observed) < len(state.draws):
    raise _Refusal(SHUFFLE_ENCOUNTERED)
```

実エンジンのドロー効果は「**3 枚引いてから**山札をシャッフルする」形が普通で
（Step 0 の実測ログ: `DRAW,DRAW,DRAW,MOVE_CARD,...,SHUFFLE`）、
この判定だと**そのステップで引いた分まで無効**にしてしまう。
ドロー自体は供給順どおりに起きているので、拒否は誤りだった。

### 修正

ログの並びで判定する（`events.sequence` の `draw` と `shuffle` の前後関係）:

- 「引いてからシャッフル」→ そのドローは有効（供給順どおり）
- 「シャッフルしてから引く」→ 無効
- 一度シャッフルが起きたら、**以降のステップ**のドローは無効（`order_lost`）

### 修正の効果（実測）

| | 修正前 | 修正後 |
|---|---|---|
| 167 outcome ノードの構築 | 0 / 167 | **167 / 167** |
| 列挙できた chance ノード（上限 600 時） | 4 | **7** |
| 構築できた outcome 合計 | 660 / 827 | **956 / 956** |
| 完全構築できたノード | 4 / 5 | **7 / 7** |

### 安全性への影響

このバグは**過剰拒否**（安全側）で、false `PROVEN_WIN` は発生しえなかった。
修正後も既定設定（`max_outcomes=24`）では実デッキのドローは列挙されないので、
**保存盤面の golden は 1 件も変化していない**（237 テスト green）。

### 構築失敗時のフォールバック回帰テスト（新規 3 件）

toy で「1 つだけ失敗 / 一部失敗 / 全部失敗」を作り、すべて `UNKNOWN` に倒れ、
`PROVEN_NO_WIN` にも `PROVEN_WIN` にもならないことを固定した。

### B5 の記録（指示 15）

> B5（outcome 粗視化）は実測の結果、現状のドロー表現では安全な outcome equivalence
> による圧縮効果が確認できなかった（827 outcome → 圧縮率 1.00x）ため、初期実装では見送る。
> 将来、別の outcome 表現（順序を持たない公開型など）で有効性が確認された場合に再検討する。

---

## Step 1-7 中間: exact enumeration のスケーリング実測

状態: **ベンチマーク完了**（2026-08-14）。証明共有（1-7b）は**未実装**。
本番 entry 未登録・Phase 3 未実装のまま。

### 1. DRAW / SHUFFLE 境界の固定（ケース A〜D）

Step 1-7a の誤拒否を再発させないため、判定を純粋関数
`cg_backend.draw_is_supply_ordered(sequence, order_lost)` へ切り出し、単体テスト化した。

| ケース | 系列 | 判定 |
|---|---|---|
| A | `DRAW×3 → SHUFFLE` | 供給順どおり（有効） |
| B | `SHUFFLE → DRAW` | 無効 |
| C | 前ステップでシャッフル済み（`order_lost`） | 以降のドローは無効 |
| D | 同一ステップ混在（`DRAW,SHUFFLE,DRAW` / `DRAW,SHUFFLE,SHUFFLE` / `SHUFFLE,DRAW,SHUFFLE`） | 最後のドローが最初のシャッフルより後なら無効 |

`tests/unit/lethal/test_draw_shuffle_order.py`（6 件）で green。

### 2. max_outcomes スイープ（対象 7 局面 / wall 5000 ms / max_nodes 20000）

| max_outcomes | chance nodes | 完全列挙できた局面 | Phase2 WIN | UNKNOWN | p50 | p95 | max | search_begin(最大) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 7 | 0 | **1** | 6 | 309 | 5180 | 5180 | 8101 |
| 32 | 7 | 0 | 1 | 6 | 287 | 5128 | 5128 | 8193 |
| 48 | 7 | 2 | 1 | 6 | 290 | 5011 | 5011 | 10175 |
| 64 | 7 | 3 | 1 | 6 | 462 | 5012 | 5012 | 10440 |
| 96 | 7 | 5 | 1 | 6 | 2257 | 5011 | 5011 | 10110 |
| 128 | 7 | 5 | 1 | 6 | 2264 | 5059 | 5059 | 10534 |
| 192 | 7 | 6 | 1 | 6 | 5009 | 5019 | 5019 | 10528 |
| 256 | 7 | 6 | 1 | 6 | 5011 | 5014 | 5014 | 10580 |
| 512 | 7 | 6 | 1 | 6 | 5010 | 5180 | 5180 | 10268 |

n=7 のため p99 = max。**上限を 24→512 にしても Phase 2 の PROVEN_WIN は 1 件のまま**で、
変わったのは「失敗理由が `OUTCOMES_NOT_ENUMERABLE` から `TIME_LIMIT` へ移った」ことと、
p50 が 309 ms → 5010 ms（約 16 倍）になったことだけ。

### 3. outcome サイズ別（全 max_outcomes で同一）

| bucket | 局面数 | 結果 |
|---|---|---|
| 36–64 | 3 | **PROVEN_WIN 1** / UNKNOWN 2 |
| 65–128 | 2 | UNKNOWN 2 |
| 129–256 | 1 | UNKNOWN 1 |
| 257–516 | 1 | UNKNOWN 1 |

唯一の Phase 2 勝ちは最小バケットにのみ存在し、上限を上げても他バケットは動かない。

### 4. wall-clock スイープ（max_outcomes=512）

| budget(ms) | 25 | 50 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|
| Phase2 WIN | 0 | 0 | 0 | 0 | 0 | **1** |

実用的な予算（≤200 ms）では**上限をいくら上げても Phase 2 は 1 件も証明できない**。

### 5. 60 秒予算での深掘り（ボトルネックの分離）

| 局面 | proof | nodes | search_begin | 時間 | 停止理由 |
|---|---|---:|---:|---:|---|
| fx012 | UNKNOWN | 92,363 | **98,607** | 61.9 s | CHANCE_DEPTH / DEPTH / TIME |
| fx028 | UNKNOWN | 4,082 | 4,123 | 2.1 s | CHANCE_DEPTH / DEPTH（**時間ではない**） |
| fx015 | UNKNOWN | 12 | 15 | 1.0 s | UNSUPPORTED_EFFECT（`reveal_without_draw` / `opponent_choice_node`） |
| fx026 | UNKNOWN | 8 | 12 | 0.0 s | UNSUPPORTED_EFFECT（`reveal_without_draw`） |

ボトルネックは 3 層に分かれる:

1. **未対応効果**（デッキ公開系の reveal・相手選択）… 予算と無関係。2/7 局面
2. **構造的な上限**（`max_chance_depth=1` / `max_depth=8`）… fx028 は時間内に探索し切ったうえで UNKNOWN
3. **materialization コスト**… outcome 1 件ごとに `search_begin` + プレフィックス再生。
   fx012 は 60 秒で **98,607 回**の `search_begin`。これが最大の律速

### 6. Phase 1 単独（36 局面 / 100 ms / max_outcomes 非依存）

| 指標 | 値 |
|---|---|
| PROVEN_WIN | **6** |
| PROVEN_NO_WIN | 15 |
| UNKNOWN | 15 |
| p50 / p95 / p99 / max | 63.7 / 103.3 / 103.7 / 103.7 ms |

（entry 経由では precheck でサイド ≤2 に絞るため 4 件。ゲートを外すと 6 件）

### 7. 安全性（全ベンチを通して不変）

false PROVEN_WIN 0 / illegal action 0 / resource leak 0 / fallback 100%。
既定 `max_outcomes=24` は変更していないので **golden も 1 件も変化していない**。
`processed_mass == 1` を満たさない限り PROVEN_WIN を出さない原則は維持。

### 8. 判断: **C（別の探索最適化が必要）**

数値根拠:

- 上限 24→512 で Phase 2 WIN は 1→1（**増えない**）。増えたのは runtime のみ（p50 16 倍）
- 実用予算 ≤200 ms では 0 件。500 ms でようやく 1 件
- 60 秒使っても最大ノードは未解決（98,607 `search_begin`）
- 2/7 局面は未対応効果で時間と無関係に UNKNOWN
- fx028 は**時間内に探索し切って**なお UNKNOWN（`max_chance_depth=1` が理由）

したがって「exact enumeration で十分（A）」ではない。
一方で純粋な B（証明共有だけで解決）でもない ── 未対応効果と深さ上限は証明共有では解けない。
**主因は探索経済（outcome ごとの `search_begin` 再生）** であり、これは C に該当する。

ただし重要な重なりがある: 証明共有（1-7b）が狙うのは「全 outcome を個別展開しない」ことで、
これは fx012 型（98k begins）の**主コストそのもの**を消す。よって次の優先順位は:

1. 未対応効果の縮小（`reveal_without_draw` = デッキ公開系の outcome 化）
2. `max_chance_depth` / `max_depth` を上げた場合の効果測定（fx028 型の切り分け）
3. その上で、なお outcome 展開が律速なら証明共有（1-7b）

---

## Step 1-8: 探索経済の調査（session 再利用 / state 共有）

状態: **調査のみ完了**（2026-08-14）。実装変更なし。本番 entry 未登録。

### Engine reuse

- `search_begin` は**すでに 1 セッション内で再利用している**（`SearchSession.begin_with`）。
  セッションは 1 回、`search_end` も 1 回。これ以上減らす余地は無い。
- 親ノードから複数の子を作ること自体は可能で、**親は破壊されない**（7/7 で `parent_ok=True`）。
  ただしそれは「同じ供給順の中での分岐」に限る。
- **outcome ごとの `search_begin` は API 上不可避**。ドロー結果は
  `search_begin(your_deck=...)` の供給順で決まるため、別 outcome を作るには
  別の順列で開き直すしかない（状態複製 API も outcome 差し替え API も存在しない）。
  コインだけは `manual_coin` で選択肢化できるが、ドローに相当する仕組みは無い。

### Cost breakdown（実測）

| 項目 | 実測 |
|---|---|
| outcome 1 件の materialization（`search_begin` + プレフィックス再生） | **0.61〜0.68 ms** |
| 516 outcome ノードの depth-1 展開だけで | **639 ms** |
| 167 outcome ノード | 224 ms |
| 60 秒探索時の `search_begin` 回数（fx012） | 98,607 → 約 43 秒相当 |

### State sharing（決定的な測定）

| id | physical outcomes | unique InfoKey | 重複率 |
|---|---:|---:|---:|
| fx012 | 167 | 167 | **0.00%** |
| fx028 | 72 | 72 | 0.00% |
| fx030 | 36 | 36 | 0.00% |
| fx031 | 36 | 36 | 0.00% |
| fx032 | **516** | **516** | **0.00%** |
| fx033 | 66 | 66 | 0.00% |
| fx035 | 63 | 63 | 0.00% |

**重複率 0%**。引いたカードは手札 multiset に入るので、子状態は必ず別 InfoKey になる。
したがって chance ノード直下での transposition / memoization による共有効果は**無い**。
（Step 1-6 の outcome equivalence が 1.00x だったのと同じ理由が、探索状態側でも成立する）

### 未測定（今回は実施していない）

- `max_chance_depth` 1/2/3/4 および `max_depth` 8/12/16 のスイープ
- session/state sharing 導入後の 100〜1000 ms 再ベンチ（共有効果が 0% のため実装しておらず、
  比較対象が作れない）

### 判断: **E（複数要因が支配的）**

内訳と根拠:

1. **session 再利用は既に最大限**（1 session / 1 search_end）。削減余地なし
2. **state 共有は効果ゼロ**（重複率 0.00%、7/7 ノード）→ 判断 B は数値で否定された
3. **outcome ごとの materialization コストは API 上不可避**で 0.65 ms/件。
   516 outcome なら depth-1 だけで 639 ms → 100〜200 ms 予算では原理的に無理。
   これを回避できる唯一の手段が**証明共有（C）**
4. **未対応効果（D）**が別集合を塞いでいる（7 局面中 2 件が `reveal_without_draw` 等で即 UNKNOWN）
5. **深さ上限**（`max_chance_depth=1`）が fx028 のように時間と無関係な UNKNOWN を生む

よって単一要因ではない。優先順位としては
**D（未対応効果）→ 深さ上限の測定 → C（証明共有）** の順に効果が見込める。
B（state sharing）は実装しない。

### Phase 1 baseline（golden として固定）

Phase 2 最適化・Phase 3 いずれも無効の状態で:
PROVEN_WIN 6/36、PROVEN_NO_WIN 15、UNKNOWN 15、p50 63.7 / p95 103.3 / p99 103.7 / max 103.7 ms。
今後の最適化でこの値が悪化していないことを毎回確認する。

### 安全条件

今回は実装を変更していないため、Step 1-7a 時点の安全性がそのまま維持されている:
false PROVEN_WIN 0 / illegal action 0 / resource leak 0 / fallback 100% / golden 不変。

---

## Step 1-9a: `reveal_without_draw` の実態調査

状態: **調査のみ完了**（2026-08-14）。1-9b 以降（実装・回帰・depth sweep）は**未着手**。
実装変更なし・本番 entry 未登録。

### 分類結果（保存盤面 36 件、深さ 1）

| SelectContext | シャッフル | 手番 | 件数 |
|---|---|---|---:|
| `TO_HAND`（デッキから手札へ = サーチして取る） | 発生せず | 自分 | **29** |
| `TO_BENCH`（デッキからベンチへ = ポフィン系） | 発生せず | 自分 | **2** |

代表例: `fx003` / `TO_HAND` / listing 11 枚 / looking 0 / shuffle なし / 自分の手番 /
効果カード `Dawn`。

ユーザ指定の capability 分類に当てはめると:

| Class | 内容 | 件数 |
|---|---|---:|
| R1 単純公開のみ | — | 0 |
| **R2 公開 + 自分の選択** | `TO_HAND` / `TO_BENCH` | **31（全件）** |
| R3 公開 + 相手選択 | — | **0** |
| R4 公開 + 同時 SHUFFLE | — | **0**（この step では未発生） |
| R5 公開 + 新 random event | — | 0 |

### 重要な含意（実装方針の根拠）

- 探索中に現れるデッキ listing は、**我々が供給した山札**（fabricated belief）である。
  したがってその公開は「相手や実物からの新情報」ではなく、
  **既知の belief multiset を見ているだけ**。
- よって R2 は **chance node ではなく自分の decision node** として扱える見込みが高い。
  outcome 列挙は不要で、選択肢は belief multiset から決まる。
- 現在の実装は `reveal_without_draw` を一律 `revealed=True` にしており、
  その結果 `enumerate_outcomes` が「ドローではない」として拒否 → `UNKNOWN` になっていた。
  つまり **31 件は「未対応」ではなく「過剰に保守的」だった可能性が高い**。

### 実装時に守るべき条件（未実装・次 Step の要件）

1. listing の**順序は使わない**（`InfoKey` は multiset のみ ── 既に実装済み）
2. 取得後の SHUFFLE 以降のドローは従来どおりクラス S（B1 / `SHUFFLE` 境界を変えない）
3. `select.deck != None` の局面から探索を**開始**する場合は従来どおり B1 扱い
   （実デッキが使われるため）。今回の 31 件は**探索の途中**で現れるケースで別物
4. サーチ結果を固定マクロ化しない。「公開 → 合法な取得対象を生成 → 選択 → 自由判断」
   として探索する
5. R3（公開 + 相手選択）が現れたら従来どおり AND ノード / `UNSUPPORTED_EFFECT`

### 未実施

- 1-9b（最小実装）、1-9c（回帰）、1-9d（depth sweep）、1-9e（Phase 2 再評価）
- `deck_delta` が計測できなかったケースの原因（`_deck_multiset` が None を返す条件）の特定

---

## Step 1-9b/c/d: R2 実装・回帰・depth sweep

状態: **完了**（2026-08-14）。本番 entry 未登録。証明共有は未実装のまま。

### 0. 実装前に見つかった重大な問題（belief の過少計上）

R2 の検証中に、`_deck_multiset()`（山札 belief）が**系統的に過少計上**していることが判明した。

| 実測（root） | 供給した山札 | 旧 belief |
|---|---:|---:|
| 例1 | 40 | **35** |
| 例2 | 44 | **38** |
| 例3 | 11 | **10** |
| 例4 | — | **None**（計算不能） |

belief が小さいと `draw_outcomes()` が**存在する outcome を列挙しない** ──
つまり「全 outcome で勝つ」の証明が穴だらけになる。**偽証明の経路**であり、
R2 より優先して修正した。

修正: 導出方法を変更し、**検証を付けた**。

1. デッキが提示されている（`select.deck`）ならそれが確定情報
2. そうでなければ「我々が供給した山札 − 経路で引いたカード」
3. どちらも `deckCount` と枚数が一致しなければ **None**（安全側で諦める）

検証（新規テスト）: 全 fixture で belief == 供給デッキ、ドロー後は引いた分だけ正確に減る。
なお、この誤りで実際に偽 `PROVEN_WIN` が出ていた形跡は無い
（当時 Phase 2 はほぼ発火していなかったため）が、**発火し始めたら直撃する**位置にあった。

### 1. R2 の実装（`KnownListingSelection`）

探索途中の `TO_HAND` / `TO_BENCH` の listing を **decision node** として扱う。
判定は仮定ではなく**検証**する:

1. 自分の手番 2. context が `TO_HAND` / `TO_BENCH` 3. listing に伏せカードが無い
4. **listing が「供給した山札の残り」の部分集合**（＝我々が知らないカードが 1 枚も無い）

4 を満たさない listing は「本当の新情報」として従来どおり公開扱い。
実測では listing = deckCount = 供給 multiset が**完全一致**（extra/missing とも空）。

同名カードの選択肢は意味（カードID）で重複排除する。健全性の根拠:
残る山札の「順序」だけが違うが、サーチ後のドローは B1 ガード（R4）か
シャッフル（クラス S）で必ず列挙対象外になるため、我々の情報モデルでは区別できない。

**B1 とは別物**として実装した。`select.deck != None` の局面から探索を**開始**する
ケース（実デッキが使われる）は従来どおり `DECK_REVEALED_AT_ROOT` で列挙拒否。
B1 ガードの条件・回帰テストは 1 行も変えていない。

### 2. R2 の効果（A/B、決定的なノード予算 30s / 5000 nodes / depth 10）

| 設定 | WIN | NO_WIN | UNKNOWN | p50 | 変化した局面 |
|---|---:|---:|---:|---:|---|
| Phase 1 R2=OFF | 6 | **24** | 6 | **42 ms** | — |
| Phase 1 R2=ON | 6 | 18 | 12 | 122 ms | 6 件が `PROVEN_NO_WIN`→`UNKNOWN` |
| Phase 2 R2=OFF | 6 | 7 | 23 | 17 ms | — |
| Phase 2 R2=ON | 6 | 7 | 23 | 42 ms | **なし（1 件も変わらない）** |

- **新しい `PROVEN_WIN` は 1 件も増えなかった**（Phase 1・Phase 2 とも）
- Phase 1 は `PROVEN_NO_WIN` を 6 件失い、p50 が 3 倍になった
- Phase 2 は**完全に不変**

理由は depth sweep で判明した（下記）: R2 は最初の障害を取り除くが、
**すぐ後ろに次の障害（ドロー outcome の列挙不能）が控えている**。
実際、R2=ON にすると `UNSUPPORTED_EFFECT` が 14→0 になる代わりに
`OUTCOMES_NOT_ENUMERABLE` が同じ局面を塞ぐ。

→ **R2 は既定 OFF**（`known_listing_as_decision=False`）。機能とテストは残し、
深さ・時間予算を増やせる段階で再評価する。

### 3. depth sweep（Step 1-9d）

**20 s / 20,000 nodes / R2=ON**（36 局面）

| depth | chance_depth | P1 WIN | P2 WIN | P2 NO_WIN | P2 UNKNOWN | DEPTH_LIMIT | CHANCE_DEPTH | TIME | NOT_ENUM | UNSUP | p50 | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 6 | 6 | 7 | 23 | 15 | 0 | 0 | 11 | 0 | 46 | 10090 | 12389 |
| 12 | 1 | 6 | 6 | 7 | 23 | 8 | 0 | 0 | 11 | 0 | 43 | 9382 | 11681 |
| 16 | 1 | 6 | 6 | 7 | 23 | 1 | 0 | 0 | 11 | 0 | 38 | 13519 | 13932 |
| **20** | 1 | 6 | 6 | 7 | 23 | **0** | 0 | 0 | 11 | 0 | 46 | 15043 | 16250 |
| 12 | 2 | 6 | 6 | 7 | 23 | 8 | 0 | 0 | 11 | 0 | 48 | 10972 | 13181 |
| 12 | 3 | 6 | 6 | 7 | 23 | 8 | 0 | 0 | 11 | 0 | 48 | 11013 | 12245 |

**depth を 8→20、chance_depth を 1→3 にしても、proof 分布は完全に不変（6 / 7 / 23）**。
depth 20 では `DEPTH_LIMIT` が 0 になる（木を辿り切っている）のに何も変わらない。
→ **深さは主なボトルネックではない**。残るのは `OUTCOMES_NOT_ENUMERABLE`（11）だけ。

参考（5 s / 8,000 nodes / R2=OFF）でも同じく WIN 6 / NO_WIN 7 / UNKNOWN 23 で不変。

### 4. Phase 1 baseline

`PROVEN_WIN = 6 / 36` は維持（R2 の ON/OFF いずれでも 6）。
時間予算 100 ms では WIN が 5〜6 で揺れるため、**baseline はノード予算で測る**ことにした
（30 s / 5,000 nodes / depth 10 → WIN 6 / NO_WIN 24 / UNKNOWN 6 / p50 42 ms）。

### 5. Safety

| 指標 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0** |
| illegal action | **0** |
| resource leak | **0** |
| fallback | **100%** |
| golden 回帰 | 全件一致（R2 既定 OFF のため変化なし） |
| B1 ガード回帰 | 変更なし・green |
| determinization 回帰 | green（listing 順序への非依存も新規に固定） |

```
tests/unit/lethal + tests/integration/lethal : 252 passed
full suite : 1 failed(既存 B6 由来), 483 passed, 11 skipped
```
リーサル探索由来の新規 failure は 0 件。

### 6. 判断: **C（R2 対応後も大量 outcome の materialization が支配的）**

- R2 で `UNSUPPORTED_EFFECT` は消せた（14→0）が、**同じ局面が
  `OUTCOMES_NOT_ENUMERABLE` に置き換わっただけ**で proof は 1 件も動かない
- depth 20・chance_depth 3・20 秒でも proof 分布は不変。深さでも時間でもない
- 残った唯一の壁は「3〜4 枚ドローの outcome が 36〜516 個あり、
  1 件あたり 0.65 ms の materialization が必要」という点

→ 次に検討すべきは **C（証明共有 = `OutcomeIndependenceCertificate`）**。
ドロー結果を使わない固定手順を全 outcome で共有できれば、
この唯一の壁を直接消せる。

---

## Step 1-10: ベンチマーク正規化と共通証明(OutcomeIndependenceCertificate)の候補測定

状態: **調査・測定完了**（2026-08-14）。証明共有は**本実装せず**。本番 entry 未登録。

### 1. ベンチマーク正規化（baseline 差の原因）

同一 fixture（36 件）で設定だけを変えて再測定した。

| config | WIN | NO_WIN | UNKNOWN | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| **旧 baseline と同一設定**（100 ms / 10,000 nodes / depth 8） | **6** | **15** | **15** | 36.7 | 102.3 |
| 100 ms / 10,000 / depth 10 | 6 | 16 | 14 | 39.6 | 102.2 |
| 30 s / 10,000 / depth 8 | 6 | 22 | 8 | 42.3 | — |
| 30 s / 5,000 / depth 10（Step 1-9c で使った設定） | 6 | 24 | 6 | 39.1 | — |

→ **旧 baseline（WIN 6 / NO_WIN 15 / UNKNOWN 15）は現在のコードで完全に再現する。**
Step 1-9c で「NO_WIN 24 / UNKNOWN 6」になったのは**時間予算を 100 ms → 30 s にしたため**で、
コード変更のせいではない。`PROVEN_NO_WIN` は「木を辿り切った」ことを意味するので、
予算を増やせば増えるのが正しい挙動。

「R2 で Phase 1 が悪化した」という比較は、**同一設定（30 s / 5,000 / depth 10）同士**の
R2 OFF（NO_WIN 24）と R2 ON（NO_WIN 18）で行っており、この結論は有効。

p50 が旧記録 63.7 ms → 36.7 ms へ改善しているのは `_deck_multiset()` 修正による
transposition の効き方の変化とみられる（proof 分布は不変なので安全側の変化ではない）。

**予算の効き方も確認した**: `time_limit=200 ms` / ノード無制限で実測 最大 205.5 ms。
Step 1-9d の p95 に出た 1,000 秒超は、バックグラウンド実行中に端末がスリープしたための
計測アーティファクトで、予算の未実施ではない。

#### 再現に必要な条件（今後の比較はこれを揃える）

```
fixture      : tests/fixtures/lethal_positions.jsonl（36 行・tags/golden 付き）
hidden state : build_dummy_search_state(obs, deck, rng=Random(0))
deck         : sample_submission/deck.csv
precheck     : 使わない（backend を直接叩く。entry 経由は max_remaining_prizes=2）
budget       : time_limit_ms / max_nodes / max_depth / max_chance_depth を明記
R2           : known_listing_as_decision（既定 False）
max_outcomes : 既定 24
```

### 2. `_deck_multiset()` の基準実装を固定

回帰テストを追加した（`test_deck_belief_and_listing.py`、9 件）:

- root の belief == 供給デッキ（36 局面中 20 件以上で検証）
- ドロー後は引いた分だけ**正確に**減る
- サーチ・公開・シャッフルを跨いでも、返す以上は必ず `deckCount` と一致する
- 検証できない導出は **None**（安全側）
- **過少計上した belief では Phase 2 が `PROVEN_WIN` を返さない**（不具合の再発防止）

### 3. 共通証明の必要条件（コードから導出）

探索が実際に参照している値から導いた。固定手順 L を全 outcome で共有するには:

| # | 条件 | 対応するコード |
|---|---|---|
| 1 | L の各行動が、各 outcome で**一意に**引き直せる | `action.reindex()`（曖昧なら None） |
| 2 | 引き直した選択がエンジンで合法 | `CgBackend.apply()` が拒否しないこと |
| 3 | 最終状態が勝ち | `is_win()`（`result == me`。エンジンが判定） |
| 4 | L の途中に新しい chance 事象が無い | `Transition.revealed` が立たないこと |
| 5 | 相手選択が現れたら全選択で L が成立 | `is_opponent_node()` → AND |
| 6 | outcome 集合が完全 | `certify_for_proof()`（質量 1・構築可能・未処理 0） |
| 7 | 隠れた山札順を使わない | `InfoKey` / `Scenario` の隔離（既存） |

**重要**: 「引いたカードを使わない」は十分条件ではない。手札枚数依存の打点、
残り山札 multiset に依存する後続 chance、相手選択のいずれかが outcome で変われば成立しない。
上記は**構造解析ではなく実際の再生で検証する**設計（推測しない）。

### 4. 実エンジンでの候補測定（決定的な数値）

各 outcome-heavy ノードで、1 つの outcome から Phase 1 で固定手順 L を取り出し、
**残り全 outcome に対して L を意味ベースで引き直して再生**した。

| id | outcomes | 候補 | \|L\| | win | no_win | reindex失敗 | 違法 | 検証時間 | 1 outcome あたり |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fx012 | 167 | **なし** | — | — | — | — | — | — | — |
| fx028 | 120 | なし | — | — | — | — | — | — | — |
| fx030 | 66 | なし | — | — | — | — | — | — | — |
| fx031 | 66 | なし | — | — | — | — | — | — | — |
| fx032 | 346 | なし | — | — | — | — | — | — | — |
| fx033 | 160 | なし | — | — | — | — | — | — | — |
| **fx035** | **129** | **あり** | 2 | **129** | 0 | 0 | 0 | 234 ms | **1.81 ms** |

- **候補率 1 / 7 = 14%**
- 候補が見つかった 1 件では **129 / 129 outcome すべてで成立**（引き直し失敗 0・違法 0）
- 検証コストは 1.81 ms/outcome（materialization 0.65 ms + L の再生 2 手）

**ただし決定的な点**: 候補が見つかった fx035 は、**通常の Phase 2 が既に `PROVEN_WIN` を
出している局面**である（Step 1-7 の測定）。残り 6 ノードは Phase 1 が
「1 つ目の outcome からでも確定手順を見つけられない」ため、共有すべき証明が存在しない。

→ **この corpus では、共通証明による新しい `PROVEN_WIN` は 0 件**。

### 5. materialization 削減量

共通証明は materialization を**減らさない**（全 outcome を構築して L を再生するため）。
減るのは「outcome ごとの部分探索」で、fx035 では
234 ms（129 outcome の検証）に対し、通常の Phase 2 探索は同種のノードで
5〜60 秒かかっていた（Step 1-7）。**成立する場合は 20〜200 倍速いが、成立が稀**。

### 6. Safety

| 指標 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0** |
| illegal action | **0** |
| resource leak | **0** |
| determinization 回帰 | green（listing 順序非依存を含む） |
| golden 回帰 | 全件一致 |

```
tests/unit/lethal + tests/integration/lethal : 255 passed
full suite : 1 failed(既存 B6 由来), 486 passed, 11 skipped
```

### 7. 判断: **C（候補がほとんど無く、別の高速化が必要）**

数値根拠:

- outcome-heavy ノード 7 件中、共通証明候補は **1 件（14%）**
- その 1 件は**既に通常の Phase 2 が証明できている**局面 → **新規 capability 0**
- 残り 6 件は「1 つ目の outcome からでも確定手順が無い」＝ outcome 依存のプレイが
  必要で、共通証明の対象外（定義上、扱えない）
- 機構自体は正しく働く（129/129・失敗 0）ので、**実装が難しいのではなく、
  この局面分布に対して当てはまらない**

→ `OutcomeIndependenceCertificate` は**本実装しない**。B5 と同じく、
「機構は妥当だが実デッキ・実局面では効かない」ことを実測で確認した記録として残す。

### 8. 次に検討すべきこと（未着手）

現時点で残っている実効的な選択肢:

1. **Phase 1 主体での本番接続**（Phase 2 は実エンジンでほぼ発火しないことを受け入れる）
   ── Phase 1 は 6/36 で `PROVEN_WIN` に到達し、再生検証も通っている
2. B8（多コイン）・新情報を伴う相手選択の対応
3. B2（Kaggle の 1 手上限）の解決

---

## Step 1-11: B8 調査・Phase 1 の実用化・precheck 偽陰性の修正

状態: **完了**（2026-08-14）。本番 entry 未登録。B8 は**調査のみ**（実装せず）。

### 0. 設計方針の固定

| Phase | 方針 |
|---|---|
| **Phase 1** | 実用候補。`PROVEN_WIN` のときだけ通常方策を上書き。最初の 1 行動だけ実行し、次の観測で再探索。false `PROVEN_WIN` = 0 を最優先 |
| **Phase 2** | 実装・安全基盤は維持。実エンジンではほぼ発火しない。`max_outcomes` 引き上げ・B5・state sharing・common proof はいずれも見送り。証明できなければ必ず `UNKNOWN` → fallback。本番既定 OFF |
| **Phase 3** | 既定 OFF。critic gate も未接続。今回も触っていない |
| R2 | 既定 OFF（feature flag として保持） |

### 1. B8（多コイン）の実測 ── **扱える**ことが判明

現行デッキには単発コイン（Dunsparce「Dig」）しか無いため、**専用デッキを組んで**測定した
（`Team Rocket's Kangaskhan ex` の Comet Punch = 「コインを 4 回投げ、表 1 つにつき 30 ダメージ」、
デッキ = ex 4 枚 + 基本エネルギー 56 枚）。

`manual_coin=True` で攻撃した結果:

| 選んだパターン | `COIN_HEAD` 選択回数 | 与ダメージ |
|---|---:|---:|
| H H H H | **4** | **-120**（表 4 × 30） |
| H T H T | **4** | **-60**（表 2 × 30） |
| T T T T | **4** | **0** |

- 「コインを 4 回投げる」は **4 つの独立した `COIN_HEAD` 選択**として提示される
- **混合 outcome（HTHT）が到達可能**。1 つの select が全フリップをまとめて決める、
  という B8 の懸念は**実測で否定された**
- 与ダメージは選んだ表の数と厳密に一致する（nested decision chain として素直に扱える）

→ **B8 は「1 フリップ = 1 決定点」として Phase 2 で扱える。**
固定 N 回のコインは 2^N outcome（各 1/2^N、全て構築可能）で、N=4 なら 16 通り
── 3〜4 枚ドローの 36〜516 outcome よりはるかに軽い。

**ただし未対応のまま残す部分**:
- 「表が出る限り続ける」系（`flip a coin until you get tails`、プール内 14 種）は
  outcome が非有界。打ち切ると残余質量が出るので Phase 2 では `UNKNOWN`
- 「場のポケモン 1 体につき 1 回」系（4 種）は N が盤面依存だが公開情報なので固定 N と同じ扱い

**実装は今回行っていない**（`manual_coin=True` でセッションを開く変更が Phase 1 の
全経路に影響するため、独立した Step で toy oracle → 実エンジンの順に入れるべき）。
現行デッキには固定 N 多コインが無いので、**現時点の capability 増加は 0 の見込み**。

### 2. 新情報を伴う相手選択

Step 1-5 の測定（相手選択ノード 25 件・その後の遷移 68 件）で
`after_opponent_choice_revealed = 0`。今回も新規の観測なし。
**「このデッキ・この corpus では未観測」**であり、
**「エンジンに存在しない」ではない**。検出コードは入っており、
発生すれば Phase 1 は `UNKNOWN`、Phase 2 は列挙拒否で安全側に倒れる。

### 3. precheck の偽陰性 ── **重大な取りこぼしを発見・修正**

Phase 1 が `PROVEN_WIN` を出した 6 件のうち **2 件（fx034 / fx035）は、
サイドが 4〜5 枚残っている**ため、旧 precheck（`残りサイド <= 2`）が
**探索を始める前に捨てていた**。

原因: どちらも**相手のベンチが空**で、バトル場を 1 体きぜつさせると
「場にポケモンがいない」で勝つ局面だった（サイド枚数と無関係な勝ち筋）。

修正: precheck に「相手のベンチが空 かつ バトル場がいる」を追加。
これは設計 §3.4 の「棄却してはいけないケース」に対応する。

| | 修正前 | 修正後 |
|---|---:|---:|
| entry 経由の `PROVEN_WIN` | 4 | **6** |
| 起動した局面 | 14 | **22** |
| 起動しなかった局面 | 22 | 14 |

回帰テストを 2 本追加した:
- ベンチが空の局面では precheck で止めない
- **起動しなかった全局面で Phase 1 を直接回し、`PROVEN_WIN` が埋もれていないことを確認**（現在 0 件）

### 4. Phase 1 の正式 baseline

**proof distribution baseline**（決定的に測るためノード予算を使う）

```
fixture = tests/fixtures/lethal_positions.jsonl（36 行）
config  = R2 OFF / Phase3 OFF / max_outcomes=24
budget  = 30 s（実質無制限）/ 5,000 nodes / depth 10 / chance_depth 1
→ PROVEN_WIN 6 / PROVEN_NO_WIN 24 / UNKNOWN 6
```

**latency baseline**（壁時計予算）

```
budget = 100 ms / 10,000 nodes / depth 8
→ PROVEN_WIN 6 / PROVEN_NO_WIN 15 / UNKNOWN 15（旧 baseline と一致）
   ALL      p50 63.9 / p95 103.5 / p99 103.7 / max 103.7 ms
   WIN      p50  8.4 / p95  33.4 / max  33.4 ms
   NO_WIN   p50 17.4 / p95  72.9 / max  72.9 ms
   UNKNOWN  p50 102.0 / p95 103.7 / max 103.7 ms
```

**注意**: 100 ms 予算では `PROVEN_WIN` が 5〜6 件で揺れる（時間切れの位置が実行ごとに変わる）。
**能力の比較は必ずノード予算で行う**こと。

### 5. Phase 1 の機能別内訳（何を解けているのか）

`PROVEN_WIN` の勝ち筋に現れた行動種別: `ATTACK` 5 / `CARD` 5（攻撃の対象選択など）/ `ATTACH` 1。

→ **現在 Phase 1 が解けているのは「今すぐ攻撃」と「エネルギーを 1 枚貼って攻撃」だけ**。
gust・進化・サーチを含む勝ち筋は 1 件も見つかっていない
（姉妹文書の想定リーサル「アメ→進化→ドロー→ボス→エネ→攻撃」は
意思決定 5〜6 手で、現在の深さ・予算では到達していない）。

### 6. 再生検証と安全指標

| 指標 | 結果 |
|---|---|
| `PROVEN_WIN` の再生成功率 | **5/5 = 100%**（1 手ずつ再探索して実際に勝ちへ到達） |
| false `PROVEN_WIN` | **0** |
| illegal action | **0** |
| resource leak | **0** |
| fallback safety | **100%** |
| determinization 回帰 | green |
| B1 ガード | 変更なし・green |

```
tests/unit/lethal + tests/integration/lethal : 256 passed
```

### 7. B2（大会の時間制約）

**外部仕様の確認が必要な未解決事項**として記録する。
Kaggle の 1 手あたり上限・プロセス永続性はリポジトリ内からは確定できない。
現時点で 4〜5 秒・700 ms などの本番値は決めない。予算は 100 ms のまま。

### 8. 判断: **B（B8 対応後に登録判断）ではなく、A に近いが 1 点保留**

本番登録条件に対する現状:

| 条件 | 状態 |
|---|---|
| Phase 1 false `PROVEN_WIN` = 0 | ✅ |
| illegal action = 0 | ✅ |
| `PROVEN_WIN` replay = 100% | ✅（5/5） |
| fallback safety = 100% | ✅ |
| resource leak = 0 | ✅ |
| determinization regression green | ✅ |
| B1 guard green | ✅ |
| disabled 時に既存 Agent と一致 | ✅ |
| R2 OFF で Phase 1 baseline 再現 | ✅ |
| Phase 1 の p95/p99 把握 | ✅（103.5 / 103.7 ms） |
| precheck の重大な偽陰性 | ✅ 発見して修正済み（0 件） |
| B8 が安全に `UNKNOWN` へ落ちる | ✅（コインが出た枝は `UNSUPPORTED_EFFECT`） |
| **B2（1 手上限）** | ❌ 外部確認待ち |

→ 技術条件は全て満たしている。残るのは **B2 のみ**で、これは外部情報が要る。

---

## Step 1-12: Phase 1 の本番候補化と限定接続・self-play A/B

状態: **完了**（2026-08-14）。`_SEARCH_MODULES` へ**登録済み・既定 OFF**。

### 1. Phase 1 production candidate として凍結した状態

| 項目 | 状態 |
|---|---|
| false `PROVEN_WIN` | 0 |
| illegal action | 0 |
| `PROVEN_WIN` replay | 100% |
| fallback safety | 100% |
| resource leak | 0 |
| determinization regression | green |
| B1 guard | green |
| `full_deck` の順序を decision 側へ渡さない | ✅（境界で `KnownDeckComposition` へ落とす） |
| R2 | OFF（feature flag 保持） |
| Phase 2 | 安全実装を保持・**既定 OFF** |
| Phase 3 | 未接続・OFF |

### 2. B8 の扱い

```
B8_FIXED_N_COIN     = capability confirmed / implementation deferred
B8_UNBOUNDED_COIN   = UNKNOWN / UNSUPPORTED
```

固定 N 回のコインは「N 個の独立した `COIN_HEAD` decision」として表現され、
混合 outcome も到達可能（実測済み、Step 1-11）。N=4 で 16 outcome。
ただし**現行デッキに固定 N 多コインが無く、実利が確認できていない**ため実装は延期。
Phase 1 の既存経路に `manual_coin` は接続していない（挙動を変えていない）。

### 3. precheck の正式化

「相手のベンチが空 かつ バトル場がいる」を起動条件に追加した。
**「ベンチが空なら常にリーサル」とは扱っていない**: precheck はあくまで起動条件で、
実際に倒せるか・場が空になるか・勝利条件を満たすかは
**エンジンの `state.result` による Phase 1 の確定判定**が決める。

回帰テスト（`test_precheck_does_not_drop_provable_wins`）:
**起動しなかった全局面で Phase 1 を直接回し、`PROVEN_WIN` が 0 件**であることを確認。
※ これは**現在の fixture（36 局面）の範囲での保証**であり、一般の保証ではない。

### 4. 本番 entry への限定接続（feature flag）

- `selector._SEARCH_MODULES` に `"lethal_phase1"` を登録した
- **登録しただけでは挙動は変わらない**。どれを使うかは config の `lethal_search.module`
- 既定 config（`rule_lethal.json`）は従来どおり `lethal_simple` = **既定 OFF**
- A/B 用に `configs/rule_lethal_phase1.json` を追加（Phase 1 のみ / Phase 2・3 は false）
- `selector` の context に `full_deck` を追加（entry 側の境界で multiset へ落とす）

追加した回帰テスト:
- モジュールは登録されているが**既定ではない**
- Phase 1 専用 config で **Phase 2 が一度も走らない**（`phase2_proof is None`）
- `enabled: false` のとき既存 Agent と**完全一致**

### 5. self-play A/B（24 試合・同一 seed・相手はランダム方策）

> ## ⛔ INVALID — 採用判断の根拠に使用しない（Step 1-16 で確定）
>
> 「同一 seed」という前提が成立していない。エンジンには seed 設定 API も state clone API も
> 無く、`BattleStart` は非決定的なので、baseline と phase1 を**同一の初期 random state から
> 比較できない**。誤った測定値ではなく、**比較手法そのものが成立していない**。
> 以下の勝率差は効果量の推定に使えない。

| | baseline（`lethal_simple`） | **phase1（新実装）** |
|---|---:|---:|
| 勝率 | 15/24 = **62.5%** | 21/24 = **87.5%** |
| 意思決定回数 | 1,455 | 1,043 |
| p50 latency | 22.4 ms | 22.7 ms |
| p95 | 27.7 ms | 123.5 ms |
| p99 | 122.5 ms | 127.8 ms |
| max | 142.8 ms | 142.1 ms |
| 1 試合の agent 時間 | 平均 1.47 s | 平均 1.40 s |
| lethal 起動 | — | 271 |
| **`PROVEN_WIN` 実行** | — | **35** |
| fallback（not_proven） | — | 236 |
| illegal action | 0 | 0 |

- 勝率 +25 pt。二標本比較で z ≈ 2.1 / p ≈ 0.04。
  **示唆的だが、24 試合・相手はランダム方策**なので確定的な結論ではない。
  実対戦相手プールでの再測定が必要。
- 意思決定回数が 1,455 → 1,043 に減っているのは、リーサルを早く決めて試合が短くなったため。
- p95 が 27.7 → 123.5 ms へ上がるのは、探索が走る手で 100 ms 予算を使うため。
  ただし **1 試合の総 agent 時間はむしろ短くなっている**（1.47 → 1.40 s）。

### 6. Phase 1 の現在の能力限界（明示）

勝ち筋に現れる行動は `ATTACK` / `CARD`（対象選択）/ `ATTACH` のみ。
**gust・進化・サーチを含む長いリーサルはまだ解けていない**。
「リーサル solver」ではなく、**「今すぐ攻撃 / エネ 1 枚で攻撃」を確実に取りこぼさない機能**
として扱うこと。

### 7. Phase 2 の位置づけ

**安全基盤は完成したが、実用 capability は未成熟**。
3 値判定・厳密 outcome 列挙・B4 AND ノード・R2・B1 ガード・構築検証・toy oracle は揃っているが、
実エンジンでは大量ドロー・materialization コスト・未対応効果・シャッフル後により
`PROVEN_WIN` は非常に限定的。既定 OFF のまま拡張基盤として保持する。

### 8. B2（時間制約）の切り分け

| 項目 | 状態 |
|---|---|
| 1 試合 10 分 / 時間切れ負け | **確認済み**（大会公式） |
| 1 手あたりの wall-clock 上限 | **未確認**（外部仕様の確認が必要） |
| プロセス永続性 / モジュール常駐 | **未確認** |

リポジトリ内の推測で解決済みとしない。予算は 100 ms のまま。

### 9. rollback

`configs/rule_lethal.json`（既定）を使う限り新実装は動かない。
A/B は config 切り替えのみで、コード変更なしに戻せる。

### 10. 最終判断: **A（Phase 1 を既定 OFF で本番接続し、A/B 測定へ進む）**

- 安全指標は全て満たしている（false PW 0 / illegal 0 / replay 100% / fallback 100% / leak 0）
- 既定 OFF なので既存挙動は 1 mm も変わらない（テストで固定）
- self-play で +25 pt の改善が観測されたが、**n=24・相手ランダム**のため確定ではない
- 次は「実対戦相手プール・より多くの試合での A/B」と B2 の外部確認

```
tests/unit/lethal + tests/integration/lethal : 258 passed
full suite : 1 failed(既存 B6 由来), 489 passed, 12 skipped
→ lethal 由来の新規 failure 0 件
```

---

## Step 1-13: 既定 OFF の安全性確認と held-out A/B

状態: **進行中**（2026-08-14）。既定 OFF のまま。Phase 2/3 は OFF、B8 は保留。

### 1. 既定 OFF の最終確認（新規テスト 13 件）

`tests/integration/lethal/test_feature_flag_safety.py`

| テスト | 内容 |
|---|---|
| 既定 config で Phase 1 が**一度も呼ばれない** | tripwire を仕込み、36 局面で呼び出し 0 回を確認 |
| 専用 config では確かに呼ばれる | テスト自体の検出力の確認 |
| 壊れた config 7 種で危険側へ倒れない | 節欠落 / 空 / module 未指定 / enabled 未指定 / 未知 module / **型違い** / null |
| entry 単体でも `enabled` が無ければ動かない | `fallback_reason == "disabled"` |
| config を戻すだけで baseline へ戻る | Phase 1 を挟んでも baseline 出力が不変 |
| Phase 1 専用 config で Phase 2/3 が OFF | |
| デッキリストの並びを変えても出力不変 | selector 経路での情報境界 |

**この過程で実際の緩さを 1 件発見・修正した**:
`enabled: "yes"`（文字列）は Python では truthy なので、
壊れた config でも Phase 1 が有効化されてしまっていた。
→ `entry` を **`config["enabled"] is True` の厳格判定**に変更し、
さらに `module` が自分向けでない場合も動かないようにした（`module_mismatch`）。

### 2. feature flag の構造

```
configs/rule_lethal.json         → module = lethal_simple   （既定・Phase 1 は動かない）
configs/rule_lethal_phase1.json  → module = lethal_phase1   （A/B 用・Phase 1 のみ）
```

`_SEARCH_MODULES` への登録は済んでいるが、**登録は有効化ではない**。
rollback は config を戻すだけ（コード変更不要）。

### 3. 回帰

```
tests/unit/lethal + tests/integration/lethal : 272 passed
full suite : 1 failed(既存 B6 由来), 502 passed, 12 skipped
→ lethal 由来の新規 failure 0 件
```

### 4. held-out paired A/B（**予備結果は再現しなかった**）

> ## ⛔ INVALID — 採用判断の根拠に使用しない（Step 1-16 で確定）
>
> `paired` と書いてあるが、**対応付けは成立していない**。seed control も state fork も
> 存在しないため、baseline と phase1 は同一の random trajectory から分岐していない。
> random 74% → 76%、rule-based 53.3% → 43.3% はいずれも効果量の推定に使えない。
> 「Phase 1 は rule-based に弱い」という主張も、この数値からは導けない。

同一 seed 列を A/B 双方に使う paired 比較。開発で使った seed（5000 番台）とは
重ならない held-out seed（90001〜90100 / 91001〜91030）を使用。

#### 相手 = ランダム方策（n=100）

| | baseline（`lethal_simple`） | phase1 |
|---|---:|---:|
| 勝率 | 74/100 = **74.0%** [64.6, 81.6] | 76/100 = **76.0%** [66.8, 83.3] |
| p50 / p95 / p99 / max | 22.4 / 29.4 / 43.1 / 136.5 ms | 22.3 / **124.2** / 127.2 / 136.8 ms |
| agent time/game | 1.30 s | **1.83 s（+41%）** |
| decisions/game | 55 | 50 |

paired: phase1 が勝ち baseline が負け **18** / その逆 **16** / discordant 34
→ **McNemar z = 0.17。有意差なし。**

Phase 1 の内訳: 起動 1,416 / `PROVEN_WIN` **140** / `PROVEN_NO_WIN` 743 / `UNKNOWN` 533。
実行 140 のうち **baseline と違う手を選んだのは 104**（36 は同じ手）。

レイテンシ内訳（Phase 1 ON）:

| 種別 | n | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| precheck のみ | 3,595 | 21.4 | 26.7 | 33.6 | 41.4 |
| 探索して UNKNOWN | 1,276 | 65.2 | 126.8 | 130.6 | 136.8 |
| `PROVEN_WIN` | 140 | 47.5 | 118.9 | 126.3 | 126.3 |

#### 相手 = ルールベース方策（n=30）

| | baseline | phase1 |
|---|---:|---:|
| 勝率 | 16/30 = **53.3%** [36.1, 69.8] | 13/30 = **43.3%** [27.4, 60.8] |
| p95 | 122.7 ms | 39.2 ms |
| agent time/game | 2.30 s | 2.30 s |

paired: phase1 勝ち/baseline 負け **6** / その逆 **9** → z = 0.52。**有意差なし（点推定はむしろ負）**。
Phase 1 起動 142 / `PROVEN_WIN` 9 のみ。

#### 解釈（重要）

- **Step 1-12 の予備 A/B（24 試合で 62.5% → 87.5%）は再現しなかった。**
  held-out 100 試合では +2 pt（信頼区間は完全に重なる）。
  予備結果は**試合数不足によるノイズ**だったと判断する。
- Phase 1 は 140 回 `PROVEN_WIN` を実行し、うち 104 回は baseline と違う手を選んでいる。
  それでも勝率が動かないということは、**その多くは baseline でも数手後に勝てていた局面**
  であり、「取りこぼしの救出」ではなく「勝ちの前倒し」になっている
  （decisions/game が 55 → 50 に減っているのと整合）。
- コストは実在する: p95 が 29.4 → 124.2 ms、1 試合の agent 時間が **+41%**。
  Step 1-12 で「総時間はむしろ短い」と書いたのも 24 試合の偶然で、再現しなかった。

### 5. 判断: **B（改善傾向は確認できず、既定 OFF を継続）**

- 安全性は全て維持（false `PROVEN_WIN` 0 / illegal 0 / fallback 100% / leak 0）
- しかし **held-out A/B で勝率改善は示せなかった**（z=0.17、相手を変えると点推定は負）
- レイテンシと総計算時間は明確に増える
- したがって既定 ON にする根拠が無い。**既定 OFF を維持**する

次に確認すべきは「Phase 1 が実際に救っている勝ちがあるのか」で、
`action_changed = 104` のうち**本当に baseline が取り逃していた勝ち**が何件かを
オフライン再生で分類する必要がある（現時点では未分離）。

---

## Step 1-14: `action_changed` の分類 ── Phase 1 は「救出」か「前倒し」か

状態: **完了**（2026-08-14）。既定 OFF 維持。新機能の追加なし。

### 分類方法（推測でなく実再生）

Phase 1 が `PROVEN_WIN` を実行し、かつ通常方策と**違う手**を選んだ決定ごとに、
探索セッションの中で baseline 側を実際に再生した。

```
root --(通常方策の手)--> 通常方策でターン終了まで rollout --> 今ターン勝てたか？
      勝てなかった場合 --> その状態に確定リーサルがまだ残っていたか(Phase 1 で確認)
```

### 結果

**相手 = ランダム方策（40 試合 / `PROVEN_WIN` 実行 51 / うち手が変わった 39）**

| 判定 | 件数 | 割合 |
|---|---:|---:|
| **Class B: 勝利の前倒し**（baseline も今ターン勝てた） | **33** | **84.6%** |
| Class A: baseline の通常方策では今ターン勝てなかった | 6 | 15.4% |
| ── うち **真の救出**（baseline の手でリーサルが消えていた） | **2** | 5.1% |
| ── うち 方策レベルの取りこぼし（リーサルは残っていたが方策が拾えず） | 4 | 10.3% |
| 判定不能 | 0 | 0% |

**相手 = ルールベース方策（30 試合 / 実行 18 / 手が変わった 14）**

| 判定 | 件数 | 割合 |
|---|---:|---:|
| **Class B: 勝利の前倒し** | **14** | **100%** |
| Class A: 救出 | **0** | 0% |

**合計（53 件）: 前倒し 47 件（88.7%） / 救出 6 件（11.3%、うち厳密な救出 2 件 = 3.8%）**

### 線の途中放棄（Class D のリスク）

`PROVEN_WIN` を実行したターンに実際に試合が終わったか:

| 相手 | 実行回数 | 発火ターンで試合終了した試合数 / 全試合 |
|---|---:|---|
| ランダム | 51 | 28 / 40 |
| ルールベース | 18 | **7 / 30** |

Phase 1 は「勝ち筋の**最初の 1 手**」だけを返し、次の観測で再探索する設計なので、
**次の決定で時間切れ（`UNKNOWN`）になると線が途中で放棄される**。
ルールベース相手で発火ターン終了が 7/30 と低いのは、この放棄が起きている可能性を示す。
放棄されると、通常方策が「Phase 1 が途中まで使った資源」の上から打つことになり、
**baseline より悪くなりうる**（Step 1-13 の rule-based 43.3% < 53.3% と整合する仮説）。
※ 直接の因果は未確認。指標が粗い（最後に発火したターンしか見ていない）。

### 結論: **B（主に「勝利の前倒し」であり、勝率改善効果は小さい）**

- 手が変わった 53 件のうち **88.7% は baseline でも同じターンに勝てていた**
- **厳密な意味での救出は 2 件（3.8%）**にとどまる
- 強い相手（ルールベース）では救出 **0 件**
- これは Step 1-13 の A/B（勝率差なし・rule-based では点推定が負）と完全に整合する
- 「140 回 `PROVEN_WIN` を出した」は capability の指標であって、価値の指標ではなかった

### 付随して分かったこと

- Phase 1 の価値は現状 **「試合短縮」** に近い（decisions/game 55 → 50）。
  ただしそのために p95 latency +320%・agent time +41% を払っている
- 「確定リーサルなら無条件採用」は**証明としては正しいが、実戦価値を保証しない**。
  baseline が同じターンに勝てる局面で発火しても、得られるものは無い
- 設計変更はしない（証明条件は不変）。**採用条件の再検討**は今後の課題として記録する

### 現時点の推奨

**Phase 1 は既定 OFF のまま、実験的機能として保持する。**
削除はしない（安全基盤・toy oracle・回帰テストは他の作業の土台になっている）。
既定 ON を検討できるのは、次のいずれかが示せたときに限る:

1. 線の途中放棄をなくす（＝発火したら必ず勝ち切る）ことで rule-based 相手の劣化が消える
2. 救出率が上がる（現在 3.8%）── gust/進化/サーチを含む深いリーサルを解けるようにする
3. tail latency を下げて、発火しない局面のコストを削る

### 付記: 分析中に見つかったテストの欠陥（修正済み）

1. `selector` 経由で「デッキリストの並びを変えても出力が同じ」を検査していたが、
   これは誤り。`build_dummy_search_state` は**呼び出しごとにグローバル random で**
   未確認カードをデッキ/サイドへ振り分けるため、同じ入力でも信念が変わる。
   → 「渡している値が deck.csv そのものであること」だけを検査する形に修正。
   デッキ並びへの非依存は、信念を固定した既存テストで担保している。
2. モジュール名を一括置換した際に「未知モジュール名」のテストが実在名になっていた
   → 本来の意図（未知なら動かない）に戻した。

この 1 点目は**本番挙動の理解としても重要**で、
Phase 1 の探索は毎回わずかに異なる信念の上で走る。
Phase 1 の証明は「公開を伴わない線」に限られるので信念に依存しないが、
**探索順序や打ち切り位置は呼び出しごとに変わりうる**（= 同じ局面でも結果が揺れる）。

---

## Step 1-15: 「線の途中放棄」仮説の検証と Phase 1 の価値確定

計測: `scratchpad/phase1_value_audit.py`（random 50 戦 seed 94001-94050 / rule-based 40 戦 seed 95001-95040、
Phase 1 ON = `rule_lethal_phase1`）、depth sweep と precheck 計測は保存盤面 36 件。

### 1. 途中放棄 — 仮説は棄却

「Phase 1 が 1 手返したあと、次の Observation で再証明できず線を捨てる」現象は **一度も起きなかった**。

| | random 50 | rule 40 |
|---|---|---|
| triggered（発火したゲーム） | 35 | 4 |
| completed（発火ターンで決着） | 35 | 4 |
| **abandoned** | **0** | **0** |
| abandonment 後 win/loss | – | – |
| 発火ターン内で再証明に失敗した決定 | 0 | 0 |

理由は depth sweep が示す:  Phase 1 が証明できる線は **残り 1〜3 手**で、
最後の 1 手（ATTACK）がその場で試合を終わらせる。再探索の機会自体がほぼ存在しない。
発火したゲームは random 35/35、rule 4/4 で勝利。

### 2. rule-based の勝率低下の真因 — **A/B 手法そのものの欠陥**

> **【Step 1-16 による訂正】** 本節の「探索がネイティブ側で本番対局と乱数列を共有している」
> という推論は**誤り**である。`BattleStart` 自体が再現しないため control 同士でも 100% 乖離し、
> 20/20 の乖離は探索の影響を示さない。結論「paired A/B は成立しない」は正しいが、
> 理由は探索ではなく**エンジンに seed 制御が無いこと**。詳細は Step 1-16 を参照。

rule-based で 53.3% → 43.3%、discordant pair 15。しかし本監査で Phase 1 が発火したのは 40 戦中 **4 戦のみ**。
「発火 ≤9 ゲーム」から discordant 15 は出ない。矛盾を追って以下を実測した。

**指す手を常に baseline に固定したまま、Phase 1 の探索だけを走らせる/走らせない**で比較（seed 96001-96020）:

```
同一行動を指しているのに結果が変わったゲーム: 20/20  （乖離開始は決定 1〜4 手目）
```

Python 側の global RNG を探索前後で save/restore しても **15/15 で乖離**。
つまり原因は Python 乱数ではなく、**`search_begin`/`search_step` がネイティブ側で本番対局と状態（乱数列）を共有している**こと。

帰結:
- これまでの **paired A/B はすべて対応が壊れている**。random 74%→76% も rule 53.3%→43.3% も、
  処置効果ではなく再ランダム化。判断 D（runtime/放棄が勝率を悪化させる）を支持する証拠は無い。
- 逆に「Phase 1 を有効化すると本番の山札シャッフル・コインが変わる」ことも意味する。
  意思決定への隠れ情報漏洩は別途テスト済みで 0 だが、**対局の乱数列は変わる**。
- 今後の A/B は非対応 2 標本として扱うか、探索用に独立した対局ハンドルを確保しない限り対応比較できない。

### 3. latency — 尾を作っているのは UNKNOWN

random 50 戦（決定 2338 件）:

| 区分 | n | p50 | p95 | p99 | max | nodes p50 | nodes p95 |
|---|---|---|---|---|---|---|---|
| precheck のみ（探索せず） | 1650 | 22.1 | 28.4 | 34.3 | 53.8 | – | – |
| PROVEN_WIN | 82 | 37.3 | 110.3 | 122.7 | 122.7 | 42 | 227 |
| PROVEN_NO_WIN | 371 | 30.1 | 91.1 | 115.5 | 123.0 | 17 | 204 |
| **UNKNOWN** | **235** | **124.8** | **134.9** | **141.6** | **145.9** | **282** | **356** |

rule-based 40 戦も同傾向（UNKNOWN n=130 p50=125.6 / p95=133.8 / max=139.4、nodes p50=223）。

- precheck のみの決定は baseline（p50 22.4 / p95 29.4）と一致 = **Phase 1 は非発火時にコストを足していない**。
- **探索した決定の 34%（random 235/688）・45%（rule 130/291）が UNKNOWN**。
  UNKNOWN は必ず 100ms 予算を使い切り、情報を返さない。これが p95 22.4→124.2ms 悪化の全体。
- 100ms 制限を超えるのは実質 UNKNOWN 時（全決定の約 10%）。B2 が未確定なので可否は判断しない。

### 4. depth sweep（保存盤面 36 件、予算 100ms）

| depth | PROVEN_WIN | PROVEN_NO_WIN | UNKNOWN | p50 | p95 | nodes p95 |
|---|---|---|---|---|---|---|
| 1 | 0 | 4 | 32 | 2.7 | 9.9 | 24 |
| 2 | 4 | 5 | 27 | 6.6 | 79.2 | 184 |
| **3** | **6** | 6 | 24 | **14.5** | 102.8 | 250 |
| 4 | 5 | 9 | 22 | 27.2 | 102.8 | 287 |
| 6 | 5 | 13 | 18 | 52.2 | 103.4 | 360 |
| 8 | 6 | 15 | 15 | 68.1 | 103.6 | 313 |

**depth 3 は depth 8 と同じ 6 勝を、中央値 1/4.7 のコストで取る。**
depth 4 以上が増やすのは PROVEN_NO_WIN だけ（行動選択には使わない = 純粋な浪費）。
depth 4/6 の 5 件は 100ms 打ち切りによる揺れ。

### 5. precheck

| 指標 | 値 |
|---|---|
| runtime | p50 **0.5 µs** / p95 0.8 µs / max 0.8 µs |
| searched | 22 / 36 |
| skipped | 14 / 36 |
| **true positive**（探索して PROVEN_WIN） | **6** |
| **false negative**（skip したが実は勝てた） | **0** |
| precision | 6/22 = 27% |

precheck は事実上ゼロコストで再現率 100%。**precheck 側に改善余地は無い**（判断 E は成立しない）。
残りのコストはすべて「探索したのに UNKNOWN」側にある。

### 6. 需要（発火ターンで選ばれた OptionType）

| 種別 | random | rule |
|---|---|---|
| ATTACK | 35 | 4 |
| CARD（グッズ/サポート = サーチ・ボスの指令を含む） | 9 | 5 |
| EVOLVE | 8 | 0 |
| NO（効果不使用） | 8 | 0 |
| PLAY | 7 | 0 |
| RETREAT | 5 | 1 |
| ENERGY | 4 | 1 |
| ATTACH | 4 | 1 |

gust 専用の OptionType は無く CARD に含まれる。需要は **ATTACK（止め）＋ 直前 1〜2 手の準備**に集中。

### 7. PROVEN_WIN の latency バケット × 判定

random（action_changed 52 件）:

| bucket | early win | true rescue |
|---|---|---|
| <25ms | 1 | 0 |
| <50ms | 25 | 1 |
| <75ms | 8 | 1 |
| <100ms | 4 | 2 |
| >=100ms | 9 | 1 |

rule-based（4 件）: <50ms early 2 / <75ms early 1 / >=100ms early 1・**rescue 2**。

50ms で打ち切ると changed 52→26、rescue 5→1。100ms なら changed 42、rescue 4。
**救出は速い決定に偏っていない**（rule-based の救出 2 件はどちらも >=100ms）。

### 8. 真の価値（本監査）

| 指標 | random 50 | rule 40 |
|---|---|---|
| phase1 executions | 82 | 12 |
| action_changed | 52 | 6 |
| **true rescue** | **5 (9.6%)** | **2 (33%)** |
| early win | 47 (90.4%) | 4 |
| harmful | 0 | 0 |
| indeterminate | 0 | 0 |

Step 1-14（rescue 6/53、rule 0）と合わせ、**救出は存在するが少数**、
かつ本監査で初めて rule-based 相手の救出 2 件を観測した。有害ケースは通算 0。

### 9. 最終判断: **B（shallow Phase 1 への簡略化を検討）**

- D は棄却: abandonment 0、有害 0。勝率悪化の根拠だった A/B は §2 の理由で対応比較として無効。
- E は棄却: precheck は false negative 0・0.5µs で、これ以上改善しても救出は増えない。
- A は時期尚早: 救出は 7 件（changed の 12%）で「十分存在」とは言えない。
- C は過小評価: 救出は 0 ではなく、rule-based 相手でも観測された。
- **B**: 価値のある線はすべて depth 3 以内にあり（§4）、コストの実体は depth 4 以上と UNKNOWN の
  予算使い切り（§3）。深さと予算を絞れば **救出を保ったまま tail を baseline 水準へ戻せる**見込みが高い。

次に検証すべきこと（未実施）:
1. depth 3 + 予算縮小での再監査（救出が保たれるか）
2. §2 のエンジン状態共有を回避した対応 A/B の手段があるか（無ければ非対応 2 標本で n を増やす）

---

## Step 1-16: RNG 非干渉の検証（Step 1-15 の結論訂正を含む）

計測: `scratchpad/rng_interference.py`, `scratchpad/rng_probe.py`。
固定したテスト: `tests/integration/lethal/test_rng_noninterference.py`（5 件）。

### 0. Step 1-15 の結論を訂正する

Step 1-15 で「同じ手を指しているのに 20/20 で結果が変わった → 探索が本番の乱数列を消費している」
と報告した。**これは誤り**だった。対照実験を先に取っていなかった。

```
同一デッキ・同一固定方策・探索なしで対局を 4 回:  4 回とも別の軌跡
別プロセスで「プロセス最初の 1 戦」を 4 回:       4 回とも別の軌跡
```

`BattleStart` 自体が再現しない。control 同士が 100% 乖離するので、
20/20 の乖離は探索の影響を一切示していなかった。

Step 1-15 の結論のうち **「paired A/B は成立しない」は正しいまま**だが、
理由は探索ではなく **エンジンに seed 制御が存在しないこと**である。

### 1. エンジンの API 面（PE export table を直接解析）

`cg.dll` の export は **13 個のみ**:

```
GameInitialize BattleStart BattleFinish GetBattleData Select AgentStart
VisualizeData AllCard AllAttack SearchBegin SearchStep SearchEnd SearchRelease
```

- **seed を設定する入口が無い**（`BattleStart` の引数はデッキ 120 枚だけ）
- **状態を複製する入口が無い**
- **RNG 状態を取得/設定する入口が無い**

`SearchBegin`/`SearchStep` は `Select` と**同じ `battle_ptr`** を第 1 引数に取る（`cg/sim.py`）。
構造上は本番対局オブジェクトの上で探索が走る。

### 2. 探索が乱数へ触れるのは SHUFFLE 時だけ

同一の live state から、同一の hidden state・同一の固定方策で探索を 3 回走らせて結果を比較（434 決定）:

| 探索 prefix | 3 回とも一致 | 不一致 |
|---|---|---|
| SHUFFLE を**含まない** | **179 / 179 (100%)** | 0 |
| SHUFFLE を含む | 172 / 255 (67%) | **83 (33%)** |

- ドローは呼び出し側が渡した山札順で決まるので完全に決定的（= determinization テストの結果と整合）。
- **シャッフルだけが、入力で決まらないエンジン側の乱数を引く。**

### 3. 探索は本番状態を破壊しない

探索の直前と直後に `GetBattleData` を取り、乱数で決まる部分（手札の実体・山札枚数）を比較:

```
探索前後で live state が変化した: 0 / 434
```

### 4. case A か case B か — **判定不能**

| | 判定 | 根拠 |
|---|---|---|
| ケース A（探索専用 RNG のみ消費 = 安全） | **証明できない** | RNG 状態を読む入口が無い |
| ケース B（本番と共有した RNG を消費 = 重大） | **反証できない** | 同上。かつ `battle_ptr` を共有している |

seed 制御が無いため、軌跡比較でも分布比較でも判別できない。
分布の偏りだけは測ったが（探索あり/なしで本番の引きを比較、chi2=24.1 / df=20）
**偏りは検出されなかった**。ただし n=60 対 60 で検出力は低く、非干渉の証明にはならない。

重要な区別: 仮にケース B でも、これは *再ランダム化* であって *偏り* ではない。
隠れ情報の意思決定への流入は別途 0 件で固定済みであり、悪用可能性は無い。
それでも**「非干渉である」ことを示せない**以上、指示 5 の条件は満たせない。

### 5. 修正方法の調査結果

| 方法 | 可否 | 根拠 |
|---|---|---|
| A: 探索専用 RNG をエンジンへ注入 | **不可** | 該当 export 無し |
| B: 探索用 engine instance を完全分離 | **部分的に可、しかし不十分** | 2 つの `BattleStart` は独立したポインタを返し、状態は互いに不変（実測）。しかし **RNG がプロセス共有か対局ごとかを判別できない**ため、分離できたことを保証できない |
| C: 探索前後で RNG state を保存・復元 | **不可** | 取得/設定の入口が無い。Python 側 seed の保存・復元では不十分（実測: 復元しても 15/15 乖離） |
| D: SDK/API 上不可能 | **これが現状の結論** | 上記より |

### 6. 本番使用可否 — **NOT SAFE**

指示 5 の必須条件

```
Exploration RNG Non-Interference = search の有無だけで本番 random outcome が変化しない
```

は **検証不能**。指示 5 のルールにより

```
Phase 1 production candidate = NOT SAFE
```

として扱う。false PROVEN_WIN = 0 は維持されているが、本番 ON は行わない。
`RNG_NON_INTERFERENCE_VERIFIED = False` をテストに定数として置き、
既定 config が `lethal_phase1` にならないことを固定した。

### 7. depth 別の品質（保存盤面 36 件・予算 100ms・オフライン解析）

指示 7 が許可するオフライン解析の範囲で測定した。**本番導入は行っていない。**

| depth | PROVEN_WIN | PROVEN_NO_WIN | UNKNOWN | UNKNOWN 率 | changed | true rescue | early win | harmful | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 4 | 32 | 89% | 0 | 0 | 0 | 0 | 2.9 | 12.6 | 17.9 | 17.9 |
| 2 | 4 | 5 | 27 | 75% | 2 | 0 | 2 | 0 | 9.0 | 85.6 | 96.8 | 96.8 |
| 3 | 5–6 | 6 | 25 | 69% | 2 | 0 | 2 | 0 | 15.7 | 103.1 | 104.3 | 104.3 |
| 4 | 5 | 9 | 22 | 61% | 2 | 0 | 2 | 0 | 31.6 | 104.7 | 105.2 | 105.2 |
| 8 | 5–6 | 15 | 16 | 44% | 3 | 0 | 3 | 0 | 66.2 | 103.0 | 103.5 | 103.5 |

**この表の読み方（指示 15）**: 「depth 3 で十分」ではなく
**「現在の保存盤面 36 件では depth 3 が depth 8 とほぼ同じ PROVEN_WIN 件数だった」**が正確な主張。
3 手以内の全リーサルを見つけられる、という意味ではない。

- PROVEN_WIN 件数が 5 と 6 で揺れるのは 100ms 打ち切りの影響（§2 の SHUFFLE 由来の揺れとは別）。
- **保存盤面では true rescue が全 depth で 0**。救出は実戦（Step 1-15: random 5 件 / rule 2 件）でのみ観測されている。
  したがって「depth 3 が救出を保つか」は保存盤面では判定できず、**有効な A/B が必要**。
- harmful は全 depth で 0。
- depth を下げると UNKNOWN 率は上がる（44% → 69%）が、UNKNOWN は予算を使い切るので
  p50 は逆に下がる（66.2 → 15.7ms）。tail は予算上限に張り付くため depth ではほぼ変わらない。
  **tail を下げるには depth ではなく予算そのものを下げる必要がある。**

### 8. 今後の A/B について

- 指示 10 のとおり、random 74%→76% と rule-based 53.3%→43.3% は**設計根拠として使用しない**。
- 「Phase 1 は rule-based に弱い」とも**断定しない**。
- 指示 11/12 の paired comparison・同一実ゲームの fork は、
  **seed 制御も state clone も存在しないため実装不能**（§1）。
  有効な比較は「非対応 2 標本で n を増やす」しか無く、
  Step 1-15 の効果量（差 2pt）を検出するには数千試合規模が要る。

### 9. 最終判断: **C（RNG を分離できず Phase 1 本番使用不可）**

- A ではない: RNG 非干渉を確認できていない。
- B ではない: 分離を「実装すべき」ではなく、**API 上実装できない**（§5）。
- D ではない: 実戦価値が低いと断定する材料が無い（有効な A/B が存在しないため）。
- E ではない: 新たな安全性問題（false PROVEN_WIN・不正行動・リーク）は 0 件。
  状態汚染も 0/434。あるのは**検証不能性**であって、観測された危害ではない。
- **C**: 指示 5 の必須条件を満たせないため、本番 Agent での Phase 1 実行を禁止する。

指示 7 に従い、探索はオフライン解析・保存盤面・別プロセスに限定する。
runtime 最適化と depth 3 導入はここが解決するまで進めない。

### 10. 安全性の状況

| 項目 | 結果 |
|---|---|
| false PROVEN_WIN | 0 |
| illegal action | 0 |
| replay verification | 100% |
| fallback | 100% |
| resource leak | 0 |
| **RNG interference** | **検証不能 → NOT SAFE** |

テスト: `1 failed（既存の B6 = value_model golden）, 507 passed, 12 skipped`。
lethal 由来の新規失敗 0。今回追加したのは
`tests/integration/lethal/test_rng_noninterference.py` の 5 件のみで、
既存コードは一切変更していない。

---

## Step 1-17: 能力マップの整理（新規探索機能の実装なし）

このステップでは**本番機能を一切追加していない**。コード変更は無く、更新はドキュメントのみ。

### 1. Phase 1 の正式な位置づけ変更 — offline research feature

```
Phase 1 lethal search
    └─→ offline analysis / saved states / benchmark
```

production Agent の action path には入れない。削除もしない。
用途は proof correctness / search efficiency / capability expansion / fixture analysis。
条件は `design.md` §Z、根拠は `step0-capability-report.md` §11。

### 2. 過去 A/B の無効化（完了）

以下 3 件に本文中で ⛔ INVALID の注記を付けた。**採用判断の根拠から除外する。**

| 結果 | 記載箇所 |
|---|---|
| 24 試合 62.5% → 87.5% | Step 1-12 §5 |
| held-out 100 試合 74% → 76% | Step 1-13 §4 |
| rule-based 53.3% → 43.3% | Step 1-13 §4 |

理由は「誤った結果」ではなく **「seed / state fork が無く、baseline と Phase 1 を同一の初期
random state から比較できないため、比較手法が成立していない」**。
同じ理由で「Phase 1 は rule-based に弱い」とも断定しない。

### 3. 保存盤面 36 件で何が解けるか（指示 9）

予算 100ms / 10,000 nodes。

| depth | PROVEN_WIN | PROVEN_NO_WIN | UNKNOWN | p50 | p95 | max | stop reasons |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 4 | 32 | 3.3 | 11.2 | 20.7 | DEPTH_LIMIT 32 |
| 2 | 4 | 5 | 27 | 8.4 | 75.2 | 101.2 | DEPTH_LIMIT 32 / TIME_LIMIT 1 |
| 3 | 5 | 6 | 25 | 15.0 | 102.2 | 102.5 | DEPTH_LIMIT 32 / TIME_LIMIT 7 |
| 8 | 5 | 15 | 16 | 66.9 | 105.1 | 111.2 | DEPTH_LIMIT 32 / TIME_LIMIT 16 |

`PROVEN_WIN` の first_action を能力タグで分類:

| depth | ATTACK | ATTACH+ATTACK | GUST | EVOLVE | SEARCH | CARD | ENERGY | RETREAT |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

**今の探索器が保存盤面で実際に解けるのは「攻撃だけ」と「エネルギー 1 枚を付けてから攻撃」の 2 種類のみ。**
GUST / EVOLVE / SEARCH / RETREAT を含むリーサルは 1 件も証明できていない。

**この範囲の明示（指示 4・15）**: 上表はすべて **現在の保存盤面 36 件での結果**である。

- 「depth 3 は実戦上 depth 8 と同等」とは**言わない**。正確には
  **「現在の保存盤面 36 件では depth 3 と depth 8 の PROVEN_WIN 件数が同程度だった」**。
- **true rescue = 0 / early win = 2〜3 も保存盤面上の結果**であり、実戦の値ではない
  （実戦では Step 1-15 で rescue が random 5 件・rule 2 件観測されている）。
- 保存盤面に救出局面がほぼ含まれていないため、**保存盤面では救出能力を評価できない**。

### 4. 何を追加すれば最小コストで PROVEN_WIN が増えるか（指示 10）

**先に「制約は能力ではない」ことを確認した。** 予算を 20 倍にして再測定:

| 予算 | PROVEN_WIN | PROVEN_NO_WIN | UNKNOWN | 総時間 |
|---|---|---|---|---|
| 100ms / 10k nodes | 5 | 15 | 16 | 2.0s |
| **2000ms / 200k nodes** | **6** | 22 | 8 | 16.1s |

20 倍の予算で増えた PROVEN_WIN は **+1 件だけ**。増分はほぼ UNKNOWN → PROVEN_NO_WIN
（行動選択には使わない）。さらに重要な点として、

- **`UNSUPPORTED_EFFECT` は全 depth・全予算で 1 件も出ていない。**
- `OUTCOMES_NOT_ENUMERABLE` / `INCOMPLETE_ACTION_SET` も出ていない。
- 出る stop reason は `DEPTH_LIMIT` と `TIME_LIMIT` のみ。

つまり**保存盤面では未対応効果に一度も当たっていない**。GUST / EVOLVE / SEARCH /
ENERGY / RETREAT のどれを追加しても、この 36 件で新しい PROVEN_WIN は増えない
（そもそも能力不足で止まっていない）。

UNKNOWN 局面の root 選択肢の内訳（100ms、UNKNOWN 16 件の合計 189 選択肢）:

| 選択肢 | 件数 |
|---|---|
| ATTACH | 91 |
| PLAY | 38 |
| EVOLVE | 18 |
| END | 13 |
| CARD | 13 |
| RETREAT | 7 |
| ATTACK | 5 |
| ABILITY | 4 |

支配的なのは **ATTACH（エネルギー付与先の選択）で全体の 48%**。これは能力の欠落ではなく
**分岐数**の問題。

**結論（指示 10 への回答）**: 現時点で「最小コストで PROVEN_WIN を増やす能力追加」は
**特定できない。保存盤面がボトルネックになっている**。
能力の優先順位を測るには、まず **本当にリーサルが存在する局面を fixture へ増やす**必要がある。
新 API も新 macro も現時点では要求されていない。

### 5. 能力マップ（指示 17）

| 項目 | currently proven | safe but unsupported | API limitation | performance limitation | needs future SDK support |
|---|---|---|---|---|---|
| **Phase 1** | 3 値証明・IDDFS・first action のみ実行・replay 検証。保存盤面で ATTACK / ATTACH+ATTACK のリーサルを証明 | GUST / EVOLVE / SEARCH / RETREAT を含むリーサル（未対応ではなく**未観測**） | — | 予算 20 倍で +1 勝。深さより分岐（ATTACH 48%）が支配的 | 本番投入には RNG 非干渉の検証が必要 |
| **Phase 2** | AND-OR・exact enumeration・確率完全性・sampling で証明しない規律 | — | outcome ごとに `search_begin` + prefix replay が必須（state clone が無い） | materialization 0.61–0.68 ms/outcome。**実エンジンでの capability は未成熟** | state clone があれば根本的に改善 |
| **Phase 3** | — | — | — | — | 既定 OFF・critic 未接続・**未評価** |
| **B1**（root でのデッキ公開） | runtime guard で条件付き解決 | — | — | — | — |
| **B4**（KO 後の相手選択） | AND ノードとして解決済み | — | — | — | — |
| **B8**（コイン） | fixed-N: engine 表現を確認、実装可能 | unbounded coin → UNKNOWN / unsupported | — | 現デッキでの利得は未測定 | — |
| **RNG** | 非 shuffle 探索は決定的（179/179）。live state 変更 0/434 | — | **seed 注入・RNG 読み書き・state clone すべて不可** | — | **非干渉の検証には seed または snapshot API が必須** |
| **precheck** | runtime 0.5µs、現 fixture で false negative 0、precision 27% | — | — | — | — |
| **`full_deck`** | 型分離で情報境界を確定。渡すのは deck.csv の multiset のみ | — | — | — | — |
| **outcome enumeration** | 多変量超幾何 + `Fraction` で厳密。`certify_for_proof()` の 5 条件 | — | — | B5 圧縮 1.00x（実装せず）。state 共有の重複 0% | — |
| **materialization** | `search_begin` + prefix replay で正確に再構成（Step 1-7a のバグ修正済み、956/956） | — | **state clone API が無いため代替不可** | 0.61–0.68 ms/outcome が下限 | clone API があれば桁で改善 |

### 6. 現在の正式な状態

```
Phase 1:
    correctness = promising
    production safety = NOT VERIFIED
    default = OFF
    usage = offline analysis

Phase 2:
    correctness infrastructure = implemented
    practical capability = immature
    default = OFF

Phase 3:
    default = OFF
    not evaluated

B8:
    capability confirmed
    implementation deferred

RNG:
    non-interference = NOT VERIFIABLE
    production use = NOT SAFE TO ENABLE

B2:
    external confirmation pending
```

### 7. このステップで保留したもの

Phase 1 production ON / Phase 1 default ON / Phase 2 production ON / B8 実装 /
Phase 3 / critic gate / self-play tuning / proof sharing — すべて未着手のまま。
precheck も現状維持（fixture が増えて偽陰性が出た場合のみ修正する）。

---

## Step 1-18: 能力評価コーパスの構築（solver 本体は未変更）

追加物: `tools/lethal_oracle.py`（独立オラクル）、`tools/build_capability_corpus.py`、
`tools/relabel_corpus.py`、`tools/analyze_capability_corpus.py`、
`tests/fixtures/lethal_capability.jsonl`（205 件）、
`tests/integration/lethal/test_capability_corpus.py`（7 件）。

### 0. 最重要: 新コーパスが Phase 1 の欠陥を 1 件あぶり出した

**`cap0197` / `cap0199` で Phase 1 が false `PROVEN_NO_WIN` を返す。**

検証済みの事実:

| 確認項目 | 結果 |
|---|---|
| オラクルの勝ち筋（4 手）を実エンジンで再生 | **勝ちに到達する** |
| `chance_free` | True（シャッフル・コイン・ドローを含まない） |
| 各深さで `legal_actions` にオラクルの手が含まれるか | **全深さで含まれる** |
| 各深さの `action_set_complete` | **全深さで True** |
| 終端で `backend.is_win` | **True** |
| `phase1.search(max_depth=8, 500ms, 20k nodes)` | **`PROVEN_NO_WIN`** |

勝ち筋が存在し、候補集合にも入っていて、行動集合も完全と主張しているのに
「今ターン勝てない」と確定している。**`PROVEN_NO_WIN` が不健全**である。

本番の行動選択には影響しない（`PROVEN_NO_WIN` は通常方策へ落ちるだけ）が、
`PROVEN_NO_WIN` の健全性を支える機構は `PROVEN_WIN` と共通なので、
**次ステップの最優先調査項目**とする。指示 18 に従い solver は変更していないので、
`test_phase1_never_contradicts_the_oracle_on_chance_free_positives` は
**意図的に失敗させたまま**残す。

旧 36 件コーパスではこの欠陥は一度も出なかった。評価基盤を作った直接の成果。

### 1. Corpus

| 項目 | 値 |
|---|---|
| total fixtures | 205 |
| unique states（board+hand+deck+turn+options） | 194（**95%**） |
| duplicate rate | 5%（重複グループは 1 つ、サイズ 12） |
| generation policies | 相手 random / rule-based を交互（**行に未記録 = 欠陥**） |

カテゴリ: `deterministic_lethal` 40 / `multi_step_lethal` 40 / `chance_lethal` 40 /
`negative` 30 / `near_miss` 30 / `opponent_choice` 25。positive 120 / near_miss 30 / negative 55。

**生成器の欠陥 2 件**（1 は修正済み、2 は未対応）:

1. **turn==0 のセットアップ局面が 20 件混入していた。** 全て `opponent_choice` に入り、
   同カテゴリ 25 件中 20 件を占め、12 件の完全重複もここだった。
   `state.turn < 1 or not prize` を除外条件として追加済み。
   → 実質使えるのは **185 件**、真の `opponent_choice` は **5 件のみ**。
2. **生成元 policy を行に保存していない。** 指示 3 の policy 別分割が現データでは不可能。
   次回の収集で記録する。現状は「random / rule-based 混合、自己対戦由来」としか言えない。

### 2. Ground Truth（指示 17）

| confidence | 件数 | 意味 |
|---|---|---|
| `GROUND_TRUTH_EXACT` | 80 | chance 無し・最小深さも厳密。確定 ground truth |
| `DETERMINIZATION_ONLY` | 40 | この 1 つの信念でしか勝てると言えない。**確定 ground truth に使わない** |
| `DEPTH_LIMITED` | 85 | depth 6 まで見つからなかっただけ。`PROVEN_NO_WIN` ではない |

`chance_free=False` の positive を確定 ground truth 扱いしないことは
`test_chance_positives_are_not_used_as_exact_ground_truth` で固定した。

### 3. Minimum Depth（指示 6/9）

| minimum_depth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 件数 | 9 | 40 | 54 | **11** | **5** | **1** |

`minimum_depth_kind`: **exact 120 / unknown 85**（upper_bound 0）。
positive 120 件すべてで depth 1..N-1 を打ち切り無しで完走しているので、
**この 120 件の minimum_depth は真の最小**。

**depth 4〜6 の positive は実在した（17 件）。** 旧 36 件コーパスには含まれていなかった。

### 4. Phase 1 depth sweep（100ms / 10k nodes、205 件）

| depth | WIN | NO_WIN | UNKNOWN | md≤d の捕捉 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 36 | 168 | 1/9 | 1.3 | 8.7 | 10.7 | 58.3 |
| 2 | 41 | 41 | 123 | 28/49 | 4.8 | 95.2 | 101.1 | 106.8 |
| 3 | 76 | 50 | 79 | 74/103 | 9.6 | 100.9 | 101.3 | 101.5 |
| 4 | 79 | 56 | 70 | 78/114 | 12.8 | 101.1 | 101.4 | 101.9 |
| 6 | 82 | 64 | 59 | 81/120 | 14.2 | 101.0 | 101.3 | 102.3 |
| 8 | **83** | 69 | 53 | 82/120 | 13.2 | 101.0 | 101.4 | 107.1 |

**Step 1-16/1-17 の「depth 3 ≒ depth 8」を撤回する。** 76 対 83 で明確に差がある。
旧コーパスで差が見えなかったのは深いリーサルを含んでいなかったため。
ただし `md≤d` の捕捉率は全 depth で約 70% 止まりで、取りこぼしの主因は深さではなく 100ms 予算。

### 5. Phase 1 vs Phase 2（同一 fixture、500ms / 20k nodes）

| | WIN | NO_WIN | UNKNOWN | p50 | p95 | max |
|---|---|---|---|---|---|---|
| Phase 1 | 93 | 77 | 35 | 12.6 | 501.6 | 507.7 |
| Phase 2 | **101** | 16 | 88 | 8.3 | 8529.9 | **9185.9** |

**Phase 1 で解けず Phase 2 で確定できた 9 件、逆は 1 件。** Phase 2 に実測可能な追加価値がある。

### 6. Phase 2 コスト分解（指示 11）

`search_begin` 回数（= materialization 回数）で分けた:

| search_begin | 件数 | p50 | p95 | max | nodes p50 | proof 内訳 |
|---|---|---|---|---|---|---|
| 1 | 2 | 2.3 | 2.3 | 2.3 | 0 | NO_WIN 2 |
| 2-5 | 40 | 1.3 | 8920.7 | **8964.7** | 2 | UNKNOWN 21 / WIN 18 / NO_WIN 1 |
| 6-20 | 70 | 3.0 | 15.1 | 82.1 | 7 | WIN 49 / UNKNOWN 12 / NO_WIN 9 |
| 21-100 | 45 | 16.3 | 85.7 | 510.8 | 30 | WIN 29 / UNKNOWN 12 / NO_WIN 4 |
| >100 | 48 | 500.7 | 509.9 | 529.1 | 1199 | UNKNOWN 43 / WIN 5 |

**コストは outcome 数にも探索深度にも比例していない。** 最も遅い 5 件はいずれも
`search_begin=3` / `nodes=2` で 8.8〜9.0 秒、停止理由は
`OUTCOMES_NOT_ENUMERABLE` + `TIME_LIMIT`。
つまり **9 秒は探索ではなく「列挙できない outcome 集合を列挙しようとして予算を食い潰す」時間**。
500ms の予算指定が効いていない（予算チェックが列挙の内側に無い）。
これは Phase 2 最適化ではなく**予算制御の欠陥**として扱う。

### 7. Capability Demand（指示 7/10）

| Capability | fixture | lethal 存在 | Phase 1 WIN | UNKNOWN | 平均 md | failure cause |
|---|---|---|---|---|---|---|
| ATTACK | 107 | 95 | 84 | 8 | 2.9 | 予算 |
| CARD | 138 | 107 | 86 | 11 | 2.9 | 予算 |
| ATTACH | 41 | 28 | 23 | 9 | 3.2 | 予算・分岐 |
| DRAW | 40 | 40 | 23 | 2 | 3.1 | **chance（Phase 2 領域）** |
| RETREAT | 21 | 3 | 2 | 11 | 4.7 | 需要が少ない |
| EVOLVE | 12 | 6 | 4 | 2 | 4.0 | 需要が少ない |
| PLAY | 24 | 0 | 0 | 11 | – | **lethal 需要 0** |
| **GUST** | **0** | **0** | 0 | 0 | – | **fixture が無く評価不能** |
| **SEARCH** | **0** | **0** | 0 | 0 | – | **fixture が無く評価不能** |
| **COIN** | **0** | **0** | 0 | 0 | – | **fixture が無く評価不能** |

実装コストは推測しない（low = 実装済み / unknown = 未調査）。
**最大の未解決は DRAW（40 件中 17 件が未証明）** で、これは Phase 2 の領域。
GUST / SEARCH / COIN は positive fixture が 0 なので、実装しても評価できない。

### 8. ATTACH の実態（指示 6）

- ATTACH が**本当に必要**なリーサル: **28 / 120**（23%）
- ATTACH が候補に出るが不要だったリーサル: 15
- positive 局面の root 選択肢: ATTACH 543 / 全 1,095 ≒ **50%**
- ATTACH を使うリーサルの minimum_depth: min 2 / 中央 3 / max 6
- ATTACH の後に使う能力: ATTACK 25 / CARD 25 / EVOLVE 3

**両方だった。** 分岐数としては過大表示（候補の 50% に対し必要なのは 23%）だが、
28 件は実際に ATTACH が無いと勝てない。
Step 1-17 の「48% は能力不足を意味しない」は半分正しく、半分誤り。

### 9. B1 / B4 タグ（指示 16）

| タグ | 件数 |
|---|---|
| `has_root_deck_reveal` | 35 |
| `has_midturn_deck_reveal` | 109 |
| `has_opponent_choice` | 77 |
| `has_multi_opponent_choice` | 74 |
| `opponent_choice_after_ko` | 30 |

### 10. near miss の不足量 / negative の細分類（指示 8）

| near miss の不足 | 件数 | | negative の種別 | 件数 |
|---|---|---|---|---|
| `bench_remaining` | 30 | | `opponent_choice_involved` | **55** |
| `unknown_deficit` | 20 | | `clearly_impossible` | 30 |
| `damage_short_50` | 10 | | | |
| `damage_short_20` | 9 | | | |
| `damage_short_60` | 9 | | | |
| `damage_short_large` | 4 | | | |
| `damage_short_30` | 2 | | | |
| `damage_short_40` | 1 | | | |

### 11. Safety（指示 13/14）

| 項目 | 結果 |
|---|---|
| **false `PROVEN_WIN`** | **0** |
| false `PROVEN_NO_WIN` | **2**（§0、新規発見） |
| illegal action | 0 |
| oracle replay failure | 0 / 120 |
| resource leak | 0 |

**`cap0064` は false PROVEN_WIN ではなかった。** オラクルはターン中の相手選択を
経由する線を探索しないため `oracle_win=False` になっていたが、実際には
1 手目でサイドを取って相手のバトル場を空にし、ベンチ 5 体すべてを AND で詰める
正しい `PROVEN_WIN` だった。テスト側を修正し、
`has_opponent_choice` の局面を負のラベルの根拠から除外した。

**この結果、negative 85 件のうち 55 件（`opponent_choice_involved`）は
「リーサルが存在しない」根拠として使えない。** オラクルに AND ノードを入れるまで、
負のラベルの射程は「相手選択を経由しない勝ち筋が無い」に限定される。

### 12. 最終判断: **F（corpus 追加が必要）＋ 先に §0 の欠陥調査**

- A（shallow Phase 1 が有望）は**否定された**: depth 3 = 76、depth 8 = 83。
- B（deeper Phase 1 が必要）は保留: 差はあるが、取りこぼしの主因は深さではなく予算。
- C（Phase 2 に追加価値）は**支持される**: 9 件。ただし §6 の予算制御欠陥を先に直す必要がある。
- D（特定 capability 実装）は**決められない**: GUST / SEARCH / COIN の positive が 0 件。
- E（実用価値が低い）は根拠なし。
- **F**: `opponent_choice` 5 件、GUST / SEARCH / COIN 0 件、policy 未記録という穴がある。

ただし corpus 追加より先に **§0 の false `PROVEN_NO_WIN`** を調べる。
証明の健全性は capability より優先する。

### 13. 維持しているもの

`RNG_NON_INTERFERENCE_VERIFIED = False` / Phase 1 既定 OFF / Phase 2 OFF / Phase 3 OFF /
R2 OFF / B8 deferred / paired self-play A/B は無効 / production safety = NOT VERIFIED。
solver 本体（phase1 / phase2 / macro generator / pruning / depth strategy）は未変更。

---

## Step 1-19: P0 健全性 2 件の修正

### P0-1 `PROVEN_NO_WIN` の不健全性（cap0197 / cap0199）

**原因**: `phase1._action_proof` が、公開を伴う枝（`transition.revealed`）を
自分の手番では `PROVEN_NO_WIN` として返していた。
docstring では「公開を伴わない範囲では勝てない」という内部的な意味で正当化していたが、
**展開していない枝に勝ち筋が無いとは言えない**。
実際 cap0197 / cap0199 では 1 手目の公開（デッキを見せるカード）の先に、
実エンジンで再生検証済みの 4 手リーサルが存在した。
`phase2_enabled=false` の本番構成では Phase 1 の `PROVEN_NO_WIN` が最終回答になるため、
この誤りはそのまま外へ出ていた。

同じ経路にもう 1 つ、`transition.state is None`（状態構築失敗）を
`PROVEN_NO_WIN` にしている分岐があった。構築失敗は我々の都合であって
盤面の性質ではないので、これも誤り。**phase2 にも同じ分岐があった。**

**修正**: 両方とも `UNKNOWN` + `OUTCOMES_NOT_ENUMERABLE` を返すようにした
（`phase1.py` / `phase2.py`）。docstring の `PROVEN_NO_WIN` の定義も
「**展開した木を辿り切って**勝ちが無いことを確認した」へ厳格化した。

**toy oracle も同じ誤りを共有していた。** `tests/toy/oracle.py::_deterministic_search`
と `tests/toy/candidates.py::phase1` が、公開手を飛ばすときに `truncated` を立てず、
「調べ尽くした」と扱っていた。これが cap0197 / cap0199 を検出できなかった理由。
両方で `truncated = True` を立てるよう修正した。

**波及**: golden 4 件（`fx004` / `fx015` / `fx026` / `fx032`）が
`PROVEN_NO_WIN` → `UNKNOWN` へ変わった。**すべて安全側で、`PROVEN_WIN` の変化は 0 件。**
これらの golden は欠陥を記録していたので更新した。

### P0-2 Phase 2 の予算制御

**原因**: `enumeration.draw_outcomes()` の列挙ループに予算チェックが無く、
組合せ爆発した集合を最後まで作っていた。実測で **500ms 指定に対し 9,186ms（18 倍）**。
遅い局面はいずれも `search_begin=3` / `nodes=2` / `OUTCOMES_NOT_ENUMERABLE` + `TIME_LIMIT` で、
**探索ではなく列挙が予算を食い潰していた**。

**修正**:
- `draw_outcomes(..., should_stop=)` を追加し、64 件ごとに予算を確認。
  打ち切り時は**部分集合を返さず `None`** を返す（中途半端な集合は
  「全 outcome を覆った」という誤った証明の根拠になりうる）。
- `CgBackend.attach_budget()` を追加し、`phase1.search` / `phase2.search` の冒頭で接続。
- 予算の定義を「**列挙 + 構築 + 探索 + 検証 + 後始末の合計**」と明文化。
- 打ち切りは常に `UNKNOWN` + `TIME_LIMIT`。`PROVEN_NO_WIN` にはしない。

| budget | actual max | Phase2 WIN | NO_WIN | UNKNOWN |
|---|---|---|---|---|
| 50ms | **52.5ms** | 95 | 16 | 94 |
| 100ms | **108.9ms** | 97 | 16 | 92 |
| 200ms | **204.8ms** | 99 | 16 | 90 |
| 500ms | **534.8ms** | 101 | 16 | 88 |

修正前は 500ms 指定で 9,186ms。超過は 5〜7% に収まった。

### 回帰テスト

`tests/integration/lethal/test_p0_soundness_and_budget.py`（8 件）:
cap0197 / cap0199 の最小再現、exact corpus 全体での false `PROVEN_NO_WIN` = 0、
公開枝が `PROVEN_NO_WIN` にならないこと、50/100/200/500ms の予算遵守、
列挙の打ち切りが部分集合を返さないこと、予算切れが `PROVEN_NO_WIN` にならないこと。

### Corpus（setup 除外後）

| 項目 | 件数 |
|---|---|
| total | 205 |
| **valid** | **185** |
| setup excluded（turn 0） | 20 |
| exact positive（`GROUND_TRUTH_EXACT`） | 80 |
| chance positive（`DETERMINIZATION_ONLY`） | 40 |
| negative 合計 | 65 |
| ├ exact negative（相手選択を含まない） | 30 |
| └ unknown negative（B4 未モデル化で根拠にできない） | 35 |
| B4（`has_opponent_choice`） | 57 |

### Phase 1 depth 再ベンチ（修正後・valid 185 件・100ms / 10k nodes）

| depth | WIN | NO_WIN | UNKNOWN | md≤d 捕捉 | false WIN | false NO_WIN | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3 | 181 | 1/9 | 0 | 0 | 1.3 | 8.2 |
| 2 | 42 | 8 | 135 | 28/49 | 0 | 0 | 5.3 | 86.4 |
| 3 | 77 | 10 | 98 | 75/103 | 0 | 0 | 11.4 | 100.8 |
| 4 | 81 | 11 | 93 | 80/114 | 0 | 0 | 16.4 | 100.8 |
| 6 | 83 | 11 | 91 | 82/120 | 0 | 0 | 24.1 | 101.0 |
| 8 | **84** | 14 | 87 | 83/120 | **0** | **0** | 25.0 | 100.9 |

修正で `PROVEN_NO_WIN` が激減した（depth 8 で 69 → 14）。これは能力低下ではなく、
**以前は証明できていないものを証明したと言っていた**ということ。

### Safety

| 項目 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0**（全 depth） |
| false `PROVEN_NO_WIN` | **0**（修正前 2） |
| illegal action | 0 |
| oracle replay failure | 0 / 120 |
| resource leak | 0 |
| RNG non-interference | **NOT VERIFIED**（変更なし） |

テスト: `1 failed（既存 B6）, 525 passed, 11 skipped`。lethal 由来の新規失敗 0。

### 維持しているもの

`RNG_NON_INTERFERENCE_VERIFIED = False` / Phase 1 既定 OFF / Phase 2 OFF / Phase 3 OFF /
R2 OFF / B8 deferred / paired self-play A/B は無効 / production safety = NOT VERIFIED。
macro redesign・pruning・transposition・B8・GUST・SEARCH・Phase 3・common proof は未着手。

---

## Step 1-20: P0 修正後の Phase 1 / Phase 2 厳密比較

計測: `tools/compare_phases.py`。**唯一変えるのは探索モジュールだけ**で、
corpus / fixture state / deck belief / node budget(20,000) / max_depth(8) /
max_chance_depth(1) / backend / process をすべて固定した。

### 1. 追加の健全性修正（`PROVEN_NO_WIN` の診断汚染）

契約テストを書いたところ、Phase 1 が `PROVEN_NO_WIN` と `DEPTH_LIMIT` を
同時に報告する事例が全予算で出た。原因は **停止理由が反復深化の反復をまたいで
累積していた**こと。証明そのものは `saw_unknown` が反復ごとにリセットされるので
以前から健全だったが、**診断からは「調べ尽くした」と「深さで止めた」を区別できない**
状態だった。停止理由を反復ごとに取り、`PROVEN_NO_WIN` はその反復のものだけを
報告するよう修正した。

`tests/integration/lethal/test_no_win_contract.py`（11 件）で共通契約を固定:

```
PROVEN_NO_WIN = 必要な探索木を完全に展開し、勝利へ到達する枝が存在しないことを確認した
```

`TIME_LIMIT` / `NODE_LIMIT` / `DEPTH_LIMIT` / `CHANCE_DEPTH_LIMIT` /
`UNSUPPORTED_EFFECT` / `INCOMPLETE_ACTION_SET` / `OUTCOMES_NOT_ENUMERABLE` /
`SHUFFLE_ENCOUNTERED` のいずれかが付いていたら `PROVEN_NO_WIN` にしない。
予算 5ms / 50ms / 500ms の 3 条件 × Phase 1 / Phase 2 で検証。
部分 outcome 集合・質量が 1 未満の集合・列挙打ち切りが証明に使われないことも固定。

### 2. Phase 1 vs Phase 2（完全同一条件）

**exact-positive subset（主評価、n=80、oracle replay 済み・chance-free）**

| budget | P1 WIN | P2 WIN | Case A | **Case B** | Case C | Case D |
|---|---|---|---|---|---|---|
| 100ms | 61 | 72 | 61 | **11** | 0 | 8 |
| 200ms | 66 | 72 | 65 | **7** | 1 | 7 |
| 500ms | 68 | 74 | 67 | **7** | 1 | 5 |

**chance-positive subset（n=40）**

| budget | P1 WIN | P2 WIN | **Case B** |
|---|---|---|---|
| 100ms | 22 | 24 | **2** |
| 200ms | 22 | 25 | **3** |
| 500ms | 23 | 26 | **3** |

**all-valid subset（n=185）**

| budget | P1 WIN / NO_WIN / UNKNOWN | P2 WIN / NO_WIN / UNKNOWN | **Case B** | Case C |
|---|---|---|---|---|
| 100ms | 83 / 14 / 88 | 97 / 16 / 72 | **14** | 0 |
| 200ms | 89 / 15 / 81 | 98 / 16 / 71 | **10** | 1 |
| 500ms | 93 / 15 / 77 | 101 / 16 / 68 | **9** | 1 |

**「Phase 2 が 17 件強い」は誤りだった。** 84 と 101 は条件の違う測定を並べたもので、
同一条件では 500ms で 93 対 101、実際に Phase 2 だけが確定できたのは **9 件**。
逆に **Phase 2 が落として Phase 1 が取れた Case C が 1 件**ある。

### 3. Case B 9 件の分類（500ms・all-valid）

| id | md | chance | B4 | reveal | outcomes | mat | Phase 1 stop | type |
|---|---|---|---|---|---|---|---|---|
| cap0008 | 1 | – | – | – | 0 | 0 | DEPTH_LIMIT, TIME_LIMIT | Type1 depth |
| cap0139 | 1 | – | – | – | 0 | 0 | DEPTH_LIMIT, TIME_LIMIT | Type1 depth |
| cap0056 | 3 | – | yes | – | 33 | 33 | DEPTH_LIMIT, OUTCOMES_NOT_ENUM | Type3 multi-outcome |
| cap0057 | 2 | – | yes | – | 53 | 53 | 同上 | Type3 multi-outcome |
| cap0136 | 2 | yes | – | yes | 4 | 4 | 同上 | Type3 multi-outcome |
| cap0179 | 2 | yes | – | yes | 12 | 9 | 同上 | Type3 multi-outcome |
| cap0058 | 2 | – | – | yes | 1 | 1 | 同上 | Type2 random outcome |
| cap0162 | 5 | – | – | yes | 0 | 0 | 同上 | Type2 random outcome |
| cap0176 | 2 | yes | – | yes | 0 | 0 | 同上 | Type5 deck reveal |

分類: **Type3 multi-outcome 4 / Type2 random outcome 2 / Type5 deck reveal 1 / Type1 depth 2**。

**9 件中 7 件が chance / reveal 系**で、これは元設計の役割分担どおり
（Phase 1 = 公開を伴わない確定手順、Phase 2 = 全 outcome 確定）。
残り 2 件（cap0008 / cap0139）は `minimum_depth=1` なのに Phase 1 が
`TIME_LIMIT` で落としており、**能力ではなく Phase 1 側の効率の問題**。

### 4. Performance（P0 修正後・500ms）

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| Phase 1 | 26.0 | 500.9 | 506.1 | 509.2 |
| Phase 2 | **6.7** | 500.9 | 501.4 | 501.9 |

**Phase 2 のほうが p50 で速い。** Phase 1 は反復深化で浅い深さを繰り返すため
中央値が高い。tail は両者とも予算に張り付く。
「Phase 2 は 9 秒かかる」という Step 1-18 の評価は**廃棄する**。

**outcome 数バケット別（500ms、all-valid 185 件）**

| bucket | n | chance nodes | materialization | search_begin | p50 | p95 | max | proof |
|---|---|---|---|---|---|---|---|---|
| 0（chance なし） | 163 | 5,010 | 0 | 38,451 | 5.0 | 501.0 | 536.3 | W89 / N15 / U59 |
| 1-16 | 19 | 38 | 66 | 11,881 | 67.7 | 514.5 | 514.5 | W9 / N1 / U9 |
| 17-32 | 1 | 1 | 22 | 251 | 89.1 | 89.1 | 89.1 | W1 |
| 33-64 | 2 | 86 | 86 | 1,847 | 368.0 | 368.0 | 368.0 | W2 |
| 65 以上 | **0** | – | – | – | – | – | – | – |

**185 件中 163 件は chance ノードを一切持たない。** outcome が 65 以上の fixture は
**1 件も無い**。したがって「Phase 2 のコストが大きな outcome 集合でどうなるか」は
**このコーパスでは測れていない**。

### 5. Corpus confidence

| 区分 | 件数 |
|---|---|
| valid | 185 |
| exact positive（`GROUND_TRUTH_EXACT`） | 80 |
| chance positive（`DETERMINIZATION_ONLY`） | 40 |
| exact negative | 30 |
| unknown negative（B4 未モデル化） | 35 |
| B4（`has_opponent_choice`） | 57 |
| setup excluded | 20 |
| GUST / SEARCH / COIN positive | **0 → `CAPABILITY_UNKNOWN`** |

### 6. Safety

| 項目 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0** |
| false `PROVEN_NO_WIN` | **0** |
| budget adherence（50/100/200/500ms） | **すべて上限内**（超過 5〜7%） |
| `PROVEN_NO_WIN` 契約（両 phase × 3 予算） | **違反 0** |
| illegal action | 0 |
| oracle replay failure | 0 / 120 |
| resource leak | 0 |
| RNG non-interference | **NOT VERIFIED**（変更なし） |

テスト: `1 failed（既存 B6）, 536 passed, 11 skipped`。lethal 由来の新規失敗 0。

### 7. 最終判断: **A（Phase 2 に実用的な追加能力がある）**

- 追加能力は **9 件 / 185**（exact-positive subset では 7 / 80）。9 件中 **7 件が
  chance / reveal 系**で、Phase 1 の設計上の担当外。**役割分担どおりに効いている。**
- runtime のペナルティは無い。むしろ **p50 は Phase 2 のほうが速い**（6.7ms 対 26.0ms）。
  p95 は両者とも予算に張り付くので、「追加 WIN / 追加 runtime」は Phase 2 側が有利。
- B（runtime が大きすぎる）は否定された。9 秒問題は予算バグで、修正済み。
- C（Phase 1 改善が費用対効果で勝る）も否定。ただし cap0008 / cap0139 は
  `minimum_depth=1` を Phase 1 が `TIME_LIMIT` で落としており、**反復深化の効率**に
  改善余地がある（能力ではない）。
- E（新しい健全性問題）は §1 の診断汚染を発見・修正済みで、残存は 0。
- D（corpus 不足）も依然として真だが、**core capability の測定はできた**。
  次に埋めるべき穴は明確になった: outcome 65 以上が 0 件、B4 negative 35 件、
  GUST / SEARCH / COIN が 0 件。

**ただし本番接続はしない。** RNG non-interference = NOT VERIFIED、
paired A/B 無効、B4 negative ground truth 不完全という状態は変わっていない。
今回の結論は offline capability evaluation としてのもの。

---

## Step 1-22: 大規模 outcome の探索 — **列挙経路に到達できないことが判明**

### 1. Phase2-only 9 件（Step 1-21 の再確認）

`tools/audit_phase2_only.py` で **9/9 replay green**。分類は
TypeC multi-outcome 4 / TypeB random 2 / TypeD deck reveal 1 / TypeE Phase1 効率 2。
うち 6 件が `GROUND_TRUTH_EXACT`、3 件が `DETERMINIZATION_ONLY`。
Phase 2 の `PROVEN_NO_WIN` 16 件も監査し、疑わしいもの 0。

replay の上限は**エンジンの選択回数**であって手数ではない。上限 12 では
cap0056 / cap0057 / cap0058 が誤って FAIL と判定された（独立全探索で
action [0] の先に勝ちが実在することを確認済み）。60 へ引き上げて 9/9 green。

### 2. 65+ outcome は理論上は自然に出る

`deck.csv` は 60 枚 / **22 種**。`draw_outcomes` の実測値:

| 残り山札 | 種類 | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|---|
| 60 | 22 | 22 | **249** | **1,932** | 11,548 | 56,638 |
| 40 | 12 | 12 | **77** | **352** | 1,282 | 3,938 |
| 30 | 8 | 8 | 36 | **120** | 328 | 770 |
| 20 | 5 | 5 | 15 | 35 | 70 | 121 |
| 12 | 3 | 3 | 6 | 10 | 15 | 18 |

**捏造しなくても、序盤の 2〜3 枚ドローで数百〜千の outcome になる。**
既存コーパスの最大が 53 だったのは、採取が終盤（山札が小さく種類も少ない）に
偏っていたため。

### 3. しかし**列挙経路に到達できない**（今回の主要な発見）

`tools/large_outcome_probe.py` で 25 試合・序盤（山札 20 枚以上）から
root の chance ノードを探したが、**17 outcome 以上の列挙可能なノードは 0 件**だった。

コーパス 185 件の root を全走査した結果も同じ:

| root chance ノードの outcome 数 | ノード数 |
|---|---|
| 1-16 | 86 |
| 17-32 | 1 |
| **33 以上** | **0** |

列挙**できなかった**理由の内訳:

| 理由 | 件数 |
|---|---|
| `UNSUPPORTED_EFFECT` | 115 |
| `OUTCOMES_NOT_ENUMERABLE` | 62 |
| `DECK_REVEALED_AT_ROOT`（B1 ガード） | 55 |

**結論**: 大規模 outcome での Phase 2 の性能は「未測定」ではなく、
**現在のバックエンドでは到達不能**。大きな outcome を生む節点は、
outcome 数が決まる前に上記 3 つの理由で `UNKNOWN` へ落ちている。

したがって判断 B（大規模 outcome でスケールしない）は**支持も否定もできない**。
Phase 2 の実効的な能力上限は、列挙のカバレッジ（`UNSUPPORTED_EFFECT` 115 件が最大要因）で
決まっており、outcome 数のスケーリングはその先の問題である。

観測できた範囲での最大は **53 outcomes / 409ms（cap0057, 深い位置の chance ノード）**で、
この範囲では予算内に収まっている。

### 4. B4 — 未着手

| 区分 | 件数 |
|---|---|
| B4 positive | 22 |
| B4 opponent-choice unmodeled | 35 |
| **B4 exact negative** | **0** |

独立オラクルが相手選択を経由する線を探索しないため、否定側の ground truth が無い。
AND semantics（全選択 WIN → `PROVEN_WIN` / 1 つでも NO_WIN → `PROVEN_NO_WIN` /
NO_WIN 無しで UNKNOWN あり → `UNKNOWN`）の Case B / Case C はまだ検証できていない。
**Step 1-22 では着手できなかった。**

### 5. Safety

| 項目 | 結果 |
|---|---|
| false `PROVEN_WIN` | 0 |
| false `PROVEN_NO_WIN` | 0 |
| budget overrun | 0 |
| illegal action | 0 |
| resource leak | 0 |
| `PROVEN_NO_WIN` 契約 | green（両 phase × 3 予算） |
| RNG non-interference | **NOT VERIFIED** |

テスト: `1 failed（既存 B6）, 536 passed, 11 skipped`。lethal 由来の新規失敗 0。

### 6. 判断: **C（B4 対応を優先）**

- B は判定不能（§3）。大規模 outcome は到達経路が無いので、
  先に列挙カバレッジ（`UNSUPPORTED_EFFECT` 115 件）を調べる必要がある。
- E（corpus を増やす）は**この問題には効かない**。局面を増やしても
  同じ 3 つの理由で弾かれる。
- A（Phase 2 最適化）は時期尚早。速度ではなくカバレッジが上限を決めている。
- D（Phase 1 最適化）は cap0008 / cap0139 の 2 件という限定的な効果。
- **C**: B4 exact negative は**現実的に作れて**、AND semantics の否定側という
  安全性の核心を検証できる唯一の未検証領域。B4 positive は既に 22 件ある。

未完了として記録: Phase2-only 9 件の golden 固定、B4 独立オラクル、
大規模 outcome fixture（到達不能のため方針変更が必要）。

---

## Step 1-23: B4 独立オラクルと列挙ブロッカーの分類

### 1. B4 独立オラクル（`tools/b4_oracle.py`）

`phase1` / `phase2` / `InfoKey` / macro generator / candidate generator /
pruning / transposition を **import していない**（実 import は
`cg.api` と `engine.SearchSession` / `EngineError` のみ）。
枝刈り無し・置換表無し・相手選択は必ず全列挙して AND を取る。

AND 契約: 全選択 WIN → WIN / 1 つでも NO_WIN → NO_WIN /
NO_WIN 無しで UNKNOWN あり → UNKNOWN / 列挙し切れない → UNKNOWN。

### 2. B4 の実測 — **真の exact negative は 0 件**

コーパス 185 件から、ターン中の相手選択ノード **139 個**を検出:

| 区分 | 件数 |
|---|---|
| `CaseA_all_win`（全選択で勝てる） | 26 |
| `trivial_all_no_win`（全選択で勝てない） | 113 |
| **`CaseB_mixed`（一部だけ NO_WIN）** | **0** |
| `CaseC_unknown_present` | 0 |

選択肢数の分布: 4 択以上 85 / 3 択 50 / 2 択 4。
context は `TO_ACTIVE` 相当が 120、`DISCARD` 相当が 18。

**このコーパスでは、相手の選択が勝敗を左右する局面が 1 つも存在しない。**
すべて「どれを選ばれても勝てる」か「どれを選ばれても勝てない」のどちらか。
したがって **AND 意味論の識別力（Case B / Case C）は依然として未検証**であり、
185 件では検証できない。

理由は構造的で、KO 後に相手が何を出しても、勝てるかどうかはサイド枚数で決まり、
昇殿するポケモンには依存しないため。Case B を作るには
**同一ターンに 2 回目の KO が必要で、かつベンチの HP がばらついている**局面が要る。
現コーパスの採取方針ではそこへ当たらない。

「エンジンに存在しない」のではなく「**今回のデッキ・採取方針では未観測**」である。

### 3. 列挙ブロッカーの分類（valid 185 件・root の chance 候補）

| Blocker | 件数 | positive fixture 由来 | 行動種別 | select context |
|---|---|---|---|---|
| `UNSUPPORTED_EFFECT` | **115** | **84** | PLAY 89 / ATTACH 26 | すべて `MAIN`(0) |
| `OUTCOMES_NOT_ENUMERABLE` | **62** | **49** | ATTACH 46 / YES 10 / ABILITY 6 | `MAIN`(0) 52 / (43) 10 |
| `DECK_REVEALED_AT_ROOT`（B1） | **55** | **33** | CARD 41 / (選択なし) 14 | (7) 55 |

**`UNSUPPORTED_EFFECT` 115 件はカード効果の未実装ではない。**
発生しているのは `MAIN` コンテキストの **PLAY（89）と ATTACH（26）**、
つまり「ポケモンを出す」「エネルギーを付ける」という最も基本的な行動である。
これらが chance ノード扱いされた上で列挙不能と判定されている。
`OUTCOMES_NOT_ENUMERABLE` も同様に ATTACH が 46 件で最多。

つまりブロッカーの主因は**特定カードの未対応ではなく、
基本行動が公開扱いされたときの列挙経路そのもの**にある可能性が高い。
指示 11 の「`UNSUPPORTED_EFFECT` = 未実装カード機能と決めつけない」は正しかった。

### 4. 優先度（件数 × positive 候補 × 安全性）

| Blocker | 件数 | positive | 安全に対応可能か | 優先度 |
|---|---|---|---|---|
| `UNSUPPORTED_EFFECT`（PLAY/ATTACH） | 115 | 84 | **要調査**（分類の誤りなら安全に解消できる可能性） | **1** |
| `OUTCOMES_NOT_ENUMERABLE`（ATTACH 主体） | 62 | 49 | 要調査 | 2 |
| `DECK_REVEALED_AT_ROOT`（B1） | 55 | 33 | **緩和しない**（B1 ガードは安全側で維持） | 3 |
| B4 Case B | **0** | – | 局面が無いので検証不能 | 4（採取方針の変更が必要） |

### 5. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / budget overrun 0 /
illegal 0 / leak 0 / `PROVEN_NO_WIN` 契約 green / RNG non-interference NOT VERIFIED。
Phase2-only 9 件は 9/9 replay green を維持。

### 6. 判断: **E（まだ分析が必要）**

- A（B4 実装・拡張）は**できない**: Case B が 0 件で、AND の否定側を
  検証する対象が存在しない。実装しても正しさを測れない。
- B（`UNSUPPORTED_EFFECT` の特定カテゴリを実装）は**時期尚早**:
  115 件の中身が PLAY / ATTACH という基本行動で、
  「未実装カード効果」という前提が崩れた。**何が unsupported と判定されているのか**を
  先に特定する必要がある。分類の誤りなら実装ではなく判定の修正になる。
- C（B1 再設計）は却下: B1 ガードは安全側で維持する。
- D（outcome enumeration 改善）も B と同じ理由で先に原因特定が要る。
- **E**: 次に必要なのは実装ではなく、
  **`UNSUPPORTED_EFFECT` 115 件で実際に何が起きているか**（どの効果・どのログ事象で
  未対応と判定されたか）の特定。ここが解ければ positive 84 件が動く可能性がある。

未完了: B4 Case B / Case C の局面採取（採取方針の変更が必要）、
Phase2-only 9 件の golden ファイル化、`UNSUPPORTED_EFFECT` の根本原因特定。

---

## Step 1-24: `UNSUPPORTED_EFFECT` 115 件の根本原因 — **R2 が OFF であること**

### 1. 発生地点の特定

`enumerate_outcomes` は `transition.revealed` の枝でしか呼ばれない。そこで
**PLAY / ATTACH がなぜ `revealed=True` になるのか**を実測した（valid 185 件・root）。

| 行動 | `revealed=True` の理由 | 件数 |
|---|---|---|
| PLAY | **`select.deck is not None`（デッキ listing が提示された）** | **89** |
| ATTACH | **同上** | **26** |
| ATTACH | ドローが発生した | 98 |
| いずれか | `current.looking`（覗き見） | **0** |

`looking` 由来は 1 件も無い。**115 件すべてがデッキ listing 由来**である。

経路は `cg_backend._reveals_without_draw()`:

```
select.deck is not None
  -> _is_known_listing_selection() が False
  -> refusals["reveal_without_draw"] += 1, revealed=True
  -> enumerate_outcomes() で「ドローが増えていない」
  -> StopReason.UNSUPPORTED_EFFECT
```

そして `_is_known_listing_selection()` の**最初の行**が

```python
if not self._known_listing_as_decision:
    return False
```

**R2（`known_listing_as_decision`）は既定 OFF**（Step 1-9b で「利得ゼロ・遅延増」と
測定して OFF のままにした）。そのため、listing の中身が
**我々が供給した belief デッキの部分集合であることを検証できる場合でも**、
一律に「新情報の公開」として扱われ、ドローが伴わないので `UNSUPPORTED_EFFECT` になる。

### 2. 分類の答え（指示 21 の表）

| action | subtype | count | root cause | fixable |
|---|---|---|---|---|
| PLAY | デッキ listing を伴う（サーチ系） | 89 | R2 既定 OFF | **既存フラグで対応可能** |
| ATTACH | デッキ listing を伴う | 26 | 同上 | 同上 |
| その他 | `looking` 由来 | 0 | – | – |

| action | reason | count |
|---|---|---|
| ATTACH | ドロー後に belief から列挙できず（`OUTCOMES_NOT_ENUMERABLE`） | 46 |
| YES | 同上 | 10 |
| ABILITY | 同上 | 6 |

**「未対応カード効果」ではなかった。** 実装が必要な新しい効果は無く、
既に実装済みの R2 判定が既定 OFF になっているだけ。
指示 11 / 16 の想定どおり「既存能力の過剰拒否」である。

### 3. なぜ Step 1-9b の測定と食い違うのか

Step 1-9b では R2 を「利得ゼロ・遅延増」と測定して OFF のままにした。
その測定は **旧 36 件コーパス**、かつ **P0 修正前**（公開枝を `PROVEN_NO_WIN` に
落としていた時期）に行ったもの。
公開枝が `PROVEN_NO_WIN` で握り潰されていたため、R2 を有効にしても
「証明可能になる枝」が見えなかった。P0 修正で公開枝が `UNKNOWN` になった今、
**同じ 115 件が Phase 2 の入口として顕在化している**。

### 4. まだ実施していないこと（重要）

**R2 を有効化していない。** 有効化は以下を伴うため、測定と安全性検証を
セットで行う必要があり、本 Step では踏み込まなかった。

- `_is_known_listing_selection()` の 4 条件（自分の手番 / context が
  `TO_HAND`・`TO_BENCH` / 伏せカード無し / listing ⊆ 供給 belief）が
  115 件すべてで実際に成立するかは**未検証**。成立しない listing は
  従来どおり公開扱いにしなければならない。
- Before/After（`UNSUPPORTED_EFFECT` 件数 / Phase 2 WIN / UNKNOWN / p95）未測定。
- positive candidate 84 件のうち何件が救済されるか未測定。
- 安全回帰（false `PROVEN_WIN` = 0、未完了枝の PROVEN への昇格が無いこと）未実施。

### 5. Safety（現状維持）

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / budget overrun 0 / illegal 0 /
leak 0 / `PROVEN_NO_WIN` 契約 green / Phase2-only 9 件 9/9 replay green /
R2 OFF / RNG non-interference NOT VERIFIED。
solver 本体は**今回も変更していない**。

### 6. 判断: **A（分類バグ修正だけで大幅改善する見込み）— ただし未検証**

- 115 件すべてが単一の原因（R2 既定 OFF）に帰着し、うち **84 件が positive candidate**。
- 新しいカード効果の実装は不要。判定フラグの問題。
- B（新 effect 対応が必要）ではない: `looking` 由来 0 件、未知効果 0 件。
- C（outcome enumeration がボトルネック）はまだ言えない。
  115 件が通ってから初めて `OUTCOMES_NOT_ENUMERABLE` の実態が見える。
- D（B4 優先）ではない: B4 は Case B が 0 件で検証対象が無い。
- E（新しい健全性問題）は無い。

**ただし「A」は見込みであって測定結果ではない。** R2 の 4 条件が
115 件で成立するかを検証し、Before/After と安全回帰を取るまで、
改善幅を主張してはならない。

---

## Step 1-25: R2 の適用可能性検証と Before/After

`tools/r2_audit.py`。**既定は R2 = OFF のまま変更していない**（指示 18）。
監査側で 4 条件を独立に再計算し、`known_listing_as_decision` を切り替えて同一条件比較した。

### 1. R2 applicability（4 条件判定）

| reason | count | positive 由来 |
|---|---|---|
| **APPLICABLE** | **115** | **84** |
| NOT_OUR_TURN | 0 | 0 |
| WRONG_CONTEXT | 0 | 0 |
| HIDDEN_CARD_PRESENT | 0 | 0 |
| LISTING_NOT_SUBSET | 0 | 0 |
| BELIEF_UNAVAILABLE | 0 | 0 |

内訳: PLAY 89 / ATTACH 26、いずれも全件 APPLICABLE。
listing ⊆ belief は multiset 差（`listing - expected` が空）で厳密に判定した。

**115 件すべてが R2 の 4 条件を満たす。** つまり過剰拒否は
「条件を満たさないものまで通していた」のではなく、
「条件を満たすものを一律に拒否していた」側の誤り。

### 2. Before / After（Phase 2, budget 500ms, valid 185 件, 同一 fixture）

| metric | R2 OFF | R2 ON |
|---|---|---|
| `PROVEN_WIN` | 101 | **104** |
| `PROVEN_NO_WIN` | 16 | 18 |
| `UNKNOWN` | 68 | **63** |
| stop `UNSUPPORTED_EFFECT` | 47 | **0** |
| stop `OUTCOMES_NOT_ENUMERABLE` | 39 | 31 |
| stop `DECK_REVEALED_AT_ROOT` | 32 | 32 |
| stop `TIME_LIMIT` | 29 | **34** |
| stop `DEPTH_LIMIT` | 46 | **65** |
| p50 | 6.3 | 6.0 |
| p95 | 500.6 | 500.6 |
| p99 | 501.2 | 522.4 |
| max | 504.3 | 526.3 |
| nodes 平均 | 270 | 320 |

**`UNSUPPORTED_EFFECT` は 47 → 0 で完全に消えた。**
runtime のペナルティも実質無い（p50 はむしろ微減、p95 同等）。

### 3. Positive 84 件の実態

| 指標 | 値 |
|---|---|
| R2 reachable（4 条件を満たし探索へ進める） | **84 / 84** |
| R2 → 新規 Phase 2 `PROVEN_WIN` | **3** |
| R2 → 依然 `UNKNOWN` | 大半 |
| R2 → 別 blocker へ移行 | `DEPTH_LIMIT` +19 / `TIME_LIMIT` +5 |

結果が変わった fixture は 5 件だけ:

```
UNKNOWN -> PROVEN_WIN     : 3  (cap0127, cap0197, cap0199)
UNKNOWN -> PROVEN_NO_WIN  : 2
```

新規 `PROVEN_WIN` 3 件は**すべて oracle が positive と判定している fixture**で、
**3/3 とも replay green**（first action だけ実行 → 再探索を繰り返して実際に勝利へ到達）。
うち cap0197 / cap0199 は Step 1-19 の P0 で見つけた 2 件そのもの。

R2 ON でも `UNKNOWN` のまま残った停止理由:

| reason | count |
|---|---|
| `DEPTH_LIMIT` | 41 |
| `TIME_LIMIT` | 34 |
| `DECK_REVEALED_AT_ROOT`（B1） | 19 |
| `OUTCOMES_NOT_ENUMERABLE` | 18 |
| `SHUFFLE_ENCOUNTERED` | 16 |
| `CHANCE_DEPTH_LIMIT` | 7 |

### 4. Safety（R2 ON で再測定）

| 項目 | 結果 |
|---|---|
| false `PROVEN_WIN` | **0** |
| false `PROVEN_NO_WIN` | **0** |
| replay（新規 WIN 3 件） | **3/3 green** |
| budget overrun | 0 |
| illegal action | 0 |
| resource leak | 0 |
| `PROVEN_NO_WIN` 契約 | violation 0 |
| B1 ガード | `DECK_REVEALED_AT_ROOT` 32 件で不変（緩めていない） |

B1（root で `select.deck != None`）と R2（探索途中の既知 listing）は
別経路のまま。R2 ON でも B1 の件数は 32 で変化していない。

### 5. 判断: **B（R2 は一部改善するが、次の blocker が支配的）**

- `UNSUPPORTED_EFFECT` は 47 → 0 で**完全に解消**した。分類の過剰拒否だったことは確定。
- しかし **positive 84 件のうち新規 `PROVEN_WIN` は 3 件だけ**。
  残りは `DEPTH_LIMIT`（+19）と `TIME_LIMIT`（+5）へ移っただけで、
  「unsupported が消えた = solved」ではない（指示 14 の警告どおり）。
- A（主要な capability 改善）とは言えない: 純増 3 件は Phase2-only 9 件と比べて小さい。
- C（効果が小さい）とも言い切れない: 対象は 115 件と多く、安全性も確認でき、
  runtime ペナルティも無い。**次の blocker を外せば効いてくる位置にある**。
- D（新しい健全性問題）は無い。

**次に支配的なのは `DEPTH_LIMIT` 41 件と `TIME_LIMIT` 34 件**、
すなわち探索効率であって capability ではない。
Step 1-21 で見つけた Phase 1 の反復深化コスト（cap0008 / cap0139 が
`minimum_depth=1` を 500ms で落とす）と同じ方向を指している。

### 6. 既定は変更していない

`known_listing_as_decision` は **既定 OFF のまま**（指示 18）。
Phase 1/2/3 既定 OFF、B8 deferred、RNG non-interference NOT VERIFIED も不変。
solver 本体は今回も未変更。

---

## Step 1-26: R2 ON 後に残る `DEPTH_LIMIT` / `TIME_LIMIT` の原因分解

solver は変更していない。R2 は既定 OFF のまま、監査でのみ ON にして測った。

### 0. まず件数の読み方を訂正

Step 1-25 で報告した「`DEPTH_LIMIT` 41 / `TIME_LIMIT` 34」は
**停止理由の出現回数**であって fixture 数ではない（1 件に複数の理由が付く）。
主因で分けると:

| R2 ON の UNKNOWN 63 件 | 件数 |
|---|---|
| `TIME_LIMIT` 主体 | 34 |
| `DEPTH_LIMIT` 主体 | 8 |
| その他（chance / reveal 系のみ） | 21 |

### 1. `DEPTH_LIMIT` — **真の深さ不足は 0 件**

8 件に予算 10 倍（5s / 300k nodes）+ `max_depth=14` を与えた結果:

| cause | count |
|---|---|
| **true depth shortage** | **0** |
| 深く探すと `PROVEN_NO_WIN`（真に勝ちが無い） | 4 |
| 予算 10 倍でも UNKNOWN（深さ以外が主因） | 4 |

**深さを上げて勝ちが見つかったケースは 1 件も無い。**
`DEPTH_LIMIT` というラベルは付くが、深さは主因ではない。
指示 15 の「`DEPTH_LIMIT` = 深さが主因と扱わない」がそのまま当てはまった。

### 2. `TIME_LIMIT` — **純粋な時間不足は 34 件中 4 件**

| cause | count |
|---|---|
| 予算 10 倍でも UNKNOWN（時間以外が主因） | **30** |
| 予算 10 倍で解決（純粋に時間不足） | 4 |
| branching | root 候補 平均 6.4 / 最大 22 |
| nodes | 平均 1,518 |

root 分岐は平均 6.4 と小さく、**分岐爆発でも反復深化コストでもない**。
30/34 は時間をいくら足しても解けない。

### 3. 解けない 44 件の**真の** blocker

予算 10 倍・`depth=14` でも UNKNOWN のままの 44 件から、
予算系の停止理由（TIME / DEPTH / NODE）を除いた内訳:

| 真の blocker | count |
|---|---|
| **`DECK_REVEALED_AT_ROOT`（B1）** | **19** |
| `OUTCOMES_NOT_ENUMERABLE` | 8 |
| `OUTCOMES_NOT_ENUMERABLE` + `SHUFFLE_ENCOUNTERED` | 7 |
| `SHUFFLE_ENCOUNTERED` | 7 |
| `CHANCE_DEPTH_LIMIT` | 2 |
| `INCOMPLETE_ACTION_SET` | 1 |

**B1（root でのデッキ公開）が単独最大の 19 件**、
chance 系（列挙不能 + シャッフル）が合計 22 件。
`chance_depth=3` / `depth=10` / 5s まで緩めても UNKNOWN 52 / WIN 4 / NO_WIN 7 で、
chance 深さも主因ではない。

### 4. R2 impact（同一 185 fixture）

| metric | R2 OFF | R2 ON |
|---|---|---|
| root branching 平均 | 8.9 | **8.9** |
| nodes 平均 | 261 | 330 |
| p50 | 6.8 | **6.0** |
| p95 | 501.7 | 501.8 |
| UNKNOWN | 68 | **63** |

**R2 は探索を巨大化させていない。** root 分岐は不変、ノードは +26%、
p50 はむしろ改善。指示 16 で懸念された「安全だが探索を巨大化させる」は起きていない。

### 5. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / replay green /
budget overrun 0 / illegal 0 / leak 0。
R2 既定 OFF・Phase 1/2/3 既定 OFF・RNG non-interference NOT VERIFIED を維持。

### 6. 判断: **F ではなく、原因は特定できた — 該当は A/B/C/D のいずれでもない**

指示 22 の選択肢に対する実測の答え:

- **A（iterative deepening 改善）— 否**: `DEPTH_LIMIT` 主体は 8 件で、
  深さを上げて解けたのは 0 件。反復深化コストは Phase 1 の p50 には効くが、
  UNKNOWN の解消には効かない。
- **B（action ordering 改善）— 否**: root 分岐は平均 6.4〜8.9 と小さく、
  順序で救えるほどの候補数が無い。
- **C（R2 branching 対策）— 否**: R2 は分岐を増やしていない（root 8.9 で不変）。
- **D（chance enumeration 改善）— 部分的に該当**: 22 件（列挙不能 15 + シャッフル 14、
  重複あり）が chance 系。ただし単独最大ではない。
- **E（複数要因）が最も近い**が、実測が示す優先順位は明確:

```
1. DECK_REVEALED_AT_ROOT (B1)          19 件  ← 単独最大
2. OUTCOMES_NOT_ENUMERABLE / SHUFFLE   22 件  ← chance 系（重複含む）
3. 純粋な時間不足                        4 件
4. 真の深さ不足                          0 件
```

**探索効率（深さ・時間・順序・分岐）はボトルネックではなかった。**
残っているのは情報境界のガード（B1）と chance の列挙可能性であり、
どちらも安全性のために意図的に置いたもの。
ここから先は「効率の最適化」ではなく
**「ガードを緩めずに扱える範囲を広げられるか」**という設計問題になる。

なお B1 の 19 件は Step 1-23 の方針どおり**緩めない**。
まず「root でデッキが公開されている局面で、何が起きているのか」を
R2 と同じやり方で条件分解する必要がある。

---

## Step 1-27: B1（root デッキ公開）の provenance 調査 — **全件が実デッキ由来**

コード変更なし。調査のみ（指示 17）。

### 1. root listing を持つ fixture

`select.deck != None` を root で持つのは **42 件**（うち Phase 2 で
`DECK_REVEALED_AT_ROOT` により UNKNOWN になるのが 19 件）。

クラスタリング結果:

| select.context | hidden card | listing size | 件数 |
|---|---|---|---|
| 7 | **無し** | 1〜45（1:4, 2:4, 3:2, 8:2, 10:3, 14:3, 38:2, 45:1 ...） | 42 |

**全件が同一 context（7）で、伏せカード（`None`）は 1 枚も無い。**
listing サイズだけが 1〜45 と幅広い。構造は同型。

### 2. Probe — listing は何に依存するか

供給する `your_deck` を差し替えて root listing が変わるかを実測（12 件）:

| probe | 結果 |
|---|---|
| **Probe D: 順序だけ変更**（reverse / shuffle） | **12/12 listing 不変** |
| **Probe B/C: 中身を変更**（半分を存在しないカード id へ差し替え） | **12/12 listing 不変** |

**供給デッキの中身を壊しても root listing は 1 文字も変わらない。**
つまり listing は我々の belief からではなく、
**エンジンが保持する実デッキから生成されている**。

Step 0 の「root で `select.deck != None` のとき `your_deck` は無視され
実物の listing が返る」という観測が、19 件（の代表 12 件）でも再現した。

### 3. 判定

| Pattern | Count | Real deck dependent | Belief reconstructable | Safe as decision? |
|---|---|---|---|---|
| root listing（context 7, hidden 無し） | 42（うち B1 blocker 19） | **Yes（12/12 実証）** | **No** | **No** |
| Case B（belief だけで説明できる） | **0** | – | – | – |
| Case C（一部だけ既知） | 0 | – | – | – |
| Case D（listing 後に shuffle/draw） | 別途 chance 系で処理 | – | – | – |

**Case A（本当に実物デッキを参照しており belief では再現不能）が全件。**

### 4. 「hidden card が無い」は安全の根拠にならない

42 件すべてに伏せカードが無く、listing ⊆ belief も多くで成立しうる。
しかし §2 の probe が示すとおり、**listing の中身は実デッキから来ている**。
belief と一致するのは「我々の belief がたまたま当たっていた」からであって、
listing を見たこと自体が新情報である。

したがって指示 6 の禁止事項

```python
if select.deck: treat_as_known_listing_decision()
if listing ⊆ belief: B1 bypass
```

はいずれも**情報漏洩を起こす**。実装しない。

### 5. R2 との違い（重要）

| | R2（探索途中の listing） | B1（root の listing） |
|---|---|---|
| listing の生成元 | **我々が供給した belief デッキ**（`search_begin` に渡した `your_deck`） | **エンジンの実デッキ** |
| 供給デッキを変えると | listing も変わる | **変わらない**（12/12） |
| 既知情報か | Yes（4 条件を検証したうえで） | **No** |
| 安全に decision 化できるか | できる（115/115 検証済み） | **できない** |

同じ「デッキ listing」に見えても情報源が違う。
**共通化してはならない**（指示 12）。

### 6. 結論: **A（全件が本当に未知情報）**

```
DECK_REVEALED_AT_ROOT -> UNKNOWN
```

を**正式仕様として維持する**。これは能力不足ではなく、
information boundary を守るための安全な拒否である（指示 11）。

`KnownRootListingDecision` は**設計しない**。
Case B / Case C が 1 件も存在しないので、作る対象が無い。

### 7. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / B1 ガード不変 /
determinization 不変（供給デッキの順序を変えても listing・結果とも不変）/
illegal 0 / leak 0。
R2 既定 OFF・Phase 1/2/3 既定 OFF・RNG non-interference NOT VERIFIED を維持。

### 8. 次の blocker

B1 19 件が原理的に扱えないと確定したので、残る改善余地は chance 系のみ:

| blocker | 件数 |
|---|---|
| `OUTCOMES_NOT_ENUMERABLE` | 15（うち 7 は SHUFFLE と併発） |
| `SHUFFLE_ENCOUNTERED` | 14（同上） |
| `CHANCE_DEPTH_LIMIT` | 2 |
| `INCOMPLETE_ACTION_SET` | 1 |

Phase 2 の到達可能な上限は、**valid 185 件のうち B1 の 19 件を除いた範囲**で決まる。

---

## Step 1-28: chance 系 blocker の根本原因調査

コード変更なし。R2 は監査でのみ ON。

### 1. `SHUFFLE_ENCOUNTERED` — **過剰拒否は 0 件**

Phase 2 の探索中に `enumerate_outcomes` が `SHUFFLE_ENCOUNTERED` で拒否した
**1,886 回すべて**について、その時点の state を調べた:

| 拒否時の state | 回数 |
|---|---|
| **経路上ですでに SHUFFLE 済み（`state.shuffled=True`）** | **1,886** |
| 当該 state は未 shuffle なのに拒否 | **0** |

| subtype | count | positive | root blocker | safe to support? |
|---|---|---|---|---|
| `DRAW → SHUFFLE`（同一 step 内、ドローが先） | **0 件が拒否されている** | – | – | 既に許可済み（Step 1-7a） |
| `SHUFFLE → DRAW` / 経路上で shuffle 済み | 1,886 | 22 fixture | シャッフル後の順序が未知 | **No** |
| SHUFFLE のみ（後続 draw 無し） | 上記に含まれ、区別可能な拒否は発生していない | – | – | – |

**`DRAW → SHUFFLE` が拒否されているケースは 1 件も無い。**
Step 1-7a で入れた `draw_is_supply_ordered()` は正しく効いており、
「同一 step 内のイベント順を見ずに一律拒否」という以前のバグは再発していない。

拒否はすべて「経路上で既にシャッフルが起きた後の列挙」で、
これはシャッフル後の順序が本当に未知であることによる**正しい拒否**である。

指示 6 の「shuffle したが後続で order を使わない」ケースについては、
拒否の入口が `state.shuffled` という**経路レベルのフラグ**なので、
現在の実装では「その後 order を参照するか」を見ていない。
理屈の上では緩和余地があるが、**実測上そこに該当する拒否は観測されなかった**
（全 1,886 件が shuffle 後の列挙要求そのもの）。したがって実装しない。

### 2. `OUTCOMES_NOT_ENUMERABLE`

拒否イベント 440 回。fixture 単位の重複:

| 区分 | fixture 数 |
|---|---|
| `SHUFFLE_ENCOUNTERED` のみ | 25 |
| `OUTCOMES_NOT_ENUMERABLE` のみ | 19 |
| **両方** | **12** |

positive fixture 由来は `SHUFFLE_ENCOUNTERED` 22 / `OUTCOMES_NOT_ENUMERABLE` 15。

`CgBackend.refusals` の内訳（全 fixture 合計）:

| refusal | 回数 |
|---|---|
| `opponent_choice_node`（B4） | 2,433 |
| `known_listing_selection`（R2 で decision 化できた） | 2,249 |
| `reveal_without_draw`（R2 対象外の公開） | 1,134 |

**`too_many_outcomes` に相当する refusal は 1 件も記録されていない。**
つまり `OUTCOMES_NOT_ENUMERABLE` は「outcome が多すぎる」ではなく、
**probe 適用の失敗・belief の再構成不能**によるもの。
指示 10 が想定した「300 outcomes で上限に当たる」型ではない。

### 3. `CHANCE_DEPTH_LIMIT`

`max_chance_depth=1` の下で 7 件。Step 1-26 で `chance_depth=3` / `depth=10` / 5s
まで緩めた測定では UNKNOWN 52 / WIN 4 / NO_WIN 7 で、
**chance 深さを 3 倍にしても解決したのはごく僅か**。
chance 段数の不足が主因ではない。

### 4. `INCOMPLETE_ACTION_SET` — 3 件（要追加調査）

`cap0027` / `cap0175` / `cap0204`。
Step 1-26 では 1 件と報告したが、R2 ON・全 fixture 走査では 3 件だった。
**どの action が欠けているのか、engine から取れないのか自作生成が落としているのかは
今回特定できていない。** chance 系とは別系統の correctness 懸念として残す。

### 5. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget overrun 0 / illegal 0 / leak 0 / B1 ガード green / R2 safety green。
テスト: `1 failed（既存 B6）, 535 passed, 12 skipped`。

### 6. 判断: **B（`SHUFFLE_ENCOUNTERED` は全て本当に未知情報で、現行 guard が正しい）**

- A（過剰拒否がある）は**実測で否定**: 1,886/1,886 が経路上シャッフル後の列挙要求。
  `DRAW → SHUFFLE` の誤拒否はゼロ。
- C（`OUTCOMES_NOT_ENUMERABLE` に過剰拒否がある）は**判定できない**が、
  少なくとも「outcome 数の上限」型ではないことは確定した。
  refusal 内訳に `too_many_outcomes` が無い。
- D（chance 系は全て正しい UNKNOWN）は shuffle については成立。
  ただし `OUTCOMES_NOT_ENUMERABLE` 19 件（うち positive 15）の
  probe 失敗・belief 再構成不能の中身は未特定。
- **E（新しい correctness 問題）が 1 つある**: `INCOMPLETE_ACTION_SET` 3 件。
  自作の action 生成が合法手を落としているなら chance とは無関係の重大問題。

したがって次に調べるべきは、実測が示す順で:

```
1. INCOMPLETE_ACTION_SET 3 件（correctness 懸念）
2. OUTCOMES_NOT_ENUMERABLE 19 件の probe 失敗理由（positive 15 件）
3. shuffle 系 25 件 → 現行 guard が正しいので、これ以上の拡張余地は無い
```

---

## Step 1-29: `INCOMPLETE_ACTION_SET` 3 件の原因特定

コード変更なし。

### 1. 発生地点

`cg_backend.legal_actions()` の中:

```python
for count in range(low, high + 1):
    for combination in itertools.combinations(indices, count):
        if len(actions) >= self._max_combinations:   # 既定 128
            complete = False
```

**engine が合法手を出せないのではなく、我々の generator が組合せを打ち切っている。**
`minCount`〜`maxCount` が広い選択（「22 枚から 19 枚選ぶ」等）で
組合せ数が上限 128 を超えると `complete=False` になる。

### 2. 3 件の実測

| fixture | engine options | min/max | 組合せ理論値 | 我々が生成 | root で incomplete | context | oracle_win | phase2 |
|---|---|---|---|---|---|---|---|---|
| **cap0027** | 22 | 19/19 | **1,540** | **128** | **Yes** | 8 | **False** | UNKNOWN |
| cap0175 | 2 | 1/1 | 2 | 2 | No（深い位置） | 4 | True (md=2) | UNKNOWN |
| cap0204 | 1 | 1/1 | 1 | 1 | No（深い位置） | 22 | True (md=3) | UNKNOWN |

| fixture | missing action | engine has? | ours has? | oracle result | impact |
|---|---|---|---|---|---|
| cap0027 | 22 枚中 19 枚を選ぶ組合せ 1,540 通りのうち 1,412 通り | **Yes** | **No（128 で打ち切り）** | **勝ち筋なし**（oracle_win=False） | **無し** |
| cap0175 | root では欠落なし。深いノードで発生 | – | – | WIN (md=2) | 能力低下のみ |
| cap0204 | root では欠落なし。深いノードで発生 | – | – | WIN (md=3) | 能力低下のみ |

### 3. Correctness への影響 — **健全性違反は無い**

| 懸念 | 判定 | 理由 |
|---|---|---|
| false `PROVEN_WIN` | **無し** | 行動が欠けても勝ちを**作る**ことはできない。見つけ損なうだけ |
| false `PROVEN_NO_WIN` | **無し** | `complete=False` は `saw_unknown=True` を立て、`PROVEN_NO_WIN` を出させない（Step 1-19 の契約） |
| 能力低下 | **有り** | cap0175 / cap0204 は oracle が positive（md 2 / 3）なのに UNKNOWN |

cap0027 は oracle が「勝ち筋なし」と判定しているので、
打ち切った 1,412 通りに勝ちは無い。実害は無い。

### 4. Pattern 判定

**Pattern A（自作 generator の欠陥）**。ただし「欠陥」というより
**意図的な安全上限（`_max_combinations=128`）が能力を制限している**状態。
engine API の制約（Pattern B）でも、fixture の誤り（Pattern C）でも、
engine のバグ（Pattern D）でもない。

上限を上げれば cap0027 は完全化できるが、C(22,19)=1,540 の組合せを
すべて展開する意味があるかは別問題（oracle は勝ち無しと判定済み）。
**今回は上限を変更しない。**

### 5. contract の確認

`action_set_complete = True` の意味は
**「engine が返した全 option から、上限内で組合せを列挙し切った」**。
上限で打ち切ったら `False` になり、`PROVEN_NO_WIN` は出ない。
この契約は Step 1-19 の NO_WIN 契約と整合しており、変更不要。

### 6. `OUTCOMES_NOT_ENUMERABLE` 19 件 — **未着手**

`INCOMPLETE_ACTION_SET` の特定に時間を使ったため、
positive 15 件を含む 19 件の根本原因調査（指示 12〜17）は**実施できていない**。
分かっているのは Step 1-28 までの範囲:

- `too_many_outcomes` = 0（outcome 数の上限型ではない）
- refusal 内訳に該当項目が無く、probe 失敗 / belief 再構成不能が疑われる
- R2 ON/OFF での first blocker 比較も未実施

### 7. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget overrun 0 / illegal 0 / leak 0 / B1 green / shuffle guard green / R2 green。
テスト: `1 failed（既存 B6）, 535 passed, 12 skipped`。

### 8. 判断: **A（action generator の問題）— ただし健全性は保たれている**

- 原因は engine ではなく我々の組合せ上限。
- **安全性は壊れていない**（false PW / false PNW とも 0、契約どおり UNKNOWN へ倒れる）。
- 影響は能力低下に限られ、しかも 3 件中 1 件は勝ち筋が存在しないので実害ゼロ。
- したがって **P0 ではない**。次の優先は `OUTCOMES_NOT_ENUMERABLE` 19 件
  （positive 15 件）の根本原因調査。

---

## Step 1-30: `OUTCOMES_NOT_ENUMERABLE` の根本原因 — **`_max_outcomes=24` による打ち切り**

### 0. Step 1-28 の結論を訂正する

Step 1-28 で「`too_many_outcomes` = 0 なので outcome 数の上限型ではない」と報告した。
**これは誤りだった。** 根拠にした `CgBackend.refusals` は、
outcome 上限の分岐で**カウンタを加算していない**。
カウンタが無いことを「その事象が起きていない」と読み替えたのが誤り。

実際の分岐は `cg_backend.enumerate_outcomes()` の末尾:

```python
if len(outcome_set.outcomes) > self._max_outcomes:   # 既定 24
    return OutcomeEnumeration(None, ..., StopReason.OUTCOMES_NOT_ENUMERABLE)
```

**`OUTCOMES_NOT_ENUMERABLE` は outcome 数の上限そのものだった。**

### 1. 実測

`_max_outcomes = 24`。R2 ON・budget 500ms・valid 185 件:

| 項目 | 値 |
|---|---|
| 打ち切られた fixture | **31 件**（うち **positive 15**） |
| 拒否イベント | 271 回 |
| 拒否された outcome 集合サイズ | 最小 **25** / 中央 **92** / 最大 **4,066** |

| outcome サイズ | 拒否回数 |
|---|---|
| 17-32 | 1 |
| 33-64 | 37 |
| **65-128** | **106** |
| **129-256** | **70** |
| 257-512 | 4 |
| 513-2048 | 49 |
| >2048 | 4 |

### 2. Root cause 分類（指示 3/15）

| root cause | 件数（拒否イベント） | positive fixture | safe fix | API limitation |
|---|---|---|---|---|
| **OUTCOME_COUNT_CAP（`_max_outcomes=24`）** | **271** | **15** | **可能**（上限を上げるだけ。情報境界は不変） | 無し |
| BELIEF_RECONSTRUCTION（belief=None） | 0 | 0 | – | – |
| OUTCOME_CONSTRUCTION（materialize 失敗） | 0 | 0 | – | – |
| PROBABILITY_COMPLETENESS（mass<1） | 0 | 0 | – | – |
| PROBE_FAILURE | 0 | 0 | – | – |

**belief 再構成も outcome 構築も確率完全性も、1 件も失敗していない。**
Step 1-28 で疑った「probe 失敗 / belief 再構成不能」は**すべて否定**された。
`_deck_multiset()` は 271 回すべてで正常に belief を返している
（過少計上の再発も無い）。

### 3. これは Step 1-22 の「大規模 outcome へ到達不能」と繋がる

Step 1-22 で「65+ outcome は自然に生成できるが、探索パイプラインの手前で
止まるので到達不能」と記録した。その「手前で止まる」正体がこれ。
実際には **outcome 集合は正しく構築できており、サイズが 24 を超えた瞬間に捨てている**。

拒否された集合の中央値は 92、最大 4,066。
つまり Phase 2 は **65〜256 outcome の領域に日常的に到達している**が、
上限 24 で入口を閉じているだけだった。

### 4. capability impact — **未測定**

`_max_outcomes` を上げたときに positive 15 件のうち何件が
`PROVEN_WIN` になるかは**測っていない**。理由:

- 上限を上げると materialization が outcome 数に比例して増える
  （実測 0.61〜0.68 ms/outcome）。92 outcome なら約 60ms、
  4,066 outcome なら約 2.7 秒で、500ms 予算では確実に `TIME_LIMIT` になる。
- したがって「上限を上げれば解ける」とは限らず、
  **上限・予算・materialization コストの三者を同時に測る必要がある**。
- 指示 18 の「原因を特定するまで max_outcomes 拡張を実装しない」に従い、
  今回は測定用の変更も本番コードへ入れていない。

### 5. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget overrun 0 / illegal 0 / leak 0 / B1 green / shuffle guard green / R2 green。
Phase2-only 9 件は 9/9 replay green。
テスト: `1 failed（既存 B6）, 535 passed, 12 skipped`。

上限による拒否は `UNKNOWN` へ倒れており、**部分集合を証明に使っていない**
（`draw_outcomes` は完全集合を作ってからサイズ判定している）。
安全側の挙動としては正しい。

### 6. 判断: **D → 訂正して「上限設計の問題」**

指示 21 の選択肢に対する実測の答え:

- A（belief reconstruction 修正が最優先）— **否**。失敗 0 件。
- B（outcome construction 修正）— **否**。失敗 0 件。
- C（probability completeness）— **否**。失敗 0 件。
- D（全て安全側の正しい拒否）— **形式的には正しいが、これは「能力の上限設定」であって
  情報境界による必然的な拒否ではない**。B1 や shuffle とは性質が違う。
- E（複数要因）— **否**。単一要因。

正確には **「`_max_outcomes=24` という能力パラメータが、
positive 15 件を含む 31 fixture を塞いでいる」**。
情報境界の問題ではないので、**安全に緩められる余地がある唯一の blocker**。

ただし §4 のとおり、緩めても materialization コストで予算に当たる可能性が高い。
次に必要なのは実装ではなく、
**`_max_outcomes` × budget × materialization コストの三次元スイープ**である。

---

## Step 1-31: `max_outcomes` × budget スイープ — **上限を上げると能力が下がる**

ベンチマーク専用にインスタンス属性 `_max_outcomes` を差し替えただけで、
**本番既定値・本番コードは未変更**。R2 は監査でのみ ON。

### 1. max_outcomes × budget（valid 185 件、Phase 2 `PROVEN_WIN` 数）

| max_outcomes | 100ms | 200ms | 500ms | 1000ms | 2000ms |
|---|---|---|---|---|---|
| **24（現行）** | **100** | **101** | **104** | **105** | **106** |
| 64 | 99 | 101 | 104 | 104 | 106 |
| 128 | 96 | 98 | 103 | 104 | 105 |
| 256 | 94 | 96 | 101 | 103 | 105 |
| 512 | 91 | 93 | 98 | 100 | 102 |

**全予算で、上限を上げるほど `PROVEN_WIN` が単調に減る。**
24 が最良（または同率最良）で、512 では 500ms で 104 → 98 と 6 件失う。

### 2. 停止理由の推移（500ms）

| max_outcomes | WIN | NO_WIN | UNKNOWN | TIME_LIMIT | NOT_ENUM |
|---|---|---|---|---|---|
| 24 | **104** | 18 | 63 | 34 | 18 |
| 64 | 104 | 18 | 63 | 34 | 16 |
| 128 | 103 | 18 | 64 | **36** | 14 |
| 256 | 101 | 18 | 66 | **40** | 8 |
| 512 | 98 | 18 | 69 | **43** | 8 |

`OUTCOMES_NOT_ENUMERABLE` は 18 → 8 へ確かに減る。
しかし減った分がそのまま `TIME_LIMIT` 34 → 43 へ移り、**さらに WIN を 6 件削っている**。

大きな outcome 集合の materialization（実測 0.61〜0.68 ms/outcome、
中央 92 outcome なら約 60ms、最大 4,066 なら約 2.7 秒）に予算を吸われ、
**他の枝で見つかるはずだった勝ちまで探索できなくなる**のが原因。

### 3. positive 15 件について

`_max_outcomes` に塞がれていた positive 15 件は、
**上限をどこまで上げても回収できない**。
上限 512（拒否された集合の中央 92・最大 4,066 の大半をカバー）でも
WIN は増えず、むしろ全体で減少した。

### 4. 未実施

指示 7/8/9 のコスト分解（enumeration / materialization / search / validation の
内訳と 500ms で処理可能な outcome 数の実測モデル）は、
スイープ本体の実行時間を優先したため**今回は出力できていない**。
ただし §2 の結果から、コストモデルを作るまでもなく
「上限を上げる方向に価値が無い」ことは確定した。

### 5. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget overrun 0 / illegal 0 / leak 0 / B1 green / shuffle green / R2 green /
Phase2-only 9 件 9/9 green。
上限 512 でも `PROVEN_NO_WIN` は 18 件で不変 = **部分集合を証明に使っていない**。
テスト: `1 failed（既存 B6）, 536 passed, 11 skipped`。

### 6. 判断: **B（上限を上げても `TIME_LIMIT` へ移るだけ）— しかも実際には悪化する**

- A（24→64/128 で新能力が増える）— **明確に否定**。64 で同数、128 以上で減少。
- C（能力は増えるが 500ms では非実用）— **否定**。1s・2s でも増えない。
- D（dynamic / adaptive outcome limit が必要）— **根拠が無い**。
  上限を動的にしても、大きな集合を処理すること自体が損なので、
  「もっと処理する」方向の適応に価値は無い。
- E（別の blocker が支配的）— 部分的に真だが、今回の測定の主結論ではない。
- **B が正解。ただし「移るだけ」より悪く、上限を上げると純減する。**

**`_max_outcomes = 24` は現行コーパス・現行予算では最適に近い。**
Step 1-30 で「安全に緩められる余地がある唯一の blocker」と書いたが、
**緩める価値は無かった**。既定値は変更しない。

これで、Phase 2 の UNKNOWN を減らす方向の候補は
B1（緩和不可）・shuffle（正しい拒否）・outcome 上限（上げると悪化）・
深さ（真の不足 0 件）・時間（10 倍でも 30/34 が未解決）が
**すべて否定**されたことになる。

---

## Step 1-32: materialization アーキテクチャの API 限界調査

コード変更なし。

### 1. API capability（実測）

| Capability | Available | Evidence | outcome branching に使えるか |
|---|---|---|---|
| State clone | **No** | export 13 個に該当なし（Step 1-16 で PE export table を直接解析） | – |
| Snapshot / restore | **No** | 同上。RNG 状態の取得・設定も無い | – |
| **同一 parent からの child branch** | **Yes** | 同一 node から `search_step` を 3 回連続で実行して全て成功、parent も生存 | **できるが不十分**（§2） |
| Observation injection | **No** | `SearchBegin` の引数はデッキ・サイド・手札・バトル場のみ。中間 observation を渡す入口が無い | – |
| Deck-only injection | **SearchBegin 時のみ** | `your_deck` は `search_begin` の引数。開始後に差し替える API は無い | – |
| Shared prefix → 複数 outcome | **No** | §2 | – |
| Same-session branch | **Yes（決定的遷移のみ）** | 上と同じ | 決定的な選択には使えるが、**引くカードを指定できない** |
| 複数 session の同時保持 | **No** | `SessionNestingError: agent_ptr is shared` — ネストすると例外 | – |

### 2. なぜ prefix を共有できないのか（構造的理由）

`search_step(search_id, select)` の引数は**選択肢のインデックスだけ**。
「どのカードを引くか」を指定する入口が無い。
outcome（= 引かれるカードの multiset）を決める唯一の手段は
**`search_begin` に渡す `your_deck` の並び**である。

したがって:

```
outcome A を作る -> your_deck を A の順に並べて search_begin -> prefix を replay
outcome B を作る -> your_deck を B の順に並べて search_begin -> prefix を replay
```

は避けられない。parent から `search_step` で分岐できても、
**その分岐先で何を引くかは既に search_begin 時点で決まっている**ので、
共通 prefix を 1 回だけ実行して複数 outcome へ分岐することはできない。

さらに `agent_ptr` が共有されているため**セッションを同時に 2 つ持てない**。
「parent を保持したまま別 outcome を構築する」実装も不可能。

### 3. Cost 分解（実測・中央値）

| Component | ms |
|---|---|
| `SearchBegin` + セッション初期化/解放 | **0.243** |
| `search_step` 1 回（深さ 1） | 0.093 |
| `search_step` 1 回（深さ 2） | 0.125 |

```
prefix 長 k の materialization ≈ 0.243 + k × 0.093 ms
  k=1 -> 0.336 ms    k=3 -> 0.522 ms
```

Step 1-8 の 0.61〜0.68 ms/outcome と整合する（k=4〜5 相当）。
**`SearchBegin` が k=3 で全体の 46% を占める。**
仮に prefix を共有できれば outcome あたり 0.09〜0.13ms まで下げられ、
**4〜5 倍の改善**になる。しかし §2 のとおり API 上不可能。

### 4. 判断: **D（API 上すべて不可 → materialization bound を確定）**

- A（clone / branch API あり）— child branch は可能だが、
  **引くカードを指定できない**ので outcome branching には使えない。
- B（snapshot / restore あり）— 存在しない。
- C（state injection あり）— 存在しない。
- **D が正解。**

正式に確定する:

```
Phase 2:
    correctness  = good
    capability   = limited
    scalability  = materialization bound（engine API の構造的制約）
```

理由は「実装不足」ではなく **`search_step` が引くカードを指定できず、
`search_begin` の deck 供給が唯一の outcome 制御手段であること**。
将来 SDK に state clone / snapshot / mid-search deck injection が
追加されない限り、この上限は動かない。

### 5. `_max_outcomes = 24` を現行 baseline として固定

Step 1-31 の実測（24 が全予算で最良、128 以上で単調悪化）に基づく。
**「24 が理論最適」ではなく「現在の corpus / budget / materialization
アーキテクチャでは最も良い測定値」**として記録する。
32/40/48 等の細かい調整は行わない（単調悪化傾向が確認済み）。

### 6. Phase 2 の正式な位置づけ

- Phase2-only 9 件、9/9 replay green、うち 7 件が chance / reveal 系
- R2 で新規 `PROVEN_WIN` 3 件（replay 3/3 green）
- `max_outcomes` を上げても総 WIN は増えない（むしろ減る）
- B1 / shuffle は安全上扱えない（緩和不可）

**Phase 2 は安全に追加能力を提供できるが、materialization コストによって
探索範囲が構造的に制限される。**

### 7. Safety

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget green / illegal 0 / leak 0 / B1 green / shuffle green / R2 green /
Phase2-only 9/9 green。R2 既定 OFF・Phase 1/2/3 既定 OFF・
RNG non-interference NOT VERIFIED を維持。

---

# Step 1-33: 最終結論（Step 1-1 〜 1-32 のまとめ）

## 結論

> **Phase 2 の能力上限は solver の実装不足ではなく、native engine API の構造的制約である。**

outcome を具体化する唯一の手段が `SearchBegin(your_deck=<順序>) + prefix replay` であり、
state clone・snapshot/restore・outcome injection・mid-search deck 差し替え・
shared prefix branching・同時セッションが**すべて存在しない**（Step 1-32 で実測）。

## Phase 2 の最終能力表

| Capability | Status | Evidence |
|---|---|---|
| deterministic search | **proven** | valid 185 件・exact positive 80 件 |
| AND opponent choice | **positive verified** | B4 positive 22 件。negative（Case B）は 0 件で**未検証** |
| known listing decision (R2) | **safe** | 115/115 applicable、positive 84/84、replay 3/3 |
| chance enumeration | **partially supported** | Phase2-only 9 件（うち 7 件が chance/reveal） |
| large outcome | **structurally bounded** | materialization 0.6ms/outcome、上限を上げると WIN が減る |
| root deck reveal (B1) | **unsupported safely** | 供給デッキを変えても listing 不変 12/12 = 実デッキ由来 |
| post-shuffle draw | **unsupported safely** | 拒否 1,886/1,886 が正しい |
| multi-coin (B8) | capability exists, deferred | engine 表現は確認済み、positive fixture 0 |

## Phase 2 の UNKNOWN 68 件の最終分類（R2 OFF / 500ms / valid 185）

Phase 2: `PROVEN_WIN` 101 / `PROVEN_NO_WIN` 16 / `UNKNOWN` 68。

| 根本原因 | 件数 | 割合 | 分類 |
|---|---|---|---|
| B1（root デッキ公開） | 21 | 31% | **Information boundary** |
| shuffle 後の順序 | 5 | 7% | **Information boundary** |
| `max_outcomes` 上限 | 18 | 26% | **Engine API limitation 由来の安全上限** |
| `max_combinations` 上限 | 3 | 4% | 意図的な安全上限 |
| `UNSUPPORTED_EFFECT`（R2 OFF のため） | 6 | 9% | R2 ON で 0 になる |
| chance depth 上限 | 6 | 9% | Solver capability |
| 予算切れ | 6 | 9% | Solver capability |
| 深さ上限 | 3 | 4% | Solver capability |

集約すると:

```
Information boundary      26 件 (38%)  -> 緩和しない（正しい拒否）
Engine API limitation     21 件 (31%)  -> SDK 改善待ち
R2 で解決可能              6 件  (9%)  -> 実装済み・既定 OFF
Solver capability         15 件 (22%)  -> 唯一の改善余地
```

**「Phase 2 が弱い」のではない。** UNKNOWN の 78% は情報境界・API 制約・
意図的な安全上限であり、探索アルゴリズムの不足は 22% にすぎない。

## 実測で否定された仮説（すべて記録として残す）

| 仮説 | 結果 | Step |
|---|---|---|
| depth を上げれば解ける | **真の深さ不足 0 件** | 1-26 |
| 時間を増やせば解ける | 10 倍でも 30/34 が未解決 | 1-26 |
| action ordering が主因 | root 分岐 平均 6.4〜8.9 で否定 | 1-26 |
| R2 が探索を巨大化させる | root 分岐 8.9 で不変、p50 は改善 | 1-26 |
| `max_outcomes` を上げれば能力が増える | **単調に悪化**（512 で −6 件） | 1-31 |
| B1 に安全なサブケースがある | **0 件**（12/12 実デッキ由来） | 1-27 |
| shuffle guard が過剰拒否 | **1,886/1,886 が正しい拒否** | 1-28 |
| `UNSUPPORTED_EFFECT` は未対応カード効果 | **R2 が OFF だっただけ**（47→0） | 1-24/1-25 |
| `OUTCOMES_NOT_ENUMERABLE` は列挙不能 | **`max_outcomes=24` の打ち切り** | 1-30 |
| `INCOMPLETE_ACTION_SET` は correctness 問題 | **意図的な上限、健全性違反なし** | 1-29 |
| 探索が本番 RNG を汚染している | **誤り**。`BattleStart` が非決定的なだけ | 1-16 |
| Phase 2 は 9 秒かかる | **予算制御バグ**。修正後 535ms | 1-19 |
| Phase 2 は Phase 1 より 17 件強い | **同一条件では 9 件** | 1-20 |

## 修正した健全性問題

| 問題 | 内容 | Step |
|---|---|---|
| **P0-1** | 未展開の reveal 枝を `PROVEN_NO_WIN` にしていた（false PROVEN_NO_WIN 2 件） | 1-19 |
| **P0-2** | 列挙に予算チェックが無く 500ms 指定で 9,186ms | 1-19 |
| 診断汚染 | 反復深化の停止理由が反復をまたいで累積 | 1-20 |
| toy oracle | 本番と同じ誤りを共有し P0-1 を検出できなかった | 1-19 |
| SHUFFLE 順序 | 同一 step 内の DRAW→SHUFFLE を誤拒否（167 outcome） | 1-7a |
| `_deck_multiset` | 過少計上（false-proof vector） | 1-9b |

## Production status（正式）

```
Phase 1: correctness good / capability limited / production NOT SAFE
Phase 2: correctness good / capability meaningful offline /
         scalability engine-API limited / production NOT SAFE
Phase 3: not implemented / not evaluated
R2     : safe mechanism confirmed, default OFF
B1     : unsupported by information boundary
Shuffle: unsupported after order becomes hidden
RNG    : non-interference NOT VERIFIED
```

## Safety（最終）

false `PROVEN_WIN` 0 / false `PROVEN_NO_WIN` 0 / NO_WIN 契約 green /
budget overrun 0 / illegal 0 / leak 0 / oracle replay 0 failure /
Phase2-only 9 件 **9/9 replay green（必須回帰として `test_phase2_capability_golden.py` に固定）**。

テスト: `1 failed（既存 B6 = value_model golden）, 555 passed, 11 skipped`。
lethal 由来の失敗 0。

## 次の判断: **C（SDK/API 改善要求として整理）を主、D（研究成果としてまとめ）を従**

判断基準は「現在の Phase 2 をさらに弄ることで、本当に新しい能力が得られるのか」。

- **A（Phase 3 が materialization bound を回避できるか調査）— 否**。
  Phase 3 は value/policy による評価であって、chance ノードの
  outcome を具体化する必要性は変わらない。同じ
  `SearchBegin + replay` が要る。**現在のボトルネックを解決しない。**
  さらに Phase 3 は確定証明ではないので、
  false `PROVEN_WIN` = 0 という最優先条件とは別の評価軸が要る。
- **B（カード能力拡張へ進む）— 否**。GUST / SEARCH / COIN の
  positive fixture が 0 件のままで、実装しても評価できない（Step 1-23）。
- **C**: 唯一の実効的な打開策は SDK 側。優先順位は
  capability report §14.1 に記録済み（state clone > outcome injection >
  shared prefix > RNG 制御）。**RNG 制御は本番評価そのものの前提条件**でもある。
- **D**: 現時点の成果（3 値証明・情報境界の形式化・独立オラクル・
  205 件の能力コーパス・否定された仮説群）は、それ自体が記録価値を持つ。

**「UNKNOWN を減らすこと」を目的にしない。** UNKNOWN の 78% は
情報境界・API 制約・意図的な安全上限であり、減らすことは
安全性を損なうか、実測上むしろ能力を下げる。

---

# Step 1-34: 凍結と整理（コード変更なし）

## 実測で否定された仮説（研究記録）

| 仮説 | 測定 | 結果 | 結論 |
|---|---|---|---|
| depth を上げれば UNKNOWN が解消 | 8 件に depth 14 / 5s / 300k nodes | **真の深さ不足 0 件** | 深さは主因ではない |
| 時間を増やせば解ける | 34 件に予算 10 倍 | 30/34 が未解決 | 時間も主因ではない |
| action ordering が主因 | root 分岐を計測 | 平均 6.4〜8.9 | 順序で救える候補数が無い |
| R2 が branching explosion を起こす | R2 ON/OFF の root 分岐 | 8.9 で**不変**、p50 は 6.8→6.0 で改善 | 否定 |
| `max_outcomes` を増やせば強くなる | 24/64/128/256/512 × 5 予算 | **単調悪化**（500ms で 104→98） | 逆効果 |
| belief reconstruction が主因 | 271 回の拒否を分解 | 失敗 **0 件** | 否定 |
| outcome construction が主因 | 同上 | 失敗 **0 件** | 否定 |
| probability completeness が主因 | 同上 | 失敗 **0 件** | 否定 |
| shuffle guard が過剰拒否 | 1,886 回の拒否時 state を確認 | **1,886/1,886 が経路上 shuffle 済み** | 現行 guard は正しい |
| B1 を R2 として扱える | 供給デッキを変えて listing を比較 | **12/12 で不変**（実デッキ由来） | 緩和不可 |
| chance depth を増やせば解ける | `chance_depth` 1→3 | WIN 4 / NO_WIN 7 のみ | 主因ではない |
| state sharing で chance 直下が高速化 | InfoKey 重複率 | **0%** | 共有できる状態が無い |
| 大量 outcome 自体が生成不能 | deck 22 種で k=2/3 を計算 | 249 / 1,932 と**自然に生成される** | 生成は可能。上限で捨てていた |
| `UNSUPPORTED_EFFECT` は未対応カード効果 | 115 件の発生地点を特定 | **R2 が OFF だっただけ**（47→0） | 過剰拒否 |
| `OUTCOMES_NOT_ENUMERABLE` は列挙不能 | 拒否時の outcome 数を計測 | **`max_outcomes=24` の打ち切り**（中央 92） | 上限の問題 |
| `INCOMPLETE_ACTION_SET` は correctness 問題 | 3 件を分解 | `max_combinations=128` の意図的上限 | 健全性違反なし |
| 探索が本番 RNG を汚染している | control 同士を比較 | **`BattleStart` が非決定的**（4/4 別軌跡） | 推論が誤り |
| Phase 2 は 9 秒かかる | 予算制御を修正 | 500ms 指定で 535ms | 予算バグだった |
| Phase 2 は Phase 1 より 17 件強い | 完全同一条件で再測定 | **9 件** | 条件違いの比較だった |

## 研究成果

| # | 成果 | 内容 |
|---|---|---|
| ① | **3 値確定証明** | `PROVEN_WIN` / `PROVEN_NO_WIN` / `UNKNOWN`。NO_WIN は「完全に展開し勝ち枝が無いことを確認した」場合のみ |
| ② | **情報境界の形式化** | root deck（B1）/ hidden shuffle order / known listing（R2）を、実測に基づいて分離 |
| ③ | **chance outcome の完全性** | 部分集合・質量 1 未満を証明に使わない。列挙打ち切りは `None` を返す |
| ④ | **opponent-choice AND semantics** | 相手選択を全列挙。有利な選択だけを見る実装を排除 |
| ⑤ | **engine capability probing** | 推測せず実測。PE export table の直接解析、供給デッキ差し替え probe |
| ⑥ | **独立オラクル** | solver を一切 import しない ground truth（`lethal_oracle` / `b4_oracle`）。P0-1 はこれが無ければ発見できなかった |
| ⑦ | **engine API limitation の特定** | state clone / outcome injection / shared prefix / RNG control の欠如を実証 |

## 正式な評価

> **安全な chance-aware 確定探索を構築でき、Phase 1 には無い追加能力を確認できた。
> 一方、大規模 chance branching の scalability は native engine API の
> state / materialization 制約によって制限される。**

Phase 2 を「失敗」とは評価しない。

## 成果物

- `step0-capability-report.md` — engine/API 能力と限界（§11 RNG、§14 Outcome Branching）
- `design.md` — 付録 W（凍結 baseline）/ X（UNKNOWN taxonomy）/ Y（max_outcomes の意味）/ Z（本番接続条件）
- `step1-progress.md` — 本書
- **`api-improvement-proposal.md`（新規）** — State Clone / Outcome Injection / Shared Prefix / RNG Control
