# PIMC本番検証 まとめ(Step5): 判断のとりまとめ

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [pimc-production-validation-implementation-plan.md](../plans/individual/shogo/pimc-production-validation-implementation-plan.md) Step5(Definition of Done)
- 依拠する計測:
  - Step2 主比較 → [2026-07-21_pimc_prod_validation_main.md](2026-07-21_pimc_prod_validation_main.md)
  - Step3 分離比較 → [2026-07-21_pimc_prod_validation_hiddeninfo.md](2026-07-21_pimc_prod_validation_hiddeninfo.md)

## 答えるべき問い

「PIMC(および estimated hidden state)を本番 `ml_policy` に入れたら、提出物は強くなるのか?」

## 計測サマリ(すべてミラー `deck.csv`、100試合、エラー0件、Wilson 95% CI)

| 比較 | A(基準) | B | A勝率 | A 95% CI | 判定 |
|---|---|---|---|---|---|
| Step2 主比較 | `ml_lethal`(現行提出) | `ml_pimc`(pimc+estimated) | 0.520 | [0.423, 0.615] | **有意差なし**(CIが0.5を含む) |
| Step3 分離比較 | `ml_lethal`(現行提出) | `ml_lethal_estimated`(estimated単独) | 0.550 | [0.452, 0.644] | **有意差なし**(CIが0.5を含む) |

(Step4 の full attribution=`ml_pimc_dummy`(pimc単独)は任意項目。Step2/Step3 の両軸が null だったため実施せず。必要なら追試可能。)

## 明示的な判断ルールへの当てはめ(計画 Step5)

1. ~~`ml_pimc` が `ml_lethal` に有意に勝ち越し → 提出候補~~ → **不成立**(Step2 有意差なし)。
2. ~~`ml_lethal_estimated` が有意に勝ち越し、かつ `ml_pimc` がそれ以下 → より安い提出候補~~ → **不成立**(Step3 有意差なし。点推定はむしろ dummy 側が高い)。
3. **どれも `ml_lethal` に有意差なし → 研究の勝ち(vs lethal)が本番に転移しなかった。** → **これに該当。**

## 結論

**現時点で PIMC・estimated hidden state のいずれも、本番 `ml_policy` の置き換えとして提出する数値的根拠は得られなかった。** Stage2 で rule_based フレームワークでは明確だった「PIMC > lethal」(61% vs 39%)が、`ml_policy` の上のミラーマッチでは消えた(52% vs 48%、有意差なし)。

## 転移しなかった理由(仮説、要検証)

1. **PolicyModel が既に強く、探索の発火域が狭い。** `ml_policy` は通常ターンの意思決定の大半を模倣 PolicyModel が担い、lethal/pimc が効くのは終盤の確定リーサル/詰め局面に限られる。その狭い領域で pimc が lethal に勝る分の利得が、平均13.7ターンの試合全体の勝率にはほとんど波及しない。rule_based では router(相対的に弱い)が広く意思決定していたため、探索の質差が勝率に大きく効いた——というフレームワーク依存の差。
2. **ミラーマッチは相手モデリングの利得が出にくい。** PIMC の本質的な強みは相手の非公開情報に対する期待値最適化だが、両陣営が同一デッキ・同一方策だと読むべき相手の分散が小さく、determinization の価値が薄まる。
3. **先手ゲーの分散が大きい。** Step3 の内訳は先手 0.68 / 後手 0.42 と強い手番有利を示す。ミラーで方策差が小さいと、勝敗が手番運に支配されて config 間の実力差が埋もれる。

## 次アクション(v1 の前提の見直し)

本計画は「提出すべきかの判断材料を出す」ところまでがスコープ。以下は人間(ユーザー)判断:

- **v1(policy prior 統合・相手ターンをまたぐ探索)の前提が変わった。** 「研究で PIMC が勝った」を根拠に v1 へ投資する前提は、本番転移が確認できなかったことで弱まった。v1 に進むなら、まず「探索の発火域を広げる」「非ミラー(多様な相手デッキ)で測る」のどちらかで転移条件を特定してからが妥当。
- **非ミラー評価は本計画の Non-goal(`match_context._load_own_deck_ids()` が常に `deck.csv` を読む制約)。** 仮説2の検証には、両陣営に別デッキを割り当てても自分デッキ推定が壊れないインフラ対応が前提になる。これが次に効く投資の候補。
- **代替の優先課題**: 実戦の主要敗因である山札切れ対策([[project_deckout_loss_cause]])を先に対応する選択肢もある。PIMC の本番寄与が小さい以上、勝率に直結する敗因潰しの方が期待値が高い可能性。

どちらに進むかはユーザーと相談して決める。

## Definition of Done(計画)

- [x] `ml_policy_agent` に後方互換の config 注入点があり、既存の `agent(obs)` 呼び出しは不変
- [x] 同一プロセス内で別configが独立して効くことの回帰テスト追加(`test_explicit_config_is_independent_per_call`)
- [x] `ml_lethal` vs `ml_pimc`(主比較)の結果を `results/` に記録
- [x] `ml_lethal` vs `ml_lethal_estimated`(hidden info単独)の結果を `results/` に記録
- [x] 明示的な判断ルールに基づく結論を本 summary に記載
- [ ] `ml_policy_agent.py`/`selector.py` への累積の追記内容を tsuoimorikaさんに共有・合意(Stage1からの持ち越し。人間側対応、未実施)
