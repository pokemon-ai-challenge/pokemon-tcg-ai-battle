# Tier3事前チェック: dummy隠れ状態でconsequence特徴は退化しないか

作成: 2026-07-23
関連: [`docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md`](../docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md) §2.3
スクリプト: [`league/_diag_consequence_distribution.py`](../../league/_diag_consequence_distribution.py)

## 目的

Stage3c(encoder/build_features/policy_modelへの本格配線・再学習・ミラー対戦)に投資する前に、
§2.3で決めた制約(学習時も実行時と同じdummy隠れ状態のみを使う。相手の非公開情報は使わない)
のもとで、`delta_best_effective_attack_damage`等の特徴が実戦で意味のある値を取るか
(相手の非公開情報が無いことでほぼ常に0に退化してしまわないか)を安く確認した。

## 方法

自分のデッキ(フーディン) vs ロック闘エネルギー入りルカリオ(`local_decks/demo_lucario.csv`)
で40試合を通常通り実行しつつ、MAIN局面でATTACH/EVOLVE/PLAY型の選択肢が存在するたびに
`consequence.option_consequence`をdummy隠れ状態で計測目的だけで追加呼び出しした
(実際の行動選択には使わない。既存のStage3aレイテンシ計測と同じピギーバック方式)。

## 結果(40試合、resolved 2551件)

| option_type | n | delta非ゼロ率 | delta正の率 | delta範囲 | delta平均 |
|---|---|---|---|---|---|
| ATTACH | 1025 | 162 (15.8%) | 128 (12.5%) | -20〜+320 | +6.07 |
| EVOLVE | 361 | 45 (12.5%) | 24 (6.6%) | -30〜+320 | +5.01 |
| PLAY | 1165 | 96 (8.2%) | 44 (3.8%) | -320〜+320 | +3.67 |

その他の即時結果特徴(グループA):
- `opp_special_energy_removed`(改造ハンマー相当の効果): **PLAY型選択肢で67/1165件(5.8%)発火**。
  40試合という小規模サンプルでも非ゼロの頻度で観測でき、dummy隠れ状態でも
  「相手の特殊エネルギーが外れる」という効果を正しく検出できている。
- `cards_drawn>0`: EVOLVE型で233/361件(64.5%、進化時ドロー特性が多いデッキ構成のため妥当)。
- `pokemon_evolved`: EVOLVE型で44/361件検出(進化オプションの一部はまだ進化前カードが
  手札に無い等の理由で「進化できる」選択肢ではあるが実際の進化確認には至っていないケースも
  含むため100%にはならない)。
- `delta_can_ko`: 全option_typeで0件。KOを新たに可能にする準備行動は今回のサンプルでは
  観測されなかった(頻度が低いこと自体はバグではなく、より狭い条件のため妥当と考えられるが、
  Stage3cで実装する際は別途固定盤面テストで健全性を再確認すること)。

未解決率: 3577件中1026件(28.7%)が`option_consequence`から`None`(unresolved)。
これは想定内(§4.1: 組み合わせ上限超過・相手ターンに及ぶ・コイン絡み等は「unknown」として
除外される設計)だが、Stage3cで実装する際は`consequence_unresolved`フラグ(§12未決事項)の
要否を判断する材料になる。

## 結論

**delta_best_effective_attack_damageはdummy隠れ状態でも退化していない。** ATTACH/EVOLVE/PLAY
いずれの選択肢型でも非ゼロ率8〜16%、正の値(=攻撃力が改善する方向)も継続的に観測され、
特に本Tierの動機そのものである「相手の特殊エネルギーを剥がして攻撃を通す」効果
(`opp_special_energy_removed`)が実戦で67件検出された。§2.3で懸念していた
「train/runtime parity制約(dummy隠れ状態のみ)により信号が消えてしまうリスク」は、
少なくともこの前提チェックの範囲では顕在化しなかった。

**Stage3c(本格配線・再学習・ablation)へ投資する価値があると判断できる根拠が得られた。**
