# Tier3 Stage3c Experiment B — 結果

作成: 2026-07-23
関連: [`docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md`](../docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md) §6 Stage3c / §7 Experiment B / §8 評価プロトコル

## 対象

グループA(即時結果)特徴のうち3つ: `opp_hp_loss` / `opp_energy_removed` / `opp_special_energy_removed`。
`delta_best_effective_attack_damage`(本Tierの主役、グループB)はExperiment Cで別途検証する
(§7の一括投入禁止方針)。

## パイプライン(Stage3a〜3cを一通り実装)

1. `ptcg_ai/board_evaluation/consequence.py`(Stage3a/3b、既存実装済み)
2. `ptcg_ai/learning/encoder.py::encode_option_consequence_features`(新規)
3. `kaggle_replays/policy_net/build_features.py --with-consequence-features`(新規フラグ、既定OFF)
4. `ptcg_ai/learning/policy_model.py`(`meta.consequence_fields`対応、旧重み完全後方互換。
   `score_options_from_state`はconsequence特徴を使う重みでは例外を出す設計にし、
   `search_begin_input`を持たない呼び出しでの誤用を防止)
5. `ptcg_ai/ml_policy/ml_policy_agent.py`(`_select_action`で`hidden_state_factory`/`deadline`を
   モデル呼び出しに常時スレッド。consequence特徴を使わない重みではコストゼロ)
6. `kaggle_replays/policy_net/train.py --consequence-fields`(新規フラグ、既定OFF)

いずれも既存テストスイート(単体テスト、全262件パス)・実エンジンでの回帰確認(45試合超、
エラー0件)を通した上で本番投入(既定OFF、後方互換)。

## データビルド

`policy_positions.jsonl.gz` 全187,690件を`--with-consequence-features`でビルド
(自分側のdummy隠れ状態はsample_submission/deck.csvを使用、§2.3のtrain/runtime parity要件)。

- 所要時間: 148.7秒、**エラー0件**
- 学習データの本体(`policy_features.npz`とは別に`consequence_features`列を追加。
  既存の`option_features`等は完全に無変更)

## 学習(同一データ・同一既定レシピでの clean ablation)

| 構成 | 入力次元(option) | test top1 | test nll |
|---|---|---|---|
| 統制群(consequence無し) | 65 | 0.5923 | 1.1711 |
| Experiment B(+3特徴) | 68 | **0.5934**(+0.11pt) | 1.1633 |

両方とも自己検証(pure_python_forward vs PyTorchモデル出力)は誤差1e-15オーダーでPASS。

**オフラインの差はごく僅か**(Tier1abcの経験(オフライン微増→実戦転移せず不採用)を踏まえ、
この時点では採用・不採用を判断しない。§8の通りミラー対戦を最終ゲートとする)。

## ミラー対戦(300試合、Wilson 95% CI)

対戦条件: 両側とも`config_base=ml_lethal_attackplan_v0only`(現行本番configベース)、
重み以外は完全に同一。先手/後手半々。

| | 結果 |
|---|---|
| 試合数 | 300(有効300、エラー0) |
| Experiment B 勝ち数 | 165 / 300 |
| Experiment B 勝率 | **55.0%** |
| Wilson 95% CI | **[49.3%, 60.5%]** |

**95% CIが50%をわずかに含む(下限49.3%)ため、統計的に有意な勝ち越しとは言えない。**
ただし点推定は55%とTier1abc(43.0%、有意に負け)とは対照的に明確にプラス方向であり、
Experiment Cを試す前にサンプルを増やして有意性を詰める価値がある水準。

## Experiment C(追記、2026-07-23)

本命特徴 `delta_best_effective_attack_damage` / `delta_can_ko` を同一データ・同一controlで検証。

| 構成 | test top1 | test nll |
|---|---|---|
| 統制群 | 0.5923 | 1.1711 |
| Experiment C(+2特徴) | **0.5944**(+0.21pt、Bの+0.11ptより大きい) | 1.1681 |

ミラー対戦(300試合、control vs Experiment C):

| | 結果 |
|---|---|
| Experiment C 勝ち数 | 164 / 300 |
| Experiment C 勝率 | **54.7%** |
| Wilson 95% CI | **[49.0%, 60.2%]** |

こちらもCIが50%をわずかに含み、単独では有意ではない。

## B・C比較

| 実験 | 特徴 | offline test top1差 | ミラー勝率 | 95% CI |
|---|---|---|---|---|
| B | opp_hp_loss, opp_energy_removed, opp_special_energy_removed | +0.11pt | 55.0% | [49.3%, 60.5%] |
| C | delta_best_effective_attack_damage, delta_can_ko | +0.21pt | 54.7% | [49.0%, 60.2%] |

**BとCが独立の実験でほぼ同じ勝率(55.0%/54.7%)に着地した。** offlineの伸び幅はCの方が
大きいにもかかわらず、ミラー勝率はほぼ同水準。2つの独立サンプルが同じ方向・同じ大きさに
揃ったことは、単なるノイズにしては偶然が過ぎる可能性があり、「真の効果量が5%pt前後で、
n=300では有意性を出すには足りない」という見方を支持する材料になる
(逆に、両方ともCIは50%付近まで重なっており、まだ確証があるとは言えない)。

## Experiment D(追記、2026-07-23)

`delta_attack_ready` / `delta_energy_shortfall` は設計書 §3.2 で定義されていたが、Stage3b
実装時にコーディングし忘れていたことが判明。`ptcg_ai/board_evaluation/consequence.py` に
`_pokemon_readiness`(静的計算、§5の方針通り仮実行不要)を追加して実装を完成させ
(単体テスト2件追加、`energy_requirements.energy_shortfall`を再利用)、features.npzを
新フィールド込みで再ビルド(187,690件、エラー0件)してから検証した。

| 構成 | test top1 | test nll |
|---|---|---|
| 統制群 | 0.5923 | 1.1711 |
| Experiment D(+2特徴) | **0.5896**(**-0.27pt、B/Cと逆方向**) | 1.1727 |

ミラー対戦(300試合、control vs Experiment D):

| | 結果 |
|---|---|
| Experiment D 勝ち数 | 105 / 300 |
| Experiment D 勝率 | **35.0%** |
| Wilson 95% CI | **[29.8%, 40.6%]** |

**CIが完全に50%を下回り、統計的に有意な負け越し。** offlineのマイナス方向(-0.27pt)と
ミラー対戦の負けが一致しており、B/Cの「弱いプラスだが不確実」とは違う、明確な悪化シグナル。
Tier1abc(43.0%)と同種の「不採用」判定になる。

**考察**: `delta_attack_ready`/`delta_energy_shortfall`は「エネルギーが足りているか」という
静的な充足判定であり、`delta_best_effective_attack_damage`(実際に通るダメージ)とは異なり
resistance/lock等の実効性を考慮しない。「エネルギーは足りたが結局ロックで0ダメージ」を
「準備完了」と誤って高評価してしまう可能性があり、これが悪化の原因という仮説は次の検証
候補になる。

## B・C・D比較

| 実験 | 特徴 | offline test top1差 | ミラー勝率 | 95% CI | 判定 |
|---|---|---|---|---|---|
| B | opp_hp_loss, opp_energy_removed, opp_special_energy_removed | +0.11pt | 55.0% | [49.3%, 60.5%] | 境界線(未確定) |
| C | delta_best_effective_attack_damage, delta_can_ko | +0.21pt | 54.7% | [49.0%, 60.2%] | 境界線(未確定) |
| D | delta_attack_ready, delta_energy_shortfall | **-0.27pt** | **35.0%** | **[29.8%, 40.6%]** | **不採用(有意に悪化)** |

B・Cの弱いプラス傾向とDの明確なマイナスが両方観測されたことで、「consequence特徴なら
何でも効く」わけではなく、**特徴の設計(特にエンジンの実効性判定を経由するかどうか)が
結果を大きく左右する**ことが分かった。B+C統合構成を試す場合はDを含めないのが妥当。

## delta_best_effective_attack_damage vs delta_can_ko の発火率(追記、2026-07-23)

Experiment Cの再学習の前に、学習データ全体(187,690行・1,455,680選択肢)で両特徴の
非ゼロ率を直接確認した。

| 特徴 | 非ゼロ選択肢数 | 非ゼロ決定点数 |
|---|---|---|
| `delta_best_effective_attack_damage` | 3,396 | 1,661(0.89%) |
| `delta_can_ko` | **1** | **1** |

`delta_can_ko`は実質ゼロ(19万行中たった1回)。**Experiment Cの54.7%は実質
`delta_best_effective_attack_damage`単体の効果とみてよい。**

## B+C統合構成(追記、2026-07-23)

| 構成 | test top1 |
|---|---|
| 統制群 | 0.5923 |
| B+C(5特徴) | 0.5934(+0.11pt、B単体・C単体より低い) |

ミラー対戦(300試合):

| | 結果 |
|---|---|
| B+C 勝ち数 | 122 / 300 |
| B+C 勝率 | **40.7%** |
| Wilson 95% CI | **[35.3%, 46.3%]** |

**B単体(55.0%)・C単体(54.7%)がどちらも弱いプラスだったのに、組み合わせると
有意なマイナスに反転した。** 想定される原因: `delta_can_ko`は学習データ上でほぼ定数
(1,455,680件中1件のみ非ゼロ)であり、標準化(train split標準偏差で割る)の際に
標準偏差が下限値(1e-6)でクリップされるため、**唯一の非ゼロ値だけが標準化後に
約100万倍という極端な値になる。** この外れ値が学習を不安定化させ、C単体では
目立たなかった悪影響が、他の特徴と組み合わさることで増幅された可能性がある。

## delta_best_effective_attack_damage 単体(追記、2026-07-23)

`delta_can_ko`(発火率ほぼゼロ)を除き、`delta_best_effective_attack_damage`だけの
1特徴構成で再検証した。

| 構成 | test top1 |
|---|---|
| 統制群 | 0.5923 |
| damage単体(1特徴) | 0.5927(+0.04pt、Cの+0.21ptより小さい) |

ミラー対戦(300試合):

| | 結果 |
|---|---|
| damage単体 勝ち数 | 141 / 300 |
| damage単体 勝率 | **47.0%** |
| Wilson 95% CI | **[41.4%, 52.6%]** |

**ほぼ完全にフラット(50%中心)。** 「Cの54.7%は`delta_best_effective_attack_damage`単体の
効果である」という仮説は**裏付けられなかった**。`delta_can_ko`を除いた途端に効果が消えた
ことになる。

## 全体像(5実験を並べて)

| 実験 | 特徴数 | ミラー勝率 | 95% CI | 判定 |
|---|---|---|---|---|
| B | 3(即時結果) | 55.0% | [49.3%, 60.5%] | 境界線 |
| C | 2(本命+can_ko) | 54.7% | [49.0%, 60.2%] | 境界線 |
| D | 2(エネルギー充足) | 35.0% | [29.8%, 40.6%] | 有意に悪化 |
| B+C | 5(統合) | 40.7% | [35.3%, 46.3%] | 有意に悪化 |
| damage単体 | 1(本命のみ) | 47.0% | [41.4%, 52.6%] | フラット |

5つの独立したミラー対戦(各300試合、異なるseed)の勝率は35.0%〜55.0%とばらついており、
**プラス方向に見えた構成(B・C)が、特徴を1つ削っただけ(C→damage単体)で消えたことは、
「真の効果がある」という説明よりも「個々の300試合の点推定にはこの程度のばらつきが
乗る」という説明の方が整合的。** 少なくとも今回試した特徴・組み合わせの中に、
安定して勝率を押し上げるものは見つからなかった。

## 結論と次のアクション

- **D・B+Cは明確に不採用**(有意に悪化)。
- **B・C・damage単体は「効果あり」と主張できる根拠が無い。** 境界線上に見えた結果は
  ノイズの範囲内である可能性が高い(1特徴に絞ったら消えたことがその傍証)。
- 現時点でTier3のconsequence特徴(グループA/B、Stage3cで検証した範囲)を本番採用する
  根拠は無い。**方針書の「本命」だった`delta_best_effective_attack_damage`単体もフラット**
  という結果は、Tier1abc同様「オフラインの僅かな伸びが実戦での安定した優位に転移しない」
  パターンの再現とみるのが妥当。
- 選択肢:
  1. ここでTier3(Stage3c以降)は一旦保留し、記録を残して撤退する。
  2. 統計的検出力を上げるため、有望に見えた構成(B等)の試合数を大幅に増やす
     (300→1000以上)。ただし今回の結果を踏まえると期待値は低い。
  3. 全く別の投資先(モデル容量拡張§9、Stage3bで未着手の静的ATTACH計算等)を検討する。

**実装自体(consequence.py Stage3a/3b、encoder/build_features/policy_model/ml_policy_agent
の配線)は既定OFF・後方互換のまま資産として残る**(§10)。今回の否定的結果は「この方向の
特徴設計では効果が確認できなかった」という知見であり、パイプライン自体は将来別の特徴を
試す際にそのまま再利用できる。
