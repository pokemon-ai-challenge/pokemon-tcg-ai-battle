# Skill Concentration — Step3: オフライン評価(上位≤200位 held-out Top-1一致率)

作成: 2026-07-21
関連: `sample_submission/docs/plans/individual/shogo/policymodel-skill-concentration-implementation-plan.md`
実行: `kaggle_replays/policy_net/evaluate_skillconc.py`(結果JSON: `kaggle_replays/policy_net/evaluate_skillconc_results.json`)

## 評価セット

既定(フィルタなし)の `features.npz`(187,690件、Step2で `rank_at_fetch` を追加して再ビルド)の
**test split(22,626件)のうち `rank_at_fetch` が 1〜200 の3,408件**を共通held-outとして使う。
split は episode_id の md5 のみで決まるため、A/B/Cどの構成で学習しても同一のheld-out集合で
公平に比較できる。

## 学習結果サマリ(各構成、自身のtrain/val/testでの学習ログ)

| 構成 | 学習データ件数 | 自身のtest top1(本命モデル) | 早期終了epoch |
|---|---:|---:|---:|
| baseline(現行、既存 policy_weights.json) | 187,690(既定重み) | 0.59(既存記録、[[project_ml_value_network_step1]]系) | — |
| A(rank<=200フィルタ) | 26,478 | 0.6218 | 20 |
| B(rank<=1000フィルタ) | 90,079 | 0.6027 | 17 |
| C(全データ+濃縮リウェイト) | 187,690 | 0.5806 | 24 |

いずれも val/test が大きく乖離する明確な過学習の兆候は無し(A は Step1で確認した
プレイヤー偏重リスクにもかかわらず val=0.622 / test=0.622 で一致しており、乖離は見えない)。

## 本題: 上位≤200位 held-out Top-1一致率(4者比較、共通評価セット)

| 候補 | top200_holdout top1 | 現行比 | pool test top1(全体) | 現行比 |
|---|---:|---:|---:|---:|
| **baseline_current(現行)** | 0.6080 | — | 0.5923 | — |
| A(rank<=200フィルタ) | 0.6218 | **+1.38pt** | 0.5636 | -2.87pt |
| B(rank<=1000フィルタ) | 0.6156 | **+0.76pt** | 0.5856 | -0.67pt |
| C(全データ+濃縮リウェイト) | **0.6229** | **+1.49pt** | 0.5806 | -1.17pt |

## 解釈

- **3構成すべてが現行より上位決定の再現率を上げる方向**(技量集中というレバー自体は機能する)。
  これは計画の仮説の方向性を支持する。
- ただし伸び幅は **+0.8〜+1.5pt** に留まり、計画の期待(勝率換算で自チーム44.6%→上位帯61%相当の
  「天井を引き上げる」規模)には遠く届かない。プール平均一致率(現行0.59)からのジャンプというより、
  「現行モデルも既に上位決定をある程度得意にしている(0.608)」ところからの微増、という実態。
- **C(全データ+濃縮リウェイト)が最良**: held-out改善が最大(+1.49pt)でありながら、volumeを
  落とさない分プール精度の犠牲が中程度(-1.17pt)。A(ハードフィルタ)は held-out改善がCとほぼ
  同等だが、volume減(26k)とプレイヤー偏重リスク(Step1、上位2人で60.7%)を抱えたままプール精度の
  犠牲が最大(-2.87pt)で、Cに対し優位性が無い。
- **B(rank<=1000)は最も保守的**: 改善も犠牲も小さい。データの多様性(261人)は最も高い。

## 次(Step4)への選定

**C を主候補、B を対照(保守的な代替)として、現行 `ml_lethal`(既定 `policy_weights.json`)との
ミラー head-to-head に進める。** A は上記の理由(Cに対しStrict優位性なし、過学習リスクは実測では
顕在化しなかったが依然構造的に高リスク)によりオンライン評価では優先度を下げる。
