# Skill Concentration — Step5: 集約サマリーと提出候補判断材料

作成: 2026-07-21
関連: `sample_submission/docs/plans/individual/shogo/policymodel-skill-concentration-implementation-plan.md`

## 結論

**提出候補あり: `policy_weights_configC.json`(全データ+濃縮リウェイト)。**

- オフライン: 上位≤200位held-out Top-1一致率で現行比 **+1.49pt**(0.6080→0.6229、
  [結果](2026-07-21_skillconc_offline.md))
- オンライン: 現行本番(`ml_lethal` + 既定 `policy_weights.json`)とのミラー head-to-head
  200試合で **58.0%勝率、95% CI [51.1%, 64.6%](0.5を除外=統計的に有意)**
  ([結果](2026-07-21_skillconc_headtohead.md))

## 経緯(Step1〜4の要約)

1. **Step1**: ≤200位データ(26,478件)は15人のプレイヤーに由来し、上位2人
   (Rmy・__Taichicchi__)で60.7%を占める強い偏重を確認([結果](2026-07-21_skillconc_configs.md))。
   これを踏まえ、ハードフィルタ(A: ≤200 / B: ≤1000)と、全データを保持したまま勾配を
   上位に集中させるリウェイト(C)の3構成を定義した。
2. **Step2**: `build_features.py`に`--rank-max`/`--weight-scheme`、`train.py`に
   `--out-weights`を追加(既定挙動は不変、既存の自己検証・回帰テストで確認済み)。
   3構成それぞれで `features_config{A,B,C}.npz` → `policy_weights_config{A,B,C}.json` を
   生成。いずれも自己検証(pure_python_forward照合)PASS。
3. **Step3**: 全構成で現行より上位held-out一致率が改善(A: +1.38pt / B: +0.76pt /
   **C: +1.49pt**)したが、伸び幅は計画の期待(プール平均54%層→上位帯61%相当の
   「天井を引き上げる」規模)よりずっと小さかった。C はプール精度の犠牲も中程度
   (-1.17pt、Aの-2.87ptより優位)で最良候補に選定([結果](2026-07-21_skillconc_offline.md))。
4. **Step4**: `ml_policy_agent.agent(obs, config=...)`に`policy_weights_path`注入点を
   新設(既定挙動は不変、回帰テスト追加)。C・Bを現行とミラー200試合ずつ対戦させ、
   **Cは有意に勝ち越し(58.0%)**、Bは正の傾向だが有意ではなかった(55.5%、
   [結果](2026-07-21_skillconc_headtohead.md))。

## 期待される効果の見立て

Step3のオフライン一致率差(+1.5pt程度)は小さく見えるが、Step4のオンライン結果は
現行に対し **58%対42%の勝率差**として表れた。ただしこれはミラー対戦(自分同士、
探索設定は完全に同一で重みだけが違う)であり、対フィールドの絶対勝率(自チーム44.6%→
どこまで伸びるか)への直接換算はできない([[project_pimc_prod_validation]]で確認済みの
「研究の勝ちが本番に転移しない」リスクは依然残る。ミラーでの58%は「現行より優れている」
ことの相対指標であって、絶対上昇幅の保証ではない)。

## 次のアクション(ユーザー判断)

計画のNon-goalsどおり、本計画は候補と証拠を出すところまで。以下はユーザー判断:

- `policy_weights_configC.json` を本番の `sample_submission/ptcg_ai/learning/policy_weights.json`
  に昇格するか(＝現行モデルの差し替え。`AGENT_TYPE`は既に`ml_policy`なので、重みファイルの
  置き換えのみで反映される)
- 昇格する場合、Kaggle実提出での絶対勝率検証(ミラーではない対フィールド)を別途行うか
- Aは今回オンライン評価を見送った(Cに対しStrict優位性なし)。必要であれば追加検証可能

## 生成物一覧(参照用)

| 種別 | パス |
|---|---|
| 候補重み(採用候補) | `kaggle_replays/policy_net/policy_weights_configC.json` |
| 候補重み(対照) | `kaggle_replays/policy_net/policy_weights_configB.json` / `policy_weights_configA.json` |
| 候補特徴量 | `kaggle_replays/policy_net/features_config{A,B,C}.npz` |
| 既定特徴量(rank_at_fetch追加で再ビルド) | `kaggle_replays/policy_net/features.npz` |
| Step1診断スクリプト | `kaggle_replays/policy_net/_diag_skillconc_player_dist.py` |
| Step3評価スクリプト | `kaggle_replays/policy_net/evaluate_skillconc.py` |
| Step4 head-to-headスクリプト | `league/_diag_skillconc_head_to_head.py` |
| 生ログ・結果JSON | `kaggle_replays/policy_net/evaluate_skillconc_results.json`、`league/results/_diag_skillconc_config{B,C}.json` |
