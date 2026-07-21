# Skill Concentration — Step4: オンライン評価(現行 ml_lethal とのミラー head-to-head)

作成: 2026-07-21
関連: `sample_submission/docs/plans/individual/shogo/policymodel-skill-concentration-implementation-plan.md`
実行: `league/_diag_skillconc_head_to_head.py`(現行 `ml_lethal` 設定 vs 候補重み、
`ml_policy_agent.agent(obs, config=...)` の `policy_weights_path` 注入点を使用。
探索設定(`lethal_search` 等)は両側とも `ml_lethal` 共通、重みJSONだけが違う)。

Step3のオフライン評価(`2026-07-21_skillconc_offline.md`)でCを主候補、Bを対照として選定。
先手後手半々、`match_context.reset()` を毎試合、Wilson 95% CI。

## 結果

| 候補 | 試合数(有効) | 勝数 | 勝率 | 95% CI | 有意(0.5を除外)か |
|---|---:|---:|---:|---|---|
| **C(全データ+濃縮リウェイト)** | 200 | 116 | **58.0%** | [0.5107, 0.6463] | **はい** |
| B(rank<=1000フィルタ) | 200 | 111 | 55.5% | [0.4857, 0.6222] | いいえ(0.5を含む) |

生ログ: `league/results/_diag_skillconc_configC.json` / `league/results/_diag_skillconc_configB.json`
(各200試合、エラー0件、平均2.6〜2.7秒/試合)

## 解釈

- **候補C(全データ+濃縮リウェイト)は、現行本番の意思決定経路(`ml_lethal` + 既定
  `policy_weights.json`)に対し、200試合のミラー対戦で統計的に有意な勝ち越し(58.0%、
  95% CI下限が51.1%で0.5を上回る)。**
- Step3のオフライン評価で示された「上位held-out一致率+1.49pt」という小さな差が、
  実際の対戦では58%対42%という無視できない勝率差に変換されている。将棋・カードゲームで
  典型的に見られる「わずかな一致率差が終局の勝率差を増幅する」パターンと整合的。
- Bも正の方向(55.5%)だが95% CIが0.5を含み、この試合数では「現行と有意差なし」の域に
  留まる。Cより改善幅が小さいというStep3の傾向とも整合。
- A(rank<=200ハードフィルタ)はStep3で「Cに対しStrict優位性なし」と判断し、オンライン評価は
  実施しなかった(過学習リスク・volume減のトレードオフに見合う根拠が無いため)。
