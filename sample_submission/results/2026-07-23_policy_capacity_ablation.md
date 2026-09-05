# PolicyModel モデル容量 ablation 結果総括(M32/M64/M128)

作成日: 2026-07-23
計画: `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`
ブランチ: `experiment/policy-capacity-ablation`

---

## 結論(1 行)

**hidden を 32→64→128 に上げても実戦勝率は改善しなかった(M64 49.0% / M128 49.3%、
いずれも 95% CI が 50% を跨ぐ)。production(M32)を据え置く。**
offline Top-1 の微増(+0.5pt)は実戦に転移せず、Tier1/Tier3 と同じパターン。
→ 次は特徴追加にも容量増にも戻らず、**Behavior Cloning 自体のボトルネック仮説**へ(別フェーズ)。

---

## 検証設計(clean ablation、hidden 以外を完全固定)

- 入力: `features_configC.npz`(= 構成C = 上位パイロット重み付け、production と同一)
- 固定: seed=42 / embed_dim=8 / lr=1e-3 / batch=64 / max_epochs=100 / patience=10 / loss(listwise softmax CE)/ 標準化
- 変えたのは `--hidden-size` のみ(train.py に追加、既定 32 で挙動不変)。
- 推論 `policy_model.py` は**変更なし**(層を動的に読むため、容量変更で 1 行も変わらない)。

### provenance 確定(推測しない)

production `policy_weights.json` は `policy_weights_configC.json` と **layer 重みがバイト一致**
(全要素 max_abs_diff=0、test_top1=0.5806152 一致、mtime=meta.created_at 一致)。
→ production = configC = `features_configC.npz` 由来を確定。

### control の完全再現(後方互換の証明)

改修後の train.py(既定パス、`--consequence-fields` なし、seed 42)で `features_configC.npz` から
M32 を再学習した `policy_weights_m32.json` は、production `policy_weights.json` と
**全層・embedding がバイト完全一致(max_abs_diff=0.000e+00)**。
→ (a) M32 control = production そのもの(再学習ノイズゼロ)、
   (b) Tier3 で入った train.py 変更が既定パスの後方互換を壊していないことを同時に証明。

---

## Stage 1: offline 評価

| model | params(mlp/total) | epochs | train Top-1 | val Top-1 | test Top-1 | train-val gap | test NLL |
|---|---|---|---|---|---|---|---|
| **M32(=production)** | 7,713 / 17,857 | 24 | 0.6122 | 0.5994 | **0.5806** | +0.0128 | 1.1149 |
| **M64** | 15,425 / 25,569 | 14 | 0.6094 | 0.6044 | **0.5852** | +0.0050 | 1.1183 |
| **M128** | 30,849 / 40,993 | 14 | 0.6149 | 0.6057 | **0.5864** | +0.0092 | 1.1142 |

- test Top-1 は容量とともに単調微増(M32→M64 +0.46pt、→M128 +0.58pt)。
- **過学習は起きていない**: train-val gap は全て <0.013、M64/M128 はむしろ M32 より小さい(早期終了も 14 epoch)。
- NLL は実質フラット(1.1149 / 1.1183 / 1.1142)。
- → offline では「hidden=32 が容量ボトルネックで、増やせば模倣精度が伸びる」という兆候はごく弱く、
  かつ Tier3 Exp B/C(+0.11/+0.21pt)と同オーダーの微差。**offline だけでは採用判断できない**帯。

## Stage 2: 破綻・過学習チェック

Stage 1 の全 split 指標で確認: 容量増で Top-1 が劣化した split はなく(train/val/test すべて微増)、
gap 拡大・NLL 悪化も無し。意思決定が壊れていないことを確認して Stage 3 へ。

## Stage 3: head-to-head(M32 control vs 候補、各 300 試合、同一 seed 群でペア比較)

config `ml_lethal_attackplan_v0only`、デッキ同一、先手後手半々、lethal/attack_plan 同一、
PolicyModel weights のみ差し替え。

| 対戦 | 候補勝率 | 95% CI(Wilson) | errors | avg turns | 判定 |
|---|---|---|---|---|---|
| **M64 vs M32** | 49.00%(147/300) | [43.4%, 54.6%] | 0 | 15.3 | 有意差なし(45–52%帯=効果薄) |
| **M128 vs M32** | 49.33%(148/300) | [43.7%, 55.0%] | 0 | 12.9 | 有意差なし(45–52%帯=効果薄) |

- どちらも 55% に届かず、増試合(600–1000)に進む閾値を満たさない。
- 採用条件(**95% CI 下限 > 50%**)を満たさない → **不採用**。
- offline の微増(+0.5pt)が実戦勝率に**転移しなかった**。これは Tier1/Tier3 と同じ現象。

## レイテンシ(pure-Python forward、features_configC.npz 2000 決定点、クリーン CPU)

| model | ms/決定 | 対 M32 倍率 | us/選択肢 |
|---|---|---|---|
| M32 | 3.85 | 1.0× | 680 |
| M64 | 9.06 | 2.35× | 1,602 |
| M128 | 17.52 | 4.55× | 3,097 |

- M128 でも 17.5ms/決定は lethal/attack_plan の探索予算(各 100ms)・Kaggle 予算(600s/agent)比では収まる。
- ただし M32 の 4.55 倍で、無視できるほど小さくはない。勝率メリットがゼロである以上、採用しない追加根拠。
- 提出サイズ: M128 で ~870KB(現行 387KB)。tarball には余裕だが、これも採用理由にはならない。

---

## 採用判定

**production 据え置き(M32 = 現行 `policy_weights.json` を変更しない)。**
M64/M128 は不採用。理由: 実戦勝率に有意差がなく(CI 下限 > 50% を満たさない)、
offline 微増が転移しないことが再確認され、レイテンシ・サイズは増えるだけ。

## 次のアクション(計画 §12 に従う)

容量を上げても実戦が改善しなかったため、**特徴追加にも容量増にも戻らない**。
「現行 Behavior Cloning 自体がボトルネック(`P(human action|state)` を最適化していて
`P(win|state,action)` ではない)」という仮説を次フェーズで検討する
(A: Policy+Value / B: Policy prior + search / C: advantage/outcome-aware / D: self-play/offline RL)。
本 ablation では実装しない。

---

## 可逆性・成果物

本 ablation のソース変更は **train.py のみ**(`--hidden-size`/`--metrics-out`/train 指標・meta 追記。
既定 32 で挙動不変、M32 が production とバイト一致することで後方互換を証明済み)。
既定重み `policy_weights.json` は**未変更**。他は新規追加ファイルのみ。不採用なので実験ファイルは
記録として残す(削除も昇格もしない)。

補足: 作業ブランチの起点(commit c94434e)時点で `policy_model.py` / `ml_policy_agent.py` /
`encoder.py` / `build_features.py` 等は既に未コミットの Tier3 作業で M 状態だった。これらは
**本 ablation では触っていない**(capacity ablation のソース差分は train.py のみ)。
回帰テストは正しい cwd(`sample_submission/`)から `test_ml_policy_agent.py` /
`test_policy_model.py` / `test_policy_model_consequence.py` 全 27 件 green
(repo root から実行すると deck.csv パス解決で落ちるが、これは既存のテスト cwd 依存で本変更と無関係)。

| 種別 | パス |
|---|---|
| 実装計画 | `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md` |
| 実験重み | `ptcg_ai/learning/policy_weights_m{32,64,128}.json` |
| offline 指標 | `kaggle_replays/policy_net/capacity_ablation/offline_m{32,64,128}.json` |
| 学習ログ | `kaggle_replays/policy_net/capacity_ablation/run_m{32,64,128}.log` |
| レイテンシ | `kaggle_replays/policy_net/capacity_ablation/latency.json` + `bench_forward_latency.py` |
| head-to-head | `league/results/2026-07-23_capacity_m32_vs_m{64,128}_300.json` |
| 実験 config | `configs/ml_capacity_m{64,128}.json` |
| negative-result 総括(Tier1/Tier3) | `results/2026-07-23_feature_addition_negative_result.md` |
