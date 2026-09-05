# PolicyModel モデル容量 ablation 実装計画書

作成日: 2026-07-23
対象ブランチ: `experiment/pimc-hidden-info-integration`(HEAD)から新規実験ブランチを切る
ステータス: **計画のみ。production 未変更。コード変更前のレビュー用。**

---

## 0. この計画の目的と非目的

### 目的

> 現行の 239 次元入力に対して **hidden=32 がボトルネックなのか** だけを、
> 特徴量・データ・sample weight・loss・seed を固定した clean ablation で検証する。

Tier1(raw identity)/Tier3(consequence)を通して
「特徴量を足して offline 模倣精度を少し上げる → 実戦勝率も上がる」という関係が
**安定して成立しなかった**(§9 に negative result を集約)。
そこで特徴量追加を一旦停止し、次の仮説を検証する:

> 現在の特徴量をモデル(hidden=32 の小型 MLP)が十分に使い切れていないのではないか。

### 非目的(この ablation では変更しない)

- Tier1 features / Tier3 consequence features / 新規 card identity / 新規 delta
- hidden information / ValueModel / 探索ロジック(lethal / attack_plan / pimc)
- 特徴量エンコーダ(`encoder.py`)の入力次元
- sample weight(構成C = 上位パイロット重み付け)
- loss(listwise softmax CE)、optimizer、lr、batch size、seed、early stopping 条件
- **production の推論経路(`policy_model.py`)** — 一切変更しない(§4 で理由を確認済み)

**hidden size 以外を変えない。** これが最優先制約。

---

## 1. 現行 PolicyModel 構造の正確な確認(コードで確認済み)

学習側 [`kaggle_replays/policy_net/train.py`](../../../../kaggle_replays/policy_net/train.py) の
`PolicyScorer`(L98–116):

```
card_id ──embedding(1268 × 8)──┐
state(166) ++ option(65) ──────┴──► concat(239) ──► Linear(239, 32) ──► ReLU ──► Linear(32, 1) ──► score
```

- **1 隠れ層 MLP**。hidden = 32。
- hidden は `_HIDDEN_SIZE = 32`(train.py L78)というモジュール定数。
  `PolicyScorer.__init__` は既に `hidden: int = _HIDDEN_SIZE` 引数を取るが、`main()`（L550）は
  常にこの定数を渡している → **CLI から可変にする配線がまだ無い**。
- card_embedding 次元 `_EMBED_DIM = 8`(L79)は「調整対象外・コーディネーター指定」。今回も固定。
- 入力内訳: `BASE_FEATURE_COUNT=166`(state) + `OPTION_FEATURE_COUNT=65`(option) + `embed 8` = **239**。
  （`encoder.py` で確認: L128 / L509）
- 損失: listwise softmax CE(train.py L184–214)。sample weight は features.npz の `weight` 列
  （= 構成C を build_features 側で焼き込み済み。学習コードは重みを読むだけ)。

### 推論側(production、変更しない)

[`sample_submission/ptcg_ai/learning/policy_model.py`](../../../ptcg_ai/learning/policy_model.py)
`PolicyModel._forward`(L252–272)は **層数・hidden size を一切ハードコードしていない**:

```python
n_layers = len(self._layers)
for layer_idx, (W, b) in enumerate(self._layers):
    z = [b[k] + sum(W[k][i] * h[i] for i in range(len(h))) for k in range(len(W))]
    is_last = layer_idx == n_layers - 1
    h = z if is_last else [v if v > 0.0 else 0.0 for v in z]  # 最終層以外に ReLU
```

`layers` を JSON から動的に読み、最終層以外に ReLU を適用するだけ。
→ **hidden を 32→64→128 に変えても、2層 MLP に変えても、推論コードは 1 行も変わらない。**
これが本 ablation を「production 無変更」で実施できる根拠。

### config 化状況

- hidden size は **config 化されていない**(train.py のモジュール定数)。
- ただし **重みファイルの差し替えは既に config 化済み**:
  `config["policy_weights_path"]` を指定すると
  [`ml_policy_agent.py`](../../../ptcg_ai/ml_policy/ml_policy_agent.py) `_get_model`(L129–144)が
  そのパスの `PolicyModel` をパス別キャッシュで読む(既定パス・グローバル `_model` は不変)。
  tier1abc 実験([`configs/ml_lethal_attackplan_tier1abc.json`](../../../configs/ml_lethal_attackplan_tier1abc.json))が
  この口を使って `policy_weights_tier1abc.json` を差している。**M ablation も同じ口をそのまま使う。**

---

## 2. 現行パラメータ数

`layers` は JSON で確認: layer0 = Linear(239, 32)、layer1 = Linear(32, 1)。

| 部品 | 式 | H=32(現行) |
|---|---|---|
| fc1 `Linear(239, H)` | 240·H | 7,680 |
| fc2 `Linear(H, 1)` | H+1 | 33 |
| **MLP 小計(容量とともに増える部分)** | 241·H + 1 | **7,713** |
| card_embedding(固定・不変) | 1268 × 8 | 10,144 |
| standardization(state/option mean·std、非学習だが JSON 格納) | 166·2 + 65·2 | 462 |
| **学習パラメータ合計(MLP + embedding)** | | **17,857** |

現行 `policy_weights.json` = 387.1 KB。

---

## 3. hidden=32/64/128(/256)のパラメータ数

MLP 部分 = `240·H + (H+1) = 241·H + 1`。embedding(10,144)と standardization(462)は全 H 共通。

| モデル | fc1 (240·H) | fc2 (H+1) | **MLP 小計** | + embedding | 学習パラメータ合計 | 概算 JSON サイズ¹ |
|---|---|---|---|---|---|---|
| **M32(control)** | 7,680 | 33 | 7,713 | 10,144 | **17,857** | ~390 KB |
| **M64** | 15,360 | 65 | 15,425 | 10,144 | **25,569** | ~550 KB |
| **M128** | 30,720 | 129 | 30,849 | 10,144 | **40,993** | ~870 KB |
| M256(保留) | 61,440 | 257 | 61,697 | 10,144 | **71,841** | ~1.5 MB |

¹ 現行 387 KB / 約 18,319 格納 float ≈ 21 byte/float から線形外挿(standardization 462 込み)。

**所見:** 容量とともに増えるのは MLP 部分のみで、パラメータの過半は容量に依存しない
embedding(10,144)。M128 でも「巨大化」ではなく、MLP だけが M32 の 4 倍になる程度。
JSON サイズ・レイテンシへの影響は §11 の通り軽微。まず M32/M64/M128 を回し、
結果を見てから M256 の要否を判断する(最初から入れない)。

---

## 4. 学習・推論コードで変更が必要な箇所

### 4.1 学習側 `train.py`(変更あり)

1. **`--hidden-size` 引数を追加**(int、既定 32)。`main()` で `PolicyScorer(..., _HIDDEN_SIZE)` の
   代わりに `args.hidden_size` を渡す(L550)。`PolicyScorer` 自体は既に `hidden` 引数対応済み → **変更不要**。
2. **train split の指標を追加取得**(overfitting 確認用、§9)。現状 `evaluate_split` は val/test のみ。
   学習後に train split でも `evaluate_split` を 1 回呼び、train NLL / train Top-1 を得る。
3. **meta に容量情報を記録**: `hidden_size`、`param_counts`(fc1/fc2/embedding/total)、
   train/val/test の NLL・Top-1、train-val gap を `weights_json["meta"]` に追記。
4. **offline メトリクス JSON を別ファイルにも dump**(§14 の保存先)。
5. `--out-weights` は既存(実験は必ず別パスを明示)。**production パスを上書きしない。**

**変更しない:** `_SEED=42`、`_EMBED_DIM=8`、`_LR`、`_BATCH_SIZE`、`_MAX_EPOCHS`、`_PATIENCE`、
loss、標準化、self-check(pure_python_forward との 1e-6 一致検証)。self-check は
hidden に依存しないので M64/M128 でもそのまま PASS するはず(層の shape が変わるだけ)。

### 4.2 推論側 `policy_model.py`(変更なし)

§1 の通り層を動的に読むため **変更ゼロ**。production 推論ファイルは無変更。

### 4.3 agent 側 `ml_policy_agent.py`(変更なし)

`config["policy_weights_path"]` 差し替え口が既存(§1)。**変更ゼロ。**

### 4.4 seed に関する注意(設計上の明記)

`torch.manual_seed(42)` は固定するが、`nn.Linear` の初期化は shape 依存で RNG を消費するため、
**M64 の fc 初期値は M32 と同一ではない**(embedding は shape 不変・生成順が先なので全 H で同一初期値)。
これは「同一初期値の比較」ではなく「同一学習手続き下での容量比較」。境界結果(§12 の 45–55% 帯)に
なった候補のみ、seed を {42, 43, 44} で 3 回学習し直して offline/実戦の分散を確認する(頑健性チェック)。

---

## 5. weights JSON 形式への影響

- **構造変更なし。** `layers[0].weight` の shape が `[32][239]`→`[H][239]`、`layers[1].weight` が
  `[1][32]`→`[1][H]` になるだけ。`card_embedding` / `standardization` / `meta` の形は不変。
- 追記のみ: `meta.hidden_size`、`meta.param_counts`(いずれも additive、既存 reader は無視)。
- スキーマ契約(policy_model.py L14–36 のコメント)は満たしたまま。

---

## 6. 旧 weights との後方互換

- **後方互換コードは不要。** `PolicyModel._load`(L130–136)は `layers` を汎用ループで読み込むため、
  H に依存しない。現行 `policy_weights.json`(H=32)も新 M64/M128 も同じコードで読める。
- `meta.hidden_size` が無い旧重みでも推論は成立(推論は meta を読まない)。
- production 既定パス `policy_weights.json` は**触らない** → 既定挙動は完全に不変。
- 実験重みは別ファイル名(`policy_weights_m{H}.json`)で共存(tier1abc と同じ運用)。

---

## 7. M32/M64/M128 clean ablation 条件

全候補で**以下を完全に固定**し、`--hidden-size` だけを変える:

| 固定項目 | 値 |
|---|---|
| 入力 features | **`features_configC.npz`**(= 構成C = concentrated 重み付け。§7.1 で provenance 確定済み) |
| consequence 特徴 | 不使用(`--consequence-fields` 指定なし) |
| seed | 42 |
| embed_dim | 8 |
| lr / batch / max_epochs / patience | 1e-3 / 64 / 100 / 10 |
| loss / 標準化 / early stopping | 現行のまま |
| train/val/test split | features.npz の `split` 列(不変) |

### 7.1 provenance 確定と control の定義(コードで確認済み・2026-07-23)

**production の由来を確定した(推測ではない):**

- production `sample_submission/ptcg_ai/learning/policy_weights.json` は
  `kaggle_replays/policy_net/policy_weights_configC.json` と **layer 重みがバイト一致**
  (layer0 sum・全要素一致、test_top1=0.5806152 一致、mtime 20:36 JST = meta.created_at 11:36 UTC 一致)。
- configC は `features_configC.npz`(= `build_features.py --weight-scheme concentrated` の出力。
  rank≤20:8.0 / ≤50:3.0 / ≤200:1.5 / ≤1000:1.0 / 1000超:0.5 / 不明:0.2)から hidden=32 で学習。
- つまり **production = 構成C = `features_configC.npz` 由来**。`features.npz`(default 重み [1.0–1.5])
  **ではない**。ablation の入力は `features_configC.npz` を使う(§7 の表)。

**control の定義:**

- **M32(control)= `--hidden-size 32` で `features_configC.npz` から今回のパイプラインで学習した
  `policy_weights_m32.json`。** M64/M128 はこの M32 control と比較する。
- **parity 健全性チェック(強化版):** production が既知の hidden=32 学習結果(configC)と
  バイト一致しているため、head-to-head ~50% ではなく **決定論的再現**で検証できる:
  今回の train.py(既定パス、`--consequence-fields` なし、seed 42)で `features_configC.npz` から
  M32 を学習し、得た重みが `policy_weights_configC.json`(= production)と数値的に一致
  (self-check と同じ 1e-6 級、Adam の非決定性があれば top1・layer sum が実質一致)することを確認する。
  ここが乖離する場合、Tier3 で入った train.py の変更が既定パスを壊している証拠なので、
  **ablation を進める前に原因を潰す**(既定パスの後方互換が崩れていないかを最優先で確認)。

---

## 8. offline 評価指標(Stage 1)

各モデル(M32/M64/M128)について train.py が出力し、JSON に保存する:

- train NLL / val NLL / test NLL(weighted)
- train Top-1 / val Top-1 / test Top-1
- **train–val gap**(Top-1 と NLL の両方)= overfitting の一次指標
- test との差
- **パラメータ数**(fc1/fc2/embedding/total、§3 の表)
- select_type 別 Top-1(train.py 既存の内訳出力、L566–574)

**判断ルール(Stage 1 単独では採用しない):** offline Top-1 の微改善は**採用根拠にしない**
(Tier1/Tier3 で offline↑ → 実戦フラット/悪化を繰り返したため)。Stage 1 は
「明らかに壊れている / 過学習している候補を落とす」フィルタとしてのみ使う。

---

## 9. overfitting 確認方法

各モデルで以下を記録・比較:

| 指標 | 過学習の兆候 |
|---|---|
| train Top-1 − val Top-1 gap | 容量を上げると train のみ上昇、gap が拡大 |
| train NLL vs val NLL | train NLL だけ下がり val NLL 横ばい/悪化 |
| val NLL の early-stop epoch | 大容量ほど早期に val が底打ち |
| test Top-1 | val と同傾向か(乖離があれば不安定) |

「train accuracy だけ上がり val/test 横ばい、実戦で悪化」パターンを Stage 1 で検知する。
gap が M32 比で明確に拡大し val/test が改善しない候補は Stage 3 に進めない。

---

## 10. head-to-head 評価方法(Stage 3)

**既存ハーネスをそのまま再利用**:
[`league/_diag_tier1abc_head_to_head.py`](../../../../league/_diag_tier1abc_head_to_head.py) は
`--baseline-weights` / `--candidate-weights` を取り、両側に `policy_weights_path` を注入して
同一プロセス内で混線なく対戦させる汎用ハーネス(L58–113)。M ablation 用に新規ハーネスは不要。

- **比較軸:** M32 control(baseline)vs 候補(M64 / M128)。
- **完全同条件:** 同一 config(`ml_lethal_attackplan_v0only` ベース)、同一デッキ、先手後手半々、
  lethal / attack_plan 同一、**PolicyModel weights のみ差し替え**、エラー/タイムアウト数を記録。
- **段階判定**(まず各 300 試合スクリーニング):

| 300 試合勝率 | 判定 |
|---|---|
| < 45% | 原則打ち切り |
| 45–52% | 効果薄 / 保留 |
| 52–55% | 増試合を検討 |
| ≥ 55% | 600–1000 試合へ増やし有意性確認 |

- **最終 production 採用条件:** **95% CI 下限 > 50%**。

### Stage 2(固定盤面 / 既知問題診断)

Stage 1 → Stage 3 の間に、既知の弱点(ATTACK 選択の一致率、山札切れ関連の意思決定など)で
M32 と候補の挙動差を固定局面で確認する。Tier3 で使った診断系(`league/_diag_*`)の枠組みを流用。
ここは「実戦の勝敗が動く前に、容量増で意思決定が壊れていないか」を安く見るゲート。

---

## 11. レイテンシ・提出サイズへの影響

### レイテンシ

- 推論 forward は pure-Python。支配項は fc1 = 1 選択肢あたり `H × 239` 回の積和。
  H=32→7,648、H=128→30,592(4 倍)。1 意思決定点あたり平均 ~7.75 選択肢 →
  H=128 でも 1 決定あたり ~237k 積和(pure Python で数 ms オーダー)。
- 対して lethal / attack_plan の探索予算は各 100ms、Kaggle 予算は 600 秒/エージェント。
  PolicyModel forward はそのごく一部 → **容量 4 倍でも実戦レイテンシへの影響は軽微**。
- **ただし推測せず計測する:** M32/M64/M128 それぞれで、サンプル決定点集合に対する
  1 決定あたり forward 実測時間と、1 ゲーム総意思決定時間を記録(§14 に保存)。
  Kaggle 600 秒/エージェント予算比の使用率を出す。

### 提出サイズ

- §3 の通り M128 で ~870 KB(現行 387 KB)。提出 tarball には余裕(過去の
  `project_submission_tarball_requires_decks` の通り decks 同梱が必須なだけでサイズ制約は緩い)。
- production 採用時のみ `policy_weights.json` を置き換え。実験中は別ファイルなので提出物に影響しない。

---

## 12. 採用 / 不採用基準

1. **Stage 1(offline):** 破綻・過学習(§9)で明らかに劣る候補を落とす。offline 改善だけでは採用しない。
2. **Stage 2(診断):** 既知局面で意思決定が壊れていないこと。
3. **Stage 3(head-to-head):** M32 control 比で
   - 300 試合 < 45% → 打ち切り
   - 300 試合 ≥ 55% → 増試合 → **95% CI 下限 > 50% で採用候補**
4. production 採用は Stage 3 の CI 下限 > 50% を満たした候補のみ。満たさなければ **production 据え置き**。

### M32/M64/M128 で改善しなかった場合(最重要)

> **特徴量追加へ戻らない。** 「現在の Behavior Cloning 自体がボトルネック」という次の仮説
> (`P(human action|state)` を最適化していて `P(win|state,action)` ではない)へ進む。
> その場合のみ、別フェーズとして設計書を作る(A: Policy+Value / B: Policy prior + search /
> C: advantage / outcome-aware / D: self-play / offline RL)。**本 ablation では実装しない。**

---

## 13. 可逆性

- production 推論(`policy_model.py`)・agent(`ml_policy_agent.py`)・既定重み
  (`policy_weights.json`)・production config は**一切変更しない**。
- 変更は train.py への `--hidden-size` 追加・train 指標追加・meta 追記のみ(既定 32 で挙動不変)。
- 実験成果物(`policy_weights_m*.json`、`configs/ml_capacity_*.json`、results)は追加のみ。
- **不採用時のロールバック = 実験ファイルを消すだけ**(production 参照は一切増やさない)。
- 採用時のみ、最終候補を `policy_weights.json` に昇格(+ meta に由来を記録)し、
  parity 再確認後にコミット。

---

## 14. 実験結果を保存するファイル構成

```
kaggle_replays/policy_net/
  train.py                              # --hidden-size 追加(既定32で挙動不変)
  capacity_ablation/
    offline_m32.json                    # train/val/test NLL・Top-1、param数、gap、実測レイテンシ
    offline_m64.json
    offline_m128.json
    run_m32.log  run_m64.log  run_m128.log

sample_submission/ptcg_ai/learning/
  policy_weights_m32.json               # control(再学習、production は据え置き)
  policy_weights_m64.json
  policy_weights_m128.json

sample_submission/configs/
  ml_capacity_m64.json                  # v0only クローン + policy_weights_path 指定
  ml_capacity_m128.json                 # (control は production config をそのまま使用可)

league/results/
  YYYY-MM-DD_capacity_m32_vs_m64_300.json
  YYYY-MM-DD_capacity_m32_vs_m128_300.json
  (増試合分の 600/1000 も同命名)

sample_submission/results/
  YYYY-MM-DD_policy_capacity_ablation.md   # 総括(offline + Stage2 + Stage3、採用/不採用結論)
```

---

## 15. Tier1 / Tier3 の negative result 保存(削除しない)

Tier3 実装(`consequence.py`、encoder/build_features/PolicyModel の consequence 配線、
transaction 解決、train/runtime parity 基盤、診断テスト、head-to-head ハーネス)は
**既定 OFF・後方互換のまま保存**する(将来 search / ValueModel / action evaluation / offline RL で再利用)。
現時点で production PolicyModel には採用しない(`meta.consequence_fields` 空が既定)。

さらに、散在している結果を **1 本の negative-result 総括ドキュメントに集約**して残す
(新規作成の deliverable):

`sample_submission/results/2026-07-23_feature_addition_negative_result.md`

内容(結論を将来の自分/他メンバーが 1 ページで参照できるように):

| 実験 | 特徴 | offline Top-1 | ミラー勝率 | 95% CI | 判定 |
|---|---|---|---|---|---|
| Tier1abc | 盤面/stadium/tool/energy/ability identity | 一部微増 | 43.0% | — | 不採用 |
| Tier1a 単独 | identity 部分 | — | 44.3% | — | 不採用 |
| Tier3 Exp B | opp_hp_loss / opp_energy_removed / opp_special_energy_removed | +0.11pt | 55.0% | [49.3, 60.5] | 境界 |
| Tier3 Exp C | delta_best_effective_attack_damage / delta_can_ko | +0.21pt | 54.7% | [49.0, 60.2] | 境界 |
| Tier3 Exp D | delta_attack_ready / delta_energy_shortfall | −0.27pt | 35.0% | [29.8, 40.6] | 悪化・不採用 |
| Tier3 B+C 統合 | B + C | — | 40.7% | [35.3, 46.3] | 悪化・不採用 |
| Tier3 delta_best 単独 | delta_best_effective_attack_damage | — | 47.0% | [41.4, 52.6] | フラット |

**結論(記録として明文化):**
特徴量を少し変えるだけで実戦勝率が 55%→47%、良さそうな B/C を統合すると 40.7%、
proxy 特徴 D は 35% と、**特徴追加による offline↑ が実戦勝率へ安定して転移しなかった**。
「カード情報/consequence を増やせば強くなる」という単純仮説は Tier1/Tier3 で不支持。
→ 次は特徴追加を止め、モデル容量 ablation(本計画)で「hidden=32 がボトルネックか」を検証する。

既存の関連資産(移動・削除しない):
- `docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md`
- `results/2026-07-23_tier3_experiment_b_results.md`
- `results/2026-07-22_policy_feature_expansion_tier1abc.md`
- `ptcg_ai/board_evaluation/consequence.py`、`tests/unit/test_consequence.py` ほか consequence テスト群
- `league/_diag_expB_head_to_head.py`、`league/_diag_tier1abc_head_to_head.py`

---

## 16. 実施順序(サマリ)

1. negative-result 総括ドキュメント作成(§15)。Tier3 は既定 OFF のまま。
2. 実験ブランチを切る(HEAD から)。
3. train.py に `--hidden-size` + train 指標 + meta 追記(既定 32 で挙動不変を確認)。
4. provenance 確定済み(production = configC = features_configC.npz、§7.1)→
   `features_configC.npz` から M32 control 学習 → **parity チェック(M32 が configC を決定論的再現)**。
   ここが乖離したら既定パスの後方互換破壊を疑い、原因を潰すまで先に進まない(§7.1)。
5. M64 / M128 学習(Stage 1 offline + overfitting、§8/§9)。
6. Stage 2 診断 → Stage 3 head-to-head 300 試合スクリーニング(§10)。
7. 段階判定に従い増試合 → CI 下限 > 50% なら採用候補、否なら production 据え置き。
8. いずれも改善せずなら **特徴追加に戻らず** Behavior Cloning ボトルネック仮説の設計書へ(§12・別フェーズ)。
9. depth 追加(2 層 MLP)は §17 の通り**別 ablation**(hidden 拡大と同時変更しない)。

---

## 17. depth(2 層 MLP)を後で試す場合の注記(別 ablation)

M32/M64/M128(1 隠れ層の hidden size 比較)が終わった**後のみ**、必要に応じて独立実験として:
`input→64→32→output` や `input→128→64→output`。

**hidden 拡大と層追加を同一実験で同時に変えない。** まず 1 層の hidden size だけ比較し、
その後 depth を別 ablation として比較する。

実装上の追加変更(depth 実験に入るときだけ):
- `PolicyScorer` を hidden の**リスト**を取れるよう一般化(現状 fc1/fc2 の 2 層固定)。
- `extract_layers_json`(train.py L330–343)は現在 `(model.fc1, model.fc2)` をハードコードしているため、
  可変層数に対応させる。
- 推論側 `policy_model.py` は**変更不要**(層を動的に読むため。§1)。self-check も層数非依存。
```
