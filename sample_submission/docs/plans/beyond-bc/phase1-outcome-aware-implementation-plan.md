# Phase 1 実装計画書: Outcome / Advantage-aware Imitation(案C)

作成日: 2026-07-23
種別: **実装計画書**(設計書 [[beyond-behavior-cloning-design]] で選定した Phase 1 = 案C を、
関数シグネチャ・データ join の具体・loss 定義・完了条件まで落とす)
ステータス: **計画のみ。production 未変更。コード変更前のレビュー用。**
control: **M32 = 現行 production**(`policy_weights.json`、features_configC 由来、容量ablで byte 一致確認済)

---

## 0. スコープと非目的

**やること**: PolicyModel の学習目的に**勝敗(outcome)**を組み込み、`P(human action)` から
`P(win|state,action)` 側へ寄せた weights を作る。2下位案を順に:

- **C1 outcome-weighted / filtered BC**(先行・ValueModel 不要): 各決定点の listwise CE を
  「その試合を勝ったか」で重み付け(勝ち強調 / 負け割引 / 勝ち試合のみ)。
- **C2 advantage-weighted BC**(AWR系): ValueModel の `V(s)` を baseline に、MC advantage
  `A = won - V(s)` で `exp(A/β)` 重み。良い局面での選択ほど強く模倣。

**変えないもの(非目的)**:
- 推論経路(`policy_model.py`)・agent(`ml_policy_agent.py`)・探索(lethal/attack_plan/pimc)。
- モデル構造(hidden=32、M32 と同一)・入力特徴(state 166 / option 65 / embed 8)。**特徴追加しない。**
- **base 重みスキームは構成C(concentrated)を維持**(control=M32 と交絡させない。outcome 係数は
  その上に掛ける)。
- production の `policy_weights.json` / `value_weights.json` / 既定 config。実験は別 weights/config/branch。

推論・レイテンシ・提出サイズは**一切変わらない**(weights が変わるだけ)。

---

## 1. データ: outcome join(コードで確認済み・2026-07-23)

### 確認済みの実データ

- `kaggle_replays/training_data/policy_positions.jsonl.gz`: 各行に
  **`episode_id` / `player_index` / `step_index`** + observation + chosen_index + rank_at_fetch。
  outcome ラベルは**無い**。
- `kaggle_replays/training_data/value_positions.jsonl.gz`: 各行に
  **`label`(win=1/loss=0)** + `episode_id` / `player_index` / `step_index`。
- join 実測: `(episode_id, player_index) → label` は **不整合 0**(9,380 対、試合×プレイヤーで一意)。
  policy 決定点 187,690 のうち **99.9%(187,582)で勝敗が引ける**(欠損 108)。

### 設計

`(episode_id, player_index) → won∈{0,1}` の lookup を value_positions から構築し、
policy 各行に `won` を付与する(policy 行の `player_index` = 決定者 = `obs.current.yourIndex`)。

- **split 整合**: 現行 `split_for_episode(episode_id)`(md5 式)を維持。won は (episode,player) 単位で
  一定、両プレイヤーは同一 episode_id なので同一 split。**train/test に同一試合が跨がずリークしない。**
- 欠損(108件、`won=-1`)は outcome 係数 1.0(中立)で残す(除外もオプション化するが既定は残す)。
- sanity: 構築時に value_positions の label 不整合が 0 であることを assert(将来データで壊れたら気付く)。

### build_features.py への変更(後方互換・既定 OFF)

```python
# 追加引数
--with-outcome                 # 有効時のみ won を features.npz に追加(既定OFF=出力は現行と完全一致)
--outcome-source PATH          # 既定: kaggle_replays/training_data/value_positions.jsonl.gz
--drop-unknown-outcome         # 任意: won を引けない行を出力から除く(既定: 残す=won -1)

# 追加ロジック(--with-outcome 時のみ)
def load_outcome_map(path) -> dict[tuple[str,int], int]:
    """value_positions.jsonl.gz から (episode_id, player_index) -> won を作る。
    同一キーで label 不整合があれば ValueError(データ健全性)。"""

# ループ内: won = outcome_map.get((episode_id, player_index), -1) を won_rows に追加
# savez に won=np.asarray(won_rows, dtype=np.int8) を追加(--with-outcome 時のみ)
```

- 既定(フラグ無し)では `won` キーは出力されず features.npz は**現行とバイト等価**(Tier3 と同じ
  後方互換方針)。
- 生成物: `features_configC_outcome.npz`(= `--weight-scheme concentrated --with-outcome`。
  base 重みは構成C、加えて won を持つ)。

---

## 2. C1 実装: outcome-weighted / filtered BC(train.py、ValueModel 不要)

現行 train.py は features.npz の `weight`(= 構成C の rank 重み)を sample weight として
listwise CE を重み付けしている(§`batch_weighted_nll`)。C1 はその weight に**勝敗係数**を掛ける。

```python
# train.py 追加引数
--outcome-weighting {none,discount,filter}   # 既定 none = 現行と完全に同一(挙動不変)
--loss-discount FLOAT                         # discount 用: 負け試合の重み係数 γ∈[0,1](既定 0.5)
                                              #   won=1 -> ×1.0 / won=0 -> ×γ / won=-1(不明) -> ×1.0
                                              # filter: won=1 のみ学習(= γ=0 と等価)
```

- 実装: features.npz から `won` を読み、`effective_weight[i] = weight[i] * factor(won[i], mode, γ)`。
  これを既存の SplitData.weight に入れるだけ。**loss 関数・標準化・モデル・self-check は不変。**
- `--outcome-weighting none`(既定)なら `won` を読まず現行と 1 バイトも変わらない(後方互換)。
- **スイープ**: γ ∈ {1.0(=control 再現), 0.5, 0.25, 0.0(=filtered)}。各で weights を書き出す。
- meta に記録: `outcome_weighting`, `loss_discount`。

### C1 の狙いと既知リスク

- 狙い: 勝ち試合の選択を相対的に強く模倣し、負け筋の手を弱める。**入力を増やさず目的を勝率へ寄せる**
  最小手。ValueModel 不要で最速。
- リスク: 1試合の勝敗は個々の手の良否と一致しない(疎・遅延・ノイズ)。強い割引(γ→0)は
  データを減らし過学習しうる → 穏当な γ から。[[feedback_conservative_confidence]]。

---

## 3. C2 実装: advantage-weighted BC(train.py + ValueModel)

MC advantage で「期待勝率 V(s) を上回る結果になった局面での選択」を強く模倣する(AWR)。

```python
# train.py 追加引数
--outcome-weighting advantage
--adv-beta FLOAT           # 温度 β(既定 1.0)。weight *= exp(clip(A/β, -c, c))
--adv-clip FLOAT           # A/β のクリップ幅 c(既定 3.0。exp 爆発防止)
--value-weights PATH       # 既定: sample_submission/ptcg_ai/learning/value_weights.json
```

- `A_i = won_i - V(s_i)`、`V(s_i) = ValueModel.predict_win_prob_from_features(state_features[i], turn[i])`。
  state_features は features.npz に既にある 166 次元 = encoder.encode_state 出力で、ValueModel の
  入力と同一(要 assert: value_weights の feature 数 == state_features.shape[1] == 166)。
- `effective_weight[i] = base_weight[i] * exp(clip(A_i / β, -c, c))`(won=-1 は係数 1.0)。
- ValueModel forward は **学習前に一度だけ全 train/val 行に対して計算**(pure-Python だが 18万行 ×
  166 次元 = 数十秒オーダー。1回だけなので許容。numpy でベクトル化した独立実装で高速化してよいが、
  **value_model.py の結果と数値一致を確認**してから使う)。
- **β スイープ**: β ∈ {0.5, 1.0, 2.0}。meta に `adv_beta`, `adv_clip`, `value_weights` を記録。

### C2 の狙いと既知リスク

- 狙い: 勝敗の粗い信号を V(s) baseline で正規化し、「予想以上に勝った局面での手」を重視。C1 より
  信号がスムーズ。
- リスク: **ValueModel AUC 0.746 のノイズが重みに乗る**。β/clip で影響を抑える。V(s) は決定“状態”の
  価値であって選択“行動”の Q ではない(MC 1サンプル近似)。過信しない。

---

## 4. weights JSON 形式・後方互換

- **schema 不変**。standardization/card_embedding/layers は現行のまま(モデル構造 M32 と同一)。
- meta に追記(additive、推論は読まない): `outcome_weighting`, `loss_discount` or
  `adv_beta`/`adv_clip`/`value_weights`, `n_outcome_known`, `n_outcome_unknown`。
- 推論 `policy_model.py` は**変更なし**(weights を読むだけ)。M32 と同一アーキで層 shape も同一。
- production `policy_weights.json` は**未変更**。実験は別ファイル名で共存。

---

## 5. 評価(Stage1/2/3、容量abl と同一ゲート)

- **Stage 1 offline(足切りのみ)**: モデルが壊れていない(NLL 有限、自明解に潰れない)ことの確認。
  **注意: C は offline Top-1(模倣一致率)が下がってよい**——目的は勝率であって模倣一致ではない。
  だから **Stage1 の Top-1 は採用根拠にも不採用根拠にもしない**(潰れ検知のみ)。
- **Stage 2 診断**: 既知局面(ATTACK 一致・山札切れ関連)で意思決定が破綻していないか。
- **Stage 3 head-to-head(決定的)**: **M32 control vs 候補**、`ml_lethal_attackplan_v0only`、
  先後半々、同一 seed 群。既存ハーネス `league/_diag_tier1abc_head_to_head.py`
  (`--baseline-weights`/`--candidate-weights`)を流用。300試合スクリーニング → 段階判定
  (45%未満打切 / 55%以上増試合 600–1000)。
- **採用条件: 95% CI 下限 > 50%。** 満たさなければ production 据え置き。
- 候補が多い(C1 γ×4 + C2 β×3 = 7)ため、まず **300試合で全候補スクリーニング**し、
  55%以上のものだけ増試合。

---

## 6. レイテンシ・提出サイズ・可逆性

- **レイテンシ/提出サイズ: 不変**(推論も weights の形も変えない)。
- **可逆性: 最高**。ソース変更は build_features(`--with-outcome`)と train.py(`--outcome-weighting`)の
  追加のみ、**いずれも既定 OFF で挙動不変**。不採用時は実験 weights/features/config を消すだけ。
  採用時のみ最終候補を `policy_weights.json` に昇格(parity 手順は容量abl と同じ)。
- 既定 OFF の後方互換は、容量abl と同様 **M32 再現 = production バイト一致**で担保する
  (`--outcome-weighting none` かつ `features_configC.npz` で学習 → configC 一致を再確認)。

---

## 7. ファイル構成

```
kaggle_replays/policy_net/
  build_features.py                      # --with-outcome 追加(既定OFFで出力不変)
  features_configC_outcome.npz           # 構成C重み + won ラベル
  train.py                               # --outcome-weighting 追加(既定noneで挙動不変)
  outcome_aware/
    offline_c1_g100.json ... g000.json   # C1 各γ
    offline_c2_b05.json ... b20.json     # C2 各β
    run_*.log

sample_submission/ptcg_ai/learning/
  policy_weights_c1_g050.json ...        # 実験weights(production据え置き)
  policy_weights_c2_b10.json ...

sample_submission/configs/
  ml_outcome_c1_g050.json ...            # v0only クローン + policy_weights_path

league/results/
  YYYY-MM-DD_outcome_m32_vs_c1_g050_300.json ...

sample_submission/results/
  YYYY-MM-DD_outcome_aware_phase1.md     # 総括(offline + Stage3、採用/不採用)
```

---

## 8. 完了条件・採用基準

1. build_features `--with-outcome`: features_configC_outcome.npz が生成でき、`won` の
   既知率 ≈ 99.9%、label 不整合 0 を確認。`--with-outcome` 無しは現行出力とバイト等価。
2. train.py `--outcome-weighting none`: features_configC.npz から M32 を再学習し
   **production と byte 一致**(後方互換の証明、容量abl と同じ手順)。
3. C1(γスイープ)/ C2(βスイープ)の weights と offline 指標が揃い、self-check PASS。
4. Stage3 head-to-head 300試合スクリーニング完了 → 段階判定。
5. **採用**: いずれかの候補が **CI 下限 > 50%**。**不採用**: 満たさなければ production 据え置きで、
   案C を negative result として記録(このとき Phase 2 = 案A へ進む判断材料にする)。

---

## 9. 実施順序

1. build_features に `--with-outcome`(+ outcome_map / label 不整合 assert)。既定OFF不変を確認。
2. `features_configC_outcome.npz` 生成(構成C + won)。既知率/不整合を検証。
3. train.py に `--outcome-weighting {none,discount,filter,advantage}` + 係数引数。
4. **後方互換確認**: `--outcome-weighting none` + features_configC.npz → M32 byte 一致(§8-2)。
5. **C1 先行**: γ ∈ {0.5,0.25,0.0} 学習 → Stage1 足切り → Stage3 300試合スクリーニング。
6. C1 に見込みがあれば増試合。無ければ **C2**: β ∈ {0.5,1.0,2.0}(V(s) 計算は value_model と数値一致確認)。
7. 総括 md・採用判定。採用なら parity 後に昇格、否なら据え置き + negative result 記録 → Phase 2(案A)検討。

---

## 10. 境界(担当領域)

- 変更は `kaggle_replays/policy_net/`(オフライン学習)と実験 weights/config のみ。
- 推論・agent・探索・`rule_based/`・`action_selection/` は**不変**([[feedback_respect_ownership_boundaries]])。
- ValueModel は**読み取り専用で再利用**(value_weights.json は変更しない)。
