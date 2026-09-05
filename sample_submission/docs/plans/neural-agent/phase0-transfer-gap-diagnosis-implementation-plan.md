# Phase 0 実装計画: 転移ギャップ診断

作成日: 2026-07-24
種別: **実装計画(関数シグネチャ・データソース・完了条件つき。方向性は `design.md` §0/§4 で決定済)**
親設計書: `docs/plans/neural-agent/design.md`
ステータス: 着手中(本セッションで read-only 分析を開始)

---

## 0. この計画の目的

`design.md` §0.2 の最大リスク —— **「offline 改善が本番勝率に乗らない」原因が未特定** —— を
3 仮説に分解して切り分ける。**全 NN 投資(Phase 1 以降)の前段の必須関門。**
成果物は「Phase 1 で信念ベクトルに何を入れるか」をデータで指す結論。

3 仮説(design.md §4 Phase 0):
- **H1 学習分布のミスマッチ**: 上位ログ(模倣対象)が本番フィールドと相手/デッキ分布が違う。
- **H2 デッキ劣位(構造)**: 山札切れ等の構造コストが勝率の天井を作っている(半分検証済)。
- **H3 評価ノイズ**: 真の小改善が 300 試合スクリーニングの検出力不足で CI に埋もれている。

**すべて read-only 分析。production・weights・config は一切変更しない。**

---

## 1. 再利用する既存資産(推測でなく実在を確認済)

| 資産 | 実体 | 使いみち |
|---|---|---|
| エピソード索引 | `kaggle_replays/index/episodes_master.jsonl`(4975 行、team_name/rank_at_fetch/leaderboard_score) | H1: 上位プレイヤーの相手分布の母集団 |
| 敗因タクソノミ | `_diag_loss_taxonomy_results.json`(363 試合中 201 敗、side_race_blowout 139 / deck_out 52 / close 10、相手アーキタイプ別) | H2/H3: 自分の敗因・相手分布 |
| 山札切れ診断 | `_diag_deckout_*_results.json`([[project_deckout_loss_cause]]) | H2: 構造コストの既存結論 |
| value網 probe | `_diag_valuenet_probe_results.json`(**self_deck_count 摂動は 200中92 しか正方向・mean_delta≈0.005**) | H2: value網が山札切れリスクをほぼ表現していない証拠 |
| policy データセット抽出 | `kaggle_replays/policy_net/build_features.py`・`extract_policy_dataset.py`(episode_id/player_index/相手アーキタイプ付き) | H1: 相手別の模倣一致率 |
| head-to-head harness | `league/_diag_tier1abc_head_to_head.py`(Wilson CI) | H3: 検出力の実測パラメータ |

---

## 2. H3 —— 評価ノイズ / 検出力(最優先・最安・数式のみ)

**なぜ最初か**: これが「転移しない」の正体である可能性がある。offline +0.5pt が真の勝率
50.5% を意味するなら、300 試合スクリーニングは **検出力ほぼゼロ**(95% CI 半幅 ≈ ±5.7pt)。
その場合「転移しなかった」ではなく **「測れていなかった」** が正しく、Phase 1 以降の評価計画自体を
見直す必要が出る(致命的に重要な切り分け)。

### 2.1 実装(`kaggle_replays/_diag_phase0_power.py`、新規・使い捨て)

```python
def wilson_halfwidth(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """n 試合・真勝率 p での Wilson 95%CI 半幅。"""

def detection_power(true_p: float, n: int, alpha: float = 0.05, sims: int = 20000) -> float:
    """真勝率 true_p で n 試合したとき『Wilson CI 下限 > 0.5』を満たす確率(=採用ゲート通過率)。
    二項サンプリングのモンテカルロで推定。"""

def required_n(true_p: float, target_power: float = 0.8) -> int:
    """検出力 target_power を得るのに必要な試合数(二分探索)。"""
```

### 2.2 出力(`_diag_phase0_power_results.json` + 表)

- true_p ∈ {0.505, 0.51, 0.52, 0.53, 0.55} × n ∈ {300, 600, 900, 2000, 5000} の検出力マトリクス。
- 各 true_p の required_n(power 0.8)。
- **判定**: 過去に採ろうとした施策の offline 差(+0.4〜0.5pt)が真勝率で何 pt に相当しうるか幅を置き、
  その検出に必要な n を現行ゲート(300 スクリーニング)と比較。

### 2.3 完了条件

- 「現行ゲートで検出可能な最小の真の優位」を数値で出す。
- **もし +0.5pt 級が 300 試合で検出不能と出たら**: Phase 1 以降の Stage 3 の試合数設計を
  「検出したい効果量 → 必要 n」で再定義する提案を design.md にフィードバック。

---

## 3. H1 —— 学習分布のミスマッチ

### 3.1 H1a: 相手アーキタイプ分布(上位ログ vs 自分の対戦)

**問い**: 模倣対象(上位プレイヤー)が戦っている相手の分布は、自分のエージェントが本番で当たる
相手の分布と一致しているか。ズレていれば、上位の選択を模倣しても「別の相手分布に最適な手」を
学んでいることになる。

- 実装 `kaggle_replays/_diag_phase0_field.py`:
  ```python
  def top_player_opponents(index_path, rank_threshold=50) -> Counter:
      """episodes_master から rank_at_fetch <= 閾値 の側の"相手"のアーキタイプ分布を集計。
      アーキタイプ判定は既存の meta_analysis / opponent 分類器を再利用(新規分類は書かない)。"""
  def my_faced_opponents(taxonomy_results) -> Counter:
      """_diag_loss_taxonomy_results.json 等、自分の対戦の相手アーキタイプ分布。"""
  def distribution_gap(a: Counter, b: Counter) -> dict:
      """L1 距離 / 上位差分 / KL を返す。"""
  ```
- 出力: 両分布の並置表 + 乖離指標。
- 注意: [[project_meta_analysis_findings]] の「出現数 vs 使用者数」の混同に注意(母数の取り方を明示)。

### 3.2 H1b: 相手別の模倣一致率と勝率の相関

**問い**: policy が上位プレイヤーと一致する度合いは相手アーキタイプで偏るか。偏るなら、
「よく当たるが一致率が低い相手」が転移ロスの温床。

- 実装(policy データセットに相手アーキタイプ列がある前提。無ければ build_features に列追加を検討):
  ```python
  def agreement_by_opponent(dataset) -> dict[str, float]:
      """相手アーキタイプ別の Top-1 一致率(現行 policy_weights.json で再評価)。"""
  def winrate_by_opponent(taxonomy_results) -> dict[str, float]:
      """相手アーキタイプ別の自分の勝率。"""
  # 一致率 × 勝率 × 遭遇頻度 を散布/相関で見る。
  ```
- 完了条件: 「一致率が低く・頻度が高く・勝率が低い」相手アーキタイプの特定(あれば H1 を支持)。

---

## 4. H2 —— デッキ劣位 / 構造的天井

**既存結論の集約が主**(新規実験は最小)。

- `_diag_deckout_*` / `_diag_loss_taxonomy` / `_diag_valuenet_probe` の結果を 1 表に統合。
- 特に **value網が self_deck_count にほぼ反応しない**(§1)事実を明記:
  → 山札切れリスクが価値評価に乗っていない = Phase 1 で **自山札残量/デッキアウト距離を信念ベクトルに
  入れる**ことの直接の動機になりうる(H2 と Phase 1 を繋ぐ最重要の観察)。
- **判定**: 構造的天井が主因なら「方策/価値/サーチのどれでも解けない ⇒ deck.csv 再設計が別トラックで必要」
  と結論(design.md §7.1 の判断ポイントへ)。deck.csv 変更自体は Phase 0 の scope 外(read-only 原則)。

---

## 5. 統合判定(Phase 0 の最終成果)

3 仮説の証拠を統合し、次を決める:

1. **Phase 1 で信念ベクトルに入れる情報の優先順位**:
   - H1 支持 → 相手デッキ事後分布(相手の脅威予測)を優先。
   - H2 支持(value網が山札切れ盲目)→ 自山札残量/デッキアウト距離/サイド落ちを優先。
2. **評価計画の是正**(H3 の結論): Stage 3 の試合数を効果量ベースで再設定。
3. **構造トラックの要否**(H2): deck.csv 再設計を別 issue にするか。

これらを `design.md` の決定ポイント(§7.1)に追記し、朝のユーザー確認に載せる。

---

## 6. 完了条件(Phase 0 全体)

- [ ] H3 検出力マトリクス + required_n。現行ゲートの検出可能下限を数値化。
- [ ] H1a 相手分布の乖離指標。H1b 相手別一致率×勝率(データが揃う範囲で)。
- [ ] H2 既存結論の統合表 + value網の山札切れ盲目性の明記。
- [ ] 統合判定(§5)を design.md §7.1 に反映。
- [ ] すべて read-only。production/weights/config 無変更を維持。

---

## 7. 非目的

- production 重み・config・deck.csv の変更(Phase 0 は診断のみ)。
- 新規の相手アーキタイプ分類器の実装(既存の meta_analysis/分類を再利用)。
- Phase 1 以降の学習・接続(本計画では扱わない)。
