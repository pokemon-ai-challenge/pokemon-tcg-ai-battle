# 実装計画: RL による対戦相手強化(PoC 先行)

作成日: 2026-07-24
種別: **実装計画(関数シグネチャ・データフロー・完了条件つき。方向性は `design.md`・
`NIGHT-SESSION-2026-07-24.md` §10-11 で決定済)**
親設計書: `docs/plans/neural-agent/design.md`(構想#4 の最初の具体化)
ステータス: 承認済(ユーザー「その方針で」2026-07-24)。M0 から着手。

---

## 0. 目的と背景(なぜこれを最初の RL にするか)

**評価の本当の障害は「対戦相手が弱すぎる」こと**(NIGHT-SESSION §10):
- round_robin の対フィールド eval で production(alakazam)は既に 68.8%。相手が弱く gate として機能しない。
- 弱い相手(対フィールド勝率: dragapult 16.4% / archaludon 32.5% / mega_lucario_ex 38.7%)。
- **rule_based に持たせても悪化**(dragapult 7.5% < 模倣ボット 21%)。**データ不足でもない**
  (dragapult 41,580 サンプル > 強い crustle 26,909)。→ **模倣学習の天井**であり、勝率目的の RL が適切。

**カリキュラムとしての位置づけ**:
1. 弱い相手を RL で強化(**伸びしろ大 → 改善が測りやすく H3 の検出力問題を回避、RLパイプライン(#4)を
   低リスクに構築、後で production RL 用の強ベンチが手に入る**)。
2. 既に強い相手(crustle 70.9%、rocket_mewtwo は alakazam に55.7%)は RL しない。
3. 強化した相手を**凍結**してベンチ化 → その上で production を RL(将来)。

## 1. スコープ

- **In**: dragapult_ex(最弱=最大の伸びしろ)を PoC 対象に、BC初期化 + PPO の self-play で
  対 production-alakazam 勝率を上げられるかを検証。RLパイプラインの最小構築。
- **Out(PoC では扱わない)**: 全アーキタイプへの横展開、production 自体の RL、
  ポテンシャルベース報酬整形(まず sparse terminal のみ)、maxCount>1 の複数選択の学習
  (既存 greedy に委譲)、Transformer 化(別フェーズ)。
- **境界**: コードは新規 `kaggle_replays/rl/`(オフライン学習インフラ。提出物 `sample_submission/` には
  一切入れない)。production の weights/config/deck.csv/cg/ は無変更。`rule_based/`・`action_selection/` は
  変更しない([[feedback_respect_ownership_boundaries]])。

## 2. 鍵となる設計判断(推奨デフォルト)

### 2.1 方策表現 — 既存模倣モデルをそのまま torch 化
既存 `PolicyModel` は「選択肢ごとにスコア → argmax」で、学習時は **listwise softmax CE**(=既に
softmax-over-options モデル)。よって **torch で同一アーキテクチャ**を組めば、
- 方策 π(a|s) = softmax(option_scores / τ)。
- **BC初期化 = 既存 `policy_weights_dragapult_ex.json` を torch にロードするだけ。**
- フォワードは `policy_model.py._forward` と厳密一致(標準化 → state'++option'++card_embedding → MLP、
  最終層線形)。

### 2.2 critic(価値) — value_weights から初期化
PPO の baseline V(s) は既存 `value_model.py` + `value_weights.json`(test AUC 0.746、encoder 共有)を
torch 化して初期化し、RL 中に fine-tune。encoder(166次元 state)は方策と共有。

### 2.3 対戦相手(PoC) — 固定 production-alakazam
PoC は **dragapult(学習側)vs production-alakazam(固定)**。最も鋭い信号(現状 ~21% vs alakazam)。
汎化(対フィールド分布サンプリング=OSFP)は PoC 成功後。相手固定なので**非定常性が無く**デバッグ容易。

### 2.4 報酬 — まず sparse terminal
勝ち +1 / 負け 0(または ±1)。割引 γ≈1(エピソード短い、平均 ~13ターン)。
ポテンシャル整形(V による reward shaping)は PoC で信号が出てから(方策不変性は保つが設計が要る)。

### 2.5 行動空間 — 単一選択のみ学習
maxCount==1 の意思決定点で方策が行動。maxCount>1 は既存 greedy(`_greedy_multi_select` 相当)に委譲
(学習対象外、環境の一部とみなす)。PoC の既知の制約として明記。

### 2.6 throughput — 並列ワーカーでロールアウト収集
run_league と同じ worker 並列で self-play を回し、torch 方策は定期同期(A2C/IMPALA 的)。
GPU は方策/価値の forward・PPO 更新に使用。env(cg)は CPU プロセス。

### 2.7 **出力の可搬性(重要)** — RL 結果を JSON schema へ書き戻す
方策アーキが単純 MLP なので、**RL 済み torch 重みを既存 `policy_weights_*.json` schema にエクスポート**
できる。→ ベンチ評価(round_robin)も将来のデプロイも**既存の pure-Python 推論のまま**(eval/提出に
torch を持ち込まない。§design.md §3.3 のデプロイ問題を回避)。

## 3. コンポーネントと関数シグネチャ(`kaggle_replays/rl/`)

```
kaggle_replays/rl/
  torch_policy.py    # 既存JSON重み <-> torch module。パリティ保証。
  torch_value.py     # value_weights <-> torch critic。
  rollout.py         # self-play でトラジェクトリ収集(cg engine 経由)。
  ppo.py             # PPO 更新ループ。
  export_weights.py  # torch -> policy_weights JSON(既存schema)。
  train_dragapult_poc.py  # PoC ドライバ。
```

```python
# torch_policy.py
class TorchOptionPolicy(nn.Module):
    def __init__(self, state_dim, option_dim, card_id_max, embed_dim, hidden_sizes): ...
    @classmethod
    def from_json(cls, weights_path) -> "TorchOptionPolicy":
        """既存 policy_weights.json を読み、標準化パラメータ・card埋め込み・各層を移植。"""
    def option_scores(self, state_feat, option_feats, card_ids) -> Tensor:  # [n_options]
        """policy_model._forward と厳密一致するスコア(標準化込み)。"""
    def distribution(self, ..., temperature=1.0) -> Categorical: ...
    def to_json_payload(self) -> dict:
        """RL後: 既存schemaの dict(standardization/card_embedding/layers)を返す。"""

# rollout.py
def collect_trajectories(policy, opponent_agent, deck_learn, deck_opp,
                         n_games, seed_start, workers) -> list[Trajectory]:
    """cg self-play。learn側の各単一選択点で (state_feat, option_feats, card_ids,
    chosen_idx, logprob, value) を記録。終局で reward を全ステップに配る(sparse terminal)。
    encoder は sample_submission.ptcg_ai.learning.encoder を再利用(train/runtime parity)。"""

# ppo.py
def ppo_update(policy, value, batch, clip=0.2, epochs=4, lr=3e-4, ...) -> dict:  # metrics

# export_weights.py
def export(policy, out_json_path) -> None: ...
```

## 4. マイルストーン(すべて検証可能・可逆)

- **M0 — torch policy パリティ**: `TorchOptionPolicy.from_json` が、複数の実 obs サンプルで
  `PolicyModel.score_options` と **数値一致**(atol 1e-5)。→ 単体テストで assert。BC初期化の土台。
- **M1 — ロールアウト健全性**: `collect_trajectories` が dragapult(torch方策, BC初期化)vs
  production-alakazam を回し、**round_robin と同水準の勝率(~21% vs alakazam)を再現**
  (torch方策 = 既存模倣と等価であることの動的確認)。エラー0・logprob/価値が妥当。
- **M2 — PPO で上昇**: PPO ループで dragapult 勝率が上昇。**PoC 成功条件: 対 alakazam を
  ~21% → 35%+**(有限予算内で単調上昇+CI下限が baseline 超え)。
- **M3 — コスト計測 & ベンチ書き戻し**: 「+10% あたり self-play ゲーム数」を計測し横展開コストを試算。
  RL済み dragapult を `policy_weights_dragapult_ex_rl.json` にエクスポートし、round_robin に投入して
  **対フィールド勝率が上がる**ことを確認(凍結ベンチ候補)。

## 5. 完了条件(PoC 全体)

- [ ] M0: パリティ単体テスト green。
- [ ] M1: BC初期化 dragapult が既存模倣と同勝率を再現。
- [ ] M2: PPO で対 alakazam 35%+ 達成 or 「天井で頭打ち」を明確化(どちらも学び)。
- [ ] M3: games-per-gain 試算 + JSON エクスポート + round_robin 反映。
- [ ] 全て `kaggle_replays/rl/` 内・production 無変更・torch は学習時のみ(eval/提出は pure-Python)。

## 6. リスクと撤退基準

- **RL が天井を破れない**(dragapult が本質的に難デッキ): M2 で上昇が見えなければ、
  「模倣天井は RL でも破れない」という結論(=別の相手強化策 or デッキ側を疑う)。**PoC の目的はこの判定。**
- **throughput**: PoC は ~10^5–10^6 env step = desktop で一晩規模。全アーキ+production へは長期。
  M3 のコスト試算で go/no-go。
- **ボット≠人間**: 強ベンチでも実フィールドの完全代理ではない。最終判定は Kaggle 提出([[project_pimc_prod_validation]] の教訓)。
- **非定常性**(将来 OSFP 化時): PoC は相手固定で回避。リーグ化は成功後に別計画。

## 7. 非目的

- production(alakazam)の RL、全アーキ横展開、Transformer化、ポテンシャル報酬整形、
  maxCount>1 の学習(いずれも PoC 後 or 別フェーズ)。
- `sample_submission/` への torch 依存の持ち込み(禁止。eval/提出は pure-Python 維持)。
