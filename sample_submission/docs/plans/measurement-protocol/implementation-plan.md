# 実装計画書: 測定プロトコル ①SPRT + 勝ち筋ロギング + ②共変量

作成日: 2026-07-25
種別: **実装計画書**(設計書 [design.md](./design.md) の lever ①(SPRT)・②(サイド差=共変量)・
勝ち筋ロギングを、関数シグネチャ・分類契約・完了条件まで落とす。[[feedback_design_vs_implementation_plan]])
ステータス: **計画のみ。production 未変更。コード変更前のレビュー用。**
control: **現行 production config**(`ml_policy` 既定 config / `policy_weights.json` 等、byte 無変更)
**push ポリシー**: 本ハーネスは**ローカル評価専用・git 追跡外**(scratchpad または未追跡ローカル dir)。
`git add` しない。提出物・共有コードに一切触れない(ユーザー要件「gitにpushされないならok」)。

---

## 0. スコープと非目的

**やること**: 候補 config と control を head-to-head で1試合ずつ対戦させ、
- 勝敗を **SPRT** に食わせて `採用 / futility / 継続` を逐次判定(固定 N ゲートの置換、§design ①)。
- 各試合の **勝ち筋種別**(prize-out / no-pokemon / deckout / error)と **サイド差**を記録。
- 最終レポートで勝率(推定量)＋ CI に加え、**サイド差を回帰調整した CI**(②、分散削減)を併記。

**非目的(変えない)**:
- production の `main.py` / weights / config / `deck.csv` / **`cg/`**(§design 1.3、改変禁止)。
- 共有評価コード(`league/run_league.py`・`round_robin.py`・`run_ablation.py`)。**新規ファイルのみ**、
  必要なら担当者へ事前確認([[feedback_respect_ownership_boundaries]])。
- CRN/fork(§design ③-b、WSL2 前提の後続フェーズ)。**本版は unpaired SPRT**。
- 全勝ち筋に単調な合成 outcome の設計(§design ②、後続)。
- **サイド差による分散削減(CUPED 等)は本版では行わない**(§1.0-3 で撤回)。サイド差・勝ち筋は
  **記述的診断としてのみ**記録する。

---

## 1. 確定した検定契約 + コード確認(2026-07-25 レビュー反映)

### 1.0 確定した検定パラメータ

**レビューで確定(2026-07-25)。以下を本ハーネスの正式既定とする。**

| 変更種別 | delta_min | α | β |
|---|---:|---:|---:|
| 小変更(重み / 特徴 / config 微調整) | 0.03 | 0.05 | 0.10 |
| 大規模変更(新アーキ / ISMCTS / DMC 等) | 0.05 | 0.05 | 0.10 |

**α/β は変更種別で階層化しない**(理由):
- α/β まで種別ごとに変えると評価契約が複雑化する。
- 「どの程度の改善を要求するか」は **delta_min のみ**が担う(大規模ほど高い bar = 0.05)。
- より厳しい最終 Champion gate が要る場合は、**後続フェーズで別途設計**する(本版の非目的)。

**1.0-2 判定は3種類(N_max 到達時に H0/H1 へ強制分類しない)**:

| decision | 意味 | 条件 |
|---|---|---|
| `PROMOTE` | 採用(H1 側境界を越えた) | LLR ≥ A = ln((1−β)/α) |
| `FUTILITY` | 不採用確定(H0 側境界を越えた) | LLR ≤ B = ln(β/(1−α)) |
| `TRUNCATED_NO_PROMOTE` | **不採用だが H0 確定ではない**(予算 N_max 到達・境界未到達) | n ≥ N_max かつ未交差 |

SPRT の境界に未到達である以上、統計的には H0/H1 いずれも確定していない。**LLR の符号による強制分類
(旧 H1_trunc/H0_trunc)は撤回**。`TRUNCATED_NO_PROMOTE` は運用上「採用しない」に倒すが、`FUTILITY`
(効果が δ_min 未満と確定)とは**区別して記録**する(再試行・N_max 引き上げ・設計見直しの判断材料)。

**名目 power と実効 PROMOTE 率は別物として扱う(重要)**:
- **名目 SPRT power = 1−β = 0.90**: 打ち切りなし(N_max=∞)の漸近保証。「β=.10」はこれを指す。
- **実効 PROMOTE 率**: N_max 打ち切り込みで実際に PROMOTE になる割合。付録C の MC 実測では
  **N_max=3000 で真 p1=0.53 のとき 84%**(打ち切り ~6% は TRUNCATED_NO_PROMOTE になり PROMOTE に数えない)。
- **∴ 「β=.10 = 実効 power .90」とは書かない**。レポート・doc では両者を区別して記す。
- **N_max は検出力に効く予算パラメータ**。既定 3000(desktop ~0.3s/試合で ~15分、ASN 中央は遥かに下)。
  p1 で実効 PROMOTE 率を名目に近づけたい大規模比較では N_max を引き上げる(§4)。

**1.0-3 サイド差 CUPED は撤回(統計的理由)**:
終局サイド差は **treatment(config)後に実現する outcome 変数**であり、CUPED が要求する
**pre-experiment 共変量(treatment に影響されない)ではない**。post-treatment 変数で勝率を調整すると
推定量が「勝率」から「サイド差条件付き量」へずれ、**config の勝率効果を bias する**(媒介/collider)。
特にサイド差経由で勝つ config はその利得が調整で消され、deckout 経由の config は逆方向にずれる。
→ **推定量は生の二値勝率のみ。SPRT も CI もサイド差で調整しない。** サイド差・勝ち筋は
「config がどのチャネルで効くか」を見る**記述的診断**として残す(§design ② の deckout 盲目性の把握用)。
将来の妥当な分散削減は **CRN(fork、§design ③-b)と手番均衡(g%2、実装済)**のみ。

### 1.1 config 注入(head-to-head の要)

### 1.1 config 注入(head-to-head の要)
`ptcg_ai/ml_policy/ml_policy_agent.py:105` `def agent(obs: Observation, config: dict | None = None) -> list[int]`。
- 明示 config を渡すと**同一プロセス内で別 config の2エージェントを混線なく**駆動できる(既存テスト
  `tests/unit/test_ml_policy_agent.py::test_explicit_config_is_independent_per_call` で独立性実証済)。
- `config["policy_weights_path"]` で候補重みを別枠キャッシュ(既定 `_model` に触れない)。
- **駆動契約**: `agent_cand = lambda obs: ml_policy_agent.agent(obs, config=cfg_cand)`、
  `agent_ctrl = lambda obs: ml_policy_agent.agent(obs, config=cfg_ctrl)`。obs は
  `to_observation_class(obs_dict)`(cg は dict を返す)。

### 1.2 対戦ループと終局(コード確認: `kaggle_replays/rl/rollout.py`)
`cg.game.battle_start(deck0, deck1)` → `battle_select(action)` ループ。
**ライブ対戦では終局時 `obs.current.result` に勝者(0/1)が入る**(rollout.py が実運用で使用。リプレイの
result=-1 は記録仕様の癖で、ライブには当てはまらない ← §design 付録B の注意はライブでは解消)。
終局盤面(`obs.current.players[i]` の `prize`/`deckCount`/`active`/`bench`)は**最終手適用後**で正確。

### 1.3 勝ち筋分類(終局 State から。フィールド名はプローブ確認済)
**併発情報を捨てない**: 各終局条件を独立に評価して `terminal_flags`(集合)に全部残し、そこから
表示用の `primary_win_condition` を優先順で1つ選ぶ。`loser = 1 - winner`。
- フラグ(併発しうる): `prize_out`=`len(players[winner].prize)==0` / `deckout`=`players[loser].deckCount==0`
  / `no_pokemon`=`len(players[loser].active)+len(players[loser].bench)==0` / `error`(1.4)。
- `primary_win_condition` の優先順: `error` > `prize_out` > `no_pokemon` > `deckout` > `unknown`
  (優先順は Step2 の実測で妥当性確認。併発は `terminal_flags` に保持されるので優先順の選択で情報は失われない)。
**完了条件で要検証(Step2)**: ライブ N 試合で primary/flags が実際の勝因と整合するか、併発の頻度。

### 1.4 サイド差(記述的診断のみ・推定量には使わない)
`prize_margin_p0 = len(players[1].prize) - len(players[0].prize)`(player0 視点、正=player0 が取り分多)。
勝ち筋種別と併せ**記録のみ**(§1.0-3 でCUPED撤回)。error 決着は margin 無効。SPRT・勝率 CI は
二値勝敗のみで計算する。

### 1.5 ファイル配置(push 無し)
`kaggle_replays/measurement/`(**新規・未追跡**)に置き `git add` しない。または scratchpad。
実運用の決定関数は §2 Step1 の `SPRT` クラスをそのまま利用。

---

## 2. ステップ分解(各ステップに完了条件)

### Step 0: config 注入スモーク(対戦少数)
`agent_cand`/`agent_ctrl` を control=control(同一 config)で数十試合走らせ、勝率が ~50% 近傍・error 0 を確認。
- **完了条件**: 同一 config 対戦で勝率が 50% を CI 内に含む・例外 0・両手番(g%2)で対称。

### Step 1: SPRT モジュール + 単体テスト
`sprt.py`:
```python
class SPRT:
    def __init__(self, p0=0.50, delta_min=0.03, alpha=0.05, beta=0.10, n_max=3000): ...
    def update(self, win: int) -> str | None:
        # 'PROMOTE' | 'FUTILITY' | 'TRUNCATED_NO_PROMOTE' | None(継続)。§1.0-2。
        # 境界未到達での N_max 到達は TRUNCATED_NO_PROMOTE(符号で H0/H1 に強制分類しない)。
    @property
    def report(self) -> dict                     # n, s, winrate, llr, A, B, decision
```
`delta_min` は呼び出し側が**変更種別で選択**(§1.0): 小変更=0.03 / 大規模=0.05。α=.05,β=.10 固定。
- **完了条件**: MC で真 0.50→PROMOTE ≒α、真(0.5+δ_min)→PROMOTE ≒(1−β)−(打ち切り率) を再現
  (付録C 値と一致。N_max=3000 では p1=0.53 の PROMOTE は ~84%)。境界 A=ln((1−β)/α)・
  B=ln(β/(1−α)) と3分類の遷移を assert するテスト green。TRUNCATED で符号強制分類しないことを確認。

### Step 2: ライブ対戦ランナー + 終局抽出
```python
@dataclass
class GameResult:
    winner: int | None; primary_win_condition: str; terminal_flags: list[str]  # 併発を捨てない(§1.3)
    prize_margin_p0: int; turns: int; error: str | None
def play_one(agent0, agent1, deck0, deck1, learner_index_first: bool) -> GameResult
```
`battle_start/select/finish` ループ。`obs.current.result != -1` で終局→ §1.3/1.4 抽出。エージェント例外は
捕捉し `error` + 相手勝ちに(1.4)。`battle_finish()` は finally。
- **完了条件**: ライブ N(≥100)試合で winner が `current.result` と一致、`terminal_flags`/`primary` の内訳が
  妥当(自デッキで deckout 比率が既知傾向 [[project_deckout_loss_cause]] と整合)、**併発の頻度を観測**、
  error ハンドリング動作。

### Step 3: head-to-head ドライバ(SPRT 配線)
```python
def run_sprt_ab(cfg_cand, cfg_ctrl, deck, delta_min, alpha=0.05, beta=0.10,
                n_max=3000, alternate_sides=True, note="") -> dict
```
`play_one` を回して勝敗を `SPRT.update()`。手番は g%2 で交互(§design ③ 層化、CRN 非依存で無条件)。
`PROMOTE`/`FUTILITY`/`TRUNCATED_NO_PROMOTE` のいずれかで停止(§1.0-2)。
**レポート JSON(再現性メタを可能な範囲で保存)**:
- 結果: decision(3種), N, winrate+Wilson CI(**二値のみ**), 勝ち筋内訳(`terminal_flags`/`primary`, 記述的), margin 分布(記述的)。
- メタ: candidate/control の config path + `policy_weights_path` + **各ファイルの hash**、delta_min, alpha, beta,
  n_max, timestamp、(取得できれば)git ref。
- **完了条件**: 候補 vs control を実走 → 3分類 decision + メタ付きレポート JSON 出力。中断/再開で N が壊れない。

### Step 4: 勝ち筋・サイド差の記述的サマリ(推定量には使わない)
```python
def outcome_breakdown(results: list[GameResult]) -> dict
# 勝ち筋種別ごとの件数/勝率、サイド差の分布。config がどのチャネル(prize/deckout/board)で
# 効くかを見る診断のみ。勝率推定量・CI・SPRT には一切影響させない(§1.0-3 で CUPED 撤回)。
```
- **完了条件**: レポートに勝ち筋内訳とサイド差分布が出る。**勝率 CI が二値のみで計算され、
  サイド差で調整されていないことをテストで固定**(将来の誤 CUPED 再導入を防ぐ回帰テスト)。

### Step 5: A/A integration smoke(+ 可能なら過去 config)
**A/A は統計テストではなく integration smoke** として扱う: cfg_cand=cfg_ctrl で `run_sprt_ab` を回し、
config 混線なし・runner・winner 抽出・手番交互・error 処理・レポート生成が**通しで動く**ことを確認する。
- **注意(統計)**: 単発 A/A が必ず FUTILITY になることは**完了条件にしない**。真 p=0.50 でも 1 回の実行は
  ~4〜5%(=α)で PROMOTE、~4%で TRUNCATED になり得る(付録C)。**α 較正の確認は Step1 の MC で行う**。
- 可能なら過去 negative の config を1つ実走し、到達 decision と N を記録(付録C のシミュレーションと突合)。
- **完了条件**: A/A が**エラー無く 3分類のどれかで停止**し、メタ付きレポートが再現可能に出力される
  (decision の値は問わない)。結果を `results/` に文書化(production 無変更を git で確認)。

---

## 3. 未着手(後続フェーズ・本計画外)

- **pre-treatment 共変量による回帰調整(将来 optional)**: 先攻/後攻・初期盤面・初期手札特徴など
  **treatment(config)前に決まる**変数は、post-treatment のサイド差(§1.0-3 で撤回)と違い回帰調整に
  使っても不偏。分散削減の妥当な候補として将来調査に残す。**MVP では実装不要**(現状の「サイド差・勝ち筋は
  診断専用」で十分)。手番は既に g%2 で均衡させているので、まずはそれで足りるか要検証。
- **合成 outcome**(全勝ち筋に単調)の設計 → サイド差を妥当な形で活かす道(§design ②)。
- **CRN**(③-b fork / WSL2)= 完全 CRN で ASN を更に下げる(§design 1.3/3-b、要 Linux)。
- **強い固定参照相手**(⑤)= worthy opponent、アルゴリズム本体トラックと合流(§design ⑤)。
- **より厳しい最終 Champion gate**(§1.0 の理由 3): 必要になれば別途設計。
- **vs-field 採用確認**(外部妥当性ガード、§design)。`match_context._load_own_deck_ids` の deck.csv 固定が障壁。

---

## 4. 残る設計判断(Step 0 開始には不要・任意)

**検定契約は §1.0 で確定済**(α/β/delta_min/3分類判定)。実装開始前に残るのは運用上の小事のみ:

- **N_max の値**: 既定 3000(§1.0 参照)。p1 で名目 power に近づけたい大規模比較は引き上げ可(検出力↑・予算↑)。
  値の変更のみで契約は不変。
- **駆動 config の置き場**: 候補重み/config をどのファイルに置くか(既存 `configs/` 慣習に倣うが**未追跡**)。

→ 上記は既定値で **Step 0 から着手可能**。
