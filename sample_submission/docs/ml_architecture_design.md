# ポケモンカードゲームAI 機械学習統合アーキテクチャ設計書

> **改訂メモ（2026-06-28）:** 本設計書は当初「Flat MCS + ランダムロールアウト」を既存実装と仮定して
> 書かれていたが、現行の `sample_submission` 実装は **探索を持たない 1 手読みのルールベース proposal 方式**である。
> このギャップを解消するため、§1 の現状認識・Phase 構成・`evaluate()` の API 名を現行コードに合わせて改訂した。
> 目標アーキテクチャ（情報推論 + ISMCTS + 評価関数 / NN）は不変。**具体的な移行手順は `docs/ml_integration_plan.md` を参照。**

## 1. 現状分析と設計思想

### 現在の構成（実コードベース）

```
Observation
  → router.py（SelectContext で振り分け）
    → main_turn.py（各 priorities/*.py が MainActionProposal(action, score, label) を提案）
      → choose_best_proposal（score 最大を 1 つ選ぶ。探索・ロールアウトは無し）
```

- **探索エンジンは未導入**（MCTS / Flat MCS は存在しない）。
- **ロールアウトも存在しない**（ランダム・ヒューリスティック問わず）。
- **全局面評価 `evaluate(state) -> float` は未実装**。`src/decision/evaluation/` にあるのは
  行動単位の特徴量ヘルパ（`attacker_priority_score`, `is_likely_knocked_out_next_turn` 等）であり、
  盤面全体の勝率評価ではない。
- 相手・自分の非公開情報の推定器も未導入（`cg.api.search_begin/step` は API として存在するが未使用）。
- デッキ知識は `src/knowledge/meta_decks.py` に採用率付きで整備済み（後述の推定器の土台になる）。

### 探索を導入する際に避けるべき既知の落とし穴（PDF調査より）

現状コードには探索が無いため、以下は「現在の不具合」ではなく **これから探索を載せる際に踏んではいけない罠**として記録する。

- ランダムロールアウトはシナジー依存型ゲームと乖離する → Q値が信頼できなくなる（だから最初からヒューリスティックロールアウトを使う）
- 深さ制限が浅い（例: 30手 ≈ 1.5〜2ターン）と先が読めない → 動的深さ制御が要る
- 終端評価（`evaluate()`）への 100% 依存はノイズを増幅する → 評価関数の質を先に上げる
- Flat MCS は Strategy Fusion（速攻 vs コントロール双方に中途半端な手）を招く → 最初から ISMCTS を採る

### 設計の基本方針

> **「探索の質（シミュレーションのリアリティ）を極限まで高める」**  
> **「不完全情報の不確実性を、推論で積極的に狭める」**

時間制約（10分）・計算資源制約（CPU/RAM）を前提に、現行のルールベース実装を土台として
以下の優先順位でコンポーネントを段階的に「追加」する（既存の proposal 方式は壊さず並走させる）。

```
Phase A（今すぐ）: 情報推論基盤（推定器）+ 全局面 evaluate() + TimeManager
                   ※ 現行ルールベースは温存。探索はまだ載せない。
Phase B（1ヶ月）: ISMCTS 導入。ロールアウト方策には既存 priorities/*.py を再利用。
                   推定器を決定化（Determinization）に統合。
Phase C（研究枠）: Policy/Value Network + 模倣学習による蒸留
```

---

## 2. システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                         agent()  エントリポイント                  │
└───────────┬─────────────────────────────────────────────────────┘
            │ Observation
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     情報推論レイヤー（Phase A）                     │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  相手デッキ推定モジュール  │  │  自分の山札・サイド推定モジュール   │ │
│  │  OpponentDeckEstimator  │  │  SelfStateEstimator             │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└───────────┬─────────────────────────────────────────────────────┘
            │ EnrichedContext（推定情報付き盤面）
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     探索エンジンレイヤー（Phase B）                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             ISMCTS（Information Set MCTS）                │   │
│  │  ・決定化（Determinization）: デッキ推定結果を使用          │   │
│  │  ・Heuristic Rollout（ルールベースでプレイアウト）           │   │
│  │  ・B1 Heuristicによる事前枝刈り                            │   │
│  │  ・残り時間に応じた動的深さ制御                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────┬─────────────────────────────────────────────────────┘
            │ 候補手 + スコア
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     評価レイヤー（Phase A強化 / Phase C）            │
│  ┌──────────────────┐   ┌──────────────────────────────────┐    │
│  │  強化evaluate()   │   │  Policy/Value Network（Phase C）  │    │
│  │  （ルールベース）    │   │  （Transformer / 模倣学習）        │    │
│  └──────────────────┘   └──────────────────────────────────┘    │
└───────────┬─────────────────────────────────────────────────────┘
            │ 最終選択インデックス
            ▼
         action返却
```

---

## 3. 情報推論レイヤー

### 3-1. 相手デッキ推定モジュール（`OpponentDeckEstimator`）

#### 役割
- 相手の使用カードから「どのデッキアーキタイプか」を確率的に推定
- ISMCTS の決定化（Determinization）に使うカードプールを精緻化
- `evaluate()` に「相手の最大打点」「ボスの指令保有確率」を提供

#### 設計

```python
class OpponentDeckEstimator:
    """
    メタデッキリスト（事前知識）をベースにベイズ推論で
    相手デッキを推定する。
    """
    
    def __init__(self, meta_decks: dict[str, list[int]]):
        # meta_decks: {"メガルカリオex": [card_id, ...], ...}
        self.meta_decks = meta_decks
        self.prior: dict[str, float]  # 各デッキの事前確率（均等初期化）
        self.seen_cards: Counter      # 相手の公開カード履歴
    
    def update(self, newly_seen_cards: list[int]) -> None:
        """相手がカードをプレイ/トラッシュするたびにベイズ更新"""
        # リストに存在しないカードが見えたら → そのデッキ確率を0に
        # 存在するカードなら → likelihoodに比例して事後確率を更新
    
    def sample_opponent_hand(self, hand_size: int) -> list[int]:
        """
        ISMCTS の決定化用: 最確デッキから公開済みカードを除いた
        プールからサンプリング（完全ランダムより大幅に精度向上）
        """
    
    def get_max_damage(self) -> int:
        """相手の推定最大打点（evaluate()に渡す）"""
    
    def get_boss_probability(self) -> float:
        """ボスの指令保有推定確率（ベンチリスク評価に使用）"""
```

#### データ（既に実装済み）

デッキ知識は **既に `src/knowledge/meta_decks.py` に整備済み**であり、設計当初に想定していた
`{name: [60枚ID]}` という単純な dict より**リッチな構造**を持つ。`OpponentDeckEstimator` はこれを
新規に作り直さず、そのまま土台として使う。

```python
# src/knowledge/meta_decks.py（抜粋・既存）
@dataclass(frozen=True)
class MetaDeckEntry:
    archetype_id: str
    rep_card_ids: tuple[int, ...]          # 代表デッキ60枚
    key_cards: tuple[int, ...]             # アーキタイプ識別シグネチャ
    card_inclusion_rate: dict[int, float]  # Card ID → 採用デッキ割合
    card_avg_copies: dict[int, float]      # Card ID → 平均採用枚数
    def is_reliable_for_estimation(self) -> bool: ...
    def probable_card_ids(self, min_inclusion=0.5) -> list[int]: ...

# 現環境の Tier 1〜2 を網羅（dragapult_ex / hydrapple_ex / olivine_ex /
# takelraiko_ex / megarucario_ex / orgepon_bullet / hatterene / crustle ...）
REALWORLD_META: dict[str, MetaDeckEntry] = { "dragapult_ex": ..., ... }
get_all_meta_decks() -> dict[str, MetaDeckEntry]
```

→ `OpponentDeckEstimator` は `get_all_meta_decks()` を読み込み、`key_cards` / `card_inclusion_rate`
を尤度に使ったベイズ更新を載せるだけでよい。

**拡張（Phase C）:** ルールベース分類 → Transformer デッキ分類器に置換し、
Tech カード（環境外採用）を含むデッキでも精度を担保。

---

### 3-2. 自分の山札・サイド推定モジュール（`SelfStateEstimator`）

#### 役割
- 盤面・手札・トラッシュ・場に出たカードから**残りの山札構成**を推定
- 残りのサイドカードを推定（何が落ちているか）
- これにより：
  - 「このターンにキーカードを引ける確率」を計算できる
  - ISMCTS/Heuristic Rollout で「ありえない手」（サイド落ちのカードを使うプラン）を事前排除できる

#### 設計

```python
class SelfStateEstimator:
    """
    デッキレシピと公開済みカード（手札・場・トラッシュ）から
    山札・サイドの残り構成を推定する。
    """
    
    def __init__(self, my_deck_recipe: list[int]):
        self.recipe_counter = Counter(my_deck_recipe)  # 60枚の構成
    
    def update(self, obs: Observation) -> None:
        """毎ターン、見えているカードを引いて山札推定を更新"""
        visible = (
            obs.current.players[my_idx].hand +
            obs.current.players[my_idx].active +
            obs.current.players[my_idx].bench +
            trash_cards  # ログから追跡
        )
        self.remaining_deck = self.recipe_counter - Counter(visible)
    
    def probability_in_deck(self, card_id: int) -> float:
        """そのカードが山札にいる確率（超幾何分布で計算）"""
    
    def sample_side_cards(self) -> list[int]:
        """
        サイドカード推定: 残り山札から6枚分をサンプリング
        → ISMCTS の決定化に利用
        """
    
    def key_card_reachable(self, card_id: int, draw_count: int) -> float:
        """draw_count ドローでそのカードを引ける確率"""
```

**活用例:**
- 「進化先がサイドに落ちている可能性が高い」→ `evaluate()` でそのラインの評価を下げる
- 「ネストボールが山札にほぼない」→ そのサーチ手を探索から枝刈り

---

## 4. 探索エンジンレイヤー（ISMCTS）

### 4-1. Heuristic Rollout（Phase B の構成要素）

> 注: 現行コードにはロールアウトが存在しないため「ランダム→ヒューリスティックへの置換」ではなく、
> **ISMCTS を載せると同時に、最初からヒューリスティックロールアウトを実装する**。
> プレイアウト方策はゼロから書かず、**既存の `src/decision/main_turn_parts/priorities/*.py` を再利用**する
> （= 現行のルールベース agent がそのままロールアウト方策になる）。導入コストが最低で効果が最大。

```python
def heuristic_rollout(state) -> float:
    """
    ランダムではなく、既存 priorities の優先順位でプレイアウトを進める:
    1. 倒せる相手ポケモンがいれば攻撃
    2. 進化できるなら進化
    3. エネルギーを貼れるなら貼る
    4. 有効なグッズ・サポートを使う
    5. どれもなければパス相当の最善行動
    """
```

### 4-2. ISMCTS（Phase B）

```
ISMCTSの1イテレーション:
  1. 決定化（Determinization）
     └ OpponentDeckEstimator.sample_opponent_hand() で相手の手札を仮定
     └ SelfStateEstimator.sample_side_cards() で自分のサイドを仮定
  2. 選択（UCB1で有望ノードを選択）
  3. 展開（新ノード追加）
  4. シミュレーション（Heuristic Rollout）
  5. バックプロパゲーション（同一Information Setのノードを共有更新）
```

### 4-3. 事前枝刈り（B1 Heuristic活用）

cabt エンジンが返す合法手リストはすでに B1 Heuristic でソート済み。
インデックス下位の「明らかな悪手」を探索前に除外する。

```python
def prune_options(options: list[Option], top_k: int = 10) -> list[Option]:
    """
    B1 Heuristicの上位 top_k 手のみを探索対象とする。
    合法手が多い局面（グッズ多数など）での探索空間爆発を防ぐ。
    """
    return options[:top_k]  # エンジンが返す順序を信頼
```

### 4-4. 時間制約への対応

10分制限 × Kaggle の CPU 制約を考慮した動的時間配分。

```python
class TimeManager:
    TOTAL_LIMIT_SEC = 580  # 10分 - バッファ20秒
    
    def __init__(self):
        self.start_time = time.time()
        self.elapsed_by_turn: list[float] = []
    
    def budget_for_this_turn(self, turn: int, remaining_prizes: int) -> float:
        """
        残り時間 / 推定残りターン数 で1ターンの予算を算出。
        サイドが残り少ない（終盤）ほど1ターンに多く割り当てる。
        """
        elapsed = time.time() - self.start_time
        remaining = self.TOTAL_LIMIT_SEC - elapsed
        estimated_turns_left = remaining_prizes * 2  # 大まかな推定
        return min(remaining * 0.3, remaining / max(estimated_turns_left, 1))
    
    def should_stop(self, budget: float, turn_start: float) -> bool:
        return time.time() - turn_start >= budget * 0.9  # 90%消費で終了
```

---

## 5. 評価レイヤー（強化 `evaluate()`）

### 5-1. Phase A: ルールベース強化 evaluate()

> **API 整合メモ:** 旧版の疑似コードは実 API（`cg/api.py`）と一致していなかったため修正した。
> - `Pokemon.hp_remaining` は存在しない → 現在 HP は `Pokemon.hp`、最大は `Pokemon.maxHp`。
> - 場のエネルギー数は `Pokemon.energies`（`list[EnergyType]`）の長さで数える。
> - 進化段階は `Pokemon.preEvolution` や `CardData.stage1/stage2` から判定する。
> - `prize` は「残りサイド」なので `6 - len(prize)` は「取った枚数」（初期 6 枚前提でよい）。
> - 評価対象の型に注意: `Observation.current` は `State` だが、探索中に `search_step` が返すのは
>   `SearchState`（別型）。Phase B で探索へ組み込む際は、評価関数がどちらの型を受けるか統一すること。
> - `is_likely_knocked_out_next_turn`（`evaluation/board_features.py`、既存）が
>   「相手の脅威打点 vs 自陣 HP」を既に概算しているので、生存リスク評価はこれを再利用できる。

```python
def evaluate(state, my_idx: int, opponent_estimator: OpponentDeckEstimator) -> float:
    me = state.players[my_idx]
    opp = state.players[1 - my_idx]

    score = 0.0

    # --- サイド差分（最重要: 重み大）。prize は残りサイドなので 6-残り=取得済み ---
    my_prizes_taken = 6 - len(me.prize)
    opp_prizes_taken = 6 - len(opp.prize)
    score += (my_prizes_taken - opp_prizes_taken) * 200

    # --- 盤面リソース（HP は Pokemon.hp、エネは Pokemon.energies） ---
    in_play_me = [p for p in me.active + me.bench if p]
    in_play_opp = [p for p in opp.active + opp.bench if p]
    score += sum(p.hp for p in in_play_me) * 0.5
    score -= sum(p.hp for p in in_play_opp) * 0.5
    score += sum(len(p.energies) for p in in_play_me) * 10

    # --- セットアップ評価（進化ラインの完成度。preEvolution / stage から算出） ---
    score += evolution_readiness_score(me) * 50

    # --- 山札圧縮度（デッキが薄い = リソースを手札/場に引き出せている） ---
    score += max(0, 30 - me.deckCount) * 2

    # --- 生存リスク評価（相手の推定最大打点 vs 自陣 HP） ---
    opp_max_damage = opponent_estimator.get_max_damage()
    boss_prob = opponent_estimator.get_boss_probability()
    for pokemon in in_play_me:
        if pokemon.hp <= opp_max_damage:
            is_bench = pokemon in me.bench
            score -= 80 * boss_prob if is_bench else 40

    return score
```

> `evolution_readiness_score()` 等の補助関数は新規に実装する
> （`src/decision/evaluation/` に追加し、`board_features.py` のヘルパを再利用する）。

### 5-2. Phase C: Policy/Value Network

```
入力: 盤面状態（各カードをIDでエンベディング）
      ┌ 自分: 場のポケモン×{HP, エネ枚数, ダメカン, 進化段階}
      ├ 相手: 場の公開情報 + 推定情報
      ├ 手札カードID列（可変長）
      └ 推定デッキ分布ベクトル

アーキテクチャ: Transformer（各カードをトークンとして扱い、Attentionで関係性を捉える）
  → CNNは固定サイズ前提のため不向き（将棋と違い盤面が可変長・順序不同）

出力:
  ├ Value Head: 現在の勝率推定（0〜1）← evaluate() の代替
  └ Policy Head: 各合法手の選択確率分布 ← 枝刈りに使用

学習方法: 模倣学習（Imitation Learning）
  ├ ISMCTS が十分な時間をかけて選んだ手を正解ラベルに使用
  └ 教師あり学習 → 収束が速く、GPU不要で CPU でも推論可能な軽量モデルを目指す
```

---

## 6. キャッシュ・計算効率化

```python
class ComputationCache:
    """
    同一局面の再評価を防ぐキャッシュ。
    メモリ制限（Kaggle RAM上限）を考慮して LRU キャッシュを使用。
    """
    
    @functools.lru_cache(maxsize=4096)
    def evaluate_cached(self, state_hash: int) -> float:
        """盤面ハッシュをキーに evaluate() の結果をキャッシュ"""
    
    def state_hash(self, state: State) -> int:
        """
        盤面の本質的な特徴量（サイド差・場の状態・エネ数）から
        軽量ハッシュを生成。手札など細部は省略してヒット率を上げる。
        """
```

**デッキ推定のキャッシュ:**
- 推定結果はターンをまたいでも有効 → ターン開始時に1回だけ更新
- `seen_cards` の差分更新のみで再計算コスト最小化

**Kaggle 環境固有の注意事項:**
- `battle_start` の連続実行・`visualize_data()` の多用 → RAM 枯渇（カーネルクラッシュ）
- A/Bテスト・学習ループでは定期的に Python プロセス再起動 + チェックポイント保存
- 有意差検証には最低 **400〜1200 試合**のデータが必要

---

## 7. 実装ロードマップ

### Phase A: 今すぐ実装（1週間）

> 現行のルールベース proposal 方式は温存したまま「追加」する。探索はまだ載せない。
> ロールアウトは Phase B（ISMCTS）で初めて登場する（現状ロールアウト自体が無いため）。

| 優先度 | タスク | 期待効果 | 難易度 |
|--------|--------|----------|--------|
| 🔴最高 | 全局面 `evaluate(state)->float` を**新規実装**（サイド差・HP・エネ枚数・セットアップ度） | 盤面評価の土台ができる | 低 |
| 🟠高 | `OpponentDeckEstimator` 実装（既存 `meta_decks.py` を土台にベイズ更新） | 決定化品質向上 | 中 |
| 🟠高 | `SelfStateEstimator` 実装（残り山札・サイド推定） | 自分の状態把握 | 中 |
| 🟡中 | `TimeManager` 実装（動的時間配分） | タイムアウト防止 | 低 |
| 🟡中 | 推定器の状態をターン跨ぎで保持する `runtime_state` シングルトン | 推定の継続性 | 低 |

### Phase B: 1ヶ月で実装（探索高度化）

| 優先度 | タスク | 期待効果 | 難易度 |
|--------|--------|----------|--------|
| 🔴最高 | ISMCTS を**新規導入**（`search_begin/step` を利用、proposal 方式とフラグで並走） | Strategy Fusion 回避・深い戦略読み | 高 |
| 🔴最高 | Heuristic Rollout 実装（既存 `priorities/*.py` をプレイアウト方策に再利用） | Q値の信頼性確保 | 中 |
| 🔴最高 | デッキ推定（Phase A の推定器）を ISMCTS の決定化に統合 | シミュレーション精度が飛躍的に向上 | 高 |
| 🟠高 | B1 Heuristic による事前枝刈り | 計算量削減・深さ向上 | 中 |
| 🟠高 | 相手の次ターン火力予測・サイドレース評価を `evaluate()` に追加 | より戦略的な評価 | 中 |
| 🟡中 | `ComputationCache` 実装 | 再評価コスト削減 | 低 |

### Phase C: 研究枠（時間があれば）

| タスク | 概要 |
|--------|------|
| Policy/Value Network 実装 | Transformer で盤面をエンベディング。合法手の事前絞り込みと価値推定 |
| 模倣学習（Imitation Learning） | ISMCTS の出力を教師データに → 軽量 NN に蒸留 → 本番では NN で高速推論 |
| デッキ分類器（ML版） | Tech カード対応のため Transformer ベースのデッキ型分類器を導入 |

**Self-Play（AlphaZero 型）は現実的でない理由:**
数千GPU日の計算資源が必要。学生チームのリソース・コンペ期間内での収束は極めて困難。
まず「ISMCTS + 模倣学習」という段階的アプローチを完成させることが最優先。

---

## 8. 各コンポーネントの関係図

```
Observation
    │
    ├──▶ SelfStateEstimator
    │        ├─ 残り山札推定
    │        └─ サイドカード推定
    │                │
    │                ▼
    ├──▶ OpponentDeckEstimator ◀── META_DECKS（事前知識）
    │        ├─ ベイズ更新（毎ターン）
    │        ├─ sample_opponent_hand() → ISMCTS 決定化
    │        ├─ get_max_damage()  ┐
    │        └─ get_boss_prob()  ┘→ evaluate()
    │
    └──▶ TimeManager
             └─ budget_for_this_turn() → ISMCTS の反復回数を制御
                      │
                      ▼
              ISMCTS（探索エンジン）
                      │ ── 決定化: SelfEstimator + OppEstimator
                      │ ── ロールアウト: Heuristic Rollout
                      │ ── 枝刈り: B1 Heuristic top-k
                      │ ── 評価: evaluate() / Value Network(Phase C)
                      │
                      ▼
              最終行動選択 → action リスト返却
```

---

## 9. 設計上の重要な判断

### なぜ CNN ではなく Transformer か
ポケカの盤面は「トラッシュにあるカードの種類と枚数」「手札（可変長・順序不同）」など、
固定サイズのテンソルで表現できない。各カードをトークンとしてエンベディングし、
Attention 機構で関係性を捉える Transformer が適している。
（将棋の 9×9 固定グリッドとは根本的に異なる）

### なぜ DQN でなく模倣学習か
行動空間が毎ターン変動する可変長構造は DQN に不向き。
また RL のオンライン学習には数十万エピソードが必要で GPU 依存が高い。
ISMCTS の出力を教師データとした模倣学習（教師あり学習）は収束が早く、
CPU のみで軽量 NN を推論可能なため本番制約に適合する。

### なぜ ISMCTS か（素朴な Flat MCS を経由せず）
現行は探索を持たないため、探索導入時に素朴な Flat MCS から始める選択肢もあるが採らない。
Flat MCS は「全シミュレーション結果の平均」を取るため、相手が速攻 / コントロールの2通りある場合に
**どちらにも中途半端な手（Strategy Fusion）**を選んでしまう。
ISMCTS は同一 Information Set のノードを共有管理するためこの問題を構造的に回避できる。
最初から ISMCTS を実装する。Hearthstone 等 TCG での実績もある。
