# 現行最高モデル「climb」(publicScore 823.5) — アルゴリズム詳細

Kaggle 提出 `55186283 submission_climb.tar.gz` / **publicScore 823.5**(自己ベスト系）。
本ドキュメントは **実ファイル**（`policy_weights_alakazam_rl_climb.json` / `torch_policy.py` / `encoder.py` /
`train_field.py` / `train_v3.py` / `abl_5_full.json` / `ml_policy_agent.py`）から再構築した正確な仕様。

---

## 0. 一言でいうと

**「小さな option-scoring MLP 方策（BC で事前学習 → field 自己対戦 PPO でファインチューン＝climb）」を、
`abl_5_full`（確定リーサル探索 + 信念決定化 PIMC 前読み）でラップした構成 × Plan A アラカザム(フーディン)デッキ。**

- ニューラルネットは **意思決定パイプラインの一部（選択肢スコアラ＝Policy prior）**であり、単独で手を決めていない。
- 「823.5」の正体は **RL方策 + 探索パイプライン + 強いデッキ** の合わせ技（後述の caveat）。

---

## 1. 意思決定パイプライン全体（`abl_5_full`）

通常ターンの1手は以下の順で決まる（`ml_policy_agent.py` / `search/pipeline.py`）:

```
obs
 │
 ├─(1) 確定リーサル探索  lethal_simple    … 今ターンに勝てる手順があれば即採用
 │        (max_remaining_prizes=3, time_limit 100ms, max_depth 20)
 │
 ├─(2) 浅い PIMC 前読み  pipeline(PIMC)   … Policy top-k を候補に、相手を推定して先読み評価
 │        top_k=4 / num_determinizations=8(=Belief N=8)/ opponent_depth=1
 │        hidden_state_source="estimated"(相手デッキ推定)/ handcrafted leaf / 400ms
 │        └── 各候補手を「信念で相手手札/山札/伏せを決定化 → 数手進めて handcrafted 盤面評価」
 │
 └─(3) Policy フォールバック                … 上記が出せないとき Policy argmax / greedy multi-select
```

- **Policy(下記NN)は (2) の候補生成 prior と (3) のフォールバックの両方**で使われる。
- config: `sample_submission/configs/abl_5_full.json`、`ml_policy_agent._CONFIG_NAME = "abl_5_full"`（env `PTCG_AI_ML_CONFIG` で上書き可）。

### 1.1 config パラメータ（abl_5_full.json）
| ブロック | 値 |
|---|---|
| lethal_search | module=lethal_simple, max_remaining_prizes=3, time_limit_ms=100, max_depth=20, max_nodes=10000, max_combinations_per_select=128, verify_shuffles=1 |
| pipeline(PIMC) | top_k=4, top1_shortcut_prob=0.9, **num_determinizations=8**, opponent_depth=1, max_rollout_steps=40, time_limit_ms=400, tie_eps=0.02, leaf_eval=handcrafted, hidden_state_source=**estimated** |
| time_budget | total_ms=540000, assumed_total_selects=400, min_ms=50, max_ms=2000 |

---

## 2. ニューラルネットワーク（Policy）= option-scoring MLP

`kaggle_replays/rl/torch_policy.py::TorchOptionPolicy`。**各“選択肢(option)”を独立にスコアリング**し、
合法選択肢集合上で softmax したものが方策。

### 2.1 構造（実重み `policy_weights_alakazam_rl_climb.json` より）

```
各 option ごとに:
  x = [ standardize(state_feat)  (166)   # 局面(全optionで共有)
        ++ standardize(option_feat) (65)  # その選択肢固有
        ++ card_embedding(option_card_id) (8) ]   # = 239 次元
  h = ReLU( W0 · x + b0 )        # W0: [32 x 239], b0: [32]   ← 中間層(隠れ層) 32ユニット・1層
  score = W1 · h + b1           # W1: [1 x 32],  b1: [1]     ← 出力 スカラー1個/option
方策 π(option) = softmax_over_legal_options( score )
```

- **入力層**：239 次元（= 166 + 65 + 8）
- **中間層（隠れ層）**：**32 ユニット・ReLU・1 層のみ**（`layers[:-1]` の bias 長 = [32]）
- **出力層**：`Linear(32 → 1)`、選択肢1件につきスカラー1個 → 選択肢集合で softmax
- **総パラメータ**：W0 32×239 + b0 32 + W1 1×32 + b1 1 ＋ card_embedding table（1268×8）≈ **約 1.8 万**（＋埋め込み1万）＝**非常に小さいネット**

### 2.2 card_embedding（学習可能ルックアップ表）
`card_embedding = {dim: 8, card_id_max, table: [1268 x 8]}`。
その選択肢の**主要カードID → 8次元ベクトル**を表引き（範囲外IDは0埋めクリップ）。カードの“素性”を8次元に圧縮した学習埋め込み。

### 2.3 標準化（standardization）
- **state(166) と option(65) は z-score 標準化**：`(x − mean)/std`。**std==0 の次元は 0 にする**（定数次元の無害化）。mean/std は重みJSON `standardization` に保存（BC学習データ由来）。
- **card_embedding(8) は標準化しない**（そのまま連結）。

---

## 3. 入力特徴量（239次元）の内訳

`sample_submission/ptcg_ai/learning/encoder.py`。**公開情報のみ**を使用（自分の手札は集約特徴として可、相手手札は枚数のみ）。

### 3.1 局面特徴 state（166次元）
| カテゴリ | 内容 |
|---|---|
| **ポケモン毎特徴 (11 × 12スロット = 132)** | 自分 active + ベンチ5、相手 active + ベンチ5 の各スロットについて **11特徴**：`present`(存在), `hp_ratio`, `remaining_hp`, `damage_counters`, `energy_count`, `best_attack_damage`, `can_ko_defender`, `min_energy_shortfall`, `has_ready_attack`, `attacker_score`, `is_likely_ko_next_turn` |
| **自分カウント (内訳あり)** | self_hand_count / hand_pokemon / hand_trainer / hand_energy, self_deck_count, self_prize_remaining, self_bench_count |
| **相手カウント (枚数のみ)** | opp_hand_count, opp_deck_count, opp_prize_remaining, opp_bench_count |
| **特殊状態(バトル場・公開)** | 両者 active の状態異常等（poison/burn/sleep/paralysis/confusion 等） |
| **ゲーム進行** | turn, supporter_played, energy_attached, retreated 等 |
| **集約** | prize_diff, bench_count_diff, self_energy_on_board, opp_energy_on_board |

（合計 166。ポケモン毎132 ＋ カウント/状態/進行/集約 で 34）

### 3.2 選択肢特徴 option（65次元）
`_build_option_feature_names()`。その選択肢が「何を・どこに・どう作用させるか」を記述：
| カテゴリ | 内容 |
|---|---|
| **選択肢タイプ one-hot** | `opttype_*`（既知option種）+ opttype_other |
| **select種 one-hot** | `seltype_*`（MAIN等の既知select種）+ seltype_other |
| **選択肢メタ (5)** | is_own, number_norm, count_norm, option_position_norm, n_options |
| **対象ポケモン (1 + 11)** | has_target_pokemon ＋ target_pokemon_{上記11特徴}（その選択肢が指すポケモンの特徴） |
| **対象カード (1 + 種別 + 5)** | has_target_card ＋ target_card_is_{カード種} ＋ target_card_hp_norm / is_basic / is_stage1 / is_stage2 / is_ex |
| **技情報 (1 + 4)** | has_attack ＋ attack_damage_norm, attack_can_ko, attack_min_shortfall_norm, attack_has_ready |

### 3.3 カード埋め込み（8次元）
`encode_option_card_ids()` で各選択肢の主要カードIDを取得 → §2.2 の table(1268×8) を表引き。

---

## 4. 価値ネットワーク（Critic）— **学習時のみ**

`train_v3.py::Critic`（デプロイ時は不使用、PPO の advantage 推定用）:
```
state(166) → standardize → Linear(166→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)
```
- 2 隠れ層 × 64 の MLP。**局面価値 V(s)** を推定し GAE で advantage を計算するためだけに使う。**提出物には含まれない**（Policy のみデプロイ）。

---

## 5. 学習アルゴリズム（= climb recipe）

### 5.1 事前学習（BC / 模倣）
- 上位パイロットのリプレイから **行動クローン(BC)** で Policy を学習（構成C）。
- 重みJSON meta: `n_train=147705, n_val=17359, n_test=22626`、`test top1_accuracy=0.581`。これが RL の初期値。

### 5.2 RL ファインチューン（`kaggle_replays/rl/train_field.py`, PPO）
- **learner（学習方策, 自デッキ）を frozen な FIELD 相手に自己対戦**させ PPO で改善（= “climb”）。
- **FIELD（開発7アーキ・メタシェア標本化, Reference Pool/Gate2 不使用）**：
  mega_lucario_ex(1257) / archaludon_ex(1078) / crustle(737) / dragapult_ex(625) / marnie_grimmsnarl_ex(591) / rocket_mewtwo_ex(247) / shirona_garchomp_ex(181)。
- **league**：champion-league 方式（meta: `rl_league=True`, `rl_champions_final=2`）＝強くなった自分をチャンピオンとして相手プールに追加。
- **アルゴリズム詳細**：PPO（clip 0.2, epochs 4）＋ GAE（γ=0.999, λ=0.95）、entropy 係数 0.005、lr(policy)=3e-4 / lr(value)=1e-3、grad clip 1.0。報酬＝ゲーム勝敗（±1）。任意で **potential-based dense shaping**（`--dense-reward`）。
- **本番ハイパー**：iters=60 / games-per-iter=512 / eval-games=400、`field_frac`（meta `rl_field_frac=0.45`）。
- **収束**：meta `rl_best_iter=57`, `rl_best_eval_winrate=0.72`（baseline BC=0.615）＝ **field 加重勝率で +10.5pt** のローカル改善。

### 5.3 出力
- 最良チェックポイントを PolicyModel JSON（`policy_weights_alakazam_rl_climb.json`）にエクスポート。提出物では `policy_weights.json` として同梱。

---

## 6. デッキ

- **Plan A アラカザム（フーディン, powerful-hand）**。提出物の `deck.csv`（60枚）。
- Policy は **デッキ非依存**（渡された60枚をそのまま操作）だが、RL は**このデッキで**自己対戦して最適化されている＝デッキ特化。

---

## 7. まとめ（層・特徴の総覧）

| 項目 | 内容 |
|---|---|
| **入力層** | **239次元** = 局面166 ＋ 選択肢65 ＋ カード埋め込み8 |
| **中間層** | **32ユニット × 1層, ReLU**（＋ 学習可能 card_embedding 1268×8） |
| **出力層** | Linear(32→1)、選択肢ごとにスカラー → softmax = 方策 |
| **標準化** | state/option は z-score（std=0は0化）、埋め込みは非標準化 |
| **価値網** | 166→64→64→1（学習時のみ・非デプロイ） |
| **学習** | BC(構成C, 14.7万サンプル) → PPO field self-play(clip0.2/γ.999/λ.95/ent.005) + league(champ2), best@iter57, field 0.615→0.72 |
| **推論の使われ方** | lethal_simple → PIMC(Belief N=8, 相手推定, top_k4, handcrafted leaf) → Policy fallback |
| **デッキ** | Plan A アラカザム(フーディン) |
| **Kaggle** | publicScore **823.5**（提出 55186283） |

---

## 8. 重要な caveat（この値の解釈）

- **823.5 は「RL方策 単独」の力ではない**。①強い alakazam デッキ、②探索パイプライン（lethal + PIMC 前読み）、③RL方策 の合わせ技。
- **local field 勝率(0.72) ≠ Kaggle score**。同 recipe を他アーキ（lucario/オーロンゲ）に適用しても Kaggle では 627.9 / 636.8 に留まり、**実ラダーはデッキが支配的**（別途検証済）。RL の +Δ は「同デッキ内での模倣→RL 相対改善」を測るもの。
- 中間層32・単層という**極小ネット**で 0.58 top1 / field 0.72 に達している＝ **表現力より特徴量設計と探索の寄与が大きい**構成。
