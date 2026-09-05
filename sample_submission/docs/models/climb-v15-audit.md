# climb 実装監査 + v1.5 診断 (2026-08-05)

`climb-823-model.md` の記述を **実コード・実重み・実ゲーム計測** で検証した結果。
doc の §2(ネット構造)/§3(特徴量)は重みJSONと**完全に一致**しており正確。
本ドキュメントは doc に**書かれていない/ズレている**点と、実測で分かったボトルネックを記録する。

情報の優先順位は「提出物へ入る実コード > 重みJSON > 学習コード > 評価コード > doc」。

---

## 1. 監査で判明した差分(重要度順)

### 1.1 【再現性】既定の `policy_weights.json` は climb ではない

| ファイル | md5 | 実体 |
|---|---|---|
| 作業ツリー `policy_weights.json` | `1275158d…` | **BC模倣**(`rl_finetuned` 無し) |
| `policy_weights_alakazam_rl_climb.json` | `c9c81f16…` | **climb**(`rl_finetuned=True`, best_iter=57) |
| 提出tarball内 `policy_weights.json` | `c9c81f16…` | **= climb**(パッケージ時にコピーされている) |

- 両者は同じBC祖先(`created_at` / `test top1=0.5806` が一致)で、climb はその上に PPO を掛けたもの。
- **ローカルで既定のまま走らせると climb ではなく BC を測ることになる。**
  climb を測るときは必ず `config["policy_weights_path"]` で明示すること。
- `deck.csv` は tarball と作業ツリーで一致(`4c4ca971…`)。

### 1.2 【本番影響なし / ローカル計測を汚染】相手側の PIMC が黙って無効化されていた

ローカルの `run_league` / `eval_field` / `measurement.runner` は **1プロセスで両陣営の ml_policy** を
動かすが、デッキは `battle_start(deck0, deck1)` でエンジンへ直接渡され、エージェント側は
`ml_policy_agent._deck_cache`(モジュールglobal)経由で **常に `deck.csv` を自分の山札だと誤認**する。

その結果 `deck.csv` 以外を持つ側は `OwnHiddenState` のサイド枚数が実盤面と食い違い、
`search_begin` が `ValueError: your_prize does not match the number of cards in your prize.` で
**必ず**例外 → `pipeline.search` が握りつぶして `None` → **その陣営だけ探索なし**で戦っていた。

実測(6試合 vs mega_lucario、`kaggle_replays/_diag_pimc_failure.py`):

| 条件 | `search_begin` 失敗率 | 探索到達時に action を返した率 |
|---|---|---|
| 両陣営合算(修正前) | **84.6%** | 41.2% |
| climb 側のみ(修正前) | 0.0% | 100% |
| ミラー(同一デッキ, 修正前) | 0.0% | 100% |
| 両陣営合算(**修正後**) | **0.0%** | **100%** |

- **本番(Kaggle)は 1プロセス1エージェントで自分のデッキ=deck.csv なので発生しない。**
- 影響は **ローカル field 評価のみ**。相手を不当に弱くした条件で勝率を測っていたため、
  過去のローカル field 勝率(climb doc の `field 0.72` 等)は楽観方向へ偏っている可能性がある。
- 修正: `match_context.set_own_deck_override(player_index, card_ids)` を追加(production は呼ばない=不変)。
  `measurement/runner.play_game` と `league/run_match.play_match` が game ごとに真の60枚を宣言する。
  併せて `run_match` には **試合間の推定状態 reset** も入れた(従来 `run_league` 経路は reset していなかった)。

### 1.3 Critic(PPOの価値網)は非デプロイ — 確定

`Critic` は `kaggle_replays/rl/train_v3.py` にしか存在せず、`sample_submission/` 配下に**一切現れない**。
自己対戦で学習した長期価値は **推論時に完全に捨てられている**(doc §4 の記述どおり)。

一方、**別物の学習 Value** は存在する:

- `ptcg_ai/learning/value_weights.json` — Kaggle上位リプレイ 486,214 局面で学習した166次元MLP。
  test AUC **0.7446** / logloss 0.593、turn_band 温度較正あり。
- `search/leaf_eval.py` は元から `kind="value"` を実装済み。だが**本番 config は `handcrafted`**。
- `feature_names` は現行 `encoder.FEATURE_NAMES` と **完全一致**することを検証済み(166/166)。
  → 学習Valueを葉に入れても特徴量skewは起きない(安全に試せる)。

### 1.4 【性能】`build_evaluator` が探索の中で毎 select 呼ばれる

`pipeline.search` は毎回 `leaf_eval_module.build_evaluator(...)` を呼ぶ。`kind="value"` にすると
`ValueModel()` が毎 select 重みJSONをディスクから読み直す。プロセス共有キャッシュを追加して解消済み
(handcrafted 経路=本番既定は ValueModel に触れないため影響なし)。

### 1.5 【測定忠実度】ローカルは本番より短い探索予算で走っている

ローカル harness は `obs.select is None`(デッキ選択ターン)を通らないため
`ml_policy_agent._match_start_perf` が `None` のままで、**動的時間予算が効かず固定 `time_limit_ms=400`** になる。
Kaggle 本番は select=None を受けるので `total_ms/assumed_total_selects ≒ 1350ms/select`。
**ローカル計測は本番の約 1/3 の探索予算**で行われている。

---

## 2. 実測したボトルネック

### 2.1 探索カバレッジ — 意思決定の約40%しか探索していない

40試合・4,475 selects の実測(`kaggle_replays/_diag_climb_coverage.py`):

| select 種別 | 割合 | 平均選択肢数 | 探索対象か |
|---|---|---|---|
| MAIN | 58.2% | 8.86 | ○(maxCount==1 のみ) |
| **CARD** | **35.0%** | 4.29 | **×(素通り)** |
| YES_NO | 5.1% | 2.0 | × |
| EVOLVE / COUNT / ENERGY | 1.7% | – | × |

- `pipeline` は **`SelectType.MAIN` かつ `maxCount==1`** にしか適用されない。
- 対象のうち **70.9%** しか探索に到達しない(残りは `top1_shortcut_prob=0.9` で即決)。
- 結果、**全意思決定のおよそ 40% だけが探索で決まる**。lethal 発火は 1.1%。
- **「どのカードをサーチ/トラッシュするか」という CARD select 35% は探索を一切通らず、
  模倣ポリシーの貪欲選択のまま**。コンボ手順の決め手はここに多い(既知の課題)。

### 2.2 候補は「1手目」だけ — ターン内の行動系列を比較していない

`pipeline.py` の候補は `candidates = [[i] for i in candidate_indices]`、つまり **first move 1手のみ**。
その後のターン残りは Policy 貪欲でロールアウトされる。
「山札圧縮→サーチ→ドロー→エネ加速→入れ替え→攻撃」のような**順序**は、
候補として並べて比較されていない。

### 2.3 仮説A(手札のカード関係が見えない)は構造的に確定

239次元の入力に含まれる**カード識別情報は「その選択肢自身のカードID」8次元のみ**。

- 手札 = `handCount` / `hand_pokemon` / `hand_trainer` / `hand_energy` の **4スカラー**(`_hand_breakdown`)
- トラッシュ = 枚数1つ、山札 = 枚数1つ、相手手札 = 枚数1つ
- → 「Aを使った後にBが使える」「Bを先に使うとAの価値が下がる」を表す情報が**原理的に入力に無い**。

---

## 3. 本番を壊さないための構成

`abl_5_full.json` は**一切変更していない**(`leaf_eval` は `handcrafted` のまま、`dynamic_top_k` キー無し)。
新しい構成はすべて別名 config として追加した。

| config | 内容 |
|---|---|
| `climb_baseline.json` | `abl_5_full` と**バイト等価**(name のみ差)の名前付きアンカー |
| `climb_v15_leaf_a05 / a08 / a10` | 葉評価を `α*learned + (1-α)*handcrafted`(α=0.5 / 0.8 / 1.0) |
| `climb_v15_dyntopk` | 確信度依存の動的候補数(min_k=3 / max_k=12) |
| `climb_v15_full` | 動的top-k + 葉ブレンド α=0.5 |

追加した実装はすべて **config-gated / 既定OFF**:

- `leaf_eval.BlendedEvaluator` + `build_evaluator(kind="blend", alpha=…)`
- `leaf_eval._get_shared_value_model()`(ValueModel のプロセス共有)
- `pipeline._resolve_top_k()`(`dynamic_top_k` キーが無ければ従来固定 `top_k`)
- `match_context.set_own_deck_override()`(production は呼ばない)

テスト: `test_leaf_eval.py` +8 / `test_match_context.py` +3 / `test_pipeline.py` +6 を追加、全て pass。
既存の失敗 19件は本変更**以前から**存在する(HEAD で 19 failed / 318 passed、変更後 19 failed / 328 passed)。

---

## 3.5 診断結果(実測)

### 3.5.1 修正後ベースライン(40試合 climb vs mega_lucario模倣, 両陣営とも探索有効)

| 指標 | 値 |
|---|---|
| climb 勝率 | 0.725 (29/40) — **95% Wilson CI ≒ [0.57, 0.84]。n=40 では何も結論できない** |
| games/min | 2.73(修正前 5.11 = 相手が実際に探索するようになり約2倍遅くなった) |
| select レイテンシ | mean 177ms / p50 14 / p90 417 / p95 428 / p99 507 / max 554 |
| 探索カバレッジ | eligible 58.0% / 全selectの 41.1% が実探索 / eligible の 70.9% |

修正前(相手の探索が無効)の 0.675 との差は **n=40 では区別できない**。
「修正で climb が強くなった」とは**言えない**(相手が強くなったので、むしろ下がるのが自然だった)。

### 3.5.2 葉評価の品質: learned value > handcrafted(ただし用途限定)

climb 自身の実局面 1,713 件を実際の勝敗でラベル付けした比較(30試合):

| 評価器 | AUC | Brier | LogLoss | 平均予測 (base=0.628) |
|---|---|---|---|---|
| handcrafted | 0.781 | 0.277 | 1.024 | **0.329**(大きく過小) |
| **learned value** | **0.808** | **0.178** | **0.527** | 0.557 |

ターン帯別 AUC:

| ターン | handcrafted | learned |
|---|---|---|
| 1-2 | 0.518 | **0.643** |
| 3-5 | **0.771** | 0.688 |
| 6-10 | 0.836 | 0.840 |
| 11+ | 0.818 | **0.884** |

- **順位付け能力(探索が実際に使う性質)は learned が +0.027 AUC** と小幅だが上。
  序盤(1-2)と終盤(11+)で明確に上、中盤序盤(3-5)だけ handcrafted が上。
  → **α ブレンドに合理性がある**(片方を捨てる理由が無い)。
- 確率としての質は learned が圧倒的(LogLoss ほぼ半減)。handcrafted は平均 0.33 と
  系統的に過小評価しており、**確率として解釈してはいけない**。
  ただし探索は argmax しか使わないので、この差がそのまま勝率にはならない。
- learned のキャリブレーションは単調で、上側がやや**過小**(0.75→実測0.88 / 0.95→実測1.00)。
  慎重側に倒れており、リスク項として使う分には安全側。

**方法上の限界(重要)**: ここで測ったのは *実局面*。探索が実際に評価するのは
*決定化された数手先の葉* であり分布が違う。この結果は「A/Bを回す価値がある」ことを示すだけで、
**採用根拠にはならない**。

### 3.5.3 dynamic top-k は実際に発火している(勝率は未検証)

4試合の engagement 確認(探索回数をそろえた比較):

| config | 実探索回数 | 平均候補数 | p95 レイテンシ |
|---|---|---|---|
| `climb_baseline` | 220 | 3.87 | 501ms |
| `climb_v15_dyntopk` | 219 | **4.74** | 422ms |

候補数が +22% に増えており、機能が**実際に使われている**ことを確認済み。
レイテンシ悪化は見られない(n=4 なので差はノイズ)。**勝率への効果は未測定**。

---

## 3.6 Phase1: 全評価経路の忠実度監査(完了・ALL_PASS)

`kaggle_replays/_audit_eval_paths.py`。各経路を**別プロセス**で走らせ、`search_begin` の
成功/失敗を陣営別に計数する(`measurement/_probe_search_begin.py`。並列 worker にも
`worker_init` ラッパ経由でプローブを仕込み、`atexit` で pid 別 JSON を吐いて親が合算する)。

デッキ組み合わせ2種(alakazam vs mega_lucario / alakazam vs crustle)× 各4試合:

| 経路 | deck分離 | `search_begin` 失敗率 | 両陣営で探索 | 判定 |
|---|---|---|---|---|
| `measurement.runner.play_game` | ✓ | 0.0 | ✓ | PASS |
| `league.run_match.play_match` | ✓ | 0.0 | ✓ | PASS |
| `run_league` workers=1 | ✓ | 0.0 | ✓ | PASS |
| `run_league` workers=2(並列) | ✓ | 0.0 | ✓ | PASS(procs=2) |
| `eval_field` | ✓ | 0.0 | ✓ | PASS |

`run_league` / `eval_field` は `run_match.play_match` を経由するため、`play_match` への
`_begin_match_state()` 追加で同時に塞がった。併せて **試合間の `match_context.reset()`** も
`play_match` に入れた(従来 run_league 経路は reset していなかった=試合をまたいで推定が残留していた)。

## 3.7 Phase2 Stage A: 修正後の climb ベースライン(n=20/アーキ, smoke)

`kaggle_replays/_phase2_baseline.py --stage A`。**全アーキで探索失敗率 0.0**(= 測定条件として有効)。

| 相手 | n | 勝率 | Wilson 95%CI |
|---|---:|---:|---|
| mega_lucario_ex | 20 | 0.60 | [0.387, 0.781] |
| archaludon_ex | 20 | 0.70 | [0.481, 0.855] |
| **crustle** | 20 | **0.05** | [0.009, 0.236] |
| dragapult_ex | 20 | 0.85 | [0.640, 0.948] |
| marnie_grimmsnarl_ex | 20 | 0.65 | [0.433, 0.819] |
| rocket_mewtwo_ex | 20 | 0.20 | [0.081, 0.416] |
| shirona_garchomp_ex | 20 | 0.75 | [0.531, 0.888] |

- **FIELD 重み付き勝率 = 0.5611** / 単純プール 76/140 = 0.5429 [0.460, 0.623]
- doc の旧 `field 0.72` から大きく下がるが、これは **climb が弱くなったのではなく相手が
  本来の強さで戦うようになったため**。旧値がどれだけ楽観側へ偏っていたかを示す。
- **crustle 0.05 / rocket_mewtwo 0.20** が全体を押し下げている。crustle は
  [[project_crustle_structural_deck_capped]] の「デッキ律速」仮説と整合するが、
  相手探索を有効にするとさらに悪化することが分かった。
- n=20 は smoke。CI は広く、**個々の値を確定値として扱わない**(Stage B で 100/アーキ)。

## 3.8 Phase2 Stage B: 正式ベースライン(n=100/アーキ)

`kaggle_replays/_phase2_baseline.py --stage B --games 100 --workers 5 --seed 5000`
→ `kaggle_replays/_p2B.json`。**`ALL_SEARCH_CLEAN = true` / `USABLE_AS_BASELINE = true`**
(全アーキで `search_begin` 失敗率 0.0)。

| 相手 | n | 勝率 | Wilson 95%CI | (参考)StageA n=20 |
|---|---:|---:|---|---:|
| mega_lucario_ex | 100 | 0.72 | [0.625, 0.799] | 0.60 |
| archaludon_ex | 100 | 0.70 | [0.604, 0.781] | 0.70 |
| **crustle** | 100 | **0.08** | [0.041, 0.150] | 0.05 |
| dragapult_ex | 100 | 0.82 | [0.733, 0.883] | 0.85 |
| marnie_grimmsnarl_ex | 100 | 0.56 | [0.462, 0.653] | 0.65 |
| rocket_mewtwo_ex | 100 | 0.42 | [0.328, 0.518] | 0.20 |
| shirona_garchomp_ex | 100 | 0.67 | [0.573, 0.754] | 0.75 |

- **主指標: FIELD 重み付き勝率 = 0.5910**(事前定義7アーキタイプ全部、crustle を含む)
- 単純プール 397/700 = 0.5671 [0.530, 0.603]
- 再現メタ: git `b505ced`(dirty) / policy `395b0248` / value `d6d7cd89` / deck `8ae7a618` /
  config `2714e32e` / seed_start 5000 / workers 5

**Stage A(n=20)との差が大きい項目がある**(rocket_mewtwo 0.20→0.42、mega_lucario 0.60→0.72)。
n=20 のノイズが実際に大きかったことの実例であり、**Stage A の値は個別に引用しない**こと。

crustle は n=100 でも **0.08 [0.041, 0.150]** で、上限すら 0.15。
[[project_crustle_structural_deck_capped]] の「デッキ律速」仮説と整合するが、
相手の探索を有効にするとさらに悪化することが分かった。

### 指標の扱い(確定ルール)
- **主指標** = 上記7アーキタイプ全部の FIELD 重み付き勝率(**crustle を含む**)。
- **副次指標** = crustle 除外値。デッキ相性で leaf 変更の効果が埋もれていないかの確認にのみ使い、
  **PROMOTE 判定・正式な採用根拠には使わない**。
- crustle は必ず baseline と候補の **paired 差**を残し、「両者とも低い(デッキ由来)」と
  「候補でさらに悪化(変更由来)」を区別する。
- 実装: `kaggle_replays/_phase4_compare_field.py`(ガードテスト
  `kaggle_replays/measurement/test_field_compare.py` 5件で固定)。

## 3.9 Phase3: leaf ブレンド(α=0.5)の効果検証

### SPRT(head-to-head mirror, 600ゲーム完走・失敗率0.0)

| 項目 | 値 |
|---|---|
| B勝率 | 0.510 (306/600) / 95%CI [0.470, 0.550] |
| LLR | **-0.3611**(PROMOTE +2.8904 / FUTILITY -2.2513) |
| **判定** | **CONTINUE**(どちらの境界にも未到達) |

### フィールド比較(主指標 = crustle を含む7アーキ FIELD重み付き)

| seed | baseline | leaf_a05 | Δ(主指標) | Δ(副次: crustle除外) |
|---|---:|---:|---:|---:|
| 5000 | 0.5910 | 0.6193 | **+0.0283** | +0.0280 |
| 9000 | 0.5854 | 0.6291 | **+0.0437** | +0.0554 |
| **合算 (n=1400/arm)** | **0.5882** | **0.6242** | **+0.0360** | +0.0417 |

- **独立seedで方向が再現**(+0.028 → +0.044)。ベースライン自体も 0.5910/0.5854 と安定。
- 合算の単純プール差 = **+0.0250、95%CI [-0.0116, +0.0615] → 非有意**。
- 副次(crustle除外)は主指標とほぼ同じ増分 = **crustle が効果を埋もれさせてはいない**。
- 速度: 候補が全アーキで一貫して約6%低速(最大 dragapult -10.6%)。Kaggle予算内。

### crustle の paired 判定(必須項目)

| seed | baseline | leaf_a05 | 差 | 95%CI | 判定 |
|---|---:|---:|---:|---|---|
| 5000 | 0.080 | 0.110 | +0.030 | [-0.055, +0.116] | both_low |
| 9000 | 0.130 | 0.110 | -0.020 | [-0.113, +0.073] | both_low |
| 合算 | 0.105 | 0.110 | +0.005 | [-0.057, +0.067] | **both_low_no_significant_change** |

→ **「両者とも低い」= デッキ相性由来**であり、**leaf 変更で悪化してはいない**。

### アーキ別(合算 n=200/アーキ)

mega_lucario のみ +0.090 [0.002, 0.176] で名目有意だが、**7マッチアップの多重比較**であり
単独では効果の根拠にしない。他6アーキはすべて CI が 0 を跨ぐ。

### n=100 のアーキ別値は seed 間ノイズと同程度

baseline を seed 5000/9000 で比べると mega_lucario 0.72→0.63、shirona 0.67→0.58 と最大 ±9pp 振れた。
**アーキ別の差分を単独の効果として解釈しないこと**。信頼できるのは 700ゲーム以上を束ねた集計値。

### 機能が実際に使われたかの確認(§15)

| config | leaf_eval | Blended呼出 | Value呼出 | Handcrafted呼出 |
|---|---|---:|---:|---:|
| `climb_baseline` | handcrafted | 0 | 0 | 23 |
| `climb_v15_leaf_a05` | blend α=0.5 | 36 | 27 | 56 |

`value_model_ready() = True`。候補側でのみ学習Valueが葉に入っていることを実呼び出しで確認。

### 判定: **サンプル不足のため継続**(PROMOTE ではない)

§5.4 の7条件のうち **6つを充足**、未充足は **SPRT PROMOTE**(=CONTINUE)と
**FIELD改善の統計的有意性**(CIが0を跨ぐ)。早期破棄条件にはひとつも該当しない。

## 3.10 Phase4A: α 短期スクリーニング(50ゲーム×7アーキ×2seed = 700/arm)

前提確認: α の式は `blended = α × learned + (1-α) × handcrafted`。
4 arm は **leaf_eval 以外すべて同一**(top_k=4 / det=8 / opp_depth=1 / rollout=40 / time_limit=400、
`dynamic_top_k` はどこにも無い)。`a10` は元 `kind:"value"` だったが、
「α以外を変更しない」を満たすため **`blend α=1.0` に統一**(等価性をテストで固定)。
Evaluator 呼び出しも実測し、config の α が実際に届いていることを確認済み。

| arm | n/arm | FIELD重み付き | Δ vs baseline | 単純プール | Δプール | 95%CI(プール差) | 有意 |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 1400 | 0.5882 | — | 0.5636 | — | — | — |
| a05 | 1400 | 0.6242 | +0.0360 | 0.5886 | +0.0250 | [-0.012, +0.062] | × |
| a08 | 700 | 0.6270 | +0.0388 | 0.6071 | +0.0436 | [-0.001, +0.088] | × |
| a10 | 700 | 0.6367 | +0.0485 | 0.5957 | +0.0321 | [-0.013, +0.076] | × |

※ a08/a10 は n=50/アーキ、baseline/a05 は n=100/アーキ(既存アンカー流用)。精度が異なる。

### seed 別(方向の一貫性)

| arm | seed5000 | seed9000 | Δ5000 | Δ9000 | 両seedで正 |
|---|---:|---:|---:|---:|---|
| a05 | 0.6193 | 0.6291 | +0.0283 | +0.0436 | ✓ |
| a08 | 0.6192 | 0.6348 | +0.0282 | +0.0494 | ✓ |
| a10 | 0.6143 | 0.6592 | +0.0233 | +0.0738 | ✓ |

**6/6 の比較で α>0 が baseline を上回る**。一方 **α 同士の順位は seed で入れ替わる**
(seed5000 は a05≈a08>a10、seed9000 は a10>a08>a05)。

### α 同士は区別できない

| 比較 | 差 | 95%CI | 有意 |
|---|---:|---|---|
| a08 vs a05 | +0.0186 | [-0.026, +0.063] | × |
| a10 vs a05 | +0.0071 | [-0.038, +0.051] | × |
| a08 vs a10 | +0.0114 | [-0.040, +0.063] | × |

### 判定: **パターンD(全てほぼ同等)** + 但し書き

α の**大小関係は決まらない**(FIELD重み付きだと単調増加に見えるが、単純プールでは a08>a10 と
逆転し、seed 別でも順位が入れ替わる = 集計方法と seed に依存する見かけの単調性)。
ただし **「learned value を葉に入れること自体」は 6/6 で正方向**。

参考(**事後的なプール・事前登録していない探索的分析**):
全 α arm 合算 1666/2800 = 0.5950 vs baseline 789/1400 = 0.5636、
差 **+0.0314**、95%CI **[-0.0002, +0.0632]** = **有意ぎりぎり届かない**。

crustle(主指標に含む): baseline 0.105 / a05 0.110 / a08 0.090 / a10 0.130 — いずれも低位で横ばい、
**どの α でも悪化していない**。

## 3.11 Phase4B: top-k 見落とし診断(46 probes / 12試合 / 86分)

本番と同じ評価器のまま、候補数16・決定化12・予算8秒へ広げて「同じ探索を全選択肢に広げたら
1位が変わるか」を測る。**真の最善手ではなく、現行Evaluatorから見た最良手**である点に注意。

| 指標 | 値 |
|---|---:|
| top-1 一致率 | 0.217 |
| top-2 内 | 0.478 |
| **top-4 内** | **0.739** |
| top-8 内 | 0.848 |
| top-12 内 | 1.000 |
| **top-4 外率** | **0.261**(95%CI [0.156, 0.403]) |
| 平均候補数 | 8.5 |

Q差: top-4外だったとき median 0.034(`tie_eps`=0.02 超)、12件中10件が閾値超。
→ §6.4 の「15%以上 = top-k固定が主要ボトルネック候補」に **CI下限でも到達**。

### ゲートに使えるのは「選択肢数」、確信度は使えない

| 選択肢数 | n | top-4外率 | | top1確信度 | n | top-4外率 |
|---|---:|---:|---|---|---:|---:|
| 5-7 | 24 | **0.125** | | <0.3 | 2 | 1.000 |
| 8-11 | 16 | **0.375** | | 0.3-0.6 | 18 | 0.167 |
| 12+ | 6 | **0.500** | | **>=0.6** | 26 | **0.269** |

**実装済み `dynamic_top_k` は確信度ゲートであり、この結果に反する**
(確信が高い局面でも27%見落とすのに、そこを狭める設計になっている)。
**そのまま勝率A/Bに掛けてはいけない**。`pipeline._resolve_top_k` の docstring に警告を明記した。
ターン帯別は 21〜33% で平坦。margin帯は「最良手がtop-4外なら確率が低い」の同語反復でゲート不可。

## 3.12 Phase4C: CARD select 診断(55 probes / 12試合)

| 指標 | 値 |
|---|---:|
| 自分の手番の select 構成 | MAIN 57.4% / **CARD 30.5%** / YES_NO 9.8% / EVOLVE・ENERGY 2.3% |
| 1試合あたり CARD select | 17.6 回 |
| CARD の平均候補数 | 4.1(probe対象は 4.65) |
| 平均 Policy entropy | 0.941 |
| **即時leafでの順位反転率** | **0.000** |
| **即時leafの候補間 spread** | **0.000** |
| **ターン終端まで進めた後の順位反転率** | **0.473**(26/55) |
| 反転時の平均 value 差 | 0.057(13件が `tie_eps` 0.02 超) |
| 後続合法手が変化した probe | **14.6%** のみ(平均変化率 0.047) |
| 1段の分岐数 | 4.65 |
| ターン終端までの推定分岐数 | median **5,007** / mean 22.4M(裾が重い)/ 残り select 平均 9.64 |

### 構造的な結論(これが本フェーズ最大の収穫)

**手作り leaf は手札の中身を一切見ない**(サイド/打点/エネ/ベンチのみ)。
そのため「どのカードを手札に加えるか」を1手進めて評価しても、候補間の値は
**厳密に 0.0 しか動かない**(n=55 で spread 0.000)。
→ **CARD select に「1段だけ」lookahead を足しても、現行 leaf では信号がゼロ**。
学習Value も同じ166次元(手札は4スカラー)なので同様に効かない。

差が出るのはターン終端まで進めた場合(反転率 47.3%)だが、これは
**CARD専用1段lookahead ではなく ターン内Beam Search そのもの**である。
なお 3決定化での測定なのでロールアウト由来のノイズを含む可能性があり、
47.3% を効果量として鵜呑みにはできない(spread 0.0 の方は構造的事実で、こちらは頑健)。

## 3.13 Phase5A: n_options ゲートの候補回収診断 → **予算律速で 5B 中止**

`_diag_topk_recall.py`(50 probes / 12試合)。leaf は A/B と同じ α=0.5 固定。
参照評価(16候補×12決定化×8秒)で「診断上の最良候補」を決め、
**本番と同じ 400ms / det=8** で fixed k=4 と n_options ゲートの到達段階を比較した。

### 候補回収

| 指標 | 値 |
|---|---:|
| fixed top-4 外だった診断候補 | 25 / 50 |
| **候補集合への回収数** | **17**(candidate recall **0.68**)✓ |
| 実評価**開始**数 | 3(0.12) |
| **実評価完了数** | **3**(completed recall **0.12**)✗ |
| **完全determinizations到達** | **0**(fully recall **0.00**)✗ |
| 最終選択された回収候補 | 2 |
| 回収候補の平均Policy順位 | 7.18 |
| 回収候補の平均leaf advantage | 0.046 |

**ゲート自体は正しく機能している**(候補集合には 68% 回収できた)。
しかし **400ms 予算では回収した候補をほぼ評価できない**。

### 予算の実態(A/B 両アームで同じ)

| 指標 | fixed k=4 | dynamic |
|---|---:|---:|
| 平均 requested 候補 | 4.00 | **8.16** |
| 平均 completed 候補 | 2.40 | **2.56** |
| 平均構築world数(要求8) | 1.44 | 1.22 |
| **timeout率** | **0.98** | **0.98** |
| 平均探索時間 | 424ms | 426ms |

| n_options帯 | n | k_fix→k_dyn | completed | fully | timeout |
|---|---:|---|---|---:|---:|
| 5-7 | 17 | 4→4 | 2.29→2.24 | 0.24 | 0.94 |
| 8-11 | 14 | 4→**8** | 2.36→2.43 | **0** | 1.00 |
| 12+ | 19 | 4→**12** | 2.53→**2.95** | **0** | 1.00 |

**requested は 4.00→8.16 に倍増したが、completed は 2.40→2.56 しか増えず、fully は 0。**
固定 top-4 ですら 4 候補中 2.4 しか完了できていない。
→ §7.1 の停止条件「requested だけ増え完了が増えない」「ほぼ常時 timeout」
「top-4候補すら十分評価できない」に**該当**するため、**Phase 5B(実ゲームA/B)は実施しない**。

### 測定忠実度の重大な但し書き

ローカル harness は `obs.select is None`(デッキ選択)を通らず `_match_start_perf` が None のため
**固定 400ms**。一方 **本番 Kaggle は動的予算 = 540000ms / 400 selects = 約 1350ms/手**(上限2000ms)。
**本番はローカルの約 3.4 倍の予算**を持つ。

したがって上記の「予算律速」はローカル条件のものであり、**本番でも同じとは限らない**。
逆に言えば、これまでの α スクリーニングを含む**すべてのローカル A/B は、本番より 1/3 に絞られた
探索(決定化 8 のうち実質 1.2〜1.4 しか回らない)で測られてきた**ことになる。
top-k を含むどの探索パラメータを測るにも、まずこの予算差を解消する必要がある。

## 3.14 Phase5.1: 本番時間予算の確定(監査完了・以後この話は closed)

実コード(`ml_policy_agent._dynamic_pipeline_time_limit_ms`)より:

```
base = pipeline.time_limit_ms                                   # 400
if not time_budget or _match_start_perf is None: return base    # ← ローカルharnessはここ
elapsed_ms        = (perf_counter() - _match_start_perf) * 1000
remaining_ms      = total_ms - elapsed_ms                       # 540000 - elapsed
if remaining_ms <= 0: return min_ms                             # 50
remaining_selects = max(1, assumed_total_selects - _selects_seen)   # 400 - seen
clamped           = max(min_ms, min(max_ms, remaining_ms / remaining_selects))   # [50,2000]
```

| 項目 | 値 |
|---|---|
| Kaggle制限 / production `total_ms` | 600,000ms / **540,000ms**(margin 60秒) |
| 初期 raw budget | 1,350ms |
| **実測平均割当** | **1,446.7ms**(median 1435 / p90 1552 / max 1613) |
| max cap 2000ms 到達率 | **0.0** / 6000ms 到達率 **0.0** |
| **実 select 数/試合/エージェント** | **平均 72.5**(assumed 400 の約 1/5.5) |

**「10分だから1手6000ms」は成立しない**。理由は2つ: ①式は実 select 数ではなく固定 400 で割る
②仮に届いても `max_ms=2000` で頭打ち。パリティテスト13件で固定
(`kaggle_replays/measurement/test_budget_parity.py`)。

### 予算感度(候補完了数 / 構築world数(要求8) / timeout率)

| budget | fixed k=4 | dynamic | completed recall |
|---|---|---|---:|
| 400ms | 1.81 / 1.22 / 1.00 | 2.43 / 1.03 / 1.00 | 0.063 |
| 1350ms | 3.38 / 2.06 / 0.94 | 4.47 / 1.44 / 1.00 | 0.267 |
| 2000ms | 3.72 / 2.74 / 0.97 | 5.15 / 2.13 / 0.97 | 0.300 |
| **production_dynamic**(実測1344ms) | 3.37 / 1.90 / 1.00 | 4.63 / 1.20 / 1.00 | **0.231** |

→ **`top_k=4 × det=8 × rollout=40` は本番上限予算でも完走しない**(fully evaluated はどの予算でも 0)。
これは前フェーズのパターンC。以後、予算・配分最適化は精度改善が確認されるまで再開しない。

## 3.15 Phase6: Hand-aware Value v1

### データ監査(§7.1)— 全ゲート PASS

| 項目 | 値 |
|---|---:|
| 総局面数 / 総試合数 | 614,209 / 4,690 |
| train / val / test | 486,214 / 59,169 / 68,826 |
| **試合が複数splitに跨る数** | **0**(split = `md5(episode_id)%100` = 試合単位) |
| 手札カードID抽出率 | **1.000**(平均8.62枚 / 322種) |
| トラッシュ非空率 / 相手公開取得率 | 0.924 / 0.968 |
| **リーク検査**(相手手札・山札) | **PASS(0件)** |
| **視点検査** | **PASS**(440/440 で両プレイヤー逆ラベル) |

既存の保存観測に手札・トラッシュ・相手公開カードが **カードID単位でそのまま入っていた**
(V0 が166次元へ集約する時点で捨てていただけ)。再収集不要。
`build_card_features.py` で V0 と同一行に突合、**match_rate = 1.000**(614,209行)。

### オフライン精度(§8、test 68,826局面 / 同一split・同一trunk・同一レシピ / 2seed平均)

| 指標 | V0′ state-only | **V1 hand-aware** | 差 |
|---|---:|---:|---:|
| AUC | 0.7400 | **0.7532** | **+0.0132** |
| LogLoss | 0.5992 | **0.5975** | −0.0017 |
| Brier | 0.2058 | **0.2028** | −0.0030 |
| Calibration error | **0.0340** | 0.0503 | **+0.0163(悪化)** |
| 序盤 / 中盤 / 終盤 AUC | 0.621 / 0.773 / 0.831 | **0.637 / 0.786 / 0.842** | +0.016 / +0.013 / +0.011 |

**2seed とも V1 > V0′**(direction_reproduced_all_seeds = True)。デプロイ済み V0(0.7446)も上回る。
コスト: 学習時間 V0′ 40秒 → **V1 約800秒(19倍)**。calibration 悪化は要温度較正。

### 反実仮想テスト(§9.3、test 3,000局面。**盤面166次元は固定**し手札IDだけ差し替え)

| 指標 | V0′ | **V1** |
|---|---:|---:|
| identity swap 平均 \|Δ\| | **0.00000** | **0.05775** |
| 0.01超変化した割合 | 0.0 | **80.4%** |
| count change 平均 \|Δ\| | 0.0 | 0.01361 |
| **identity 効果 / count 効果** | — | **4.24 倍** |

→ V1 は「手札の枚数」ではなく **カードの中身**に反応している。V0′ が厳密に 0.0 なのは
入力に手札IDが無いためで、テストが意図どおり効いていることの確認になる。

## 3.16 Phase6C: CARD select 識別は「失敗」ではなく「測定不能」だった

Phase6 の結論は当初「V1 は候補順位を改善しない(pairwise 0.428 < 0.5)」だった。
しかし **V0′ も V1 も** 0.5 未満・Spearman 負という揃い方は、モデル側でなく
**教師側**を疑うべき兆候だった。そこで教師の**自己一致**を測った
(`_gen_actionq_teacher.py`。候補ごとに K=6 の独立決定化を2群に割り、群間の順位一致を見る)。

| 指標 | **CARD** (22 groups) | **MAIN** (45 groups) |
|---|---:|---:|
| 教師 自己 top-1 一致 | **0.364** | 0.622 |
| **教師 自己 pairwise** | **0.560** | **0.782** |
| 教師 自己 Spearman | **0.162** | 0.630 |
| 平均 teacher spread(信号) | 0.0873 | 0.110 |
| **平均 teacher std(ノイズ)** | **0.0907** | 0.0799 |
| 高margin(≥0.02)での自己 pairwise | **0.474** | 0.728 |

**CARD では教師のノイズ(std 0.0907)が信号(spread 0.0873)より大きい。**
教師自身の pairwise が 0.560 = ほぼ偶然(0.5)であり、**これが測定の上限**。
モデルの pairwise はこの上限を超えられないので、
Phase6 で得た「V1 CARD pairwise 0.428」は **モデルの失敗を意味しない**(解釈不能)。

高margin サブセットで自己 pairwise が **さらに下がる(0.474)**のは、
CARD の "margin" が実差でなくノイズの産物であることの傍証。

一方 **MAIN の教師は使える**(自己 pairwise 0.782 / Spearman 0.630、信号>ノイズ)。

### なぜ CARD だけノイズなのか
CARD 選択が変えるのは主に**手札の中身**だが、ターン終端で評価する leaf は handcrafted で
**手札を一切見ない**。したがって候補間の差は「引き・シャッフルのばらつき」に埋もれる。
= 教師の構造上、CARD の差が出にくい。

## 3.17 Phase6.5 §3: Action-Q 教師データ監査

| 項目 | 値 |
|---|---:|
| `policy_positions.jsonl.gz` | 187,690 行 / 2,380 試合 |
| select種別 | MAIN 114,036 / **CARD 50,260** / YES_NO 16,490 / EVOLVE 3,181 / ENERGY 3,086 |
| 候補2以上の割合 | 0.913 |
| **outcome 結合可能率** | **1.000**(2,810/2,811) |

- Dataset A(Logged Return)/ C(Behavior Auxiliary): **即構築可能**
- Dataset B(Counterfactual Ranking): **生成が必要、かつ CARD では現行設計だと成立しない**

制約: policy_positions は `maxCount==1` かつ **alakazam 単一アーキタイプ**。
Dataset B 生成時はアーキタイプ多様性を別途確保すること。

## 3.18 Phase7 Track B: CARD 教師の再設計は不成立(ゲート FAIL)

同一 rollout を使い回して **leaf 種別だけ**を変え、K は入れ子で切り出す統制比較
(`_diag_card_teacher_redesign.py`、34 groups)。rollout をやり直すと leaf 差と
rollout ノイズが交絡するため、末端 state を保存して3評価器で採点した。

| 教師 | horizon | K | 自己pairwise | 自己Spearman | top-1 | spread | std | spread/std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **T0(現行 handcrafted)** | 次ターン開始 | 6 | **0.642** | **0.395** | **0.529** | 0.0493 | 0.0550 | 0.983 |
| T1 V1(hand-aware) leaf | 同 | 6 | 0.555 | 0.062 | 0.294 | 0.0603 | 0.0730 | 0.863 |
| T2 blend 0.5 | 同 | 6 | 0.603 | 0.180 | 0.353 | 0.0514 | 0.0567 | 0.882 |
| T3 長い地平 | opp_depth=2 | 6 | 0.525 | 0.060 | 0.353 | 0.0940 | 0.1009 | 0.999 |
| T5 K=12 | 次ターン開始 | 12 | 0.647 | 0.269 | 0.382 | 0.0423 | 0.0604 | 0.796 |
| T5 K=24 | 同 | 24 | 0.552 | 0.239 | 0.147 | **0.0391** | 0.0659 | **0.716** |

**ゲート(pairwise≥0.70 / Spearman≥0.50 / top-1≥0.55 / spread>std)は全構成 FAIL。**

### 前フェーズの仮説は否定された
「leaf を手札が見える評価器(V1)にすれば CARD 教師は安定する」という提案は誤りだった。
**T1 は T0 より悪化**(0.642 → 0.555)、地平延長 T3 も最悪(0.525)。

### 本質: 教師がノイジーなのではなく、差そのものが小さい
K を 6→12→24 と増やすと **spread が 0.0493→0.0423→0.0391 と縮み**、
std は 0.055〜0.066 のまま下がらない。K=6 で見えていた「候補間の差」の相当部分がノイズであり、
**真の候補間差は rollout ノイズより小さい**(K=24 で spread/std = 0.716)。
= K を増やして解決する問題ではない。

**CARD 候補は、この基準(自ターン終端〜次ターン開始の盤面価値)ではほぼ等価**。
手札にカードを加えても、使うまで盤面は変わらず、ターン終端までに差が薄まる。
V1 leaf でも改善しないのはこの構造のため。

### T4(CRN強化)は実施不可
cg のシャッフル/相手乱数ストリームが API 非公開。共有できるのは root 決定化まで。
乱数基盤の全面改修は本フェーズの禁止事項に該当するため未実施として報告する。

→ §14 に従い **CARD Action-Q は実装しない**。

---

## 4. 未了・次の判断

- 葉 α スイープ(`climb_v15_leaf_a*`)と動的top-k の**実ゲーム A/B は未実施**。
  config と評価基盤は揃っている(`measurement/driver.run_sprt_ab` が SPRT + resume 対応)。
- top-k 見落とし率の実測(`_diag_topk_miss.py`)は実装済み・未実行。
- **1.2 の修正で相手が強くなったため、過去のローカル field 数値とは直接比較できない。**
  ベースラインは修正後に取り直す必要がある。
