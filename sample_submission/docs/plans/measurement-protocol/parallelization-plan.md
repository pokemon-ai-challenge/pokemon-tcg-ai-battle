# 方針書: 測定ハーネスの並列化・実用速度化

作成日: 2026-07-25
種別: **方針書**(短い仕上げフェーズの方向づけ。Step0 調査結果に基づく設計方針＋検証計画。
関数シグネチャ・完了条件まで要るなら別途 implementation-plan に落とす)
前提: 測定ハーネス MVP 完成([[project_measurement_protocol]]、`kaggle_replays/measurement/`、git 未追跡)
ステータス: **方針のみ。コード未着手。production/cg/shared league code は変更しない。**

---

## 0. 目的とスコープ

`abl_5_full` 等の重い agent は ~36 秒/試合。数百〜1000 試合要る SPRT 裁定や今後の
multi-opponent field evaluation では実用上重い。**既存の並列実行資産を最大限再利用し、
測定ハーネスを安全に複数試合同時実行できるようにする**のが目的。

**これは「短い仕上げフェーズ」**。並列基盤そのものを大規模プロジェクト化しない。今回の並列化は
今後の worthy opponent / field evaluation / self-play league / ISMCTS・Transformer・RL 等の
大規模介入を**現実的速度で測るための土台**であって、+0.5〜1pt の微差 config を大量裁定するのが
主目的ではない([[project_measurement_protocol]] §0.3 と整合)。

---

## 1. Step0 調査の結論(read-only 調査済み)

### 既存並列資産
| 資産 | 場所 | 種別 | 用途 | 追跡 |
|---|---|---|---|---|
| **`run_league.py`** ★本命 | `league/` | process(`ProcessPoolExecutor`) | 2 config の並列 head-to-head(measurement とほぼ同型) | tracked(**改変禁止**) |
| `collect_parallel.py` | `kaggle_replays/rl/` | process(`multiprocessing.Pool`) | RL 収集(純 PolicyModel) | 未追跡 |
| `collect_field.py` | `kaggle_replays/rl/` | process(`multiprocessing.Pool`) | マルチ相手 field 収集＋per-opp 勝率 | 未追跡 |

subprocess 並列・thread 並列・独立した replay 生成並列は**評価パスに存在しない**。

### 確定した事実
- **全て process 並列**。cg エンジンは**プロセス内シングルトン**(`cg/sim.py` の `Battle.battle_ptr`/
  `Battle.obs` はクラス属性＝プロセス毎の単一グローバル)。`run_league` も「スレッドではなくプロセスで
  分散する」と明記。→ **thread 並列は不可、process 並列が必須**。
- worker-init パターン(3 資産共通): Pool/Executor の **initializer で worker 起動時に 1 回だけ**
  agent/model/deck を構築し worker-global に保持。**worker 内で `os.chdir(sample_submission/)`**
  (spawn の子は cwd 非継承。私の `ensure_production_cwd` と同じ対処を run_league も実施)。
- **集計は index 昇順レコードのみから**行い実行順序非依存(`run_league._aggregate_records`)。
- `run_league.build_agent(name, weights, config_base)` は **ml_policy の config 注入そのもの**
  (deepcopy load_config ＋ policy_weights_path)＝ 私の `AgentSpec.build_agent` と同一思想。
- 評価パスは**完全に CPU 律速**(cg native＋純 Python/numpy、torch/GPU 不使用)。GPU は RL 学習のみ。
  → measurement 並列化は **CPU コア数でスケール**。Windows は `spawn`(fork 不可)。

---

## 2. 方針

**`run_league` の実証済みパターン(ProcessPoolExecutor＋worker-init＋順序非依存集計)を
`kaggle_replays/measurement/` 内に mirror する。** 直接 import・改変はしない(理由: run_league は
tracked＝改変禁止、かつ勝ち筋分類を持たず固定 N。shared code への結合は [[feedback_respect_ownership_boundaries]]
の観点でも避ける)。

- worker 本体 = measurement の **`runner.play_game`**(勝ち筋分類 terminal_flags つき)。run_league の
  `_play_game`(winner のみ)ではなく measurement 版を使うことで、記述診断を並列でも保持する。
- main は **group-sequential SPRT**: `workers` 件を並列実行 → **index 昇順で** SPRT に投入 → 判定 →
  未決なら次バッチ。SPRT の早期停止を**バッチ粒度**で維持(overshoot は最大 workers-1 試合＝ASN~1000 に対し無視可)。
- 既存の逐次 driver(`run_sprt_ab`)と**同一の推定量・レポート・resume 契約**を保つ。並列は実行方式の差だけ。

---

## 3. 設計(要点)

### 3.1 worker
- `pool_init`(initializer): `os.chdir(sample_submission)` → `AgentSpec.build_agent` で cand/ctrl agent を
  worker-global に構築(process 毎に独立コピー＝混線なし)。
- task = `(game_index, seed)`。worker: `cand_side = game_index % 2` で p0/p1 割当 → `runner.play_game`
  → `cand_won = (winner == cand_side)` → **outcome レコード dict**(既存 outcomes.jsonl と同一スキーマ:
  index/cand_side/winner/cand_won/primary/flags/margin_p0/error/loser_by_error)を返す。

### 3.2 main(group-sequential ループ)
```
outcomes = resume で outcomes.jsonl を復元(あれば) → SPRT 再生
while SPRT 未決 and N < n_max and N < max_games:
    batch = 次の workers 件の task
    results = ProcessPoolExecutor で batch を並列実行
    for rec in sorted(results, key=index):        # ← 必ず index 昇順(順序非依存・決定論)
        outcomes.jsonl に追記
        SPRT.update(rec["cand_won"])
        if SPRT.decision: break                    # バッチ内で判定が出たらそこで停止(残りは記録済み)
report = candidate_estimand + outcome_breakdown + 再現メタ(既存と同一)
```
- **outcomes.jsonl への追記は必ず canonical index 昇順**(並列完了順ではない)。これで SPRT 決定・
  resume が実行方式・worker 数に非依存で決定論的になる(run_league の順序非依存集計と同原理)。
- resume・checkpoint・推定量・レポートは既存 `run_sprt_ab` と共通化(並列/逐次でコード分岐は最小)。

### 3.3 worker 数
- `resolve_workers(None/0/負) = max(1, CPU数-1)`(run_league 準拠)。既定は控えめ(CPU-1)。
  cg は CPU 律速なので理想は物理コア数程度。過剰 worker はメモリ(各 process が cg＋model ロード)に注意。

---

## 4. 危険な共有状態と対策(process 分離で解決)

| 共有状態 | 実体 | thread 並列だと | 対策 |
|---|---|---|---|
| cg `Battle.battle_ptr`/`obs` | `cg/sim.py` クラス属性=単一 global | 2 試合並走で battle_ptr 上書き=破綻 | **process 分離必須** |
| ml_policy module global | `_config_cache`/`_model`/`_model_cache_by_weights_path`＋pipeline タイマー `_match_start_perf`/`_selects_seen`＋`match_context` | 同時対戦でタイマー・信念状態が混線 | process 分離で各 worker 独立 |
| CWD | agent が deck.csv/configs を cwd 相対で読む | 子は cwd 非継承 | **worker-init で chdir(sample_submission)** |
| native RNG | `GameInitialize()` が import 時に random_device seed | — | プロセス毎に別シャッフル=試合独立(既知・非再現)。並列でばらつきは増えない |

---

## 5. 非目的(やらないこと)

- **`run_league.py`(shared/tracked)を改変・import しない。** パターンを mirror するのみ。
- **thread 並列にしない**(cg シングルトンで不可)。
- **fork ベース CRN はやらない**(Windows は spawn。設計 §3-b の完全 CRN は Linux/WSL2 前提で本フェーズ外)。
- **微差 config の大量自動裁定を主目的にしない**(§0)。
- **並列基盤を汎用大規模化しない**(measurement 内の最小 mirror に留める)。
- production/cg/weights/deck.csv/shared 変更なし。全て `kaggle_replays/measurement/`(未追跡・push 無し)。

---

## 6. 検証計画(並列版が逐次版と同値であることを固定)

1. **決定論**: 同一 outcomes.jsonl(同一 cand_won 列)を SPRT 再生 → 逐次版と decision/N/llr が厳密一致
   (既存 test_sprt の replay 一致テストと同型。worker 数を変えても outcomes.jsonl の index 昇序が同じなら
   SPRT 結果は同じ、を固定)。
2. **統計的同値**: A/A を workers=1 と workers=k で複数 run → 勝率分布・decision 分布が有意差なし
   (cg RNG は seed 制御外なので run ごとにばらつくが、並列がばらつきを増やさないことを確認。run_league と同主張)。
3. **較正の並列再現**: [[project_measurement_protocol]] の ground-truth 較正(学習済み vs first_choice)を
   並列で再走 → PROMOTE/FUTILITY の向きが再現。
4. **速度**: workers=1 と workers=CPU-1 で同 N の wall-clock を実測(期待 ~コア数倍、cg CPU 律速)。
5. **無破壊**: production/cg/shared 無変更を git で確認、既存 18 テスト green 維持。

---

## 7. 実装時に確定する未解決点

- **バッチ粒度と overshoot**: batch=workers 固定か、判定接近時に縮めるか(既定=workers 固定、overshoot≤workers-1 許容)。
- **spawn の起動コスト**: worker-init で cg＋model ロード(~数秒)×worker 数。長い run では償却されるが、
  短い run では逐次の方が速い場合あり → **workers=1 で従来逐次に完全一致(フォールバック)**を保証。
- **checkpoint 頻度**: outcomes.jsonl はバッチ毎 flush(既存の 1 試合毎 append と同等の耐障害性)。
- **worker 数既定**: CPU-1。重い config はメモリ上限で worker 数を絞る判断。
- **将来拡張(本フェーズ外)**: collect_field の「相手を share でサンプル＋per-opp 勝率」パターンは
  multi-opponent field evaluation にそのまま応用可(worthy opponent トラックで再利用)。

---

## 8. 実装・検証結果(2026-07-25)

**実装済**(全て `kaggle_replays/measurement/`、未追跡・push 無し): `driver.run_sprt_ab` に `workers` 引数を
追加。`workers<=1` は従来の逐次(完全フォールバック)、`workers>1` は `ProcessPoolExecutor`＋
`_par_worker_init`(chdir＋build_agent)＋`_par_play`(=`runner.play_game`)で group-sequential。batch を
並列実行 → `ex.map` の入力順(index 昇順)で SPRT 投入・outcomes.jsonl 追記 → 判定が出たら残 batch 破棄
(jsonl 行数 == sp.n を維持)。`resolve_workers`(CPU-1)。report.meta に `workers` 追加。`run_league` は
import も改変もしていない(パターンの mirror のみ)。検証: `parallel_check.py`。

**検証結果(§6)= 全 ALL OK**:
- 同値: A/A を workers=1 と workers=auto で同 max_games 実行 → N 一致・errors 0・outcomes.jsonl 行数一致。
- 決定論: 並列 run の outcomes.jsonl を独立再生 → driver の SPRT(n,s,llr)と一致(index 昇順追記ゆえ完了順非依存)。
- 較正の並列再現: 学習済み vs first_choice を workers=7 で → **PROMOTE @N=73, winrate 0.726**(逐次と同結論)。
- 無破壊: production/cg/shared 無変更、既存 18 テスト green 維持。

**速度(重要な実測)**: このノート PC(logical 8 コア)で **speedup ~1.65〜2.0x** に留まる。64 試合 63.8s→38.7s。
逆算すると **cg ワークロードの実効並列度は ~2 コア相当**(Iris Xe 級・CPU/メモリ律速・thermal で contention)。
→ **並列化は正しく動くが、本機はサチる。真の throughput はコア数の多いホスト(desktop/クラウド)で得る。**
n_max は SPRT が truncate、workers はコア数でスケール。

**運用注意(Windows spawn)**: 並列は**本物の .py ファイル＋`if __name__=="__main__"` ガードから呼ぶこと**。
`python -c`/heredoc(stdin=__main__)で呼ぶと子プロセスが __main__ を再 import できず `BrokenProcessPool`。
`agent_fn` を使う場合は **picklable(None か top-level 関数)**であること(lambda/closure は不可)。

---

## 付録: 参照した既存資産(read-only)
- `league/run_league.py`: `build_agent`(config 注入)、`_worker_init`(chdir＋build)、`_worker_play`、
  `_aggregate_records`(順序非依存)、`run_league`(ProcessPoolExecutor)、`resolve_workers`(CPU-1)。
- `kaggle_replays/rl/collect_parallel.py` / `collect_field.py`: `multiprocessing.Pool`＋initializer＋
  worker-global `_W`＋`pool.map(tasks)`。純 PolicyModel worker(measurement は full ml_policy を使う点が差)。
- `cg/sim.py`: `Battle` クラス属性(単一 global)、`GameInitialize()` at import。
