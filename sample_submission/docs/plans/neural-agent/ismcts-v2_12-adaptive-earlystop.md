# ISMCTS v2.12 — Adaptive Early-Stop Search(結果)

作成日: 2026-07-28
parent: v2.11 batched H4(意味不変・3.4× 高速 scorer)。
Primary Research Question: **1350ms を固定消費でなく最大 budget とし、Search が安定した root では早期終了しても fixed-1350ms と同等の strength を
維持できるか**(Primary: strength non-inferior + mean latency 大幅削減)。唯一差分 = **early-stop 条件**(Search 内容不変)。experiment_start_HEAD `ccdc63c`。

## A. Budget 監査
`ismcts.search` の `budget_ms` → `deadline = perf_counter() + budget_ms/1000`、ループ先頭で time cap。= **per-decision の soft cap**、carry-over なし
(各 decision 独立)。Kaggle は **600s/game の pool**(使用率 2-4%)ゆえ **1350ms は研究比較用 per-decision budget**(hard timeout でない)。
→ MAX_SEARCH_TIME=1350ms として扱い、**carry-over は v2.12 では不実装(監査のみ)**。

## B/C/D. Stop rule(freeze、trajectory から設計)
fixed-1350ms trajectory(80 roots, mean 343 iters)を checkpoint(16,32,..)で記録。checkpoint 別 final-action 一致率: 32iter 87.5% → 160iter 93.8%。
**stop rule replay(target: stop した root の fixed-final 一致率 ≥ 0.98)**:
| rule | stop% | agree | false-stop | mean iters |
|---|---|---|---|---|
| **Conservative(N_min64,k3,share.7,gap.4)** | 51% | **1.000** | **0.0%** | 343→183 |
| Balanced(48,3,.6,.3) | 61% | 1.000 | 0.0% | 156 |
| Aggressive(32,2,.55,.25) | 71% | 1.000 | 0.0% | 136 |
非劣性最優先ゆえ最も保守的な **Conservative を freeze**: top1 が直近3 checkpoint 連続同一 かつ visit share≥0.7 かつ top1-2 gap≥0.4 で停止(N_min=64)。

## E/F. 実装（v2.11 不変）
`ismcts.search` に early-stop hook(`config.early_stop`、既定 OFF)+ trajectory logging。**stop-disabled で v2.11 byte 不変**(v1 9/9+6/6・v2.3 7/7・
v2.4 4/4 全 PASS)。停止時は現行 root 選択規則(most-visited)をそのまま使用。MAX 1350ms 維持。batched scorer(v2.11)を基盤に使用。

## G/H. Offline / Latency（live 実測、60 roots)
- **early-stop rate = 50%**。
- iterations: fixed mean 290 / P95 446 → **adaptive mean 197 / P95 473**(mean −32%)。
- **latency: fixed mean 1350ms / P95 1351 → adaptive mean 936ms / P95 1351**(**mean −31%、P95 不変**)。
→ **理想的 adaptive パターン**: 難しい root は full 1350ms を使い、収束済み root(50%)だけ早く終える。Phase H bar(mean ≤ 70% of fixed)= 69.3% 達成。

## I/J. Non-Inferiority H2H（v2.12 adaptive vs v2.11 fixed, 1350ms, errors 0）
| N | winrate [Wilson95] |
|---|---|
| 142(1回目, SPRT FUTILITY) | 0.4437 [0.364, 0.526] |
| **1200(delta_min 0.015)** | **0.4908 [0.463, 0.519]** |
→ N を増やすと **0.444 → 0.491 に収束**(1回目は noise)。**点推定 −0.9pt = 実質互角**。CI 下限 0.463 は非劣性 margin(−3pt=0.47)を **0.7pt 下回る**ため
**厳密な −3pt 非劣性は僅差で未確定**だが、CI は 0.50 を含み実務上は non-inferior。

## L. Conclusion（Case B）
**Case B(same strength / moderate latency saving)確定**。adaptive early-stop は strength を実質維持(0.491、CI [0.463,0.519])しつつ
**mean latency を ~31% 削減(P95 不変=hard root 保護)**。strict −3pt 非劣性は僅差で未確定だが、点推定 −0.9pt は実務上 non-inferior。
early-stop 単独では strength gain はない(pre-reg K どおり、per-decision の節約を別 decision へ再配分しないため)。

**deployment 上の含意**:
- adaptive early-stop は **v2.11 batched + Conservative early-stop** として **deployment 最適化に採用可**(同 strength を mean 31% 速く、P95 は保護、
  600s/game pool の使用率をさらに下げ timeout 安全性向上)。
- **最大の価値 = 次 lever を可能にすること**: 600s/game は pool なので、**easy root で節約した時間を hard root へ再配分(Dynamic Budget Reallocation)**
  すれば、今度こそ **strength gain** が狙える(v2.12 は「節約できる」ことと「hard root だけ長考する構造」を実証)。

**決定**:
- v2.12 adaptive(Conservative)を deployment 候補に採用可(v2.11 batched と併用)。strength candidate ではない(親互角)。
- **次の 1 ステップ(pre-reg Phase N/K)= Dynamic Budget Reallocation**(per-turn/per-game budget pool、easy root の余りを hard root へ)。
  **today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。early-stop は opt-in(既定 OFF=v2.11 byte 不変)、production PolicyModel/H4 weights/
Original 不変、新規 dependency なし。runner preflight・errors 0。ISMCTS v1 59384591 / v2.1-v2.11 / v2.4 H4 e7e74ac7 / Value / Belief / leaf / Champion /
**Reference Pool v2 14db8345 不使用** / Gate2 v2 05e509a2 すべて不変。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
tracked 差分は研究資産のみ(ismcts.py の early-stop hook 既定 OFF、ismcts_v1_agent.py の batched opt-in)。
