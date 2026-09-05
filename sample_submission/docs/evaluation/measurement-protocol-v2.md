# Measurement Protocol v2

小効果（+1〜2pt）を現実的な時間で検出できる評価基盤。Measurement Resolution Audit v1 の結論を運用に落とす。
**strength アルゴリズムは一切変えない。measurement のみ。**

---

## 0. なぜ MDE が重要か（Audit v1 要約）

過去の neutral/FUTILITY の大半は「失敗」ではなく **UNDERPOWERED（測定不能）** だった。

fixed-N 80% power MDE（H0: p=0.50, α=0.05 two-sided）:

| N | MDE(80%) |
|---|---|
| 300 | 8.1pt |
| 449 | 6.6pt |
| 1200 | 4.0pt |
| 2000 | 3.1pt |
| 3000 | 2.6pt |
| 8700 | 1.5pt |

必要N: +1pt→**19,620** / +1.5pt→8,719 / +2pt→4,904 / +3pt→2,178。

→ v2.5〜v2.13 が使った N=300〜1200 では **+1〜2pt を原理的に判定できない**。ISMCTS vs PIMC の +7〜8pt は検出できた（大効果だから）。

## 1. Evaluation Contract（freeze）

- **Primary metric = binary win/loss を維持**。prize margin / terminal type / deckout / latency / iterations / Search override はすべて **diagnostic**。実験中に primary を変えない。
- 統計 default: **alpha=0.05, power=0.80**（calculator で 90% も出せる）。**小効果は fixed-N を primary**。
- SPRT（`sprt.py`, δ_min=0.03/0.05）は **削除しない**が用途を限定（§4）。

## 2. Tools

| ツール | 役割 |
|---|---|
| `measurement/power_calc.py` | MDE / 必要N / CI半幅 / 所要時間 / SPRT OC。`--validate` で MC 検証（emp.power 0.80/0.90 一致） |
| `measurement/gauntlet_manifest.py freeze` | fast-gauntlet 相手プールを artifact hash 付きで凍結 |
| `measurement/gauntlet_manifest.py prereg` | 実験前登録（target effect→N, hashes, machine） |
| `challengers/algo_gauntlet.py` | fast development gauntlet 本体（candidate 各セル vs frozen field） |

使用例:
```
python power_calc.py --validate           # MDE/必要N 表 + MC 検証
python power_calc.py --plan 1.5            # +1.5pt の必要N・所要時間
python gauntlet_manifest.py freeze         # 相手プール凍結
python gauntlet_manifest.py prereg --name X --parent P --candidate C --effect 2.0
```

## 3. Evaluation Modes（estimand が違う点に注意）

| Mode | 内容 | 速度(desktop 15w) | estimand |
|---|---|---|---|
| **Direct H2H** | Candidate vs Parent（同deck, mirror） | ~4.6 g/min | 最も直接的な strength |
| **Fast Gauntlet** | 各セルを frozen fast opponents へ | ~24 g/min | field 相対 strength（surrogate） |
| **Stratified Field** | 複数 archetype 別 + macro/weighted | ~24 g/min | matchup 分解 |

Fast Gauntlet は Direct H2H と **同一 estimand ではない**。高速だが、重要な production 昇格は Direct H2H かより deployment 代表な field で確認する（§5, §6-Tier3）。

## 4. SPRT の位置づけ（repositioning）

- **SPRT = large-effect screen（≥ ~3pt）専用**。δ_min=0.03 は「≥+3pt か」を早く判定する道具。
- **FUTILITY は「候補が悪い」ではない**。真に中立(p=0.50)でも FUTILITY 率は n_max=600/1200/3000 で 33/64/**91%**。FUTILITY = 「≥+3pt ではない」。
- **+1〜2pt 評価に SPRT を使わない**（真 +1pt の PROMOTE 率は 3〜20%）。小効果は **fixed-N**（§6 Tier3）。

## 5. Fast Gauntlet Calibration（Audit の positive/neutral control）

既存の Cross-Evaluation gauntlet（同deck, N=100/相手×8, errors=0）で校正済み:

- **Positive control**: ISMCTS(61.6%) vs PIMC(53.9%) macro = **+7.8pt, p=0.001**。gauntlet は既知の大差を **正しい符号で検出**。個別も rocket_mewtwo +20(p=0.004)。
- **Neutral control（同一 run 内）**: 真に中立な matchup（mega_lucario 58→58, dragapult 80→79）で **Δ≈0**。gauntlet は **架空の大差を作らない**。

**結論: Fast Gauntlet は Direct H2H の candidate ranking 用 development surrogate として使用可能（PARTIAL→YES: rank/sign/large-effect は再現。数値一致は要求しない = Phase E4）。** ただし production 最終昇格は Direct H2H / deployment field で確認（1指標だけで昇格しない）。

## 6. Tier Protocol v2

| Tier | 目的 | default N | MDE(80%) | mode |
|---|---|---|---|---|
| **0 Correctness** | tests/smoke/errors/illegal/fallback/artifact identity PASS | — | — | — |
| **1 Cheap Screen** | 大改善・明確な regression のみ | ~400 | ~7pt | Fast Gauntlet |
| **2 Confirmation** | ~3pt 級 | ~2,200 | ~3pt | Fast Gauntlet / Stratified |
| **3 Formal small-effect** | +1.5pt（高価・有望候補のみ） | ~8,700 | ~1.5pt | fixed-N（+ Direct H2H 確認） |

- **N は hard-code しない**。candidate ごとに `prereg` で target effect→N を freeze（Adaptive N, Phase H）。例: 「+2pt あれば価値」なら N≈4,900 で十分。
- **Tier1 で 50〜53% だから neutral/reject と判断しない**。`small effect unknown` = **UNDERPOWERED** と記録するだけ。

## 7. Decision Vocabulary（label 統一）

**禁止: 「CI が 0.5 を跨ぐ → neutral/failure」。**

| label | 定義 |
|---|---|
| **PROMOTED** | 事前 N で有意に正（CI 下限 > 0.50 or SPRT PROMOTE） |
| **REGRESSION** | 有意に負（CI 上限 < 0.50） |
| **LIKELY_NEUTRAL** | 十分 power（MDE ≤ target）で点推定 0 付近、CI が target 級効果を排除 |
| **UNDERPOWERED** | achieved MDE > 想定効果。**「効果なし」と書かない** |
| **INCONCLUSIVE** | error 過多 / protocol 逸脱 で無効 |
| **ENGINEERING_SUCCESS** | strength 中立だが latency/throughput/保守性を改善（例: v2.11/v2.12） |

## 8. Hardware Rules（Phase K）

Search を伴う評価は **machine / wall-time budget / mean iterations/root / P50・P95 iterations** を必ず保存。

既知: **Laptop ≈ 70 it/1350ms、Desktop ≈ 240 it/1350ms**。異なる hardware の結果を同 pool に混ぜない。formal strength は reference/deployment regime（desktop）優先。

## 9. Pre-registration / Result Schema / Resume

- **Pre-registration**（`prereg`）: 実験前に name/parent/candidate/unique-diff/target-effect/N/mode/alpha/power/machine/artifact-hashes/start_HEAD を固定。**開始後に target/N/opponent/primary を結果を見て変えない**。
- **Result schema（標準 JSON）**: games, wins, losses, winrate, Wilson CI, SE, CI half-width, target effect, planned N, completed N, machine, wall-time, games/min, errors, illegal, fallback, per-opponent, terminal distribution。
- **Resume/crash safety**: 8k〜20k game run では incremental 保存必須。resume 時に manifest/parent/candidate hash 照合、違えば resume 禁止、completed games を重複しない。
- **Errors ≠ 敗北**: errors / loser_by_error / invalid / timeout を別分類。error 率が閾値超なら strength 結果を無効化（INCONCLUSIVE）。

## 9.1 Long-run Auto-Resume（v2.1、Tier3 用）

8k〜20k game の Tier3 formal run が中断（crash / 電源断 / Ctrl+C）しても、**同一 experiment identity を保証した上で
重複・欠損・条件変更なく再開**する。`measurement/resume.py` + `algo_gauntlet.py --resume-dir DIR`。

- **どう動くか**: 起動時に `run_manifest.json` を照合。無ければ **NEW**（manifest を freeze）、semantic hash 一致なら
  **RESUME**（`progress.jsonl` の completed game_id を読み、schedule から除外して remaining のみ実行）、1 項目でも
  semantic mismatch なら **REFUSED**（既存へ絶対に追記しない、差分を表示）。
- **hash に含む（semantic identity）**: experiment_name / parent / candidate / config sha1（parent・candidate・opponent）/
  policy weights sha1 / deck sha1 / gauntlet manifest sha1 / evaluation_mode / primary_metric / **planned_n** / target_effect /
  alpha / power / **allocation**（cell×opponent の予定games、順序非依存）/ macro definition / search_budget_ms / **machine_class**。
- **hash に含まない**: timestamp / output path / absolute path / worker 数 / logging 用フィールド（semantics 無関係）。
- **Game ID**: `(exp_hash|cell|opponent|game_index|side)` の決定的文字列。同 manifest なら再起動後も同じ schedule から
  同じ id が出る。**⚠️ native full-game seed API が無いため bitwise 再現は保証しない**（Phase V）。保証するのは
  「同 contract の planned slot を重複なく最後まで埋める」こと。
- **durable 記録**: `progress.jsonl` に 1 game 1 行 append + fsync。**durable 保存後のみ completed**（worker crash で
  未保存の game は resume で再 dispatch）。source of truth = manifest + per-game records（summary は再構築物）。
- **重複防止**: 同一 game_id を二重集計しない（起動時 dedupe + record 時 skip）。
- **corruption**: broken JSON / 未知 game_id を検出したら **RESUME REFUSED**（silent ignore しない）。
- **inspect**: `progress.jsonl` を数える（= completed）/ `run_manifest.json` の `_resume_manifest_hash` を確認。
- **restart**: 同じコマンド（同 `--resume-dir` + 同 args）を再実行するだけ。
- **性能**: record overhead ≈1.5ms/game（fsync）= game 時間の 0.01〜0.03%（≪5%）。20k schedule scan ≈0.4s / 8MB。
- **backward compat**: `--resume-dir` 未指定なら従来挙動（count のみ partial）。旧 partial format は自動変換しない。
- **hardware**: `machine_class`（desktop_reference / laptop）を hash に含む → 原則 desktop→laptop resume は REFUSED。

## 10. What changes from previous practice

- N=300 で small effect を判定しない。
- FUTILITY を failure 扱いしない。
- 実験前に target effect を決め、必要 N を先に計算する。
- small effect は SPRT でなく fixed-N。
- hardware / iterations を必ず保存。
- Fast Gauntlet を formal Direct H2H と同一視しない。
- Reference Pool v2 / Gate2 は held-out（calibration にも使わない）。
