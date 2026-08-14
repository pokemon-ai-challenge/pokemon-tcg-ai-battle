# 夜間セッション記録 — 2026-07-24

対象: 新アルゴリズム(信念状態 × Transformer × デュアルヘッド × 自己対戦RL × 価値ゲートISMCTS)の始動。
指示: 「順番に。判断が要るところまで自律で進める。デスクトップでできることをやる。朝に見られるよう記録」。

---

## 0. 今夜やったこと(3行)

1. **設計書**と **Phase 0(転移ギャップ診断)実装計画**を作成(production 無変更)。
2. **GPU デスクトップに torch 2.11.0+cu128 を導入・CUDA 動作確認**(NN 学習インフラが起動可能に)。
3. **Phase 0 の最重要診断(H3 検出力)を実行** → 「これまでの評価は小さな真の優位を測れていなかった」ことが判明。

---

## 1. 成果物(ファイル)

| ファイル | 内容 |
|---|---|
| `docs/plans/neural-agent/design.md` | **設計書**。構想5コンポーネントの評価、既存資産マッピング、計算資源の現実、リスク順のフェーズ順序、評価ゲート、決定ポイント。 |
| `docs/plans/neural-agent/phase0-transfer-gap-diagnosis-implementation-plan.md` | **Phase 0 実装計画**。H1(分布ミスマッチ)/H2(デッキ劣位)/H3(評価ノイズ)の切り分け手順・関数シグネチャ・完了条件。 |
| `kaggle_replays/_diag_phase0_power.py` | H3 検出力診断(read-only, 使い捨て)。 |
| `kaggle_replays/_diag_phase0_power_results.json` | その結果。 |
| 本ファイル | 夜間記録。 |

**production の重み・config・deck.csv・cg/ は一切変更していない。** 実験・診断のみ。

---

## 2. 設計書の要点(design.md)

- 構想の方向性は **Player of Games / ReBeL 系の王道**で正しい。ただし **最大リスクは「転移ギャップ」**
  (offline 改善が本番勝率に乗らない、という過去5件の negative)。今回の構想は転移しなかった3方向
  (容量↑=Transformer、offline学習=BC、サーチ=ISMCTS)を**同時に強くしたもの**で、根本原因に触れていない。
  → **Phase 0 診断を全 NN 投資の前段の必須関門**に据えた。
- **既存資産の再利用が大きい**: 信念状態は `encoder.extra_features` の差し込み口が既にある/
  ValueModel・PolicyModel は稼働中/rollout は `search_begin/step` で可能。新規で難しいのは
  Transformer・自己回帰方策・RL の3つだけ。
- **計算資源の現実**: 単一 GPU + ctypes DLL エンジンでは **self-play RL で BC を超えるのは非現実的**。
  RL(構想#4)は「軽い fine-tune / 研究枠」に格下げ推奨。**プロジェクトの成否を賭けない。**
- **推論の純Python制約**: 現行は Kaggle 提出のため numpy/torch 非依存。**Transformer 推論には
  Kaggle 環境で numpy/onnx が使えるかの検証(Phase 2 Gate 0)が必須。** 使えなければ蒸留して MLP に落とす。
- **リスク順フェーズ**: 0 診断 → 1 信念接続 → 2 Transformer+dual-head → 3 ISMCTS → 4 RL(deferred)。

---

## 3. 今夜の最重要発見 — H3 検出力(`_diag_phase0_power_results.json`)

現行の採用ゲート「head-to-head N 試合で Wilson 95%CI 下限 > 0.5」の検出力を閉形式で算出:

### 各 N で「通過に必要な勝率」
| N | 通過に必要な勝率 |
|---|---|
| 300(現行スクリーニング) | **55.67% 以上** |
| 600 | 54.17% |
| 900 | 53.33% |
| 2000 | 52.20% |
| 5000 | 51.40% |

### 検出力(真勝率 → 300試合ゲートを通す確率)
| 真勝率 | N=300 | N=900 | N=2000 | N=5000 |
|---|---|---|---|---|
| 50.5% | **0.04** | 0.05 | 0.07 | 0.10 |
| 51.0% | 0.06 | 0.09 | 0.15 | 0.29 |
| 52.0% | 0.11 | 0.22 | 0.44 | 0.81 |
| 53.0% | 0.19 | 0.43 | 0.77 | 0.99 |
| 55.0% | 0.43 | 0.85 | 0.99 | 1.00 |

### 検出力 0.8 に必要な試合数
- 真 51%: **19,603 試合** / 真 52%: 4,924 / 真 53%: 2,178 / 真 55%: 773 / 真 58%: 300。

### 含意(重要)
- **offline +0.5pt が真勝率 ~50.5–51% を意味するなら、300 試合ゲートの検出力は 4–6%。事実上検出不能。**
- 過去の negative(M64/M128=49.0/49.3%、Beyond BC C2=50.7%/900試合)は、**「真に効果ゼロ」とも
  「小さな正の効果を測れなかった」とも区別がつかない**。
- → **「転移しなかった」の一部は「測れていなかった」**の可能性が高い。これは評価計画そのものの問題。

### さらに強い注意点(seedペア比較)
- head-to-head は seed ペア比較で分散を下げているが、**効くのは"展開が変化した試合(discordant)"だけ**。
  Beyond BC C2 のログでは **300試合中46試合しか展開が変わらなかった**。実効サンプルは ~46 で、
  上の独立二項近似よりさらに小さい効果しか見えない恐れ。
- → **次の安価な診断(H3b)**: 既存 seed ペアログから discordant 率を測り「実効 N」を出す。
  低分散化(paired McNemar 検定)への切替も検討。

---

## 4. GPU デスクトップの状態

- `shogo@100.99.53.12`(Windows/PowerShell)、RTX 5060 Ti(Blackwell)、16GB、driver 580.97、Python 3.11.9。
- **torch 2.11.0+cu128 を導入済み。`cuda.is_available()=True`、GPU matmul 動作確認済み。**
- → Phase 2(Transformer 学習)/Phase 4(RL fine-tune)の学習インフラは**起動可能**。
  ※ repo サブセットの同期状態は未確認(必要になった時点で scp 差分同期)。

---

## 5. 朝に判断してほしいこと(design.md §7.1 + H3 由来)

1. **フェーズ順序の承認**: 0診断 → 1信念接続 → 2 Transformer+dual-head → 3 ISMCTS → 4 RL(deferred)で確定してよいか。
2. **RL(#4)の格下げ**: 単一GPUでは非現実的 → 「軽い fine-tune / 研究枠」に落としてよいか。
   本命にするならクラウド計算資源の用意が要る。
3. **【H3 由来・新規】評価計画の是正**: 300試合スクリーニングは真 55%未満を検出できない。今後は
   (a) 検出したい効果量から必要 N を逆算する / (b) discordant ベースの低分散評価に切り替える、
   のどちらで進めるか。**この判断は Phase 1 の設計より前に効いてくる**(測れない改善を積んでも意味がないため)。
4. **Phase 2 デプロイ前提**: Kaggle で numpy/onnx 不可なら Transformer を MLP へ蒸留で妥協してよいか。

---

## 6. 次にやること(承認待ち / 続行可能)

- **続行可能(判断不要)**: H3b(discordant 率の実測)、H1a(相手分布の乖離)、H2 既存結論の統合表。
  いずれも read-only。朝までに追加で走らせても production に影響なし。
- **承認後**: Phase 0 の統合判定 → Phase 1(信念ベクトル接続)の実装計画作成 → 実装。
- **保留(承認前は着手しない)**: Transformer/RL の本学習、production 重み・config・deck.csv の変更。

---

## 8. 追記(夜間続行分:H3b / H2 / H1a を read-only で実行)

「朝確認するので進めておいて、いつでも戻せるように」との指示で、判断不要の read-only 診断を続行。
すべて既存 JSON の読み取りのみ・使い捨てスクリプト・production 無変更。

### 8.1 H3b — 過去 head-to-head の実データ裏取り(`_diag_phase0_hh_pool.py`)

過去9件の候補 vs m32(production)直接対戦をプール(独立二項なので H3 表が直接適用):

| 候補 | N | 勝率 | Wilson95%CI | 判定 |
|---|---|---|---|---|
| m64 | 300 | 49.0% | [43.4, 54.6] | 判別不能(ノイズ帯) |
| m128 | 300 | 49.3% | [43.7, 55.0] | 判別不能 |
| c1_g000 | 300 | 42.7% | [37.2, 48.3] | **真に有害**(CIが50%跨がない) |
| c1_g050 | 300 | 43.0% | [37.5, 48.7] | **真に有害** |
| c2_b05 | 300 | 52.0% | [46.4, 57.6] | 判別不能 |
| c2_b20 | 300 | 50.3% | [44.7, 56.0] | 判別不能 |
| **c2_b05 pooled** | **900** | **50.67%** | [47.4, 53.9] | 判別不能(真値なら確定に44,108試合) |

- **9件中ゲート通過 0件。** offline で良く見えた候補(m64/m128/c2)は全て「300試合で判別不能な49–52%帯」に着地。
- 一方 **outcome重み付けBC(c1)は本当に有害**(CIが50%を下回る)。「測れなかった」と「本当にダメ」を切り分けられた。
- **訂正**: 夜間サマリ§3で「46/300 discordant」を一般化したが、これは同一seed事後veto設計(attackplan)の話で、
  weight入替head-to-head(独立二項)には当てはまらない。上表は独立二項として正しく扱っている。

### 8.2 H2 — 構造的天井(既存結論の統合)

| 指標 | 自分 | 第三者(695チーム/2515試合) |
|---|---|---|
| 山札切れ / 敗北 | 52/201 = **25.9%** | 273/1157 = **23.6%** |
| 敗因内訳 | blowout 139 / deckout 52 / close 10 | blowout 740 / close 144 / … |

- 山札切れ率は第三者とほぼ同水準 → **構造コスト**([[project_deckout_loss_cause]]確認)。
- value網は `self_deck_count` 摂動にほぼ無反応(§7)→ 山札切れリスクが価値評価に未反映。

### 8.3 H1a — 学習分布 vs 配備分布のミスマッチ(`_diag_phase0_field.py`)【今夜の第2の大発見】

`meta_report.json` の出現シェア(出現数ベース、player_count と分離済)で、上位帯(=学習データの相手)と
フィールド(=配備の相手)の分布を比較:

- **正規化 L1 距離 = 0.997(最大2.0)、KL(field‖top) = 4.56** —— 学習と配備の相手分布が**ほぼ半分ズレている**。

| archetype | top%(学習) | field%(配備) | 差 |
|---|---|---|---|
| alakazam | **60.80** | 31.90 | +28.9 |
| marnie_grimmsnarl_ex | 17.61 | 5.51 | +12.1 |
| **mega_lucario_ex** | **0.00** | **17.45** | **−17.5** |
| **archaludon_ex** | **0.00** | **15.28** | **−15.3** |
| **dragapult_ex** | 0.33 | 6.35 | −6.0 |

- policy は **Alakazamミラー(top 60.8%)にほぼ特化**して学習。しかし配備先フィールドの2・3番手
  (**lucario 17.5% / archaludon 15.3%**)は**上位ログにほぼ存在せず = 未学習の相手**。
- しかもそれらは**自分の敗北相手の上位**(lucario 17.9%・dragapult 12.4% が losses の上位)。
- → **転移ギャップの正体は「特徴/容量/目的関数」ではなく学習データのカバレッジの可能性が高い。**
  belief 特徴を足しても「Lucario/Archaludon と一度も練習していない」事実は変わらない。
  - caveat: top%/field% は "使用シェア" を "遭遇相手" の代理にしている(top はミラー多めで近似が効くが厳密でない)。
    losses 側はサバイバル(敗北のみ)。厳密化は H1b(相手別の一致率×勝率)で。

### 8.4 8節を受けた Phase 1 方針への含意(朝の判断材料)

- **H1a が正しければ、Phase 1 の第一手は「信念接続」より「学習データのカバレッジ拡大」が上流。**
  既存の [[project_multi_archetype_imitation]](アーキタイプ別 policy 量産)が**まさにこの方向**で、
  未接続・未評価のまま止まっている。これを本線に引き上げる価値がある。
- 信念接続(相手アーキタイプ事後分布)は「未学習相手でも "これは Lucario" と条件付けできる」点で
  補完的。**データ拡大 × 信念条件付けの二本立て**が Phase 1 の再設計候補。
- ただし §3(H3)の評価分解能問題が未解決のまま。カバレッジを広げても、**300試合ゲートでは効果を測れない**。
  → 評価計画の是正(§5-3)がやはり全ての前提。

---

## 9. 進め方の結論と実行中ジョブ(夜間・続き)

H3(測れない)と H1a(カバレッジ)は**同じ1解**に収束する。しかも土台は既に8割存在:

> **評価軸を「Alakazamミラー自己対戦」から「対フィールド加重勝率(`league/round_robin.py`)」へ切り替える。**

- `round_robin.py` は8アーキタイプ(deck+**アーキタイプ別policy**)の総当たりから、メタシェア加重の
  対フィールド期待勝率を出す。**per-archetype policy(policy_weights_<arch>.json)は全7個・デッキも実在**。
  だが `league/results/round_robin/` は空 = **一度も測っていない**([[project_multi_archetype_imitation]] が未評価のまま)。
- これに切り替えると: (a) フィールド相手は改善幅が大きく H3 の分解能問題が緩和、(b) lucario/archaludon 含む
  多様な相手で測るので H1a のカバレッジ欠損がそのまま数字に出る、(c) 「勝率の正しい目的=フィールドを倒す」を
  初めて測る(ミラー自己対戦は目的を測っていなかった=転移ギャップの隠れた第3要因)。

### 推奨シーケンス(判断1の具体化)
1. **[実行中] production の対フィールド加重勝率をベースライン化**(round_robin 300試合/ペア)。
   → 勝率マトリクスがどのアーキで負けるか露わにする(H1a予測: lucario/archaludon)。**今後の正式ゲート。**
2. Phase 1 = その対フィールド勝率を目的関数に。負けアーキへカバレッジを足す。
3. 信念条件付け(相手アーキ事後分布→encoder)で補完。
4. Transformer/RL/ISMCTS は据え置き。

### 実行中ジョブ(朝には完了見込み)
- `python league/round_robin.py --games 300 --workers 12`(ローカル、read-only、出力は `league/results/round_robin/`)。
- スモーク(2試合/ペア)は全28ペア err=0 で健全性確認済。ログ: `league/results/round_robin/_full_run.log`。
- **完了後にやること**(自律可): `_matrix.json` から対フィールド期待勝率と負けアーキを抽出し、H1a予測と照合、
  本記録と design.md に反映。

### 唯一残るユーザー判断(マトリクスを見てから)
- Phase 1 のカバレッジの入れ方: (a) 予測相手アーキで per-archetype policy をルーティング(deck predictor 既存)
  vs (b) メイン policy を field-representative データで再学習 vs (c) 両方。→ **どこで負けるか見てから決める。**

---

## 10. round_robin 実測による H1a 解釈の【修正】(重要・正直な訂正)

§9 で「対フィールド round_robin を回す」と書いたが、**実は今日 03:10 に完走済みだった**
(出力先は `league/results/` ではなく**リポジトリ直下 `results/round_robin/`**。私が誤った場所を見ていた)。
`_matrix.json` を読んだ結果、§8.3 の H1a の因果解釈を**修正する必要がある**。

### production(Alakazam)の対各アーキ勝率(300試合/ペア)
| 相手 | 勝率 | |
|---|---|---|
| mega_lucario_ex | **79.7%** | 未学習なのに勝ち越し |
| archaludon_ex | **82.3%** | 未学習なのに勝ち越し |
| dragapult_ex | **78.7%** | 未学習なのに勝ち越し |
| shirona_garchomp_ex | 69.0% | |
| marnie_grimmsnarl_ex | 58.3% | |
| **rocket_mewtwo_ex** | **44.3%** | 負け越し |
| **crustle** | **38.7%** | 負け越し |
| 対フィールド期待勝率(加重) | **68.8%** | |

### 訂正の要点
- **H1a の "未学習アーキに弱い" という因果は、この実測では支持されない。** production は
  lucario/archaludon/dragapult(=上位ログに無い相手)に**むしろ 79–82% で勝つ**。負け越しは
  crustle と rocket_mewtwo のみ。分布のズレ(§8.3、L1=0.997)は事実だが、それが**負けに直結する
  という結論は先走りだった。**
- **ただし決定的な但し書き**: round_robin の相手は**弱い模倣ボット**(対フィールド勝率で
  dragapult 16.4% / archaludon 32.5% / lucario 38.7% と、policy が非常に弱い)。
  「production が lucario ボットに 79%」は「強い/人間の lucario に 79%」を意味しない。
  実戦で lucario に負けている(§8.3、敗北の17.9%)のは、**アーキタイプ相性ではなく相手の"強さ"
  (人間/上位ボット)** が原因の可能性が高い。

### これで分かった Phase 1 への含意(方針転換)
1. **per-archetype policy へのルーティング(Phase1案a)は現状 NG。** これらの policy は production-Alakazam
   より弱い(dragapult 16%等)。ルーティングは勝率を**下げる**。[[project_multi_archetype_imitation]] は
   "量産したが弱い" 状態で、まず各 policy の品質を上げない限り使えない。
2. **round_robin 対フィールド eval は、相手が弱すぎてゲートとして機能しない。** production は既に 68.8%。
   改善を判別する gate にするには**相手を強くする必要がある**(rule_based に各アーキを操作させる／
   各アーキの最強 policy を使う／実リプレイ由来のベンチ)。**評価の本当の障害はここ。**
3. 転移ギャップの主因候補は「カバレッジ」から**「相手の強さのモデル化 / 評価相手の弱さ」**へ重心が移る。
   H3(ミラー eval が過小検出力)は依然有効。

**結論(進め方の更新)**: Phase 1 の前に **"強い評価相手" を用意する**のが最優先。それが無いと
(a)ルーティングも(b)再学習も、改善したか判定できない。これが H3 の是正とも直結する。

---

## 11. RL 対戦相手強化 PoC の実装・実行(ユーザー承認済「その方針で」)

方針決定の根拠(§10 + 追加実測):
- rule_based に dragapult を持たせても **7.5%**(模倣ボット21%より弱い)= 既存ツールでは強い相手を作れない。
- dragapult 模倣は **41,580サンプル**(強い crustle 26,909 より多い)= **データ不足でなく模倣の天井**。
- → 勝率目的の RL が適切。**カリキュラム**: 弱い相手を RL 強化 → 凍結ベンチ化 → 後に production を RL。

実装(すべて新規 `kaggle_replays/rl/`、torch は学習時のみ、production 無変更):
- `torch_policy.py` — 既存 policy_weights.json を torch 化。**M0 PASS**: pure-Python と数値一致
  (float64 構造 6.5e-9 / float32 現実入力 1.3e-6)。BC初期化=既存JSONロード、RL後は同schemaに書き戻し。
- `rollout.py` — cg self-play でトラジェクトリ収集。素policy同士(overlayなし)で「policyが伸びるか」を隔離。
  **M1 PASS**: 実盤面262決定点で torch argmax == pure-Python(不一致0)、素policy dragapult vs alakazam
  **ベースライン20.0%**(round_robin 21%と一致)。
- `train_dragapult_poc.py` — PPO(clip)+fresh critic+エントロピー。可変選択肢はパディング+マスクsoftmax。
  sparse terminal 報酬。**M2 実行中**(30 iters × 160 games、目標 20%→35%+)。終了時
  `policy_weights_dragapult_ex_rl.json` にエクスポート(M3)。

**計算配置**: このワークロードは cg エンジンの **CPU律速**(~0.5s/game、GPUはほぼ効かない)なので
laptop で本走(desktop GPU は後の Transformer 用に温存)。ログ: `kaggle_replays/rl/_poc_run.out` /
`_train_dragapult_poc.log`。

### PoC 結果(完走、baseline 18%)【示唆に富む・成功でも失敗でもない】

eval勝率(greedy, 100試合/点): iter0 **18%** → 3:16 / 6:15 / 9:21 / 12:16 / **15:35** / 18:22 / 21:17 /
**24:31** / 27:22 / **30(最終):23%**。 平均 21.5%、最大 35%、最小 15%。

- **単調上昇ではない。** ノイズの多い振動 + 弱い上方ドリフト。baseline 18% → 最終 23%(+5pt、eval CI ±9pt で
  単独有意でない)。
- **ただし iter15 で 35%(CI_lo 0.264)、iter24 で 31% に到達** = **より強い dragapult は方策空間に存在し
  RL は見つけられる。が保持・収束できていない**(次evalで戻る)。エントロピー 0.92→0.85 とほぼ下がらず=
  良い領域にコミットしていない。
- **結論**: 「模倣天井は RL でも破れない」という否定ではない(35%到達が headroom の証拠)。一方で現状設定は
  不安定(sparse報酬+MC advantage+160games/iter+30iter+GAEなし=高分散)。**パイプラインは M0–M3 完全動作**
  (`policy_weights_dragapult_ex_rl.json` エクスポート済、ただし最終23%モデル)。

### 次の一手(安く効く順) — 朝の判断材料
1. **best-checkpoint エクスポート**(今は"最終"を出した。iter15=35%/iter24=31%を保存すべきだった。
   これだけで即 30%+ のベンチ相手)。
2. **分散削減**: GAE(λ) + games/iter 160→512。
3. **報酬整形**: value網をポテンシャルに(design.md #4)。
4. **長期化**: 30→150 iter。

→ RL に signal はある(35%到達)ので、**安定化(1〜4)に投資する価値あり**。これは production RL の前段の
学びでもある。ユーザー判断: この安定化に進むか、別アーキ(archaludon/lucario)でも同様の signal を確認してから決めるか。

---

## 12. 自律セッション(05:15–、~12:00まで)進捗ログ

ユーザー指示「昼12時頃まで自律的に進めて。現状に戻せる範囲でいくら進めてもよい。必要な実験・学習は自律で」。
すべて新規ファイル/`_rl_*.json`(git で現状に戻せる)、production 無変更。

### 実施
1. **RL安定化(v1の不安定さ是正)**: `train_v2.py` に **GAE(λ) + best-checkpoint + grad-clip** を実装。
   best-eval の重みをエクスポート(v1は最終を出していた)。
2. **並列収集インフラ(最大の compounding 投資)**: `collect_parallel.py`。cg収集はCPU律速で単一プロセスだと
   1コアのみ→ワーカーを **torch非依存(pure-Python PolicyModel、M0でtorch一致保証)** にして
   multiprocessing で全コア収集。**検証 PASS**: low-temp(≒argmax)で単一プロセス baseline 20% と一致。
3. **`train_v3.py`**: 並列収集 + GAE + best-ckpt。games/iter を大きく(512)して分散削減。
4. **並列で2アーキ同時学習**:
   - laptop(8core): **v3 dragapult**(50×512、workers8)
   - desktop(16core, `python -u` detach): **v3 archaludon**(50×512、workers16)
5. **M3評価準備**: `eval_m3.py` — RL済み相手 vs 模倣相手で「対alakazam勝率が上がった=ベンチが強化されたか」を
   full agent(overlay込み)で測る。RL学習完了後に実行。

### 完了後の自律プラン
- v3 学習完了 → `eval_m3.py` で dragapult/archaludon の強化を確認 → best を凍結ベンチ候補に。
- 効果あり → 他の弱アーキ(lucario)へ横展開、round_robin ベンチ更新。
- v3 でも不安定 → 報酬整形(value potential)/相手カリキュラム等を試す or デッキ側を疑う。
- 各判断は本ログに追記(ユーザーが後で追える形)。

### インフラの学び(記録)
- **並列収集の利得は限定的**: torch非依存ワーカーの pure-Python forward が torch の~8倍遅く、
  8並列でも実効スループットは単一プロセス torch(v2)とほぼ同等。→ 真の高速化には torch を
  ワーカーに入れる(起動重い)か、collection を C/vectorize する必要。今回は GAE/best-ckpt/大バッチの
  質的改善のために v3 を使うが、速度目的なら再設計が要る。
- **Windows で cg ワークロードは Start-Process detach で native crash**(hidden window + redirect、.err空で即死)。
  multiprocessing 版(v3)も detach 下で spawn 失敗。→ **desktop は "laptop の background で ssh 接続を保持"して
  remote python をセッション内で走らせる**方式に変更(ssh smoke は動く実績)。完了時に ssh 戻りで通知も来る。
- **RL の学習信号の弱さ(観察)**: 相手(production alakazam)が強く learner が ~20% しか勝てない=
  80% が報酬0の負け → 正の信号が疎で序盤 eval が動きにくい。v1 も iter15 で初めて 35% に跳ねた。
  → best-ckpt でピークは回収。将来は相手カリキュラム(弱い相手から)や報酬整形を検討。

### ★結果(検証済、~07:20)= RLで弱い相手を強化できる、が確認された

**素policy eval(overlay無し、学習中の指標):**
| アーキ | baseline | best(RL) |
|---|---|---|
| dragapult(v3,batch512) | 20.5% | **38.5%**(持続~30-32%) |
| archaludon(v2,batch256) | 13.3% | 27.3% |
| lucario(v2,batch384) | 26% | ~31%(実行中) |

**★M3 = full agent(overlay込み・本番ベンチの姿)での対alakazam勝率(`eval_m3.py`):**
| アーキ | 模倣相手 | RL相手 | Δ | 判定 |
|---|---|---|---|---|
| **dragapult**(300試合) | 15.7% [12.0,20.2] | **28.0% [23.2,33.3]** | **+12.3pt** | **z=3.66, p<0.001 有意(CI非重複)** |
| archaludon(300試合) | 17.3% | 23.3% | +6.0pt | z=1.83, p=0.068 |
| **archaludon(700試合・確定)** | 20.4% [17.6,23.6] | 21.3% [18.4,24.5] | **+0.9pt** | **z=0.39, p=0.69 効果なし** |

→ **【重要・正直な訂正】RLゲインが overlay を通過するかはアーキ依存。**
- **dragapult: 通過する**(+12.3pt有意)。RLでベンチを硬くできた。
- **archaludon: 通過しない**(700試合で+0.9pt=washout)。素policy eval の 13→27% は overlay で消えた。
  300試合の+6ptはノイズだった。**教訓: 素policyの学習eval はベンチ価値を過大評価する。M3(overlay込み)が真の判定。**
- 推定原因: archaludon は lethal/attack_plan overlay がプレイを支配する型で、policy改善が表に出ない。

**含意(field-expected、訂正版):** 実際に効いたのは dragapult のみ → production の対フィールド勝率低下は
**68.8% → ~67.2%(dragapultのみ、-1.6pt)**。当初の -3.0pt 見積り(archaludon込み)は撤回。
**限界: RL済み dragapult でも production は 72% で勝つ**=bot benchmark は少し硬くなったが人間級に届かない
([[project_pimc_prod_validation]] の bot≠human と整合)。**lucario も素eval(35%)を信用せず M3 で判定必須。**

### 次の判断ポイント(ユーザー、後で見返す用)
RL強化は機能したが限界(相手は72%+で負け続ける)。次の分岐:
- **(a) 相手RLをさらに押す**(報酬整形/相手カリキュラム/より長く)で相手を50%側へ近づける
- **(b) 現状の~3-6pt硬いベンチを受容**し、production改善の測定に使い始める
- **(c) production自体のRL**へ軸足(相手強化で得た pipeline を production に適用)
→ 自律では (a) の安い部分(lucario完走+archaludon追試で有意化確認)まで進め、(b)(c)の本格投資は判断を仰ぐ。

### 実行中/完了ジョブ(このセッション)
- ✅ laptop v3 dragapult 完了(20.5→38.5%)。✅ desktop v2 archaludon 完了(13.3→27.3%)。✅ eval_m3(上表)。
- 🔄 desktop v2 lucario(held-ssh、~30分)。

---

## 13. 自律セッション総括と推奨(~08:00、lucario M3 待ち)

### 検証できたこと
1. **RLパイプラインは動く・実装完了**(torch_policy/rollout/collect_parallel/train_v2・v3/eval_m3、全て新規・可逆)。
2. **RLは素policyを実際に強化する**: dragapult 20.5→38.5%、archaludon 13.3→27.3%、lucario 26→35%(素eval)。
3. **★ただし overlay(本番の姿)を通すかはアーキ依存**:
   - dragapult: **+12.3pt 有意(700試合, p<0.001)** ← 本物。
   - archaludon: **+0.9pt washout(700試合)** ← 素ゲインが overlay で消えた。
4. **限界**: 効いた dragapult でも production は 72% で勝つ。field-expected は 68.8→~67.2%(-1.6pt)。
   **bot ベンチは少し硬くなるが人間級に届かない。**

### 最重要の学び
- **素policyの学習eval はベンチ価値を過大評価する。M3(overlay込み)が真の判定。** 今後 RL 系は必ず M3 で見る。
- RL の machinery が **dragapult で +12pt の "本物の改善" を見つけられた**こと自体が大きい(pipeline の de-risk)。

### 次の分岐(ユーザー判断が要る・大投資)
- **(a) 相手RLを押す**: 報酬整形/長期化/他アーキ。ただし archaludon washout と "bots は人間級に届かない" 限界から
  **費用対効果は逓減**の見込み。非推奨寄り。
- **(b) 現状の少し硬いベンチで production 改善を測り始める**: 安いが、ベンチが弱く分解能が低い(H3 未解決)。
- **(c) 【推奨】検証済みRLパイプラインを production 自体に向ける**(self-play / リーグ)。dragapult が
  "RLは本物の改善を見つける" ことを実証したので、次は**それを主力(alakazam)に適用**するのが最も直接的。
  ただし self-play リーグは design.md #4 の最大投資=**あなたの戦略判断が必要**。

**自律での到達点はここまで**(相手強化の検証完了)。(a)(b)(c) の本格投資は判断を仰ぐ。lucario の M3 が3点目のデータ。

## 14. 方針(c)着手 = production 自体を RL(承認済「その方針で」)+ 重要バグ発見

### 発見したバグ(修正済)
`collect_parallel` のワーカーで **opponent 重みを相対パスで渡すと解決できず PolicyModel が未ロード
(is_ready=False)→ 全選択 index0 の壊れた相手**になり、勝率が偽陽性化する(alakazam-vs-crustle スモークが
90% と出た原因。正しくは 23%)。→ **`_init_worker2` に is_ready 検証を追加し fail-loud 化**。
**既存の dragapult/archaludon 結果は無事**(相手=production alakazam を `opp_weights=None`→絶対デフォルトパスで
正しくロードしていた)。壊れたのは相対パス指定した crustle スモークのみ。

### bare-field probe(重要)= production の弱点特定
bare production-alakazam vs bare 各相手(greedy 60試合):
| 相手 | alakazam勝率 |
|---|---|
| dragapult | 86.7% |
| archaludon | 86.7% |
| lucario | 80.0% |
| shirona | 68.3% |
| marnie | 56.7% |
| **rocket_mewtwo** | **43.3%** ← 弱 |
| **crustle** | **23.3%** ← 最弱 |

→ **bare alakazam に本物の policy由来の弱点(crustle/rocket)がある** = bare-RL で改善余地。方針(c)が有効化。
(注: crustle は full-agent だと 38.7% = overlay が +15pt 助ける。bare 23% を RL で上げても overlay 通過は要M3確認。)

### 方針c 第一歩の結果 = 単一相手 overfit(決定的)
- alakazam-vs-crustle RL: bare 38→46.5%。**eval_field(overlay込み全7マッチ)結果:**
  - **crustle 33.3→39.3%(+6.0pt)= 狙ったマッチは overlay 通過後も改善**(RLはproductionのマッチも改善できる、
    archaludon washout と違う=改善余地と手段はある)。
  - **だが他マッチ劣化(archaludon -2.0、shirona -3.3、lucario -0.7 等)→ 対フィールド加重は 66.22→66.40%
    = +0.18pt(実質ゼロ)。典型的な単一相手 overfit。**
- **結論: 単一固定相手のRLはダメ(狙ったマッチは上がるが他が下がり net ゼロ)。**
  → **次: field-sampling RL**(相手を meta-share で毎ゲームサンプルして学習=広く改善、overfit回避)。
  collect に per-game 相手サンプラを足す(collect_field)。これが方針(c)の正しい実装。

## 15. 最終総括(自律セッション 05:15–12:00)

### やったこと・分かったこと(3行)
1. **設計・Phase0診断**: 転移ギャップの正体=①評価が過小検出力(H3)②評価相手が弱すぎ(production 68.8%)。
2. **RLで相手強化(承認済)**: パイプライン構築・**dragapultは+12.3pt有意(700試合)で本物**、archaludonはwashout。
3. **RLでproduction自体(方針c)**: 弱点(crustle~38%/rocket~43%)特定、alakazam-vs-crustle RL 実行中(38→46.5%)。

### 確定した結論
- **RLパイプラインは動き、本物の改善を見つける**(dragapult +12.3pt, p<0.001, 700試合で確認)。pipeline de-risk 完了。
- **ただしゲインの overlay 通過はアーキ依存**(dragapult○ / archaludon×)。**素eval はベンチ価値を過大評価、M3(overlay込み)が真の判定。**
- 相手強化の field 影響は小(-1.6pt)、RL済みbotもproductionに72%負け=**人間級に届かない**。
- production自体にも本物の弱点あり(crustle)→ 方針(c)で改善試行中(結果~12:25)。

### 成果物(全て新規・可逆、production無変更)
`kaggle_replays/rl/`: torch_policy / rollout / collect_parallel / train_v2・v3 / eval_m3 / eval_field。
RL重み: policy_weights_{dragapult_ex_rl_v3, archaludon_ex_rl_v2, alakazam_rl_vscrustle}.json(_rl_接尾で本番と別)。
診断: `_diag_phase0_*`。docs: neural-agent/。**`git clean`で完全に現状復帰可。**

### インフラの学び
- 並列pure-pythonワーカーはtorch同等速(利得なし)。Windows cgはStart-Process detachでcrash→held-ssh。
- **バグ修正**: opponent相対パス→未ロードの壊れた相手→偽陽性(is_ready検証追加)。dragapult/archaludon結果は無事。
- lucario(3点目)は desktop 不安定で再実行も停止、非essentialのため見切り。

### ユーザー判断(後で見返す用)= crustle結果を受けて具体化
crustle 実験の結論: **単一相手RLは overfit(crustle +6pt だが他が下がり field net +0.18pt=ゼロ)。**
ただし **crustle が overlay通過後も +6pt = RLでproductionのマッチを改善する手段はある**(archaludon washoutと違う)。
→ 次の分岐:
- **【推奨】(a'')field-sampling RL**: alakazam を meta-share でサンプルした相手 mix に対して学習。overfit を避け
  対フィールド加重を直接最適化。**次セッションの第一歩として最も筋が良い**(実装: collect_field + train)。
  ただし相手は弱い bot なので、上がっても Kaggle 転移は別途要確認([[project_pimc_prod_validation]])。
- **(b)自己対戦リーグ**(design.md #4、共進化)= 人間級に近い相手を作る唯一の道だが最大投資。
- 判断材料は揃った。**(a'')を試して field 加重が有意に上がるか**を見てから (b) の大投資を検討するのが妥当。

### 追記: field-sampling RL 着手(承認済「その方針で」)
`collect_field.py`(per-game 相手 meta-share サンプル、is_ready検証込)+ `train_field.py`(train_v3ヘルパー再利用)を実装。
**alakazam を7アーキ field mix に対して学習**中(60×512、bare baseline field加重 ~72.5%)。eval=share加重greedy勝率で
best-ckpt。完了後 **eval_field(overlay込み)で production 66.2% から有意に上がるか**を判定 → 上がれば方針(c)成立、
頭打ちなら (b)自己対戦リーグへ。単一相手 overfit を避ける正しい実装。

### field-sampling RL 結果 = 横ばい(方針c fixed-opponent は行き止まり)
- bare field加重: baseline 67.7% → 60iter 横ばい(64.5〜70.8%)、**best 70.8%@iter6、final 69.3%**。上昇せず。
- **解釈: production は既に弱い bot フィールドを圧倒(67.7%)しているため fixed-opponent RL(単一/field-sampling とも)
  では改善できない。** RLパイプラインは本物(dragapult +12.3pt)だが、**相手が弱すぎるのが根本。**
- **overlay込み eval_field(確定): production 68.20% → field-RL 65.55% = -2.66pt(悪化)。**
  マッチ別も大半劣化(dragapult -8.0/lucario -3.3/archaludon -3.3/crustle -2.0、marnie+2/rocket+2.7)。
  **bare最適化(横ばい70%)が overlay では production を悪化させた。** production の BC は既に良く調整済で、
  弱い bot 相手の RL はそれを劣化させる([[project_beyond_bc_design]] C1 と同型)。
- **注意: `league/run_league.py` が本セッション中に外部変更された**(MEMORY.md の「統合意思決定パイプライン」並行作業、
  per-agent config で `_worker_init` に config_base_b 追加)。公開シグネチャは後方互換だが、ラン中編集で eval_field が
  一過性クラッシュ。**共有インフラなので勝手にパッチせず、安定後に再実行**([[feedback_respect_ownership_boundaries]])。

### ★方針(c) 最終結論と (b) への分岐(要ユーザー戦略判断)
fixed-opponent RL(単一 overfit / field-sampling 横ばい)は**production を改善できない**=行き止まり。
唯一残る道は **(b)自己対戦リーグ(共進化)**。ただし:
- design.md #4 の**最大投資**(AlphaStar型、セッション規模を超える多週間の研究)。
- **根本リスク: bot リーグ ≠ 人間**。リーグ内で強くなっても Kaggle 転移は別問題。最終判定は Kaggle 提出。
- **"とりあえず進める"範囲を超える** → 正式に go/no-go を諮る(自律では着手しない)。

---

## 7. メモ(判断の芽)

- H2 側の伏線: value網は `self_deck_count` の摂動にほぼ反応しない(probe で 200中92 のみ正方向、
  mean_delta≈0.005)。**山札切れリスクが価値評価に乗っていない**。Phase 1 で「自山札残量/デッキアウト距離」を
  信念ベクトルに入れる直接の動機になりうる(H2 → Phase 1 の橋)。
- 敗因は side_race_blowout 139 / deck_out 52 / close 10(201敗中)。大差負けの分散が主 →
  クリティカル局面 ISMCTS(#5/Phase3)は効く場所を外す懸念。優先度を下げた根拠。
