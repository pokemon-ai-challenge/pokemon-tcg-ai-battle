# BC（模倣ポリシー）再学習 — 2026-08-03

> **状態: 有効（最新）** — 2026-08-03。現行の BC 重み3件（marnie / alakazam / crustle）は
> この文書の手順で作られたもの。容量アブレーションの再検証結果も §5.5 に含む。

リプレイ 7,444件時点で、模倣ポリシーを 3アーキタイプ分やり直した記録。
旧BC との比較、変更点、確認できていない点をまとめる。

対象: `marnie_grimmsnarl_ex` / `alakazam` / `crustle`
未実施: `archaludon_ex`（理由は §5）

---

## 1. 結果（test split、旧BC と同一分割）

各アーキタイプ自身の test split で、旧BC の重みを退避版から読み直して同じ条件で採点した。
指標は Top-1 一致率（上位プレイヤーが実際に選んだ選択肢を argmax で当てる率）。

| アーキタイプ | 旧BC | 新BC | 差 | test n |
|---|---|---|---|---|
| marnie_grimmsnarl_ex | 0.6305 ±0.0032 | **0.7388 ±0.0029** | **+10.8pt** | 85,218 |
| alakazam | 0.5778 ±0.0090 | **0.6643 ±0.0086** | **+8.7pt** | 11,500 |
| crustle | 0.5024 ±0.0118 | **0.5830 ±0.0117** | **+8.1pt** | 6,853 |

3件とも 95%信頼区間が重ならない。

参考として、同じ test split での「選択肢特徴のみ・隠れ層なし」ベースライン:

| アーキタイプ | ベースライン | 新BC |
|---|---|---|
| marnie_grimmsnarl_ex | 0.6036 | 0.7388 |
| alakazam | 0.5464 | 0.6643 |
| crustle | 0.4560 | 0.5830 |

### select_type 別（marnie、test split）

| select_type | n | 新BC Top-1 |
|---|---|---|
| 0（MAIN） | 38,915 | 0.6779 |
| 1 | 36,309 | 0.7394 |
| 2 | 64 | 1.0000 |
| 4 | 1,318 | 0.8907 |
| 7 | 818 | 0.9694 |
| 8 | 5,278 | 0.9898 |
| 9 | 2,516 | 0.9829 |

難所は MAIN（0.678）で、簡単な確認系（8/9）はほぼ飽和している。

---

## 2. この比較が公平である理由

- **分割は episode_id の md5 で決まる固定式**（`build_features.py: split_for_episode`、
  `h = md5(episode_id) % 100`、`<80` train / `<90` val / それ以外 test）。
  エピソード単位なので同一試合が train と test に跨らない。
  さらにこの式はコーパスが増えても各エピソードの所属を変えないため、
  **旧BC が今回の test エピソードを学習に使うことは構造上ありえない**。
- 旧BC の重みは `bc_backup_2026-08-02/` から読み直し、新BC とまったく同じ
  test 行に対して採点した（採点器: `score_weights.py`、train.py の
  `pure_python_forward` を numpy でベクトル化したもの）。
- 採点器の妥当性は、新BC で train.py が報告した値（marnie 0.7388）を
  厳密に再現することで確認した。
- 各重みの標準化パラメータは JSON 内に自己完結しているので、
  新旧で別々の正規化が正しく適用される。
- 書き出した重みは train.py 内の自己検証（JSON 再読込による純Python 再計算と
  PyTorch 出力の一致）を 3件とも PASS。最大誤差 1.4e-14 / 3.6e-15 / 4.4e-15。

> 補足: 当初「旧BC が test 局面を学習済みの可能性があるので差は控えめ」と述べたが、
> これは誤り。上記の md5 固定分割によりその混入は起きない。差はそのまま読んでよい。

---

## 3. 旧BC から変えたこと

### 3.1 レシピは変えていない

`run_archetype_pipeline.py` の既定（＝確立済みの本番レシピ）をそのまま使った。

| 項目 | 値 |
|---|---|
| 特徴量 | `build_features.py --weight-scheme concentrated`（＝本番＝構成C） |
| モデル | `Linear(in, 32) -> ReLU -> Linear(32, 1)`、選択肢間で重み共有 |
| card_embedding | `nn.Embedding(card_id_max+1, 8)`、index 0 = 識別なし |
| 損失 | 決定点内リストワイズ softmax 交差エントロピー（sample weight 付き） |
| 最適化 | lr 1e-3 / batch 64（決定点数）/ max_epochs 100 / patience 10 |
| seed | 42 |
| 早期終了 | val split の重み付き平均 NLL |

**ハイパーパラメータ・特徴量設計・モデル構造はいずれも変更していない。**
変えたのは入力データだけである。

### 3.2 変えたのは教師データ

リプレイ 7,444件・`deck_labels.jsonl` 14,888行の現行版から抽出し直した。

| アーキタイプ | 対象 episode-player | 抽出された意思決定点 |
|---|---|---|
| marnie_grimmsnarl_ex | 8,912 | 818,350 |
| alakazam | 1,530 | 110,759 |
| crustle | 1,327 | 69,782 |

marnie は 818,350点と圧倒的に多く、実メタでの使用率（約60%）を反映している。

### 3.3 確認できていないこと（重要）

**旧BC がどのコーパスで学習されたかの記録は残っていない。** 旧重みは
`policy_weights.json` ほかが 2026-07-30 04:52 付で、当時の抽出物や実行ログは
上書き・削除されている。したがって「データが増えたから良くなった」と
断定はできない。実際、alakazam については逆向きの観測がある:

- 旧BC と同時刻の `evaluate_results.json` は test_n = **22,626**。
- 今回の alakazam の test split は **11,500**（全 110,759点の 10.4%）。
- `outcome_aware/offline_*.json` に旧データセットの総数が記録されていた:
  `n_outcome_known` 187,582 + `n_outcome_unknown` 108 = **187,690点**
  （n_test も 22,626 で一致）。

つまり旧 alakazam は 187,690点、今回は 110,759点で、**データは約4割減っている。
それでいて精度は +8.7pt 上がっている**。
最も素直な説明は `deck_labels.jsonl` の再生成でアーキタイプ判定が変わり、
旧データセットには alakazam でない試合が相当数混入していた、というものだが、
当時のラベルが残っていないため**これは仮説であって検証されていない**。

同様に marnie / crustle についても旧データセットの規模は不明で、
今回の改善のうちどれだけが「量」で、どれだけが「ラベル品質」によるものかは
分離できていない。

再発防止として、今後の学習では抽出件数とラベルのスナップショットを
実行ログとともに残すことを推奨する（現状 `archetype_runs/*.log` に
抽出件数は残るようになった）。

---

## 4. 成果物と退避

新規・更新:

```
sample_submission/ptcg_ai/learning/policy_weights_marnie_grimmsnarl_ex.json
sample_submission/ptcg_ai/learning/policy_weights_alakazam.json       (新規作成)
sample_submission/ptcg_ai/learning/policy_weights_crustle.json
kaggle_replays/policy_net/features_{marnie_grimmsnarl_ex,alakazam,crustle}.npz
kaggle_replays/training_data/policy_positions_{...}.jsonl.gz
kaggle_replays/policy_net/archetype_runs/*_{extract,features,train}.log
```

**変更していないもの:**

- `policy_weights.json`（本番既定、2026-07-30 のまま）
- `policy_weights_*_pool_k60.json`（対戦系RLの起点、2026-07-31 のまま）
- `league_runs/league1/` の各世代重み

旧BC の退避先: `sample_submission/ptcg_ai/learning/bc_backup_2026-08-02/`
（`policy_weights.json` = 旧 alakazam BC、`policy_weights_{crustle,marnie_grimmsnarl_ex,archaludon_ex}.json`）

---

## 5. archaludon_ex を実施しなかった理由

`deck_labels.jsonl` 上の教師量が他と桁違いに少ない。

| アーキタイプ | 教師エピソード |
|---|---|
| marnie_grimmsnarl_ex | 6,228 |
| alakazam | 1,450 |
| crustle | 1,272 |
| rocket_mewtwo_ex | 863 |
| **archaludon_ex** | **55** |

55試合では模倣学習が成立しない。archaludon_ex は凍結プール評価でも一貫して
最下位（0.422）だったが、これはRL側の問題ではなく**模倣する対象が存在しない**
ことに起因する。実メタでもほぼ不在のため優先度は低い。
改善するならリプレイ収集が先。

---

## 5.5 容量アブレーション再検証（2026-08-03）

過去に未決着だった「hidden を増やす」案を、新しい marnie データセット
（818,350点、旧 alakazam の 4.4倍）で再検証した。特徴量は使い回し、
`--hidden-size` 以外は一切変えていない。同一 test split（85,218点）。

| hidden | test Top-1 | test NLL | train-val gap (top1) | パラメータ数 |
|---|---|---|---|---|
| **32（現行）** | **0.7388** | 0.7255 | +0.0108 | 17,857 |
| 64 | 0.7362 | 0.7225 | — | 25,569 |
| 128 | 0.7367 | 0.7239 | +0.0149 | 40,993 |

**結論: 容量を増やしても改善しない。** 3点とも 95%CI（±0.003）の内側で、
むしろ 32 が最良。過学習指標は 128 で +0.0108 → +0.0149 とわずかに悪化する。

旧環境（alakazam 187,690点）では hidden=128 が hidden=32 に対し +0.6pt
（0.5864 vs 0.5806）だったが、**この優位はデータ規模を 4.4倍にすると消える**。
事前には「データが増えたぶん容量不足がより顕著になる」と予想したが、逆だった。

したがって現行モデルは容量では律速していない。hidden=32 を維持する
（レイテンシも最小: 3.85ms/判断 対 hidden=64 の 9.06ms）。
精度をさらに上げるなら投資先は容量ではなく、特徴量または学習信号の側になる。

未検証で残っている改善案:

- **outcome-aware weighting**（`--outcome-weighting discount/filter/advantage`、実装済み）。
  旧環境の6構成はいずれも素のBCより Top-1 が低かった（0.578〜0.583 対 0.5923）が、
  この手法は敗北局面を意図的に軽く扱うので **Top-1 が下がるのは想定内**であり、
  勝率で評価しないと是非を判定できない。対戦評価とセットで行う必要がある。
- **特徴量の拡充**。select_type=0（MAIN）が 0.678 と唯一の弱点で、
  容量が効かない以上、表現力ではなく入力情報の不足が疑われる。

---

## 6. 限界

**Top-1 一致率は勝率ではない。** 測っているのは「上位プレイヤーの選択をどれだけ
再現するか」であり、対戦での強さは別途測定が必要。上位プレイヤーの選択が
最適とは限らない点も含めて、この指標は代理指標にとどまる。

現行の対戦系の重み（`*_pool_k60.json`、`league_runs/league1/`）はすべて
**旧BC を起点に育てたもの**なので、今回の改善はまだ対戦性能に反映されていない。
反映には新BC を起点としたプール学習のやり直しが必要。

---

## 7. 再現手順

```bash
python kaggle_replays/policy_net/run_archetype_pipeline.py --archetype marnie_grimmsnarl_ex
python kaggle_replays/policy_net/run_archetype_pipeline.py --archetype alakazam
python kaggle_replays/policy_net/run_archetype_pipeline.py --archetype crustle
```

所要時間の目安（CPU、4コア相当）: marnie は抽出約20分・特徴量約10分・学習約40分。
alakazam / crustle は各 15〜25分程度。

新旧比較の採点:

```bash
python score_weights.py <features_<ARCH>.npz> <新weights.json> <旧weights.json>
```
