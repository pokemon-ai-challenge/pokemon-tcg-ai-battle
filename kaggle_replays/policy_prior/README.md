# policy_prior

上位エージェントのリプレイから **方策**（Policy Prior / Behavior Cloning）を学習するためのオフライン資産。

関連文書:
- 要件定義 `test_plan/ptcg_top_replay_utilization_requirements_v2.md` §4（Semantic Action）/ §7（リーク防止）/ §10（分割）
- 実測レポート `test_plan/ptcg_replay_schema_report.md`

---

## なぜこれが必要か

ルールベース方策のランダム相手勝率は **56.1%**（1000試合、95%CI 53.0-59.1%）にとどまる。土台の方策そのものが弱いため、探索や Belief をいくら改善しても実戦勝率に反映されにくい。上位エージェントの模倣が最もレバレッジの大きい打ち手である。

一致率の現在地（`MAIN` context / 単一選択 / 強制手を除く 24,150 decision で実測）:

| | 一致率 |
|---|---:|
| 自明ベースライン（type 事前分布順に選ぶ） | 32.2% |
| バックオフ付き lookup 方策（学習なし） | 41.8% |
| **上位エージェント間の一致率**（≒ 現実的な天井） | 約 80% |

**ACCEPT-BC-001**: 学習方策の test 一致率が **41.8% を上回ること**。下回るなら状態を条件付ける以上の情報を学べていない。

---

## ファイル

| ファイル | 役割 |
|---|---|
| `build_dataset.py` | リプレイ → BC データセット(JSONL) |
| `build_card_attributes.py` | `data/EN_Card_Data.csv` → カードID別の静的属性表 |
| `train.py` | 線形 pointwise ranking の学習。`sample_submission/ptcg_ai/learning/policy_weights.json` を出力 |
| `evaluate.py` | 一致率の評価。ベースラインと lookup 非ヒット限定の一致率を併記する |
| `output/` | 生成物。本体は `.gitignore` 済み（再生成可能・107MB）。`*.summary.json` のみ追跡する |

### 実装は提出側にある

**特徴抽出と Option 解決の実装は `sample_submission/ptcg_ai/learning/` にしか存在しない。** 本ディレクトリはそれを import する。

| 提出側のモジュール | 役割 |
|---|---|
| `ptcg_ai/learning/semantic_action.py` | Option の位置参照を `observation.current` で解決する（v2 §4.2 の唯一の実装） |
| `ptcg_ai/learning/observable_state.py` | observation → 観測可能な状態 dict |
| `ptcg_ai/learning/policy_features.py` | 特徴抽出 |
| `ptcg_ai/learning/policy_model.py` | 純Python推論 |

学習側にコピーを置いてはならない。学習時と推論時で特徴がズレるのは最も見つけにくい欠陥で、精度が出なかったときにモデルの問題か特徴の問題かを切り分けられなくなる。`kaggle_replays/tests/test_policy_parity.py` が、学習時のスコアと `PolicyModel` のスコアの一致を検証している。

## 使い方

```bash
python policy_prior/build_dataset.py
```

既定の出力は `policy_prior/output/main_decisions.jsonl`。

```bash
# MAIN 以外の context も含める / 複数選択も含める / 強制手も含める
python policy_prior/build_dataset.py --all-contexts --include-multi --include-forced
```

---

## Option の解決（`semantic_action.py`）

`cg/api.py` の `Option` は **カードが何であるかを持たない**。持つのは位置参照だけである。実測（option 延べ 302,791件）で `cardId` / `serial` を持つ Option は **471件（0.16%、`OptionType.SKILL` 専用）** しかない。

カード実体は `observation.current` 側にある。

```jsonc
// hand / deck / discard / prize の要素
{"id": 7, "playerIndex": 0, "serial": 3}
// active / bench の要素
{"id": 646, "serial": 20, "hp": 70, "maxHp": 70, "energies": [], "tools": [], ...}
```

したがって「位置参照 → `current` を引く」手順が必須になる。`resolve_option()` がそれを行う。

**踏みやすい罠**: `current.stadium` は単一オブジェクトではなく **`list`**（不在時 `[]`）。単一オブジェクトとして扱うと `ABILITY` の解決が 46% 失敗する。

**検証状況**: `MAIN` context の全 option **161,866件で解決失敗 0件**。

### `index` を特徴量にしないこと（v2 §4.3 FR-ACT-006）

`index` はエリア内の位置であり、ドロー・シャッフル・トラッシュで容易に動く。同じ `index` が別ターンには別のカードを指すため、そのまま特徴量にするとモデルが偽の相関を学ぶ。

- 特徴量には解決結果の `card_id` を使う
- `serial` は個体追跡には使えるが、試合ごとに振り直される任意の番号なので**特徴量にはしない**
- 出力スキーマは位置参照グループ（再マッピング用）と解決結果グループ（特徴量用）を意図的に分離している

なお「常に `index=0` を選ぶ」だけで一致率 27.6%（ランダム 16.7%）に達する。位置に信号はあるが、これは「手札の先頭を出しがち」という表層的な癖であり盤面が変われば崩れる。**ベースラインとしてのみ扱い、特徴量には入れない。**

---

## 既定のスコープ（MVP）

実測にもとづき、既定では以下に絞る。

| 絞り込み | 理由 |
|---|---|
| `SelectContext.MAIN` のみ | 全 decision の 52.4%。最も判断が重い。context は49種あり尾が長いので、まず1つに集中する |
| 単一選択のみ | 複数選択は 5.1%。pointwise ranking で表現できない |
| 強制手を除外 | 選択肢1個は 8.1%。方策の情報を持たず精度を水増しする |

結果として **24,150 decision**（300エピソード / 131エージェント）。

## 分割（v2 §10）

**エピソード単位**で train/val/test = 80/10/10。decision 単位で分けると同一エピソード内の強い相関でリークする。

割り当ては乱数シードではなく `episode_id` の MD5 で決める。リプレイを追加しても既存の割り当てが動かないようにするため（シード + shuffle だと母集合が変わるたびに全件の割り当てが変わり、過去の評価値と比較できなくなる）。

## リーク防止（v2 §7 FR-LEAK-001）

格納する状態は **その手番のプレイヤーに提示された `observation`** に限る。リプレイには両プレイヤーの observation が入っているが、相手側の観測は参照しない。

相手の手札は `null`（`handCount` のみ公開）、サイドは `[null]×6` として渡ってくるため、observation をそのまま使う限り非公開情報は混入しない。`state.opponent` に入るのは公開情報（場のポケモン、トラッシュ、各種カウント、特殊状態）だけである。

## 出力スキーマ（1行 = 1 decision）

```jsonc
{
  "episode_id": "88086972", "player_index": 0, "step_index": 12,
  "split": "train",

  // 教師の出所。サンプル重み付けに使う
  "agent": "THIRD PTCG Club", "rank_at_fetch": 3, "score_at_fetch": 1190.1,

  // decision
  "select_context": 0, "select_type": 0,
  "min_count": 1, "max_count": 1, "n_options": 7, "forced": false,
  "actions": [ /* 解決済み Semantic Action。v2 §4.4 のスキーマ */ ],
  "chosen": [4],
  "chosen_label": [7, 646, null],   // (option_type, card_id, target_card_id)

  // 観測（Observable Features のみ）
  "state": { "turn": 3, "own": {...}, "opponent": {...}, ... },
  "remaining_overage_time": 597.3
}
```

`chosen_label` は**選択肢集合に依存しない正準形**であり、別 decision・別エージェント間で「同じ行動か」を比較できる。上位同士の一致率はこのラベルで測っている。`serial` を含めないのは、試合ごとに振り直されるため含めると常に不一致になるからである。

## 不変条件のチェック（v2 §4.1 FR-ACT-005）

リプレイ300件で成立を確認した性質を `check_invariants()` が検査する。破れても例外にせず件数を集計し、サマリJSONに記録する（コンペ期間中のエンジン更新で破れ得るため）。

- `MAIN` の decision には必ず `END` が1つだけ含まれる
- `ATTACH` / `EVOLVE` の `area` は常に `HAND`
- `ABILITY` の `area` は `ACTIVE` / `BENCH` / `STADIUM` のみ

サマリJSONの `invariant_violations` が非ゼロになったら、データかエンジンの変化を疑うこと。
