# Stage1 (Tier1a/1b/1c) 実装計画書 — 可逆性最優先

作成: 2026-07-22 / 親方針書: [`design-and-implementation-plan.md`](./design-and-implementation-plan.md)

**スコープ**: Tier1a(盤面ポケモン identity)+ Tier1b(スタジアム identity)+ Tier1c(ポケモン毎スカラー:
道具有無・基本/特殊エネ枚数・特性有無)を **1 つの学習・検証サイクル**として実装する。
特殊エネ identity(Tier2b)・技 identity(Tier2a)は **本計画のスコープ外**(第二弾以降・別計画)。

**最重要要件(ユーザー指定)**: いつでも現状の最善パラメータ/学習物へ戻せること。
本計画は §1 の可逆性アーキテクチャを最初に定義し、全ステップがそれを壊さないことを DoD にする。

---

## 1. 可逆性アーキテクチャ(設計の中心)

「戻せる」を 4 層で保証する。**どの層単独でも現行に戻せる**冗長設計にする。
(バージョン管理 = git は本計画のスコープ外。作業ツリー上で直接実装する。)

### 1.1 層1 — 現行の学習物を一切上書きしない(不変スナップショット)

- 現行の最善重み [`ptcg_ai/learning/policy_weights.json`](../../../ptcg_ai/learning/policy_weights.json)
  (構成C, ref 54883922, publicScore 749.8)は **本計画で一切変更しない**。
- 新しい学習物は **別ファイル名**で出力する:`policy_weights_tier1abc.json`。
- さらに保険として、現行重みを **不変スナップショット** `policy_weights_baseline_v4.json` として
  複製し(中身は現行と同一)、A/B 対戦の「王者」側に固定参照させる。将来誰かが既定重みを
  再学習しても baseline が動かないようにする。
- 学習データ側も同様:現行 `features.npz` を上書きせず `features_tier1abc.npz` へ出す。

### 1.2 層2 — 既定 config を変更しない(config で切替)

- 既定は現行のまま `ml_lethal_attackplan_v0only`(重み未指定=既定 `policy_weights.json`)。
- 新実験は **新 config** `ml_lethal_attackplan_tier1abc.json` を追加し、そこにだけ
  `"policy_weights_path": ".../policy_weights_tier1abc.json"` を書く。
- 既存の [`ml_policy_agent._get_model`](../../../ptcg_ai/ml_policy/ml_policy_agent.py#L128) は
  `config["policy_weights_path"]` があればそのパスを、無ければ既定を読む(パス毎キャッシュ済み)。
  **→ 既定 config を選ぶ限り、コードが新機能を積んでいても挙動は現行と完全一致**。
- 採用が決まるまで既定 config は変えない。採用時のみ既定を新 config に差し替える(その差し替え自体も
  1 行で戻せる)。

### 1.3 層3 — 重み JSON が「必要な特徴」を自己記述(encoder はそれに従う)

- encoder を **tier 対応(パラメータ化)** にする。どの Tier を有効化するかを表す
  `tiers: frozenset[str]` を受け取り、**`tiers` が空なら現行とビット同一のベクトルを返す**。
- 重み JSON の `meta.feature_tiers`(新キー)に有効 Tier を記録。
  [`PolicyModel`](../../../ptcg_ai/learning/policy_model.py) はロードした `feature_tiers` を
  encoder に渡す。
- **旧重み(`feature_tiers` 無し)→ 空集合 → 現行の 166/65 ベクトル + card 埋め込みのみ**。
  つまり旧重みは新コードでもそのまま動く(後方互換)。

### 1.4 層4 — append-only レイアウト(旧ベクトルは新ベクトルの厳密な prefix)

- Tier1c の追加スカラー(4×12=48)は **既存 166 次元の後ろに追記**する(`_pokemon_features` の
  11 特徴には**手を入れない**。中間に挿入するとレイアウトが崩れ prefix 不変が壊れるため)。
- Tier1a/1b の identity は元々 forward で card 埋め込みの後ろに連結する **追記系**。
- 結果:`state_vec[0:166]` は Tier 有無に関わらず**同一値**。旧標準化配列・旧重み行列が
  そのまま先頭次元に対応する。**MLP は特徴順に依存しない**ので、append-only による順序変更は
  モデル品質に影響しない(意味的な綺麗さより可逆性を優先する明示的判断。親方針書 §3.3 の
  「interleave」案からの逸脱)。

> **戻し方の要約**: ①既定 config を選ぶ(コード変更不要でランタイム現行化) /
> ②`policy_weights_tier1abc.json` を消す(学習物のみ現行化)。
> どちらか 1 つでも現行に戻る。加えて encoder/policy_model の変更は後方互換
> (旧重み=Tier 空で現行動作)なので、コードを残したままでも現行挙動を害さない。

---

## 2. Tier 対応 encoder の設計(可逆性の実装核心)

[`ptcg_ai/learning/encoder.py`](../../../ptcg_ai/learning/encoder.py):

### 2.1 定数と関数

- **モジュール定数は現行値のまま温存**:`FEATURE_NAMES`(166)/`BASE_FEATURE_COUNT`(166)/
  `OPTION_FEATURE_COUNT`(65)は「Tier 空集合=ベースライン」の値。既存 import 元
  (build_features/policy_model)が壊れない。
- Tier 対応の長さは関数で提供:
  ```python
  TIER_IDS = ("tier1a", "tier1b", "tier1c")  # Stage1 で扱う識別子

  def feature_names(tiers: frozenset[str]) -> list[str]:   # 末尾に tier1c ブロックを追記
  def base_feature_count(tiers: frozenset[str]) -> int:    # 166 + (48 if tier1c else 0)
  def option_feature_count(tiers: frozenset[str]) -> int:  # 65 + (4 if tier1c else 0)
  ```

### 2.2 状態エンコード(Tier1c 追記 + Tier1a/1b の id ストリーム)

```python
def encode_state_from_state(
    state: State | None,
    extra_features: list[float] | None = None,
    tiers: frozenset[str] = frozenset(),      # 追加。空=現行と完全一致
) -> list[float]:
    # ... 現行の 166 次元をそのまま構築 ...
    if "tier1c" in tiers:
        feats += _tier1c_state_block(state)   # 4×12=48 を末尾追記(空スロットはゼロ)
    if extra_features:
        feats += [float(x) for x in extra_features]
    return feats

def _tier1c_state_block(state: State) -> list[float]:
    """自分/相手 × (active + bench0..4) の 12 スロットを _side_pokemon_block と同順で、
    各スロット [tool_present, basic_energy_count, special_energy_count, ability_present]。"""

# Tier1a: 盤面 12 スロットの Pokemon.id(_side_pokemon_block と同順、空/伏せ=0)
def encode_state_pokemon_ids(state: State | None) -> list[int]:   # len 12

# Tier1b: stadium[0].id or 0
def encode_state_stadium_id(state: State | None) -> int:
```

`_pokemon_features`(11 特徴)には**触れない**。Tier1c は別ヘルパ `_tier1c_pokemon_scalars(pokemon, card)`
で 4 値を返し、state 側(12 スロット)と option 側(対象ポケモン 1 体)から呼ぶ。

### 2.3 選択肢エンコード(Tier1c の対象ポケモン分だけ追記)

```python
def encode_options_from_state(
    state, select, tiers: frozenset[str] = frozenset(),
) -> list[list[float]]:
    # 現行 65 次元を構築後、"tier1c" in tiers なら各選択肢の末尾へ
    # _tier1c_pokemon_scalars(target_pokemon) の 4 値を追記(対象なしはゼロ4個)。
```

### 2.4 Tier1c スカラーの定義(公開情報のみ)

| スカラー | 定義 | 入力元 |
|---|---|---|
| `tool_present` | 0/1 | `len(pokemon.tools) > 0` |
| `basic_energy_count` | 枚数 | `pokemon.energyCards` を card_cache で引き `cardType==BASIC_ENERGY` を計数 |
| `special_energy_count` | 枚数 | 同上 `cardType==SPECIAL_ENERGY` |
| `ability_present` | 0/1 | `CardData.skills` 非空(カード静的属性) |

> 未決(§8): 枚数スカラーの事前スケール。既定は生値+標準化(std が吸収)。要検討だが Stage1 は生値で開始。

---

## 3. 変更ファイルとステップ順(各ステップ: 変更 / DoD / 戻し)

作業ツリー上で直接実装する(git はスコープ外)。ステップは論理的な区切り。

### Step 1 — encoder(Tier 対応 + Tier1a/1b/1c 出力)

- 変更: §2 の関数追加。既存関数はデフォルト引数 `tiers=frozenset()` を足すだけ(呼び出し元は無改修で現行動作)。
- DoD:
  - `encode_state_from_state(s)`(tiers 省略)が**現行とビット同一**(既存の決定性/parity テスト緑)。
  - `base_feature_count(frozenset({"tier1c"})) == 214` / `option_feature_count(...) == 69`。
  - `encode_state_pokemon_ids` の順序が `_side_pokemon_block` と一致(固定盤面 fixture で突合)。
  - 空/伏せ/未知 id で例外なくゼロ/0。
- 戻し: 追加関数は未使用なら影響なし(既存呼び出しはデフォルト引数 tiers=空で現行動作)。

### Step 2 — build_features(新 npz へ、行順・fail-fast 維持)

`kaggle_replays/policy_net/build_features.py`:
- 変更: `--tiers tier1a,tier1b,tier1c`(既定=空=現行と完全一致)。tiers 有効時のみ
  `state_pokemon_ids`(N,12)/`stadium_id`(N,)/ 拡張 state・option 特徴を `features_tier1abc.npz` へ。
  出力パスは `--out` で明示(既定 `features.npz` は上書きしない)。
- DoD: `--limit 2000` で走破。tiers 空なら現行 npz とバイト一致(回帰)。tiers 有効で新配列が
  期待 shape/dtype。行順不変・encode 失敗 fail-fast を維持。
- 戻し: `--tiers` を付けない/新 npz を消すだけ。

### Step 3 — train(Tier フラグ、埋め込み共有、新重みへ)

`kaggle_replays/policy_net/train.py`:
- 変更: `--tiers` で読み込む特徴を切替。**既存 `nn.Embedding`(card)を board 12 + stadium 1 の
  lookup にも流用**(新テーブルなし)。forward の連結順を §4 に固定。エクスポートに
  `meta.feature_tiers` を追加。出力は `policy_weights_tier1abc.json`(既定重みは上書きしない)。
- DoD: 全 Tier OFF で現行 top1 一致率(58.06% 近傍)を再現(対照の再現性)。Tier ON で収束し
  新スキーマ(§4)で出力。
- 戻し: `--tiers` なしで学習=現行重み再現。新重みファイルを消す。

### Step 4 — policy_model(自己記述ロード + tier 対応 forward)

[`ptcg_ai/learning/policy_model.py`](../../../ptcg_ai/learning/policy_model.py):
- 変更: `_load` で `meta.feature_tiers`(無ければ `frozenset()`)を読み `self._tiers` に保持。
  `score_options_from_state` は encoder に `tiers=self._tiers` を渡す。`_forward` は
  §4 の順で board(12)/stadium(1)埋め込みを card テーブルから引いて追記。
- DoD:
  - **旧重み回帰**: `feature_tiers` 無しの現行 `policy_weights.json` を読み、現行と**同一スコア**
    (既存の推論テストが緑)。
  - **学習/実行 parity**: 同一 obs で train 側 forward と model 側 forward のスコア一致。
- 戻し: 変更を残しても旧重みなら現行動作(自己記述で Tier 空に落ちる)。

### Step 5 — config 追加(既定は不変)

`configs/ml_lethal_attackplan_tier1abc.json`:
- 内容: 既定 config を複製し `"policy_weights_path": "<repo>/ptcg_ai/learning/policy_weights_tier1abc.json"` を追加。
- 既定 config `ml_lethal_attackplan_v0only.json` は**変更しない**。
- DoD: `PTCG_AI_ML_CONFIG=ml_lethal_attackplan_tier1abc` で新重みが読まれ 1 ゲーム走る。
  既定 config では従来どおり `policy_weights.json`。
- 戻し: 環境変数/既定を戻すだけ。

### Step 6 — 検証(オフライン + ミラー対戦ゲート)

- オフライン: `features_tier1abc.npz` の test split で top1 一致率を測定(現行 58.06% が対照)。
- **ミラー対戦(採用の最終ゲート)**: baseline(`policy_weights_baseline_v4.json`)vs
  candidate(`policy_weights_tier1abc.json`)を head-to-head。既存 league / `policy_weights_path`
  機構で両モデルを同時ロード。95% CI で有意勝ち越しのみ採用候補。
- 勝てない/差なし時は §7 に従い **Tier1a 単独**へ切り戻して再測(1b/1c のノイズ切り分け)。
- DoD: 判定結果を `results/` に記録。採用/不採用を明記。

---

## 4. 契約(`features_tier1abc.npz` / `policy_weights_tier1abc.json`)と forward 連結順

```jsonc
// policy_weights_tier1abc.json(旧 policy_weights.json は無改修で共存)
{
  "meta": { "feature_tiers": ["tier1a","tier1b","tier1c"],
            "state_feature_count": 214, "option_feature_count": 69, ... },
  "standardization": { "state_mean":[214], "state_std":[214], "option_mean":[69], "option_std":[69] },
  "card_embedding": { "dim":8, "card_id_max":..., "table":[...] },   // Tier1a/1b が共有。新テーブルなし
  "layers": [ ... ]                                                  // 入力次元=下記連結長
}
```

forward 連結(Tier1a/1b/1c 有効時):
```
x = state'(214, 標準化) ++ option'(69, 標準化)
    ++ card_embedding(option_target_id)                      # 既存 8
    ++ concat[ card_embedding(pid) for pid in 12 board ]     # Tier1a 96
    ++ card_embedding(stadium_id)                            # Tier1b 8
```
旧重み(feature_tiers 空)は `state'(166) ++ option'(65) ++ card_embedding(target)` のみ=現行と一致。

---

## 5. テスト計画

| テスト | 目的 | 合格条件 |
|---|---|---|
| encoder 回帰 | append-only の prefix 不変 | tiers 省略で現行とビット同一ベクトル |
| encoder Tier1 単体 | 新出力の正しさ | 固定盤面 fixture で pokemon_ids 順序・スカラー値・stadium_id を突合 |
| build_features 回帰 | 既定 npz 不変 | `--tiers` なしで現行 npz とバイト一致 |
| policy_model 旧重み回帰 | 後方互換 | 現行 `policy_weights.json` で現行と同一スコア |
| 学習/実行 parity | 連結順一致 | 同一 obs で train forward == model forward |
| 提出健全性 | import/実行 | 空 `decks/` 展開で 1 ゲーム走破(既知の tarball 要件) |

---

## 6. 成果物一覧(新規/変更、現行不変を明示)

**新規(現行に影響なし)**:
- `ptcg_ai/learning/policy_weights_tier1abc.json`(学習後)
- `ptcg_ai/learning/policy_weights_baseline_v4.json`(現行の不変スナップショット複製)
- `kaggle_replays/policy_net/features_tier1abc.npz`(学習後)
- `configs/ml_lethal_attackplan_tier1abc.json`

**変更(後方互換・デフォルトで現行動作)**:
- `ptcg_ai/learning/encoder.py`(デフォルト引数追加、既存出力不変)
- `ptcg_ai/learning/policy_model.py`(自己記述ロード、旧重みで現行動作)
- `kaggle_replays/policy_net/{build_features,train}.py`(既定フラグ=現行)

**不変(絶対に触らない)**:
- `ptcg_ai/learning/policy_weights.json`(現行最善重み)
- `configs/ml_lethal_attackplan_v0only.json`(既定 config)
- `kaggle_replays/policy_net/features.npz`(現行学習データ)

---

## 7. 検証・採用ゲート(親方針書 §7 準拠)

1. オフライン top1 一致率(対照 58.06%)。改善なしは一次不採用シグナル(ただし単独で採用は決めない)。
2. **ミラー対戦 95% CI 勝ち越しが採用の必要条件**([[project_pimc_prod_validation]] の非転移教訓)。
3. 1abc まとめて負け/差なし → Tier1a 単独へ切り戻して再測(1b/1c のノイズ切り分け)。
4. 採用時のみ既定 config を新 config へ差し替え(その 1 行も戻せる)。非採用でもコードはフラグ OFF で温存。

---

## 8. 未決事項(着手前に確認 or 実装中に決定)

- Tier1c 枚数スカラー(basic/special energy count)の事前スケール。Stage1 は生値+標準化で開始、
  効果を見て調整。
- Tier1c を option 側対象ポケモンにも入れるか。本計画は「入れる(append-only)」。効果薄なら state 限定に縮小可。
- board identity の学習コスト(埋め込み lookup 12 倍)。Step 3 でバッチ構築時間を実測し、
  過大なら board を active+ 相手 active の 2 スロットに縮小する縮退案を検討。

---

## 9. 実装順の要約(このあと着手する順)

Step 1(encoder)→ Step 2(build_features)→ Step 3(train)→ Step 4(policy_model)→
Step 5(config)→ Step 6(検証)。各ステップ完了ごとに DoD を満たす。
Step 1・4 の**回帰テスト(旧重みビット同一)を最優先**で通し、可逆性を最初に固定する。
