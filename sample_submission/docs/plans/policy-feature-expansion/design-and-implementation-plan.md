# PolicyModel 特徴量拡張(Tier1/Tier2) 設計・実装方針書

作成: 2026-07-22 / 対象: `AGENT_TYPE="ml_policy"` の PolicyModel(模倣学習)

現行アルゴリズムは残したまま、盤面のカード identity・付属物(道具/エネルギー)・技の
非ダメージ効果などを特徴量に追加した「拡張版」を**別重み**として学習・検証する。

関連ドキュメント:
- 現行概要: [`docs/architecture/current-algorithm-overview.md`](../../architecture/current-algorithm-overview.md) §2
- 既存の card_embedding 設計: [`docs/plans/ml-value-network/step2-algorithm-selection.md`](../ml-value-network/step2-algorithm-selection.md) §7
- 実装対象: [`ptcg_ai/learning/encoder.py`](../../../ptcg_ai/learning/encoder.py) /
  [`ptcg_ai/learning/policy_model.py`](../../../ptcg_ai/learning/policy_model.py) /
  `kaggle_replays/policy_net/{build_features,train,evaluate}.py`

---

## 0. TL;DR(フェーズ地図)

| フェーズ | 内容 | 追加の埋め込みテーブル | 契約変更 |
|---|---|---|---|
| **Tier1a** | 盤面ポケモン identity 埋め込み(12スロット) | なし(既存 card テーブルを共有) | 特徴 id ストリーム追加 |
| **Tier1b** | スタジアム identity 埋め込み(1) | なし(既存 card テーブルを共有) | 同上 |
| **Tier1c** | ポケモン毎スカラー追加(道具有無 / 基本・特殊エネ枚数 / 特性有無) | なし | `BASE_FEATURE_COUNT` 増加 |
| **Tier2a** | 技 identity 埋め込み(ATTACK/SKILL 選択肢の attackId) | **新規** `attack_embedding` | 選択肢 id ストリーム追加 |
| **Tier2b** | 特殊エネルギー identity(active に付いた特殊エネの pool 埋め込み) | なし(既存 card テーブルを共有) | 特徴 id ストリーム追加 |

各 Tier は**独立の学習・検証サイクル**にし、Tier 単位で ablation する(§7)。まとめて全部入れない。

**設計上の重要判断(§3.4 に詳述):**
- 「特性の identity」専用埋め込みは**やらない**。特性はカードの静的属性なので Tier1a の
  ポケモン identity 埋め込みが暗黙に捕捉する。
- 「特性の使用済み状態」は**スコープ外(実装不可)**。公開 `Observation` に per-pokemon の
  ability-used フラグが無く、`obs.logs` からしか復元できないが、encoder は学習/実行 parity の
  ため logs 参照を禁じている。
- 「弱点・抵抗・軽減・無効」は**追加しない**。弱点/抵抗は既に `resolve_damage` 内で打点へ反映済み
  ([encoder.py:195](../../../ptcg_ai/learning/encoder.py#L195))。軽減/無効の多くは特性由来で、
  Tier1a のポケモン identity が担う。
- 「option を 1 手仮実行した結果の特徴」は**この方針書のスコープ外**(別扱い)。学習/実行の
  hidden-info 乖離とレイテンシのリスクが質的に異なるため、Tier1/2 完了後に独立実験として扱う。

---

## 1. 背景と現行アーキテクチャの制約

現行 PolicyModel の入力は 3 系統(合計 239 次元):

1. **状態特徴** `encode_state` … 166 次元。標準化(mean/std)される。
2. **選択肢特徴** `encode_options` … 選択肢ごと 65 次元。標準化される。
3. **カード identity 埋め込み** `encode_option_card_ids` → `card_embedding.table`(dim=8)…
   **選択肢の対象カード 1 個ぶんだけ**。標準化せず生の埋め込みを連結。

制約として現状こうなっている:

- 盤面 12 スロットのポケモンは `_POKEMON_FEATURE_NAMES` の**数値 11 特徴のみ**で、
  「どのポケモンか」を捨てている(identity 埋め込みは選択肢対象にしか存在しない)。
- ポケモンに付いた**道具**の概念が特徴に一切無い。
- エネルギーは `energy_count`(生枚数)のみ。基本/特殊の区別・型・identity が無い。
- スタジアムは `has_stadium`/`stadium_played` の有無フラグのみで、どのスタジアムかが無い。
- 技は damage/can_ko/shortfall/ready の 4 スカラーのみ。非ダメージ効果が不可視。

**parity 制約(全 Tier で厳守)**: encoder は「公開情報のみ・`obs.logs`/`obs.select` 非依存・
決定的」でなければならない([encoder.py](../../../ptcg_ai/learning/encoder.py) 冒頭 docstring)。
学習データでは `logs` が削除されているため、logs から復元する情報は追加できない。

---

## 2. 設計判断

### 2.1 2 トラック方式(スカラー / identity 埋め込みを分ける)

追加情報を性質で 2 経路に分ける。既存アーキテクチャに素直に乗る:

- **標準化スカラー・トラック**: 有無フラグや枚数など連続値は、既存の状態/選択肢特徴ベクトルに
  足して**通常どおり標準化**する。→ Tier1c(道具有無・基本/特殊エネ枚数・特性有無)。
- **identity 埋め込みトラック**: 「どのカード/技か」は id を埋め込みテーブルで引き、
  **標準化せず**生ベクトルを連結する(既存 card_embedding と同じ扱い)。→ Tier1a/1b/2a/2b。

この分離により、識別情報を安易に one-hot 化して次元爆発させない。

### 2.2 id 空間と埋め込みテーブルの共有

- **card-id 空間**(`all_card_data()` の `cardId`): ポケモン・道具・スタジアム・特殊エネは
  すべて同じ card-id 空間。→ **既存の `card_embedding.table` を共有**して引く。
  新しいテーブルを作らない(パラメータ増を抑え、同一カードが複数文脈で勾配を受けるので
  むしろ汎化に有利)。Tier1a/1b/2b はテーブル追加ゼロ。
- **attack-id 空間**(`all_attack()` の `attackId`): カードとは別 id 空間。→ Tier2a で
  **新規テーブル `attack_embedding`** を 1 つだけ追加。

### 2.3 pooling(複数個持てるスロット)

道具・エネルギーは 1 スロットに複数付きうる。埋め込みは**平均 pool**(存在しなければゼロ)。
枚数情報はスカラー・トラック側(Tier1c)が別に持つので、pool 側は identity の代表値だけ担う。

### 2.4 未学習埋め込みの安全性

学習データに出ない id の埋め込みは初期値(ゼロ近傍)のまま。既存 card_embedding と同じ挙動で
安全側に劣化するだけ([policy_model.py](../../../ptcg_ai/learning/policy_model.py) §card_embedding)。
id が 0 / 範囲外は既存同様 index 0(「識別なし」予約枠)へフォールバック。

---

## 3. 追加特徴の定義

### 3.1 Tier1a — 盤面ポケモン identity(id ストリーム、既存 card テーブル)

- 入力元(公開情報): 自分/相手 × (active + ベンチ 5 スロット) = 12 スロットの `Pokemon.id`。
  空/伏せスロットは 0(識別なし)。
- 順序: `encode_state` の `_side_pokemon_block` と**完全に同じスロット順**
  (self active, self bench0..4, opp active, opp bench0..4)。
- 出力: 長さ 12 の `list[int]`。forward で各要素を `card_embedding.table` で引き、
  12×dim を状態側の埋め込みとして連結。

### 3.2 Tier1b — スタジアム identity(id ストリーム、既存 card テーブル)

- 入力元: `state.stadium`(size 0/1)。あれば `stadium[0].id`、無ければ 0。
- 出力: 単一 int。forward で 1×dim を連結。

### 3.3 Tier1c — ポケモン毎スカラー追加(標準化トラック)

`_POKEMON_FEATURE_NAMES` に以下を追加(12 スロット全てに効く。標準化される):

| 新スカラー | 定義 | 入力元 |
|---|---|---|
| `tool_present` | 道具が付いているか(0/1) | `len(Pokemon.tools) > 0` |
| `basic_energy_count` | 付属エネのうち基本エネ枚数 | `Pokemon.energyCards` を card_cache で引き `cardType==BASIC_ENERGY` を計数 |
| `special_energy_count` | 付属エネのうち特殊エネ枚数 | 同上 `cardType==SPECIAL_ENERGY` |
| `ability_present` | 特性を持つカードか(0/1) | `CardData.skills` が非空(静的属性) |

注: `Pokemon.energies`(`EnergyType` 列)は「現在このポケモンが供給できるエネ型」を表すが、
基本/特殊の区別は付かない。基本/特殊の判別は `energyCards`(実カード)→ `cardType` で行う。

> **契約影響**: `_POKEMON_FEATURE_NAMES` が 11→15 になり `BASE_FEATURE_COUNT` が
> 166 → 166 + 4×12 = 214 に増える。選択肢側の `target_pokemon_*`(同じ 11 特徴を流用)も
> 自動的に 15 になり `OPTION_FEATURE_COUNT` も増える。標準化 mean/std は再計算(再学習)必須。

### 3.4 Tier2a — 技 identity(選択肢 id ストリーム、新規 `attack_embedding`)

- 対象: `OptionType.ATTACK` / `SKILL` の選択肢。`Option.attackId` を id とする。
- それ以外の選択肢は 0(識別なし)。
- 出力: `encode_option_card_ids` と並列の、選択肢と同順・同長の `list[int]`。
- forward で新規 `attack_embedding.table`(`nn.Embedding(attack_id_max+1, dim)`)を引き、
  選択肢埋め込みとして連結。
- **狙い**: 「ダメージ以外の効果(状態異常付与・ベンチダメ・エネ加速・ドロー等)」を、
  効果テキストを手書きパースせず attackId からデータで学習させる。ポケモン identity(Tier1a)は
  「どのポケモンか」までしか分けないが、同一ポケモンの複数技のうち**今どの技を選ぶか**の
  非ダメージ効果差はこの埋め込みが担う。

### 3.5 Tier2b — 特殊エネルギー identity(id ストリーム、既存 card テーブル、優先度低)

- 対象: 自分/相手 active の `energyCards` のうち `cardType==SPECIAL_ENERGY` のカード id 群。
- pool: 平均 pool(§2.3)。無ければゼロ。self/opp active の 2 個ぶん。
- 優先度は Tier1/2a より低い。効果が出にくければ見送り可。

### 3.6 やらない/できないもの(再掲・根拠)

- **特性の使用済み状態**: 公開 `Observation` に無い。`obs.logs` からしか取れず parity 違反。→ 不可。
- **特性専用 identity 埋め込み**: 特性はカード静的属性 → Tier1a に内包。冗長のため不採用。
- **弱点/抵抗の明示特徴**: 既に打点計算に反映済み。二重のため不採用。
- **軽減/無効**: 多くが特性由来 → Tier1a が担う。専用特徴は不採用。
- **option 1 手仮実行特徴**: 別実験(hidden-info 乖離・レイテンシのリスク管理が別問題)。

---

## 4. 契約変更(`policy_weights.json` / `features.npz`)

学習パイプラインと共有する契約([policy_model.py](../../../ptcg_ai/learning/policy_model.py) 冒頭、
[build_features.py](../../../../kaggle_replays/policy_net/build_features.py) 冒頭)を以下のとおり拡張。
**後方互換**: 新キーが無い旧 `policy_weights.json` を読んだ場合は拡張分を無効化(Tier 判定は
meta のフラグで行い、現行 v4 重みはそのまま動く)。

```jsonc
{
  "meta": {
    "state_feature_count": 214,          // Tier1c で更新
    "option_feature_count": <new>,       // Tier1c で更新
    "feature_tiers": ["tier1a","tier1b","tier1c"],  // 有効化した拡張の記録(新規)
    ...
  },
  "standardization": { ... },            // 次元数は state/option の新カウントに一致
  "card_embedding": { "dim": 8, "card_id_max": ..., "table": [...] },  // 既存。Tier1a/1b/2b が共有
  "attack_embedding": {                  // Tier2a のみ新規
    "dim": 8, "attack_id_max": ..., "table": [...]
  },
  "layers": [ ... ]                      // 入力次元は下記 forward の連結長に一致
}
```

`features.npz` 追加配列(build_features):
- `state_pokemon_ids`: shape (N, 12) int32 — Tier1a。
- `stadium_id`: shape (N,) int32 — Tier1b。
- `option_attack_ids`: object 配列(ragged, 選択肢数可変) int32 — Tier2a。
- `special_energy_ids_self` / `_opp`: object 配列(可変長) int32 — Tier2b。
- `attack_id_max`: スカラー — Tier2a(`all_attack()` から動的計算、ハードコード禁止)。

forward 連結(policy_model):
```
x = state' ++ option'
    ++ card_embedding(option_target_id)                 # 既存
    ++ concat[ card_embedding(pid) for pid in 12 board ] # Tier1a
    ++ card_embedding(stadium_id)                        # Tier1b
    ++ attack_embedding(option_attack_id)                # Tier2a
    ++ mean_pool[ card_embedding(e) for special_energy ] # Tier2b(self/opp active)
```

---

## 5. 実装フェーズと関数シグネチャ

各フェーズは PR を分ける。encoder → build_features → train → policy_model → 検証 の順。

### Phase A: encoder(Tier1a/1b/1c)

[`ptcg_ai/learning/encoder.py`](../../../ptcg_ai/learning/encoder.py):

```python
# Tier1c: _POKEMON_FEATURE_NAMES に 4 特徴を追加、_pokemon_features の戻り値を 15 要素へ。
#   引数に card(既に _card_or_none 済み)を使い tool/energy/skill を集計。
def _pokemon_features(pokemon, defender, attacker_hand_size, state, owner_index) -> list[float]:
    ...  # 末尾に tool_present, basic_energy_count, special_energy_count, ability_present

# Tier1a: 盤面 12 スロットの Pokemon.id を _side_pokemon_block と同順で返す。
def encode_state_pokemon_ids(state: State | None) -> list[int]:  # 長さ 12(空/伏せは 0)

# Tier1b:
def encode_state_stadium_id(state: State | None) -> int:  # stadium[0].id or 0
```

DoD(Phase A):
- `BASE_FEATURE_COUNT == 214`、`FEATURE_NAMES` 末尾に新 4 特徴 ×12 が正しい名前で並ぶ。
- `encode_state_pokemon_ids` の順序が `_side_pokemon_block` と一致(単体テストで固定盤面を突合)。
- 空スロット/伏せ/未知 id で例外を出さずゼロ/0 フォールバック(既存の決定的性質を維持)。
- 既存テスト(encoder の parity/決定性)が新次元で緑。

### Phase B: encoder(Tier2a/2b)

```python
# Tier2a: encode_option_card_ids と同順・同長。ATTACK/SKILL 以外は 0。
def encode_option_attack_ids(state: State | None, select: SelectData | None) -> list[int]:

# Tier2b: self/opp active の特殊エネ card_id 群(可変長)。
def encode_active_special_energy_ids(state: State | None) -> tuple[list[int], list[int]]:
```

DoD(Phase B): ATTACK 選択肢で `Option.attackId` を返し、非攻撃選択肢は 0。長さが
`select.option` と一致。特殊エネ抽出が `energyCards` の `cardType` 判定で行われる。

### Phase C: build_features(`kaggle_replays/policy_net/build_features.py`)

- 各 Phase の新 encoder 出力を呼び、`features.npz` へ §4 の配列名で保存。
- `attack_id_max = max(a.attackId for a in all_attack())` を動的計算し保存(ハードコード禁止)。
- **行順を変えない / encode 失敗は fail-fast**(既存の row_index 対応の制約を厳守)。
- 長さ検証(`state_pokemon_ids` は必ず 12、`option_attack_ids` は n_options)を追加。

DoD(Phase C): `--limit 2000` で走り、npz に新配列が期待 shape/dtype で入る。`allow_pickle=True`
読込ヘルパの注記を更新。

### Phase D: train(`kaggle_replays/policy_net/train.py`)

- 本命モデルへ埋め込み lookup を追加: 既存 `self.embedding`(card)を **board/stadium/special-energy の
  追加 lookup にも流用**、Tier2a は `self.attack_embedding = nn.Embedding(attack_id_max+1, dim)` を新設。
- `build_batch` に新 id 列の連結を追加。ragged は既存 card_id と同じ扱い。
- エクスポートに `attack_embedding` と `meta.feature_tiers` を追加(§4)。
- **フラグで Tier を切替**(`--tiers tier1a,tier1b,...`)。既定=全 OFF で現行 v4 と完全一致
  (ベースライン再現性の担保 = ablation の対照)。

DoD(Phase D): 全 Tier OFF で現行 top1 一致率を再現。Tier ON で学習が収束し、
`policy_weights.json` が §4 スキーマで出力される。

### Phase E: policy_model(`ptcg_ai/learning/policy_model.py`)

- `_load` で `attack_embedding` と `meta.feature_tiers` を読む(無ければ拡張無効=後方互換)。
- `_forward` を §4 の連結順に拡張。`score_options_from_state` で新 encoder 出力を呼ぶ。
- 未ロード/範囲外 id は既存同様フォールバック。

DoD(Phase E): 旧重み(拡張キー無し)で現行と同一出力(回帰テスト)。新重みで
学習時の連結順と実行時が一致(train と model の forward が同じ次元・同じ順序)。
**学習/実行 parity テスト**: 同一 obs に対する train 側 forward と model 側 forward のスコアが一致。

---

## 6. 完了条件(全体 DoD)

- [ ] 各 Phase の単体 DoD を満たす。
- [ ] parity: 学習データ(logs 削除済み)とランタイム obs で新 encoder 出力が同一。
- [ ] 決定性: 同一入力で同一出力(既存 encoder の保証を新特徴でも維持)。
- [ ] 後方互換: 現行 `policy_weights.json`(v4)を新コードで読んで挙動不変。
- [ ] 提出健全性: 空 `decks/` を展開して 1 ゲーム走らせ import/実行が通る
  (submission tarball の decks 必須の既知事項を踏襲)。

---

## 7. 検証・計測プロトコル(Tier 単位 ablation)

過去、**オフライン一致率の改善が対戦勝率に転移しない**前例がある(構成 C はオフライン +1.5pt でも
ミラーで有意勝ち / PIMC は研究で勝ったが本番 ml_policy 上は有意差なし)。したがって:

1. **オフライン**: 各 Tier の重みで test top1 一致率を測る(現行 58.06% が対照)。改善が無ければ
   その Tier は不採用の一次判断。ただし top1 だけで採用は決めない。
2. **ミラー対戦(採用の最終ゲート)**: 現行既定重み vs 拡張重みを head-to-head で対戦させ、
   95% CI で有意に勝ち越したものだけ採用(構成 C と同じ判定様式)。
3. **ablation は Tier 単位で加算的に**: baseline → +Tier1a → +Tier1b → +Tier1c → +Tier2a …
   と 1 つずつ足し、どの Tier が効いたかを切り分ける。まとめて全部入れて測らない。
4. 採用した構成のみ既定 config / `policy_weights.json` へ反映。非採用 Tier のコードは
   フラグ OFF で温存(削除しない)。

---

## 8. リスクと対策

| リスク | 対策 |
|---|---|
| 次元増による過学習(特に board 12×dim + 新スカラー 48) | Tier 単位 ablation で寄与を確認。効かない Tier は入れない。埋め込み共有でパラメータ増を抑制 |
| 学習/実行の forward 不一致(連結順ズレ) | §5 Phase E の parity テストを必須化。連結順を §4 に一元化し train/model 双方が参照 |
| 未学習 id の劣化 | 既存 card_embedding と同じゼロ近傍フォールバック。index 0 予約枠を維持 |
| 契約破壊(旧重みが読めない) | 新キーは任意・meta.feature_tiers で分岐。後方互換テストを DoD 化 |
| オフライン改善が勝率に非転移 | ミラー対戦を採用の最終ゲートに(§7-2) |
| 特性使用済みを諦めることによる情報欠落 | 現状は許容(公開 obs に無い)。将来 logs 由来の別経路が要るなら hidden_information 側で別途検討 |

---

## 9. 未決事項(実装着手前に確認)

- 追加スカラーの正規化: 枚数系(basic/special energy count)のスケール。既存 `energy_count` と
  同じ生値+標準化で足りるか、10 で割る等の事前スケールを合わせるか。
- Tier2b(特殊エネ identity)を初回スコープに含めるか、Tier2a の結果を見て判断するか。
- 学習コスト: board identity で埋め込み lookup が 12 倍。train.py のバッチ構築時間の実測が必要。
- `attack_embedding` の dim を card と同じ 8 に固定するか(coordinator 指定の踏襲)。
