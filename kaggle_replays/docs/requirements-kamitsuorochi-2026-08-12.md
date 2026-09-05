# カミツオロチex 移行と、対面別AI／リーグRLへの計画（2026-08-12）

> **状態: 有効（現在の作業指示書）。作成 2026-08-12、同日深夜に全面改訂。**
> 上位方針は [roadmap-2026-08-05.md](roadmap-2026-08-05.md)。本書はその §8（メインデッキ確定）を
> カミツオロチex でやり直し、さらに Phase E（リーグRL）への具体的な段取りまでを含む。
> **数値はすべて 2026-08-12 に実測したもの。** 推定値には「推定」と明記する。

---

## 1. 最終的に目指す形（ユーザー方針、2026-08-12 確認）

```
① デッキごとに BC で AI を作る
② その AI 同士を戦わせて強化学習する
③ デッキごと・対面ごとの個別 AI を作る
④ 相手に応じてモデルを切り替える
```

本書はこれを4つの step に分解し、**どれが実測で裏づけられていて、どれがまだ未検証か**を区別して並べる。
Transformer / Set Transformer は**やらない**（ユーザー判断、2026-08-12）。

---

## 2. 現在地（2026-08-12 深夜、完了済み）

| | 状態 |
|---|---|
| メインデッキ | **カミツオロチex（L3、60枚）**。`sample_submission/deck.csv` と `archetype_decks/kamitsuorochi_ex/06.csv` が md5 一致 |
| 教師コーパス | 12,047 リプレイ（旧 7,444 ＋ 新 4,603）。kamitsuorochi 教師 **551 ep-player / 35,347 決定点** |
| 打点計算 | **可変打点3パターンを修正済み**（§5-1）。みつあめストームが 30 → 最大330 |
| 提出重み | `policy_weights.json` md5 `fa19b32c…` = カミツオロチ専用BC（test top-1 0.6623） |
| テスト | 369 passed / 14 skipped |
| 本番経路 | `run_league` で rule_based に 28/30、エラー0、6分07秒 |

**この状態はそのまま Kaggle に提出できる。** 以降はすべて上積み。

---

## 3. 実測データ（判断の土台）

### 3-1. メタが入れ替わっている

新ログ `archive (3)` = **2026-08-10 の1日分**、4,603エピソード / 9,198デッキ / 21.5GB。
旧コーパスは episode 88.0M〜89.0M（7月中旬〜下旬）。

| アーキタイプ | 新ログ | 旧コーパス | 新ログでの勝率 |
|---|---:|---:|---:|
| marnie_grimmsnarl_ex | 29.5% | 59.9% | 45.0% |
| alakazam | 19.6% | 10.3% | 47.4% |
| **mega_froslass_ex** | **9.6%** | **0.01%** | 51.8% |
| dragapult_ex | 9.1% | 2.6% | 57.8% |
| other | 6.1% | 1.6% | 49.6% |
| crustle | 5.8% | 8.9% | 42.8% |
| **kamitsuorochi_ex** | **5.6%（515）** | **0.24%（36）** | **59.8%** |
| omatsuri_ondo | 4.4% | 3.0% | 56.0% |
| mega_lucario_ex | 4.0% | 0.50% | 58.5% |
| ogerpon_teal_ex | 2.7% | 2.9% | 58.2% |
| yadoking | 1.5% | 0.01% | 65.7% |
| shirona_garchomp_ex | 1.2% | 2.9% | 46.4% |
| rocket_mewtwo_ex | 0.38% | 6.0% | 28.6% |

ユキメノコexは旧コーパスに2件しかないのに現在9.6%、ロケット団ミュウツーは 6.0%→0.38% で消滅。
**旧コーパスのメタ分布はもう使えない。** 評価プール（`rl/pools.py:85` の POOL8）も現メタとずれている
（rocket_mewtwo を1/8の重みで評価し、froslass 9.6% が入っていない）。

**腕前の分布**: 新ログは1日分の全体なので中位帯が厚い（両者1100点超は 5.4%、旧コーパスは 22.8%）。
`manifest.csv` の `avg_score` でフィルタできる（`>=1050` で 18.6%）。今回は使わず、
代わりに順位ベースの重み付け（→ §9-2）で対処した。

### 3-2. カミツオロチの対面別成績（新ログ実測）

| 対面 | 件数 | 決定点 | 勝率 |
|---|---:|---:|---:|
| marnie_grimmsnarl_ex | 129 | 8,645 | 76.0% |
| **alakazam** | 92 | 7,260 | **35.9%** ← 最悪 |
| dragapult_ex | 69 | 4,522 | 52.2% |
| mega_froslass_ex | 64 | 3,927 | 70.3% |
| crustle | 30 | 2,548 | 53.3% |
| mega_lucario_ex | 22 | 1,448 | 45.5% |
| omatsuri_ondo | 20 | 1,445 | 45.0% |
| ogerpon_teal_ex | 18 | 1,071 | 77.8% |

**alakazam は実メタ19.6%を占め、勝率35.9%。ここが唯一の構造的な弱点。**
次点は「特性持ちの打点を無効化する壁」を含む群（§5-2）。

### 3-3. デッキリスト（L3 を採用）

515 ep-player の中に**ユニークなリストは6種類しかない**。

| リスト | n | 勝率 | 使用者の平均スコア | L1との差 |
|---|---:|---:|---:|---:|
| L1（最頻） | 186 | 57.5% | 1057 | 0 |
| L2 | 122 | 59.8% | 1034 | 7 |
| **L3（採用）** | **116** | **69.0%** | **1077** | **3** |
| L4 | 63 | 66.7% | 1055 | 7 |
| L5 | 16 | 37.5% | 1000 | 30 |
| L6 | 12 | 0.0% | 1013 | 31 |

L3 − L1 = セレビィ(655) / カプ・ブルル(920) / ブライア(1201) を抜いて基本【草】エネ +3。
**基本草エネが17枚と最多で、これは打点式（§7-1）と整合している。**
代表エピソード 91802249（team "palsystem"、avg_score 1189.1）。

L3 の60枚:

```
1 x17 基本【草】/ 96 x4 オーガポンみどりのめんex / 1094 x4 むしとりセット /
1121 x4 ハイパーボール / 1227 x4 リーリエの決心 / 1261 x4 活力の森 /
93 x2 カミッチュ / 150 x2 カミツオロチex / 710 x2 メガニウム / 917 x2 チコリータ /
1071 x2 ニャースex / 1182 x2 ボスの指令 /
140 キチキギスex / 149 カジッチュ / 346 カジッチュ / 709 ベイリーフ / 918 ベイリーフ /
1080 アンフェアスタンプ / 1097 夜のタンカ / 1152 ポケパッド / 1184 スイレンのお世話 /
1188 暗号マニアの解読 / 1231 ヒカリ
```

> 記事（ハレルヤ2）は「オーガポンいしずえのめんex 突破にカプ・ブルルが必須」とするが、
> L3 は不採用。実測では 117 を積む相手は 4.9%（crustle 246 / ogerpon_teal 149）で、
> カプ・ブルル入りの L2/L4 が L3 を上回る結果は出ていない。**117 対面が測定で
> 問題として出てきたら再考する。**

---

## 4. 確定した設計判断（すべて実測に基づく）

| # | 判断 | 根拠（実測） |
|---|---|---|
| D1 | **新ログを使う。旧コーパス単独では不可能** | 旧コーパスの kamitsuorochi 教師は 36 ep-player、新ログに 515 |
| D2 | **対面別に BC をファインチューンするのは効かない。やらない** | 3クラスタで +0.17 / +0.50 / −0.13 pt。シード間SD 0.0038 と同程度（§4-1） |
| D3 | **対面特化は RL でやる。これは D2 とは別物** | 7月の `alakazam_rl_vscrustle` が 0.380 → 0.465（§4-2）。ただし best-checkpoint 由来の水増しに注意 |
| D4 | **カード同一性の追加ゾーン（C2/T1）は入れない** | alakazam −13.9 Elo、marnie −11.7 Elo。カミツオロチでも再現（フルT1 0.6531 対 ablate版 0.6623） |
| D5 | **`decks/new_deck/*`（ルールベースのプロファイル）は書き換えない** | 本番経路に載っていない（§4-3） |
| D6 | Transformer / Set Transformer / Deep Sets | ユーザー判断で中止 |

### 4-1. なぜ対面別 BC ファインチューンが効かないのか

`train.py` に `--init-weights` / `--lr` / `--row-mask` を追加し、ベースBC（27,347行で学習）を
初期値に、クラスタ別の部分集合を lr 1/8 で追加学習した。**同じ部分集合でベースと比較**した結果:

| クラスタ | 相手 | train行数 | ベース top-1 | FT top-1 | 差 |
|---|---|---:|---:|---:|---:|
| A 壁 | crustle / ogerpon / lucario / starmie | 4,031 | 0.5571 | 0.5588 | +0.0017 |
| B ex殴り合い | marnie / dragapult / froslass | 13,022 | 0.6713 | 0.6763 | +0.0050 |
| C alakazam | alakazam | 6,014 | 0.6965 | 0.6953 | **−0.0013** |

**理由は「新しい情報が1ビットも入っていない」から。** 同じ教師の同じ手を、部分集合で
もう一度なぞっているだけなので、精度が上がる余地が無い。

**副産物として重要な発見**: ベースの時点で**クラスタA が極端に苦手**（0.5571、C との差 14pt）。
追加学習で埋まらなかった＝「この対面の教師が足りないから弱い」のではない。
壁対面の正しい立ち回りは「主砲を諦めて非特性アタッカーに切り替える」で、
これは**手の順位付け（BCが学ぶもの）ではなく打点評価の問題**（→ §5-2）。

### 4-2. なぜ対面特化 RL は別物なのか

BC ファインチューンと違い、**RL は特定の相手と実際に戦って勝敗を受け取る**ので、
教師データに存在しない情報が入る。7月の実測（重みJSONの `meta` に残っている）:

| 重み | RL前 | RL後（best iter） | 差 |
|---|---:|---:|---:|
| `policy_weights_alakazam_rl_vscrustle.json`（**対面特化**） | 0.380 | 0.465（iter 21） | **+8.5pt** |
| `policy_weights_alakazam_rl_field.json`（全体） | 0.6775 | 0.7075（iter 6） | +3.0pt |

**対面特化のほうが伸びている。** roadmap §2 の「出発点が低いほど伸びる」と整合
（alakazam 0.410→0.600 +19pt / crustle 0.710→0.730 +2pt）。

roadmap §2 の情報量の議論とも矛盾しない:

```
RL 60反復 ≒ 31 kbit を1試合80決定に薄く拡散  → 学べるのは粗い行動傾向だけ
BC ≒ 170 kbit を決定点に直付け              → 細かい対象選択はこちらの領分
```

「壁相手には主砲を諦める」は**粗い行動傾向**なので RL の帯域に乗る。
一方「17択のどれを選ぶか」は乗らない。**役割分担が明確。**

> ⚠ **上の +8.5pt は額面どおりではない。** `rl_best_eval_winrate` は best-checkpoint の値で、
> roadmap §1 欠陥#1 の実測（marnie best 0.800/n=200 → 最終 0.757/n=1200、−4.3pt、理論値+6pt）
> からすると、**真の伸びは +2〜3pt 程度の可能性が高い**。§8 の修正を入れてから測り直すこと。

### 4-3. `decks/new_deck/*` を書き換えなくてよい根拠

`sample_submission/decks/new_deck/` は `741/742/743`（フーディン線）等 alakazam の card_id で
全部書かれており、`profile_registry.py:104-131` は `.get()` なので未知 card_id で静かに `None` を返す。
書き換えは人手で数時間〜1日。**しかし本番の意思決定経路に載っていない**:

| 経路 | 依存 |
|---|---|
| `core/agent.py:18` `AGENT_TYPE="ml_policy"` → `ml_policy_agent` | — |
| `ml_policy_agent.py:88` `_CONFIG_NAME="abl_5_full"` | `configs/abl_5_full.json` に **`attack_hybrid` キーが無い** |
| `_try_attack_hybrid`（`:268-270`） | `attack_hybrid.enabled` が False → **発火しない** |
| `_try_lethal` → `search/lethal_simple.py` | rule_based を import しない |
| `_try_pipeline` → `search/pipeline.py` / `leaf_eval.py` | `leaf_eval` は `board_features.attacker_score` のみ |

影響が出るのは `run_league --agent-a rule_based` のベースライン相手と、BC 未ロード時のフォールバックだけ。

---

## 5. 既知の欠陥（打点・評価の嘘）

このデッキで最も効くのはここ。roadmap の実績では **crustle の打点修正が +15.0 Elo** で、
データ倍増（+4.9 Elo）より大きかった。

### 5-1. 【修正済み 2026-08-12】可変打点が全部過小評価だった

`attack_features.py` の可変打点パターンは
`does (\d+) damage` と `for each card in your hand`（フーディン専用）の2つしか無かった。

| ワザ | 修正前 | 修正後 |
|---|---:|---:|
| みつあめストーム 基本草2枚（メガニウム無） | 30 | 90 |
| 同 5枚（メガニウム無） | 30 | 180 |
| 同 2枚（メガニウム有） | 30 | 150 |
| **同 5枚（メガニウム有）** | **30** | **330**（HP330のexをワンパン） |
| ともだちのわ ベンチ0/3/5体 | 20固定 | 0 / 60 / 100 |
| まんようしぐれ 両バトル場エネ計3 | 30 | 120 |

**メガニウム「おいしげる」の2倍化は cg エンジンが `Pokemon.energies` に既に織り込んでいた**
（カミツオロチexミラー403サンプルで実測。メガニウム在場の104サンプルは全部 `energies/energyCards = 2.0`、
不在は 298/299 が 1.0）。自前の倍化ロジックは不要——入れていたら二重に掛けていた。

配線は A10 と同じ5箇所（`encoder.py`×2 / `priorities/attack.py` / `attack_turn.py` /
`board_features.py`）。後方互換とフーディン回帰（140）を確認。メモ化により実行時間の増分は実質ゼロ
（約2.3µs/回）。回帰テスト `tests/unit/test_attack_features_variable_damage.py` 19件。

### 5-2. 【未修正・次の最優先】テラスタルのベンチ無敵が評価されていない

`cg/api.py:478`:

```
tera: bool  # True if Tera Pokémon. Tera Pokémon take no damage from attacks as long as they are on the Bench.
```

**`ptcg_ai/` 全体で、この `tera` をダメージ無効の判定に使っている箇所はゼロ。**
`attack_features.py:420,498` の2箇所は「ワザのテキストに *tera pokemon* と書いてあるか」を
見ているだけで、ベンチ無敵とは無関係。

**影響**: ベンチのオーガポン みどりのめんex(96) が「倒されうる」と誤って評価される。
選択肢特徴には `target_pokemon_is_likely_ko_next_turn` があり（＝「次ターン倒される対象に
エネを付けない」は学習可能な形になっている）、その値が嘘をついている。

**このデッキの正しいエネの置き場所は、ベンチのオーガポン。**
打点への寄与は付け先に依らないのに、ベンチのテラスタルに置いたエネだけは絶対に失われない。
**教師データを何倍にしても直らない。表現の問題。**

修正: `damage_prevented()` に「defender がベンチにいて `card.tera` が真なら無効」を1条件追加。
`is_likely_ko_next_turn` は方策・値ネット・葉評価の全部が使う特徴なので効き幅が広い。

### 5-3. 117 オーガポン いしずえのめんex（ハードカウンター）

```
[特性]いしずえのかまえ「このポケモンは、相手の特性を持つポケモンからワザのダメージを受けない」
```

L3 で特性を持つのは **カミツオロチex / カミッチュ / メガニウム / オーガポンみどりのめんex /
キチキギスex / ニャースex** ＝ 主要アタッカーが全部無効化される。突破できるのは
チコリータ(917) / ベイリーフ(709,918) / カジッチュ(149,346) だけで最大60打点。

実測: 117 を積む相手は **4.9%**（crustle 246 / ogerpon_teal 149 / other 57）。
対 117 入りの成績は 24戦 50.0%（サンプル小）。

`damage_prevented()` は A8 で「攻撃側が特性持ち」条件に対応済みなので、§5-1 の修正で
**この対面で打点0を正しく0と評価できる**ようになった。**プラン変更（非特性アタッカーへの切替）は未実装**
——ここが §4-1 で見つかった「クラスタAが 0.5571 と極端に苦手」の正体である可能性が高い。

### 5-4. 【2026-08-12 実測・最大の欠陥】crustle 対面で攻撃の6割が打点0

> **状態: 確定**。100試合を本番エージェント経路で計測（`kaggle_replays/diagnostics/diag_crustle_shard.py`、
> `run_league.py` の `build_agent` / デッキ読み込み / 手番割当を複製、`config_base=abl_5_full`、エラー0）。
> **これは「打点評価のバグ」ではない。評価器は正しい。行動選択とプランの問題。**

#### 構造（カード原文）

```
345 イワパレス HP150  特性 Mysterious Rock Inn
    「相手の ex ポケモンのワザによる、このポケモンへのダメージをすべて防ぐ」
756 メガガルーラex HP300  きぜつで相手はサイド3枚（Mega ex）
```

L3 のアタッカーは **カミツオロチex(150) / オーガポンみどりのめんex(96) / キチキギスex(140) /
ニャースex(1071) が全部 ex** ＝ イワパレスに 0 打点。
`damage_prevented()` を実カードで検証した結果、**4体すべて True、非ex 4種すべて False、
メガガルーラex には全員 False** ——**評価器は完全に正しく見えている**。

#### 壁は殴り倒せない（デッキ構成から確定）

crustle 側は Grow Grass Energy(+20HP) / Hero's Cape(+100HP) / **Jumbo Ice Cream ×4（毎ターン80回復）**。
こちらの非ex最大打点は メガニウム Solar Beam **140**、カミッチュ Do the Wave 最大100。
**回復に追いつかず、イワパレスは倒せない。** 「非exアタッカーに切り替える」は解ではない。

#### 唯一の勝ち筋（打点を実測して確認）

カミツオロチex の Syrup Storm は「自分の全ポケモンについた{G}1個につき+30」。
メガニウムの Wild Growth で基本G1枚が{G}2個になる（`Pokemon.energies` に反映済み・検証済み）:

| 盤面の基本G | 打点 |
|---:|---:|
| 4枚 | 270 |
| **5枚** | **330** ← メガガルーラex(HP300)をワンパン |

**メガガルーラex は Mega ex なので1体でサイド3枚。2体倒せば6枚＝勝ち。イワパレスは無視してよい。**

#### 実測（100試合、A=カミツオロチ）

| 指標 | 値 |
|---|---:|
| A 勝率 | 4/100（別途 `run_league` では 12/100。§下注） |
| **A の攻撃 1,022回のうち 打点0** | **642回（62.8%）** |
| **うち ex→イワパレス の完全な空振り** | **608回（全攻撃の59.5%）** |
| 1攻撃あたり平均打点 | 46.1 |
| A が取れたサイド（平均） | **0.95 / 6**（**69試合で0枚**） |
| 平均ターン数 | 19.6 |

攻撃の内訳（上位）:

| 攻撃側 → 相手バトル場 | 回数 | うち0打点 |
|---|---:|---:|
| オーガポンみどりのめんex → イワパレス | 390 | **379** |
| カミツオロチex → イワパレス | 219 | **217** |
| オーガポンみどりのめんex → メガガルーラex | 65 | 2 |
| カミッチュ → イワパレス | 50 | 4 |
| メガニウム → イワパレス | 30 | 0 |

**敗因内訳（96敗）**:

| 終局理由 | 件数 | 割合 | 平均ターン |
|---|---:|---:|---:|
| サイドを取り切られた | 52 | 54.2% | 19.4 |
| **山札切れ** | **41** | **42.7%** | 21.0 |
| 場のポケモン切れ | 3 | 3.1% | 8.0 |

山札切れ 42.7% は、7月の marnie 対 crustle（17.6%、`docs/rl/crustle-matchup-2026-08-06.md`）の**2.4倍**。
0打点で殴り続けて長期戦になり、自滅している。

#### 勝ち筋が唯一であることの証明（層別）

| 層 | n | 勝率 |
|---|---:|---:|
| メガガルーラex を **2体** 倒せた | 3 | **100%** |
| 1体 | 22 | 4.5% |
| 0体 | 75 | **0%** |

**2体倒した試合は全勝、0体の試合は全敗。** 上の「Syrup Storm でメガガルーラexを2回」が
このマッチアップの勝ち筋そのものであることが実測で確定した。それが100試合中3回しか起きていない。

#### 盤面到達率（もう一つの欠陥）

| カード | 到達 | 中央値ターン |
|---|---:|---:|
| オーガポンみどりのめんex | 98/100 | 1 |
| カミッチュ | 78/100 | 5 |
| メガニウム | 63/100 | 8 |
| **カミツオロチex** | **50/100** | 8.5 |

**メガガルーラexをワンパンできる唯一のカードが半分の試合で場に出ていない。**
ただし両方揃った36試合でも勝率8.3%なので、**盤面構築だけでは足りず、狙う相手を変える必要がある**。

#### 対策の優先順（この対面）

1. **0打点の攻撃を選ばせない**——`resolve_damage()==0` のアタッカーで、正の打点を出せる
   選択肢が存在するときは選ばない。評価器は既に正しいので**ルールベースのガードで済む**。
   全攻撃の59.5%が対象なので、効果は大きいはず（未検証）。 → **【実装・実測済み】§5-5**
2. **ベンチのメガガルーラexを狙う**——ボスの指令×2、キチキギスex の Cruel Arrow
   （ベンチに100。イワパレスの特性は自分自身しか守らない）。1手先の評価では出てこない。
   → **【実装・実測済み】§5-5**
3. **カミツオロチex ＋ メガニウムの到達率**（50% / 63%）を上げる。 → **§5-5 の実測で
   次の最優先ボトルネックと判明**
4. これらを入れてから **step 3 の対面特化RL**。順序を逆にすると、0打点で殴る方策の上に
   RLを重ねることになる。

> **注（勝率の再現性）**: 同一設定で `run_league` は 12/100、この診断は 4/100（2標本検定 p=0.037）。
> `cwd` による config 差は否定済み（`load_config('abl_5_full')` は同一）。
> `run_match.py` の docstring どおり**ネイティブエンジン内部のシャッフルは `random.seed()` で
> 制御されていない**ため、同じ seed でも別の試合列になる。2つは反復ではなく独立標本。
> 上の攻撃内訳・敗因内訳・層別はいずれも勝率の推定値に依存しない。

#### 教師データにこの対面がほとんど無い

カミツオロチの教師 551 episode のうち **crustle 戦は 34（6.2%）**
（marnie 143 / alakazam 105 / dragapult 69 / froslass 64）。
しかも正解手（ボスの指令→ベンチのメガガルーラex）はこの対面でしか意味を持たない稀な行動で、
BC の損失は残り94%の局面に支配される。**BCがこれを学べる道理がない**——§4-1 の
「対面別ファインチューンは効かない」と整合する（同じ教師をなぞるだけでは新情報が入らない）。

### 5-5. 【2026-08-12 実装・実測】ルールベースガード（wall_guard）— 効果は限定的、次のボトルネックを特定

> **状態: 実装済み・既定オフ。** `sample_submission/ptcg_ai/search/wall_guard.py`（新規）＋
> `ml_policy_agent.py::_select_action` への配線。カード ID をハードコードせず
> 「解決打点が0」「サイド価値」で判定するため、他の壁デッキ（117 いしずえのめん / ルカリオ）にも
> 汎化する設計。既定は無効（`DEFAULTS = {"enabled": False}`）、`abl_5_full_wallguard` という
> 新configでのみ有効。既存の `abl_5_full`（本番）は無変更。

**Guard A**（0打点で突っ立たない→ベンチに撤退）と **Guard B**（壁の向こうのメガガルーラexを
ボスの指令で突く）の2本。実装1回目は Guard A が `pipeline` 探索の後ろに配線されており
**到達不能**（`guard_a_checks=0`）というバグがあり、修正して再実装・再測定した。

**注意**: 修正版の1回目の再測定（48/100・0打点率46%）は**対戦相手が `rule_based`
（crustleの学習済みAIではない）という設定ミス**によるもので、無効と判断し破棄した。
以下は `ml_policy` vs `ml_policy`（crustle学習済み重み）という正しい構成での実測。

| 指標 | baseline | Guard A死んでいた版 | **Guard A修正版** |
|---|---:|---:|---:|
| 勝率 | 4/100 [1.6,9.8] | 10/100 | 14/100 [8.5,22.1] |
| 0打点率 | 62.8% | 60.8% | **61.0%（ほぼ不変）** |
| Guard A 発火 | 0 | 0 | **34**（チェック2,571回中＝1.3%） |
| Guard B 発火 | — | 47 | 51 |
| 平均獲得サイド | 0.95枚 | 1.65枚 | 1.70枚 |
| Kangaskhan撃破 0/1/2+ | 75/22/3 | 56/38/6 | 62/26/**12** |

**Guard A は正しく動くようになったが、0打点率への効果はほぼ無い。** 発火条件（ベンチに
正の打点を出せるポケモンがいる）を満たす場面自体が2,571チェック中34回（1.3%）しかない。
これは**§5-4 の盤面到達率（カミツオロチex 50%・メガニウム63%）が低いこと**と符合する
——「殴らず退く」判断が直っても、退く先の駒自体が場にいないことが多い。

**改善の大部分は引き続き Guard B 由来。** Kangaskhan 2体撃破（＝勝率100%の層、§5-4参照）が
3→6→**12** と単調に伸びている。

**回帰確認**（同じ正しい構成、dragapult/marnie、それぞれ n=100）: 勝率75/100→75/100、
68/100→67/100 でいずれも変化なし。壁クラスタ以外への悪影響は無い。

**次の最優先課題**: カミツオロチex・メガニウムの盤面到達率を上げること（マリガン運用・
サーチカードの優先度など）。手段（撤退/ボスの指令の判断）を直しても、その手段を実行する
ための駒が場にいなければ発火しようがない。

### 5-6. 【2026-08-13 実装・実測】BC が複数選択を一切学習していなかった

> **状態: 実装済み・効果確認済み。** §5-5 が特定した「盤面到達率の低さ」の直接原因。

**原因**: `extract_policy_dataset.py` が `maxCount > 1` の決定点を**全部スキップ**していた
（`multi_select` スキップ理由）。カミツオロチの教師データでこれは全決定点の**14.5%**、しかも
**TO_HAND（サーチ）34.8%・DISCARD（Ultra Ballのコスト）** に集中——つまり**進化ラインの
組み立て方そのものを、BCは一度も見たことがなかった**。教師の Stage2到達率（rank≤50・
78.7%/89.2%）と AIの到達率（50%/63%）の差は、これで説明がつく。

**修正**: 3ファイルの変更。
1. `extract_policy_dataset.py` — 複数選択行を `chosen_indices`（選ばれた集合）付きで採用。
   既存の `chosen_index` フィールドは維持（後方互換）。
2. `policy_net/build_features.py` — 選ばれた集合を npz に追加（`chosen_mask`）。
   既存配列は不変。
3. `policy_net/train.py` — 損失をリストワイズ softmax の複数正解版に拡張
   （`-w * mean(logp[c] for c in chosen_set)`、**集合サイズ1で既存の損失と完全一致**）。
   推論側（`ml_policy_agent.py::_greedy_multi_select`）は無変更——各選択肢を独立スコアする
   構造は元から正しく、欠けていたのは学習だけだった。

**後方互換の検証**（複数選択行を除いて再学習）: `state_features` がバイト単位で完全一致
（`np.array_equal` True）、test top-1 **0.65754039497307** / val top-1
**0.6585778781038375** / test nll **0.9875202487105975** / 16 epochs、
本番の学習run（`kamitsu_tera_s42`）と全桁一致。リファクタは挙動を完全に保存している。

**エンコーダの検証**（最大のリスク——選択肢を区別できなければ意味がない）:
`TO_HAND`/`DISCARD`/`SKILL_ORDER`/`TO_DECK`/`SETUP_BENCH_POKEMON` の option 特徴を
実データで確認。`card_id` embedding ＋ HP/種族/exフラグ等の連続特徴で**選択肢は区別できている**
（例: Ultra Ball探索でメガニウム vs チコリータが HP・stage・exフラグで異なるベクトルになる）。
`TO_HAND` の22%が「card_id未解決」に見えたが、追跡した結果**全件がサイド落としの選択**
（`players[i].prize` は伏せられており、ゲーム上そもそも本人にも見えない非公開情報）——
エンコーダのバグではなく、学習不能な部分が混じっているだけで実害はない。

**実測（100試合、crustle戦、`abl_5_full`、複数選択学習込みの新重みのみ差し替え）**:

| カード | 修正前 | **修正後** | 教師（rank≤50） |
|---|---:|---:|---:|
| **Hydrapple ex(150)** | 50%・8.5T | **77%・7T** | 78.7%・5T |
| メガニウム(710) | 63%・8T | **73%・8T** | 89.2%・5T |
| カミッチュ(93) | 78%・5T | 92%・4.5T | — |
| オーガポンex(96) | 98%・1T | 100%・1T | — |

**Hydrapple ex（メガガルーラexを唯一ワンパンできるカード）が教師水準にほぼ到達**
（50%→77%、教師78.7%）。メガニウムも63%→73%と改善（教師89.2%までまだ差はある）。
到達の速さ（中央値ターン）はまだ教師より2-3ターン遅い。

**単一選択の回帰チェック**: test top-1 **0.6625**（旧基準0.6575から+0.5pt、悪化なし）、
val top-1 **0.6703**（+1.2pt）。**複数選択の precision@k**: test 0.676 / val 0.678 —
チャンスレートより明確に高く、学習が機能している証拠。

**勝率**: 4/100（旧基準4/100と同値、ネイティブ乱数の非決定性により参考程度）。
盤面到達率が大きく動いたのに勝率が動かなかったのは、§5-4で特定した「0打点攻撃62.8%」
「ex→イワパレスへの空振り」がこの修正の対象外だったため——**BC修正と wall_guard(§5-5)は
別の欠陥を直しており、両方を重ねて初めて勝率に効くと見込まれる**（未検証）。

**次のアクション**: (a) `abl_5_full_wallguard` の重みをこの新BCに差し替えて重ね掛け効果を測る、
(b) 相手プールの他4アーキタイプにも同じ修正を適用するか判断（複数選択の比率はデッキ依存）、
(c) 到達の速さ（中央値ターン）がまだ教師より遅い点はサーチの優先順位（Ultra Ball vs Bug
Catching Set のどちらを先に切るか）の学習不足の可能性があり、要追加調査。

### 5-7. 【2026-08-13 実測】wall_guard × 新BC の重ね掛け — 単純な足し算にはならなかった

> **状態: 実測済み。原因未特定のまま次段階へ。** `abl_5_full_wallguard` の agent A 重みを
> `policy_weights_kamitsuorochi_ex_multisel_s42.json` に差し替え、100試合・crustle戦・エラー0で計測。

| 指標 | baseline（旧BC・guard無） | wall_guard単独 | 新BC単独（guard無） | **重ね掛け** |
|---|---:|---:|---:|---:|
| 勝率 | 4/100 [1.6,9.8] | 14/100 [8.5,22.1] | 4/100 [1.6,9.8] | 8/100 [4.1,15.0] |
| 0打点率 | 62.8% | 61.0% | 未測定 | **63.0%（改善なし）** |
| Hydrapple ex 到達率/中央値ターン | 50%・8.5T | 未測定 | **77%・7T** | **65%・7T（新BC単独より-12pt）** |
| メガニウム 到達率/中央値ターン | 63%・8T | 未測定 | 73%・8T | 80%・9T（改善） |
| Kangaskhan撃破 0/1/2+ | 75/22/3 | 62/26/12 | 未測定 | 57/35/8（2体撃破が-4） |
| guard_a／guard_b 発火 | 0/0 | 34/51 | n/a | 32/65 |
| 平均獲得サイド | 0.95 | 1.70 | 未測定 | 1.70 |

**単純な足し算にはならなかった。** 特に **Hydrapple ex 到達率が新BC単独(77%)より重ね掛け(65%)
で12pt低い**のが未解決の疑問点。仮説（未検証）: guard A の撤退判断が、新BCの展開計画と
噛み合っていない可能性。ただし全指標のCIが広く、100試合の標本ノイズ（ネイティブ乱数が
非決定的、§5-4の注参照）で説明できる範囲を出ない可能性もある。**原因の切り分け
（同一seed列で新BC単独 vs 重ね掛けを直接比較する等）は未着手。**

**この時点でstep 0（相手プールBC・評価器修正・ルールベースガード・BC複数選択）は
実質的に完了とみなす。次段階の方針をユーザーに確認（2026-08-13）:**
**「step1（測定欠陥7件）を先に直してからRLに入る」「対面特化RLは crustle（壁クラスタ）から
着手」の2点が決定。** 7月に測定欠陥を放置してRLが機能しなかった前例があるため、
step1を飛ばさない。wall_guardと新BCの相互作用の原因調査は保留し、必要なら後で再訪する。

### 5-8. 【2026-08-13 実測】新BC(multisel) vs 旧BC(本番) ミラー戦 — 有意差なし、悪化なし

`eval_mirror_symmetric.py`（役割バイアスなしの直接比較、§8-4手順どおり6並列×200試合）で
`policy_weights_kamitsuorochi_ex_multisel_s42.json`（新BC）と
`policy_weights_kamitsuorochi_ex_tera_s42.json`（旧BC＝現行本番と同一）をL3ミラーで対戦させた。

| | 新BC勝率 | 95%CI |
|---|---:|---|
| 全体（n=1200、エラー0） | 52.25%（627/1200） | [49.4%, 55.1%] |
| 先攻時（n=600） | 56.5%（339/600） | [52.5%, 60.4%] |
| 後攻時（n=600） | 48.0%（288/600） | [44.0%, 52.0%] |

**vs 0.5 で有意差なし**（z=1.56, p=0.119）。6分割の勝率は45〜56%とばらつき（stdev 3.8pt）、
ネイティブ乱数の非決定性の範囲内。ただし**点推定は一貫して新BC優勢、有意な悪化は無い**。
先攻時のみCIが0.5を上回る（56.5% [52.5,60.4]）。

これは§5-6のcrustle戦での盤面到達率改善（Hydrapple ex 50%→77%等）と矛盾しない。
ミラー戦は「総合的な強さ」を見る粗い指標で、複数選択学習（進化ラインの構築）の効果は
壁デッキのような特定の相性でこそ顕著に出るため、一般ミラー戦で悪化していないことが
確認できれば十分。**新BCを引き続き採用する根拠は保たれている。**

### 5-9. 【2026-08-13 移植】rough_predictor のオーガポン認識バグ修正を取り込み

`feature/gen2-bc-data-update` ブランチ（チームメンバー Showgo1130、commit `e195121`）から
`sample_submission/ptcg_ai/opponent_modeling/rough_predictor.json` の修正をcherry-pick。
エンコーダのバージョン差分（166次元 vs 現行715次元）とは独立な設定ファイルのため
コンフリクトなく適用できた。

**原因**: オーガポンみどりのめんex(96)を3アーキタイプ(カミツオロチex/オリーヴァex/
オーガポン本家)が`shared_anchor`(7点)で共有していたが、確信に必要なしきい値が不揃い
（本家14点 vs 他10点）。オーガポンが見えると、本家より「相方として採用しているだけの
デッキ」が上位に来る構造だった。

**修正内容**: `ogerpon_teal_ex`のconfident_score 14→10、オーガポンの役割を`shared_anchor`→
`core`/`anchor`に格上げ、現行型専用札（テラスタルオーブ/エネルギー回収/ブライア）を追加、
非該当カード（タケルライコex/ナゲツケサル）を除外。

**自分たちのリプレイ300戦で再検証**（`evaluate_rough_predictor.py`も合わせて移植）:

| 指標 | 修正前 | 修正後 |
|---|---:|---:|
| オーガポンtop1（T4/T8/T12） | 22%/28%/17% | **61%/56%/67%** |
| 誤確信「オーガポン→タケルライコex」 | 9/12/5件 | 3/2/1件 |
| 全体top1正解率 | 96.0〜96.8% | 97.1〜97.8% |

他アーキタイプへの悪影響なし。ユニットテスト21件PASS。**本番`rough_predictor.json`に適用済み。**
これは要件書step4（相手によるモデル切替の判定にrough_predictorを使う計画）に直接活きる。

---

## 6. 段階計画

### step 0 — 相手プールの BC を現エンコーダで学習し直す（**RLの前提。最優先**）

**いま「BCのAI同士を戦わせる」ことができない。** 相手側の重みが全部次元ガードで弾かれる:

| アーキタイプ | 次元 | is_ready | 新コーパスの教師 |
|---|---:|---|---:|
| alakazam | 715 | **True** | 3,336 |
| marnie_grimmsnarl_ex | 251 | False | 11,625 |
| crustle | 251 | False | 1,860 |
| dragapult_ex | 251 | False | 1,216 |
| mega_froslass_ex | — （未学習） | — | **880** |
| ogerpon_teal_ex | 251 | False | 678 |
| mega_lucario_ex | 166 | False | 441 |

**1体あたり約25分**（extract 20分＋build 1分＋train 3分）。カミツオロチと同じ手順（§7）。
優先順は実メタ比率順に **marnie → dragapult → froslass → crustle → lucario**。
`mega_froslass_ex` は旧コーパスに2件しか無く**今回はじめて学習できる**（実メタ9.6%）。

完了したら `rl/pools.py:43-70 LEARNER_REGISTRY` と `:85 POOL8`、
`encoder.py:63-77 POOL8_ARCHETYPES` を現メタに合わせて更新する
（`pools.load_extra_registry()` で JSON から追加もできる）。

### step 1 — Phase E の測定欠陥を直す（**これを飛ばすと7月の再現になる**）

roadmap §1 と §5 Phase E に列挙済み。最低限:

| 項目 | 現状 | 変更 |
|---|---|---|
| checkpoint | 200試合evalのbest | **最終イテレート**（winner's curse を消す） |
| iters | 60 | **25**（20〜25で飽和） |
| Critic 初期化 | ランダム（`train_pool.py:240`） | **学習済み価値関数から**。最大の欠陥 |
| Critic warmup | なし | **方策固定で5反復** |
| 報酬 | 終局のみ | **サイド差のポテンシャルシェーピング** `Φ = c(相手残り−自分残り)`, c≈0.1、`Φ(終局)≡0` を強制 |
| 正則化 | entropy のみ | **+ KL(π‖π_BC)** β=0.01〜0.05。crustle/lucario が RL で悪化したのはこれが無いため |
| train/eval プール | 同一 | **分離** |
| 評価 | 勝率 pt | **Elo差**（`173.7 × logit(p)`）。検定統計量はサイド差 |

**【2026-08-13 実装・検証済み】** 7件すべて `train_pool.py`/`train_v3.py` に実装し、
単体テスト（`test_reward_shaping.py` 等7本、全PASS）で個別に検証した。特筆点:

- **Critic初期化**: `kaggle_replays/value_net/value_weights_v251.json`（251次元、715次元の
  非ablate部分列と完全一致する教師あり学習済み価値関数）を発見し `Critic.from_value_net()` で
  転写。ランダム初期化(常に≈-0.098)と違い、サイド差に単調に応答する（優勢0.71/互角0.47/劣勢0.28）。
  **注意**: この価値関数は8/8学習＝8/12の可変打点修正より前。多少古い可能性がある
- **報酬シェーピング**: `Φ(終局)≡0` はコード上「終局遷移では常にリテラル0.0」として実装
  （特徴量から計算しない）。telescoping性質（γ=1で `adv[0]==R` に一致）を代数的に証明する
  テストまで書かれている

**Kaggle分散実行への移植（`distributed/learner.py`、2026-08-13）**: 上記7件は元々
`train_pool.py`（ローカル実行）にのみ実装されており、**Kaggle notebook経由の実行入口
`distributed/learner.py` は完全に別実装で反映されていなかった**。このままKaggleに投げると
step1の修正が空振りになるため、ユーザー判断で `learner.py` にも移植（`test_distributed.py`
拡張＋新規 `test_distributed_defects.py` で検証、全PASS）。
KL項は `learner.py` 側で温度Tを考慮した独自実装が必要だった（`train_v3.py` 版は意図的にT=1
前提——README「train_v3.pyとの違い」節を参照——そのまま流用すると温度不一致のバグを
再導入するため）。

**副次的に発見・修正**（同じく2026-08-13）: `rl/pools.py` と `rl/distributed/common.py` が
デッキパスを `01.csv` 固定にしていたバグ。`crustle/01.csv` と本番評価で使う `crustle/06.csv`
はmd5が別物（`alakazam`/`dragapult_ex` も同様）。両ファイルとも「`06.csv` があれば必ず
そちらを使う、無ければ `01.csv` にフォールバック」に修正し、md5照合で検証済み。
`marnie_grimmsnarl_ex` は元々 `06.csv` が無いため `01.csv` のままで正しい。
**`kamitsuorochi_ex` は `pools.LEARNER_REGISTRY` にまだ未登録**——crustle対面RLを実行する前に
登録が必要（次のアクション）。

### step 2 — 単腕RL で「最終 vs 初期」を判定

カミツオロチのベースBCを初期値に、step 0 のプールを相手に25反復。
**最終イテレートと初期を各1,200試合以上で比較し、有意差が出なければRLは撤退**（roadmap E3）。

現在のベースBC の対プール勝率は**まだ測っていない**。7月のデータから言えるのは
「出発点が低いほど伸びる」だけで、カミツオロチがどちら側かは未知。**step 2 の最初にこれを測る。**

**【2026-08-13 着手】crustle対面の単腕RLをKaggle分散基盤で開始。**
run `kaggle_replays/rl/runs/kamitsuorochi_vs_crustle_v1`（learner=kamitsuorochi_ex、
初期値=`policy_weights_kamitsuorochi_ex_multisel_s42.json`、opponent=crustle
`policy_weights_crustle_v715b_s42.json`、games_per_worker=250）。

**Kaggle投稿は5回目でようやく成功。** 原因を段階的に特定・修正（§9-4に詳細記録予定）:
1. 診断セルが既定で166次元の古い重みを参照→dim不一致でPoolが無限再spawn（見かけ上30分ハング）
2. `git archive HEAD`は未コミットファイルを送らない構造的欠陥→本番`policy_weights.json`
   （166→715次元に再学習済みだが未コミット）が古いまま送られていた
3. 上記2の対策でoverlay機構を入れたが、zip内に同名エントリが2つ残り、Kaggle側のDataset
   自動展開ロジックが（ローカルPythonのzipfileとは逆に）**古い方**を採用していた
   →zip構築を「重複ゼロを保証」する設計に変更

修正の副産物として `collect_parallel.py` に fail-fast preflight（重みのdim不一致等を
即座に検知）、診断セルのtimeoutを1800秒→180秒に短縮。以後の投稿は数分で結果が分かる。

**gen0完了**: 250試合・決定点18,688件、Kaggle実行408秒。
**gen0→gen1 PPO更新完了**: critic_init=value_net（value_weights_v251.json、
feature coverage 0.351）、critic warmup 5回（val_loss 0.0407→0.0329）、
kl_bc=0.0015（π_BCから大きく逸脱していない）。

**【2026-08-14 完了】25反復完走 → E3判定、結論「有意差なし・RL撤退」。**
`eval_e3_gate.py`（`learner.py`の`evaluate()`をそのまま再利用、温度0.01でcrustleに対し
各1,200試合、独立2標本）:

| | 勝率 | 95%CI | Elo差 |
|---|---|---|---|
| 初期（model_v0、BC） | 8.25%（99/1200） | [6.82%, 9.94%] | -418.4 |
| 最終（model_v25、25反復RL） | 9.25%（111/1200） | [7.74%, 11.02%] | -396.6 |

勝率差 +1.00pt、Elo差の差 +21.8。**2標本比率検定 z=0.867, p=0.386 — 有意差なし。**
roadmap の事前登録基準（有意差が出なければRLは撤退）どおり、**この対面のRLは撤退**と判定する。
CIが大きく重なっており（8.25%も9.25%もほぼ同じ「crustleにほぼ勝てない」状態）、
`collect_winrate`が25世代通じて0.028〜0.096の範囲で明確な上昇トレンドを見せなかった
（§train log参照）ことと整合する。**単腕PPO・25反復・現在の報酬設計では、crustle対面の
構造的な弱さ（§5-4で特定した「0打点の攻撃を選ぶ」等の評価器無視の癖）は直らなかった。**
step 4 のレジストリでは `"crustle": null` のまま（＝汎用モデルにフォールバック）が
実測に基づく正しい状態になる。

**根本原因、コード確認で確定（2026-08-14）**: `kaggle_replays/rl/collect_parallel.py:119-140`
（`_play_one`、`worker.py`の収集と`learner.py`/`eval_e3_gate.py`の評価が両方これを使う唯一の
rolloutループ）は `pm._forward()`/`pm.score_options()` の生スコアだけで行動を決めており、
`wall_guard`は一度も呼んでいない（`grep -rl wall_guard kaggle_replays/rl/*.py
kaggle_replays/rl/distributed/*.py` はヒット0件）。つまり**この対面のRL训练・評価は
25反復すべて、§5-4で確定した「攻撃の59.5%が0打点」の欠陥を一度も踏まえずに行われた**。
本番（`ml_policy_agent.py`）は `wall_guard.guard_a_retreat`/`guard_b_boss_orders` を
**盤面から直接判定するpre-step**（`obs`だけが引数、`PolicyModel`のスコアは不要、
`:361-378`/`:347-359`）として呼んでいるため、RL側の rollout ループに同じ2関数を
同じ位置（行動を決める直前、pre-stepとして）で足すだけで揃えられる——設計変更は不要、
配線するだけ。**初期(8.25%)も最終(9.25%)もほぼ同じ低勝率だったのは、両方とも
同じ0打点問題の上で評価していたから、という説明と矛盾しない。**

**【2026-08-14 実装】wall_guardを配線してcrustleで再挑戦、進行中**:
`kaggle_replays/rl/collect_parallel.py::_play_one`のlearner側分岐に、本番
（`ml_policy_agent._select_action`）と同じ呼び出し順序（pending→Guard A→Guard B、
盤面`obs`だけで判定するpre-step、モデルのスコアは使わない）で3関数を追加。相手側
（crustle）には適用しない（wall_guardは自分の0打点攻撃を直すためのもので相手の忠実さを
変える理由が無いため）。ガードの発火はモデルのサンプリングではないので `steps`（PPO学習対象）
には積まない。例外は`ml_policy_agent.py`の`_try_wall_guard_*`と同じく握りつぶし、1試合が
丸ごとerror扱いで捨てられないようにした。

**配線の動作確認**（20試合、単一プロセスで`wall_guard.get_stats()`を直接確認）:
`guard_a_checks=349 guard_a_fired=2` / `guard_b_checks=90 guard_b_fired=9` /
`pending_checks=11 pending_consumed=11 pending_dropped=0`（孤立したpending状態なし）。
発火が0でないことを確認済み——配線ミスで常にNoneを返しているだけ、ではない。

新run `kamitsuorochi_vs_crustle_wallguard_v1` を作成（`model_v0`は旧runと同じ
`policy_weights_kamitsuorochi_ex_multisel_s42.json`、sha256=`5d0cae0fbbcbbf7c...`で
一致——初期重みは変えず rollout の仕組みだけを変えた比較になっている）。
worker-idは`local`（Kaggleを介さない）。gen0から25反復のループを開始（進行中）。
旧run（`kamitsuorochi_vs_crustle_v1`、wall_guardなし、E3「有意差なし」）は削除せず
比較対象として保持する。

**【2026-08-14 完了】E3再判定 → 結論「有意に改善・RL採用」。仮説（wall_guard配線漏れが
主因）は正しかった。**

`kamitsuorochi_vs_crustle_wallguard_v1`、gen0→gen25完走（PC再起動を1回挟んだが
`run.json`の世代番号で機械的に再開、データ欠損なし）。`eval_e3_gate.py`で再測定
（初期model_v0・最終model_v25とも今回はwall_guard込みのrolloutで評価——「初期」も
旧run（wall_guardなし）とは別物である点に注意）:

| | 勝率 | 95%CI | Elo差 |
|---|---|---|---|
| 初期（model_v0、BC＋wall_guard） | 11.25%（135/1200） | [9.58%, 13.16%] | -358.8 |
| 最終（model_v25、25反復RL＋wall_guard） | 14.33%（172/1200） | [12.46%, 16.43%] | -310.6 |

勝率差 +3.08pt、Elo差の差 +48.2。**2標本比率検定 z=2.261, p=0.0237 — 有意に改善。**
roadmapの基準どおり、**この対面のRLは採用**。

**旧run（wall_guardなし）との比較で分かること**:
| | 初期勝率 | 最終勝率 | 有意差 |
|---|---:|---:|---|
| wall_guardなし（`_v1`） | 8.25% | 9.25% | p=0.386（なし） |
| wall_guardあり（`_wallguard_v1`） | 11.25% | 14.33% | p=0.0237（あり） |

wall_guard自体が初期勝率を8.25%→11.25%に押し上げ（§5-5の単体測定と整合）、**かつ**
その土台の上でRLがさらに有意に伸ばした（+1.00pt/n.s. → +3.08pt/p<0.05）。§5-4で
想定した「評価器の嘘（0打点攻撃）を先に直してからRLを重ねる」の必要性が、今回
数値で裏付けられた形。

**ただし過大評価はしない**: 14.33%はまだ大きく負け越す対面（Elo差-310.6）。
「crustleに勝てるようになった」ではなく「crustleへの負け方が有意に改善した」が
正確な言い方。壁を完全に崩せてはいない。

**次**: `model_v25.json`（`kamitsuorochi_vs_crustle_wallguard_v1`）を
`policy_weights_kamitsuorochi_ex_vs_crustle_rl.json`として確定し、step 4の
`opponent_model_registry.json`に登録する（crustleが初の実エントリになる）。
ogerpon/lucarioへの展開は、**このrolloutループの修正（wall_guard配線）を引き継いで
同じレシピで進める**——対面共通の欠陥だったので再度踏む心配はない。
25反復・shaping-c/kl-betaのハイパーパラメータ調整は、今回E3が通ったため優先度を下げる。

### step 3 — 対面特化RL

step 2 が通ったら、伸びしろの大きい順に:

1. **壁クラスタ**（crustle / ogerpon / lucario、実メタ12.5%）← **最優先に変更（2026-08-12）**。
   §5-4 で crustle 対面 12%（教師53.3%）の原因を実測で特定した。CIが教師と重ならない唯一の対面。
2. **alakazam 対面**（実メタ19.6%・勝率28%）← 降格。教師35.9%[0.268,0.461] と AI 28%[0.201,0.375] の
   **CIが大きく重なる**ため、AIの弱点とは言えない（デッキ相性の可能性が高い）。
3. B クラスタ（実メタ48.2%だが既に64〜80%。dragapult は AI 80% > 教師 52.2% で既に上回っている）

§5-2 のテラスタル修正と、**§5-4 の対策1（0打点の攻撃を選ばせないガード）・対策2（ベンチの
メガガルーラexを狙う）を先に入れておくこと。**
**評価器が嘘をついたままRLを回すと、嘘に最適化された方策ができる。**
§5-4 の場合は評価器は正しく、**行動選択が評価器を無視している**——0打点で殴る方策の上に
RLを重ねると、その癖ごと強化される。

**個別重みのレジストリ化（step 4 と接続）**: 各対面のRLが roadmap E3 ゲート（最終 vs 初期、
1,200試合以上、有意差）を通過したら、その世代の `models/model_v<final>.json` を
`sample_submission/ptcg_ai/learning/policy_weights_kamitsuorochi_ex_vs_<opponent>_rl.json`
という固定名にコピーし、step 4 の `opponent_model_registry.json` に登録する。**通過しなければ
登録しない**（レジストリに無い＝汎用のまま、が既定の安全側。§5-7 で wall_guard×新BC の
組み合わせが単純加算にならなかった前例があるため、「個別RLが有効」は対面ごとに実測で確認した
ものだけ載せる）。

**進捗（2026-08-13〜14、完了）**: `kamitsuorochi_vs_crustle_v1`、gen0→gen25 完走。
`collect_winrate` は0.028〜0.096で推移し明確な上昇トレンドは無し。**E3判定は「有意差なし・
撤退」**（結論と数値は §step2 参照）。**Kaggle経由の投入はこの対面では以後使わないことにした**——
gen3投入中にセッション切断が起き、`push_kaggle.py` が再接続後に `.kaggle_stage/out/` に
残っていた**1世代前の古いダウンロードキャッシュ**を新しい世代のものとして
`shards/v3/kaggle.npz` に上書きコピーする不具合を実際に踏んだ（`learner.py` の世代ハッシュ照合
が正しく弾いたため実害は無し、trap として §9 に記録）。250試合1世代がローカルで約160〜170秒
で終わるのに対し、分散基盤側のオーバーヘッド（git archive/zip構築/Dataset反映待ち90秒/
カーネル起動）の方が重く、往復も不安定だったため、**この対面（および今後の対面特化RL）は
`worker.py`/`learner.py` をKaggleを介さずローカルで直接ループさせる**方針に変更。
（分散基盤自体は Colab 等で計算資源を足したくなったときのために残す。）

**【2026-08-14 着手】ogerpon_teal_ex・mega_lucario_ex の対面特化RL、進行中（一旦停止）**

まず両者の対戦相手重みを715次元に再学習（§8-2 のレシピをそのまま適用、crustleと同じ）:

| アーキタイプ | 決定点 | test top1 | 出力 |
|---|---:|---:|---|
| ogerpon_teal_ex | 44,728（678 episode-player） | 0.6549 | `policy_weights_ogerpon_teal_ex_v715_s42.json` |
| mega_lucario_ex | 29,945（441 episode-player） | 0.5876 | `policy_weights_mega_lucario_ex_v715_s42.json` |

両方とも自己検証PASS（最大誤差 1e-15 台）、`is_ready=True` を確認済み。

run はそれぞれ `kamitsuorochi_vs_ogerpon_teal_ex_v1` / `kamitsuorochi_vs_mega_lucario_ex_v1`
（初期重みはcrustleと同じ `policy_weights_kamitsuorochi_ex_multisel_s42.json`、
sha256=`5d0cae0fbbcbbf7c...`で一致）。**wall_guardの配線は`collect_parallel.py`本体に
入っているので、新しいrunは何もしなくても最初からwall_guard込みで収集・評価される**
（crustleのときのような後からの配線漏れ修正は不要）。

**gen0のcollect_winrateがcrustleと大きく違う**（同じ壁クラスタでも対面ごとに難易度が
全く異なることの実測）:

| 対面 | gen0 collect_winrate |
|---|---:|
| crustle（wall_guardあり、参考） | 11.25%（E3評価時、温度0.01） |
| **ogerpon_teal_ex** | **84.0%**（250試合中210勝、温度1.0） |
| **mega_lucario_ex** | **32.8%**（250試合中82勝、温度1.0） |

ogerponは既にBCの時点でかなり勝っている可能性がある（25反復後のE3判定で「有意差なし」に
なる可能性も込みで見ておく——伸びしろが小さい対面をRLでさらに伸ばすのは難しくて当然）。

**Kaggle並行投入のためのインフラ整備**:
1. `push_kaggle.py` の `DATASET_SLUG`/`KERNEL_SLUG` が定数だったため、複数runを同時に
   Kaggleへ投げると同じDataset/Kernelを取り合って競合すると判明。`--dataset-slug`/
   `--kernel-slug`（`push_train_pool.py`と同じ流儀）に変更し、run ごとに別スラグを渡せば
   真の並行実行ができるようにした。`kaggle_worker.ipynb` 側はもともと `/kaggle/input/**/` の
   glob でファイルを探す設計だったため**変更不要**（スラグ非依存）。
   ogerponは既定スラグ、lucarioは `ptcg-distributed-selfplay-lucario`/
   `ptcg-worker-kaggle-lucario` で実際に同時実行を確認済み。
2. crustleで踏んだ「`.kaggle_stage/out/`の古いダウンロードが新世代として誤って使われる」
   不具合が、**セッション切断を伴わない正常終了でも再現**した（ogerpon gen1で実際に発生、
   `learner.py`の世代ハッシュ照合が正しく弾いたので実害なし）。手動でのクリアに頼らず
   `push_kaggle.py`自体を修正: ダウンロード前に`out/`を毎回空にし、ダウンロード直後に
   `common.check_shard()`（`learner.py`と同じ検査関数を再利用）で世代・モデルハッシュ等を
   検証してから`shards/v<gen>/`へコピーするようにした。ずれていれば`shards/`へは
   コピーされず、その場で分かりやすいエラーで止まる。
3. `kaggle_loop.py`（新規、`kaggle_replays/rl/distributed/`）: collect(push_kaggle.py)→
   learn(learner.py)を目標世代まで無人で繰り返すドライバ。**ローカルのPythonプロセスとして
   動き続ける必要がある**（Kaggle側は試合生成だけで、回収・PPO更新・次世代投入をする
   ループ本体はローカル。PCをシャットダウンすると実行中のKaggleカーネル自体はおそらく
   完走するが、回収と次世代投入は止まる。`run.json`の世代番号ベースで安全に再開できるので
   実害はない）。

**【2026-08-14 深夜、方式を切り替え】`distributed/`(worker.py+learner.py+push_kaggle.py)経由は
ogerpon 3/25・lucario 2/25まで進めて一旦停止したが、その後 `train_pool.py`+
`push_train_pool.py` 経由に切り替えて gen0 からやり直した**（2〜3反復ぶんの計算は捨てたが、
下記の理由で妥当な判断）。

**切り替えた理由**: `distributed/` の設計は「収集はKaggle、集約とPPO更新はローカル」を
1世代ごとに繰り返す構成で、**ローカルのPythonプロセスが動き続けている必要がある**
（`kaggle_loop.py`で無人ループにはできるが、それでもローカルプロセスは生きていないと
次世代の投入が止まる）。一方 `train_pool.py` は「学習側1モデル vs 固定相手」の
PPO学習ループ全体（収集も更新も）を1つのスクリプトで完結させる設計で、
`push_train_pool.py` で丸ごとKaggle Notebookとして投げると、**指定した`--iters`ぶん
（25反復）が1回のKaggleカーネル実行の中で全部終わる**。`--no-wait`で投げれば
ローカル側はpushした時点で用済みになり、**あとはKaggle側だけで完走する**
（ユーザーの「前はいちいち投げなおさず長時間放置できた」という記憶と一致する構成）。

**collect_pool.py はcollect_parallel._play_oneを直接importして呼ぶ設計**
（ロジックを複製しない、モジュールdocstringに明記）なので、今夜配線した wall_guard の
修正は追加作業なしでそのまま効く。念のため`push_train_pool.py`が作るバンドルzip
（git archiveではなく作業ツリーの全ファイルリスト方式）に、修正済み`collect_parallel.py`
（`wall_guard`呼び出し込み）と`model_router.py`/`opponent_model_registry.json`が
実際に入っていることを展開して直接確認済み。

**`pools.LEARNER_REGISTRY` も更新**（`train_pool.py --train-opponents`はこのレジストリ経由で
相手重みを解決するため）: `ogerpon_teal_ex`・`mega_lucario_ex`・`crustle` の3エントリが
旧次元（251/166/251）の壊れた重みを指したままだったのを、今夜作った715次元版に差し替え。

**踏んだ罠（`push_train_pool.py`固有、2件とも解消済み）**:
1. `--out-dir`の既定値が`notebooks/.stage_train_pool`固定で、2本同時に(オプション無指定で)
   実行するとローカルのバンドルzipを取り合って`PermissionError`になる
   （lucarioの1回目の投入で実際に発生）。2本目には`--out-dir .stage_train_pool_lucario`の
   ように別ディレクトリを明示することで回避。
2. `--dataset-wait`の既定90秒では、180MB級のバンドルzipのDatasetへの反映が間に合わず
   `Notebook を push`まで到達せずに終了することがある（ogerponの1回目の投入で実際に発生、
   バージョン26のまま更新されなかった）。`--dataset-wait 240`で解消（29〜28秒で反映され、
   実際にはそこまで待たなかったが、余裕を見て240を指定）。

**投入先**（両方 `--no-wait`、`kaggle kernels status <slug>` で "RUNNING" を確認済み）:
| 対面 | Kernel | Dataset |
|---|---|---|
| ogerpon_teal_ex | `koshin953/ptcg-train-pool-run` | `koshin953/ptcg-train-pool` |
| mega_lucario_ex | `koshin953/ptcg-train-pool-run-lucario` | `koshin953/ptcg-train-pool-lucario` |

**確認方法**（ローカルプロセスは不要、いつでも軽いCLI呼び出しだけで見に行ける）:
```bash
kaggle kernels status koshin953/ptcg-train-pool-run
kaggle kernels status koshin953/ptcg-train-pool-run-lucario
```
`COMPLETE`になったら`kaggle kernels output <slug> -p <出力先>`で学習済み重み一式を回収する
（`push_train_pool.py`の`--no-wait`を外して再実行すれば待って自動回収もできる）。
25反復完走後は`eval_e3_gate.py`相当の手順（またはtrain_pool.py自体の最終評価ログ）で
E3判定を行う（crustleと同じ考え方、§step2参照）。

**（参考、もう使わない）`distributed/`経由でogerpon 3/25・lucario 2/25まで進めた分は
`kamitsuorochi_vs_ogerpon_teal_ex_v1`/`kamitsuorochi_vs_mega_lucario_ex_v1`のrunディレクトリに
残したまま（削除していない）。`kaggle_loop.py`は「ローカルプロセスを保つ必要がある」という
制約自体は解消されないので、今後この対面特化RLは基本的に`push_train_pool.py`経由を使う。**

### step 4 — 相手によるモデル切替【2026-08-13 kamitsuorochi_ex 向けに再設計】

旧ドラフト（A/B/C の3クラスタ表）は alakazam がメインデッキだった頃のもので、
`rough_predictor` がオーガポンを誤認識するバグ（§5-9）が直る前の設計だった。**判定に
`rough_predictor` を使う方針そのものは維持**しつつ、対応表と実装の注入点を現状に合わせて
更新する。設計のみで**未実装**（下記どれもまだコードに書いていない）。

**デッキ予測器（`HybridDeckPredictor`/21クラス）を使わない理由は変わらず有効**:
- デプロイ済み `deck_predictor_weights.json` は2026-07-19学習で事前分布が現メタと乖離
  （archaludon 13.2%対実測0.03%、marnie 6.9%対実測29.5%）
- 補正用ベースが repo に残っていない（`output/` gitignore）。再学習は 12,047リプレイ×2視点＝数時間
- 必要なのは21クラスの確率分布ではなく「個別RL済みの対面かどうか」の判別だけで、要求精度が
  桁違いに低い

→ **ルールベースの `rough_predictor`（キーカード一致、チューニング不要、§5-9で
  オーガポン認識を修正済み）で十分。** かつ `rough_predictor.json` の `archetypes` キー
  （`crustle` / `ogerpon_teal_ex` / `mega_lucario_ex` / … 全21種、`python -c` で実測確認済み）は
  `pools.LEARNER_REGISTRY` や RL run のディレクトリ名（`kamitsuorochi_vs_<archetype>_v1`）と
  **同じアーキタイプ文字列を使っている**ので、旧ドラフットの3クラスタのような独自の変換表は
  不要——`rough_predictor` が返す `deck_type` をそのままレジストリのキーとして引ける。

**注入点**（2026-08-13時点の行番号、`sample_submission/ptcg_ai/ml_policy/ml_policy_agent.py`）:
- `_get_model(config)`（`:153`）が既に `config["policy_weights_path"]` を見てパスごとに
  `PolicyModel` をキャッシュしている（`_model_cache_by_weights_path`、`:102`）。**モデル切替の
  受け口はもう存在する**——ここに渡す `policy_weights_path` を試合中に動的に決める部分が無いだけ。
- `agent()`（`:105`）の `obs.select is None` 分岐（`:140-149`、新しい試合の開始）が、
  `wall_guard.reset_pending_target()` と同じパターンでルーティング状態をリセットする場所。
- `_select_action(obs, config)` 呼び出し（`:150`）の直前が判定を差し込む場所。

**相手情報の取得は新規に作らず `match_context` を再利用する**: `sample_submission/ptcg_ai/
hidden_information/match_context.py` は既に `_knowledge: dict[int, OpponentKnowledge]` を
プレイヤー別に保持し、`update(obs)`（`ml_policy_agent.py:130` から毎ターン呼ばれ済み）の中で
`update_from_logs`→`update_from_state` の順に更新している（`:139-142`）。**ただし現状これを
外から読む公開関数が無い**（`get_own_state`/`get_opponent_state` はあるが `get_knowledge` は
無い）——`get_own_state`/`get_opponent_state` と同じパターンで `get_knowledge(player_index)`
を1つ足す必要がある（実装時のTODO）。これがあれば `evaluate_rough_predictor.py` と同じ呼び方
（`rough_predictor.predict(state, knowledge)`）で、二重に状態を持たずに判定できる。

**新モジュール案**: `sample_submission/ptcg_ai/opponent_modeling/model_router.py`
（wall_guard と同じ「モジュールグローバルの試合スコープ状態＋configで有効化」パターン）

```python
_routed_deck_type: str | None = None  # このプロセスの「今の試合」で確定したら以後固定

def reset() -> None: ...  # ml_policy_agent.py の obs.select is None 分岐から呼ぶ

def route(obs, config) -> str | None:
    if not (config or {}).get("opponent_model_routing", {}).get("enabled"):
        return None  # 既定 off。wall_guard / value_shadow_logging と同じ流儀
    global _routed_deck_type
    if _routed_deck_type is not None:
        return _routed_deck_type  # 一度切り替えたら試合中は戻さない
    knowledge = match_context.get_knowledge(obs.current.yourIndex)  # 要追加の公開関数
    result = rough_predictor.predict(obs.current, knowledge)
    if result["status"] != "confident":
        return None  # insufficient_evidence / ambiguous / no_candidate は汎用のまま
    weights = REGISTRY.get(result["deck_type"])  # opponent_model_registry.json
    if weights is None:
        return None  # 未登録＝その対面はまだ個別RLが無い(or E3で有意差なし)。汎用のまま
    _routed_deck_type = result["deck_type"]
    return weights
```

`_select_action` の先頭付近で `route()` の戻り値があれば、それを
`config = {**config, "policy_weights_path": weights}` のように差し込んでから
`_get_model(config)` を呼ぶ（既存の受け口をそのまま使う）。

**レジストリ**: `sample_submission/ptcg_ai/learning/opponent_model_registry.json`
（**2026-08-14 作成・`model_router.py`から実際に読まれる**。詳細は本セクション末尾の
「実装完了」参照）

```json
{"crustle": "policy_weights_kamitsuorochi_ex_vs_crustle_rl.json", "ogerpon_teal_ex": null, "mega_lucario_ex": null}
```

crustleがE3を通過した初のエントリ（`kamitsuorochi_vs_crustle_wallguard_v1`のmodel_v25、
sha256=`ad87b19bb5c9e11b...`、§step2参照）。他は未着手のため `null`（＝rough_predictorが
確信を持ってもレジストリに無いので汎用にフォールバック）。各対面のRLが E3 を通過した時点で
そのキーだけ `policy_weights_kamitsuorochi_ex_vs_<opponent>_rl.json`
のパスに更新する。**rough_predictor.json の21アーキタイプキー空間をそのまま使う**ので、
crustle 以外（alakazam・dragapult_ex 等）も個別RLをやれば同じ仕組みでそのまま追加できる。

**config**: `{"opponent_model_routing": {"enabled": false}}` — 既定 off。有効化は
「切替あり vs 汎用のみ」のA/Bで有意勝ちを確認してから（下記ゲート条件のまま）。

ゲート条件（旧ドラフトから変更なし）:
- **そのアーキタイプの主軸カードが実際に見えた**（＝`rough_predictor` の `status=="confident"`）
  ときだけ切り替える
- **一度切り替えたら試合中は戻さない**（プランの不連続を避ける）
- **レジストリに載る＝その対面のRLがA/Bで有意に勝った場合のみ**（載っていないアーキタイプは
  ずっと汎用方策のまま）
- 切替機構自体の有効化も **A/B で有意勝ちを確認してから**（config のフラグ、既定 off）

**【2026-08-14 実装完了】** 設計どおりに実装した:
- `match_context.py::get_knowledge(player_index)` を追加（`get_own_state`/`get_opponent_state`
  と同じフォールバックパターン）。ユニットテスト2件追加（フォールバック安全性・
  `update()`の蓄積を反映すること）、既存6件と合わせて計8件PASS。
- `sample_submission/ptcg_ai/opponent_modeling/model_router.py`（新規）。設計どおり
  `route(obs, config)` が重みパス文字列 or None を返すだけ、PolicyModelの差し替えは
  呼び出し側の責務。ユニットテスト8件（config無効時None・未確信None・確信+登録あり→パス・
  確信+未登録None・未知アーキタイプNone・試合中ロックイン・reset()で解除・current=None安全）
  全PASS。
- `ml_policy_agent.py`: import追加、`obs.select is None`分岐に`model_router.reset()`、
  `_try_model_router`（wall_guardの`_try_wall_guard_*`と同じ例外握り潰しパターン）、
  `_select_action`内の`pipeline_action`チェック直後・`_get_model()`呼び出し直前に配線
  （ルーティング結果は`policy_weights_path`だけを差し替えた別dictとして`_get_model`にのみ渡し、
  `hidden_state_source`等の他のconfig参照には影響させない）。
- `opponent_model_registry.json` を作成、`crustle`にstep2で確定した重みを登録
  （他は`null`のまま）。
- 検証用config `sample_submission/configs/abl_5_full_wallguard_routing.json`
  （wall_guard＋opponent_model_routingの両方enabled）を追加。**routingで切り替わる重みは
  wall_guard込みでRL学習・評価されたものなので、本番投入時はwall_guardも一緒に有効化する
  こと**（wall_guardなしで個別重みだけ使うと、学習・評価時に無かった0打点攻撃の判断を
  本番で初めて経験することになり、測定した効果が再現される保証がない）。
- **実ゲームでの動作確認**（手組みスクリプト、crustleデッキ相手に5試合・計472手番、
  `sample_submission/ptcg_ai/learning/policy_weights_kamitsuorochi_ex_multisel_s42.json`を
  base、`opponent_model_routing.enabled=true`）: **5試合全てturn5〜9でcrustleを認識し、
  以降ずっとcrustle専用重みを使用**（`ml_policy_agent._model_cache_by_weights_path`に
  base重みと`policy_weights_kamitsuorochi_ex_vs_crustle_rl.json`の両方がロード済みとして
  記録されていることを確認）。設計だけでなく実際に機能することを確認済み。
- `sample_submission`全テストスイート384件PASS（14 skip、既存分）。
  副産物: `test_golden_scores_match`が本件と無関係に失敗しているのを発見
  （`policy_weights.json`が2026-08-12以降に再学習されたのにgolden fixtureが未更新、
  `regenerate_policy_fixture.py --dry-run`で9局面中4局面のスコア変化を確認）、
  同スクリプトで再生成して解消（本件の変更とは独立、pre-existingの取りこぼし）。

**まだやっていないこと**: 「切替あり vs 汎用のみ」のA/B自体（config既定offのままなので
現状は本番挙動に影響なし）。ogerpon/lucarioの個別RLが済んでレジストリに複数エントリが
揃ってから、まとめて測るのが効率的。

---

## 7. デッキの立ち回り（記事＋カード原文＋実測）

出典: [ハレルヤ2](https://www.hareruya2.com/blogs/news/just-nowv4) /
[ポケざんまい](https://www.pokeca-zanmai.jp/archives/91885) / `data/JP_Card_Data.csv`

### 7-1. 理想盤面は「良い盤面の目安」ではなく打点式そのもの

```
バトル場: 150 カミツオロチex (HP330)
ベンチ:   96 オーガポン みどりのめんex ×2   ← テラスタル: ベンチにいる限りワザのダメージを受けない
          710 メガニウム ×1                ← 基本草エネを2個ぶんにする（重ならない）
盤面全体に基本草エネを蓄積する
```

**みつあめストーム = 30 + 60 ×（盤面の基本草エネ枚数）**（メガニウム込み）。
**エネを盤面に置く行為が、そのまま打点の蓄積になる。** 後半になるほど強い。

### 7-2. そこから導かれる方針（探索のソフト prior / deck_plan 候補）

| 方針 | 根拠 |
|---|---|
| **エネは可能な限りベンチのオーガポン(96)に付ける** | ベンチのテラスタルはワザのダメージを受けない＝**そのエネは絶対に失われない**。打点への寄与は付け先に依らない。**※ §5-2 の修正が前提** |
| バトル場のカミツオロチexに過剰に付けない | KOされるとそのエネ＝打点が消える |
| **メガニウムを最優先で立てる** | 全エネの価値が2倍。1体で十分 |
| エネ加速は じゅくせいチャージ(150)＋みどりのまい(96)＋手貼りで1ターン最大3枚 | 記事の「実質8枚加速」はメガニウム込みの換算 |
| **活力の森(1261)** で出したターンに進化できる | チコリータ→ベイリーフ→メガニウム を1ターンで駆け上がれる。**相手も進化できる**点に注意 |
| 序盤は カミッチュ「ともだちのわ」(20×ベンチ数) で繋ぐ | ベンチ4体で80、5体で100 |
| じゅくせいチャージは **HP30回復**も持つ | ダメカンを置くタイプ（マシマシラ等）に強い |
| 勝ち筋は **2-2-2 か 1-2-3 の3回攻撃** | 記事 |
| **リーリエの決心(1227)** は「出し切ってから」「手札が少ないとき」。8枚ドローは**サイド残6枚＝実質1ターン目限定** | カード原文 |

### 7-3. ユーザー観察への回答（2026-08-12）

「エネの付け先／盤面の展開／リーリエのタイミングが微妙。機械学習で直るか」への整理:

| 観察 | 機械学習で直るか | 理由 |
|---|---|---|
| **エネの付け先** | **直らない（表現の問題）** | 正解を示す `target_pokemon_is_likely_ko_next_turn` が、テラスタルのベンチ無敵を知らずに嘘をついている（§5-2）。BCでもRLでも学習側では直らない |
| **リーリエのタイミング** | **直る余地あり（データ量）** | 必要な入力は全部ある（手札3カウント・手札card_idスロット43種に1227を含む・サイド残り・ターン）。表現上は学習可能なので、教師量の問題 |
| **盤面の展開順** | **半分は表現、半分は探索** | `self_board_card_slot` を ablate しているので方策は**自分のベンチに誰がいるかを card_id で見られない**（「メガニウムがもう立っているか」が分からない）。ただし入れた版（C2/T1）は実測で弱い（D4）。**探索のソフトprior か deck_plan 側に入れるのが筋** |

---

## 8. 手順（コマンド）

`<REPO>` = `C:\Users\rinnz\Documents\pokemon\pokemon-tcg-ai-battle`

### 8-1. 新しい日次ダンプを取り込む（追加分が手に入ったとき）

```powershell
# ハードリンク（同一ドライブなのでディスク消費ゼロ）。命名規則が違う点に注意
$src = "C:\Users\rinnz\Downloads\<新しいダンプ>"
$dst = "<REPO>\kaggle_replays\replays"
Get-ChildItem $src -Filter *.json | ForEach-Object {
  $t = Join-Path $dst ("episode-" + $_.BaseName + "-replay.json")
  if (-not (Test-Path $t)) { New-Item -ItemType HardLink -Path $t -Target $_.FullName | Out-Null }
}
```

続けて `scratchpad/add_master_rows_archive3.py` と同じ方式で
`index/episodes_master.jsonl` に行を追加する（**必須**。理由は §9-2）。

### 8-2. アーキタイプ1体分の BC（step 0 でこれを繰り返す）

```bash
# 1) ラベル再生成（リプレイを増やしたときだけ）
cd <REPO>/kaggle_replays/deck_predictor
python extract_decks.py && python label_decks.py --min-score 10.0

# 2) 決定点抽出（約20分／12,047リプレイ）
cd <REPO>/kaggle_replays
python extract_policy_dataset.py --archetype <ARCH> \
  --out training_data/policy_positions_<ARCH>.jsonl.gz \
  --audit --audit-out policy_net/audit_<ARCH>.md

# 3) 特徴量（約620 rows/s）
cd <REPO>/kaggle_replays/policy_net
python build_features.py \
  --in ../training_data/policy_positions_<ARCH>.jsonl.gz \
  --out features_<ARCH>_v715.npz \
  --weight-scheme concentrated \
  --deck-csv ../meta_analysis/archetype_decks/<ARCH> \
  --opponent-vocab-dir ../meta_analysis/archetype_decks

# 4) 学習（約3分）※ --out-weights は必須
python train.py --features features_<ARCH>_v715.npz \
  --out-weights <REPO>/sample_submission/ptcg_ai/learning/policy_weights_<ARCH>_v715_s42.json \
  --ablate-features self_board_card_slot,self_discard_card_slot,opp_board_card_slot,opp_discard_card_slot \
  --seed 42 --hidden-size 32 \
  --metrics-out archetype_runs/<ARCH>_v715_s42_metrics.json
```

### 8-3. ファインチューン（`train.py` に追加済み、2026-08-12）

```bash
python train.py --features <features>.npz \
  --init-weights <ベース重み>.json \
  --row-mask <mask>.npy \
  --lr 1.25e-4 \
  --out-weights <出力>.json \
  --ablate-features ... --seed 42
```

- `--init-weights`: 標準化パラメータ（`state_mean/std`・`option_mean/std`）を**部分集合から再計算せず
  ベースのものをそのまま使う**。語彙・次元・ablate 指定が一致しないとエラーで停止する
- `--row-mask`: bool の .npy。**train/val/test の全splitに適用**
- クラスタ別マスクの作り方は `scratchpad/build_cluster_masks.py`（npz の `row_index` 経由で対応付け）

### 8-4. A/B と提出

```bash
# 対称ミラー戦（Windows は --workers>=2 でデッドロックするので seed0 をずらして分割）
cd <REPO>/kaggle_replays/rl
for i in 0 1 2 3 4 5; do
  python eval_mirror_symmetric.py --model-a <A>.json --model-b <B>.json \
    --archetype kamitsuorochi_ex \
    --deck-csv <REPO>/kaggle_replays/meta_analysis/archetype_decks/kamitsuorochi_ex/06.csv \
    --games 200 --workers 1 --seed0 $((10000 + i*200)) > ab/part$i.log 2>&1 &
done; wait

# 本番経路の確認（リポジトリルートで実行）
cd <REPO>
python league/run_league.py --agent-a rule_based --agent-b ml_policy \
  --config-base abl_5_full --weights-b <候補>.json --games 30 --workers 1 \
  --out league/results/<名前>.json

# デプロイ
cp <候補>.json sample_submission/ptcg_ai/learning/policy_weights.json
python kaggle_replays/policy_net/regenerate_policy_fixture.py --dry-run
python kaggle_replays/policy_net/regenerate_policy_fixture.py
cd sample_submission && python -m pytest tests -q
```

---

## 9. 罠（実際に踏んだ／踏みかけたもの）

### 9-1. `hand_card_vocab` の不一致は警告なしで手札特徴を潰す

`policy_model.py:161-173` の次元ガードは**715次元しか見ない**。語彙が完全に別物でも
`is_ready=True` で通り、`self_hand_card_slot_*` 56次元が常にほぼ全0になる。

実例: alakazam の提出重みの語彙24種と、カミツオロチ L3 の23種の**重なりは6種だけ**。
**`deck.csv` だけ差し替えるのは禁止。BC 再学習とセットで行う。**
A/B の対照群に別デッキの重みを置くのも禁止（不公平になる）。

### 9-2. `--weight-scheme concentrated` は rank が無いと重み 0.2 に落ちる

`build_features.py:228-241`: `rank_at_fetch is None → 0.2` / `<=20 → 8.0` / `<=50 → 3.0` /
`<=200 → 1.5` / `<=1000 → 1.0` / それ以外 → 0.5。

新ログは `episodes_master.jsonl` に無いので放置すると**全行が 0.2**。今回は同日相対順位を
付けて対処し、決定点の **99.4%** に rank が入った（1-50位 62.6% / 51-200位 32.5% / 201-1000位 5.0%）。

> **スナップショットへの内挿はしてはいけない。** 手元の leaderboard は 2026-07-30・上位200のみで
> 200位が 956.5点。一方 8/10 のダンプは 97.9% が 950点以上（11日でスコアが全体的に上がっている）。
> **異なる日のスコアを絶対値で比較できない。** 同日内の相対順位を使うこと。

### 9-3. その他

| 罠 | 内容 |
|---|---|
| `train.py` の `--out-weights` 省略 | **提出中の `policy_weights.json` を無警告で上書き**（`:77,593`） |
| `run_archetype_pipeline.py` | `--deck-csv` も `--opponent-vocab-dir` も渡さない → card_id 特徴が全0の npz が黙って出来る。**使わない** |
| `eval_mirror_symmetric.py` の `--deck-csv` 省略 | `01.csv`（別リスト）にフォールバック（`:151-153`） |
| `meta_analysis/extract_archetype_decks.py` | `--top-decks 5` 既定で NN.csv と manifest.json を再生成し、**手動追加した 06.csv を消す**。実行しない |
| `policy_net/evaluate.py` | パスがハードコードで使えない。オフライン指標は `train.py --metrics-out` を読む |
| `collect_pool` | 素の PolicyModel だけを呼び、リーサル探索も pipeline も通らない。本番経路の測定にならない |
| `collect_parallel._play_one` の役割非対称 | learner役だけ確率的サンプリング。必ず `eval_mirror_symmetric.py` を使う |
| features.npz の陳腐化 | 元データや `attack_features.py` を変えたら派生 npz を必ず作り直す（PASS/FAIL が反転した実例あり） |
| `collect_parallel.py`（RL収集・評価の唯一のrolloutループ）に `wall_guard` の呼び出しが無い | 2026-08-14に発覚。§5-4/§5-5の対策（0打点攻撃ガード等）は本番(`ml_policy_agent.py`)にしか配線されておらず、RLは常にそれ抜きで学習・評価していた。詳細と修正は step 2「根本原因、コード確認で確定」 |

### 9-4. `push_kaggle.py` はKaggle側の出力が更新されなくても古いダウンロードを黙って使い回す

**2026-08-13、crustle対面RLのgen3投入で実際に踏んだ。** セッション切断からの再接続後、
`.kaggle_stage/out/kaggle.npz`（gen2投入時にダウンロード済みの古いシャード、`model_sha256`が
model_v2のもの）が残っている状態で gen3 を再投入した。ダウンロードステップ自体は完走した
（ログの `.kaggle_stage/out/ptcg-worker-kaggle.log` はタイムスタンプが更新されていた）が、
`kaggle.npz` だけはタイムスタンプ・サイズとも**古いgen2のものと完全一致**のまま
`shards/v3/kaggle.npz` にコピーされていた——つまり `kaggle kernels output` が今回の実行で
`kaggle.npz` を出力しなかった（何らかの理由でカーネル側が新しいシャードを書き出せなかった）
にもかかわらず、`push_kaggle.py` はステージング先に前回分が残っていることを検知せず、
古いファイルをそのまま「今回の結果」として配置した。

**実害は無かった**: `learner.py` の世代・モデルハッシュ照合（README「世代がずれないための
仕組み」1-2番）が `[警告] kaggle.npz: 世代が違う(v2 != v3)` を出して弾いたため、誤った
シャードがPPO更新に混ざることはなかった。ただし**気づかずに放置すると「gen3の収集に見えて
中身はgen2の使い回し」がずっと再現し、無限にリジェクトされ続ける**（学習が進まないだけで
エラーにはならない）ので、`--status` で `rejected` が続く場合はこれを疑うこと。

対処: `shards/v<N>/` の疑わしいファイルと `.kaggle_stage/out/*` を消して撮り直す。
根本修正（`push_kaggle.py` 側でダウンロード前後の `kaggle.npz` ハッシュ比較や `.kaggle_stage/out/`
の事前クリアを入れる）は**未着手**——この対面（および今後の対面特化RL）はKaggleを介さず
ローカルで回す方針（step 3 参照）にしたため、優先度は低い。Colab等で計算資源を足す段になったら
直すこと。

---

## 10. スケジュール

| いつ | 内容 |
|---|---|
| **済（8/12 深夜）** | 打点修正 / コーパス更新 / デッキ移行 / BC 3シード / 本番経路確認 / デプロイ |
| **木（8/13）** | ① §5-2 テラスタル修正＋回帰テスト（半日）② step 0 の相手BC（marnie / dragapult / froslass、各25分） |
| **金（8/14）** | ③ step 0 の残り（crustle / lucario）④ ベースBCの対プール勝率を測る（step 2 の出発点）⑤ 提出を固める |
| **来週** | step 1（Phase E の測定欠陥7項目）→ step 2（単腕RL 25反復）→ step 3（対面特化RL）→ step 4（切替） |

**金曜までに RL 本体には入らない。** step 1 を飛ばして step 2〜3 をやると、7月と同じく
「best-checkpoint で +8.5pt 出たが本当かは分からない」に戻る。**step 0 は測定基盤なので、
それが無いとRLの効果自体が測れない。** ここを固めるのが今週の正しい終わり方。

**追加の日次ダンプが手に入ったら最優先で取り込む。** 1日分でカミツオロチが約500 ep-player 増え、
3日分で alakazam 専用BC の教師量（110,759決定点）を超える。§7-3 の「リーリエのタイミング」は
まさにこれで直る類の問題。

---

## 11. 付録: 2026-08-12 の実測ログ

### BC 学習（3シード、ablate 版）

| seed | test top-1 | test nll | val top-1 | epochs |
|---|---:|---:|---:|---:|
| **42（採用）** | **0.6623** | **0.9846** | 0.6586 | 14 |
| 2 | 0.6598 | 0.9982 | 0.6597 | 17 |
| 1 | 0.6533 | 0.9916 | 0.6676 | 16 |
| 平均 / SD | 0.6584 / **0.0038** | | | |

**教師データが alakazam の 1/3（27,347 対 89,722）なのに top-1 は 1pt 差**（0.6623 対 0.6723）。
train/val gap 0.019 で過学習の兆候なし。SD 0.0038 は B1 のフルデータ時（0.0042）と同等。
カミツオロチのリストが6種類しかなく教師の分布が狭いことが効いていると見ている。

参考: `--ablate-features` を外したフル715次元版は **0.6531**（−0.9pt）。
alakazam の C2/T1 実験（0.6560 対 0.6723、−1.6pt）と同じ向きで再現した（→ D4）。

### データ

| 工程 | 結果 |
|---|---|
| ハードリンク | 4,603件 → replays 12,047件、ディスク消費0 |
| ラベル | 24,094デッキ中 kamitsuorochi_ex **551件**（旧36＋新515、事前推定と完全一致） |
| 決定点 | **35,347件**（64.2点/ep-player） |
| 特徴量 | 57秒・618 rows/s、`hand_card_vocab` 43種、train 27,347 / val 3,544 / test 4,456 |
| デッキ健全性 | 60枚 / たね11枚 / マリガン率 22.2% / 教師551 → **OK** |

### 提出物

```
sample_submission/ptcg_ai/learning/policy_weights.json
  md5 fa19b32c2896b76246f7d4b430950e9a （旧 616727d9… = alakazam BC）
  = policy_weights_kamitsuorochi_ex_ctl_deck06_s42.json
旧重みは scratchpad/policy_weights_alakazam_backup.json に退避
```

`run_league`（rule_based vs ml_policy、abl_5_full、30試合）:
**ml_policy 28/30 = 93.3%、エラー0、平均15.8ターン、366.9秒。先攻14/15・後攻14/15。**
※ 相手の rule_based は alakazam のプロファイルを使うためハンデ付き。alakazam版の 27/30 とは
直接比較できない。確認したかったのは「エラー0で完走すること」。
