# 相手デッキ推定システム（OpponentModel）

AI が盤面の公開情報から相手のアーキタイプを推定し、行動戦略を変化させる仕組みの全体説明。

**対象ファイル:**
- `src/knowledge/opponent_model.py` — 推定ロジック本体
- `src/knowledge/meta_decks.py` — デッキレシピ・軸ポケモン定義
- `src/decision/target_policy.py` — 推定結果を使う意思決定

---

## 1. 全体フロー

```
毎ターン main.py で update(obs) 呼び出し
       ↓
OpponentModel._revealed に相手の公開カード ID を蓄積
       ↓
estimate_arch() → estimate_deck_candidates() → 推定アーキタイプ名
       ↓
boss_target_ids() → ARCH_BOSS_TARGETS[arch] → ボスの指令ターゲット優先リスト
       ↓
choose_switch() でボス対象を決定
```

将来的にはアーキタイプに応じてベンチ上限・エネルギー付け先・にげる判断なども変化させる予定（`docs/axis_card_analysis.md` 参照）。

---

## 2. 公開情報の収集（`OpponentModel.update()`）

```python
# 毎ターン呼び出して相手の公開カード ID を蓄積
model.update(obs)
```

収集対象：
- 相手のバトルポケモン（`.id`、エネルギー・どうぐ・進化前）
- 相手のベンチポケモン（同上）
- 相手のトラッシュにあるカード

**注意:** ゲーム内ポケモンオブジェクトは `.id` 属性（in-play ID）を使う。`CardData.cardId` とは別物。

ターン番号が巻き戻ったとき（`turn < last_turn`）は新しい対戦と判断してリセット。

---

## 3. アーキタイプ推定（`estimate_deck_candidates()`）

`meta_decks.py` の `CARD_TO_DECKS`（全デッキのレシピから自動生成）を使った重み付きスコアリング：

```
各公開カード ID に対して:
  そのカードを採用しているデッキ数 N を調べる
  各デッキに 1/N の重みを加算（排他性が高いカードほど強く貢献）

最終スコアを最大値で正規化 → (アーキタイプ名, 0.0~1.0) のリストを返す
```

例：ドラメシヤ(119) が見えたとき
- `CARD_TO_DECKS[119] = ["dragapult_ex"]` → dragapult_ex に +1.0
- ドラパルト専用カードなので即時ほぼ確定

例：オーガポン+みどりのめんex(96) が見えたとき
- 複数デッキに採用 → 重みが分散、単体では確定しない
- ドラメシヤなど組み合わせで絞り込む

**閾値:** `MIN_CONFIDENCE = 0.3`。正規化スコアが 0.3 未満は `None`（不明）。

---

## 4. アーキタイプ別ボスターゲット（`ARCH_BOSS_TARGETS`）

| アーキタイプ | 優先ターゲット（先頭ほど優先） | 理由 |
|------------|--------------------------|------|
| hydrapple_ex | メガニウム(710) → ベイリーフ → チコリータ → オーガポン(96) | エネ加速役を先に倒す |
| olivia_ex | メガニウム(710) → 同上 → オリーニョ(402) | 同上 |
| dragapult_ex | ドラメシヤ(119) → ドロンチ(120) → バシャーモex(326) → ドラパルトex(121) | 進化ラインを崩し将来の打点を潰す |
| raging_bolt_ex | オーガポン(96) → テツノイサハex(75) → コライドン(62) | ベンチの補助ポケモンを引き出す |
| ogerpon_bullet | マシマシラ(112) → ラティアスex(184) → オーガポンいどのめん(108) → オーガポン(96) | ドロー役(マシマシラ)を先処理 |
| terasta_bullet | リーリエのピッピex(272) → マシマシラ(112) → … | 同上 |
| mega_lucario_ex | リオル(677) → ルナトーン(675) → ソルロック(676) → マクノシタ(673) | 進化ラインとドローエンジン妨害 |
| crustle | イシズマイ(344) → メガガルーラex(756) | 進化前は EX で倒せる（アビリティ回避） |
| maries_obstagoon_ex | マリィのベロバー(646) → ギモー(647) → ユキワラシ(103/104) | 進化ライン崩し |
| alakazam | ケーシィ(741) → ユンゲラー(742) → ノコッチ(65/66) | 進化ライン崩し |

---

## 5. 推定アーキタイプの利用箇所

### 現在利用している場所
- `target_policy.py:choose_switch()` — ボスの指令時に `boss_target_ids()` でターゲット選択
- `target_policy.py:choose_switch()` — トレースログに `opp_arch=xxx` を出力（デバッグ用）

### 将来的に利用を予定している場所（`docs/axis_card_analysis.md` の MatchupPolicy 参照）
- `target_policy.py:choose_to_bench()` — vs Dragapult はベンチ上限を 2 枚に制限
- `main_policy.py:_pick_best_play()` — ベンチ上限超過時はポケモン展開を抑制
- `main_policy.py:choose_main_action()` — アーキタイプ別にリトリート判断を変化

---

## 6. アーキタイプを追加するには

1. `meta_decks.py` に `MetaDeck` エントリを追加（`key_pokemon_ids` + `recipe`）
2. `opponent_model.py:ARCH_BOSS_TARGETS` に新アーキタイプのターゲットリストを追加
3. `DECK_CATALOG` への追加で `CARD_TO_DECKS` が自動的に更新される

デッキレシピの元データは `docs/axis_card_analysis.md` の再抽出手順を参照。

---

## 7. 現在の制限と既知の課題

| 課題 | 内容 | 対策案 |
|-----|------|-------|
| 推定遅延 | ゲーム序盤は公開カードが少なく推定精度が低い | key_pokemon_ids の優先度付けで加速 |
| 多デッキ共通カード | オーガポン(96)などが複数デッキに採用 → 単体で確定しない | 組み合わせカードが出るまで待つ |
| 新デッキ未対応 | DECK_CATALOG に存在しないデッキは推定できない | min_count 以上のデッキが登場したら追加 |
| イワパレス対策 | EX 攻撃を無効化するアビリティ → 非 EX アタッカーに切り替えが必要 | `src/knowledge/ex_immune.py` インフラ実装済み（未接続） |

---

## 8. アーキタイプ別対戦ベンチマーク（基準値）

メガルカリオexデッキ vs 各アーキタイプ（30試合、相手ランダムAI、2026-06-26）：

| アーキタイプ | 勝率 | 強みと弱み |
|------------|------|----------|
| terasta_bullet | 83% | 弱点なし、ベンチバレットで勝ちやすい |
| dragapult_ex | 77% | ファントムダイブ散布は許容範囲内 |
| mega_lucario_ex | 77% | ミラー、互いに2プライズ取り合い |
| olivia_ex | 73% | 草自己回復だがメガニウムをボスで対処 |
| maries_obstagoon_ex | 73% | 悪妨害は受けるが押し切れる |
| raging_bolt_ex | 90% | 格闘弱点を突けて最強マッチアップ |
| ogerpon_bullet | 67% | マルチタイプ対応が難しい |
| crustle | 63% | EX 無効化アビリティが直撃 |
| hydrapple_ex | 57% | 最弱、自己回復+メガニウム加速が辛い |
| alakazam | 60% | 手札ダメージで削られる |

→ 改善余地大: hydrapple_ex（57%）、crustle（63%）、ogerpon_bullet（67%）
