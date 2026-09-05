# 設計書: 戦術プレイング修正（config-gated, 手書きルールなし） — 2026-08-03

## 0. 背景（ユーザーのプレイ観察による戦術ミス）
1. **クセロシキのたくらみ**（相手が使用時、自分は手札を3枚に減らす＝**残す3枚を自分で選ぶ**）で、立て直しに必要な札を残せていない。
2. **せいなるはい**（トラッシュのポケモンを山に戻す＝唯一の deckCount 回復）のタイミングが微妙。
3. **ノコッチ（にげあしドロー：3ドロー＋自分を山に戻す）ループ**を山札切れ間際に回せていない（0枚では不可・1枚なら可の境界）。
4. **改造ハンマー**を「このターンKOできる相手」に使って浪費する（剥がしても結果が変わらない）。
5. **ロケット団ミュウツー（フリーザー）**：たねロケットにフーディンは0ダメージ→進化を倒す立ち回りができていない。

## 1. 原則
手書きヒューリスティクスは足さない。使うレバーは2種のみ、**いずれも config-gated（既定 `abl_5_full` にキー無し＝本番不変）**：
- **(A) consequence特徴**：各選択肢を1手だけ仮実行し「KO可否が変わるか / 特殊エネを剥がせるか / ダメージ量 / …」を特徴として方策に渡す（＝"この手が実際に効くか"を見せる）。**推論側は配線済み**（`policy_model.PolicyModel._consequence_fields` ← `meta.consequence_fields`、`_append_consequence_features` が option 特徴に連結。encoder.`encode_option_consequence_features` が本体、対象型 = {ATTACH, EVOLVE, PLAY}、fail-soft）。
- **(B) ゲームルール上の探索**：確定リーサル探索（`_try_lethal`/`lethal_simple`）の**対称版＝延命探索**。ヒューリスティクスでなく探索なので「ルール手書きを避けたい」に抵触しない。

## 2. ワークストリーム

### WS3 — 延命探索（逆リーサル）※設計見直し済：期待値は低い
**⚠️設計上の注意（lethal精読で判明）**: lethal は「このターン内に `result==me`」というクリーンな終端を探すが、**deckout回避には同種の終端が無い**（自分の山札切れ負けは"次の自分のターン開始時に引けない"時に発生し、自ターン内では終端化しない）。したがって WS3 を実直に作ると＝**「ターン終了時 deckCount>0 を保つ/最大化する手順の浅い探索」**になり、これは実質 **既存 deck_sustain ヒューリスティクスの"探索精密版"**。そして **deck_sustain は過去 Δ≈0（中立）**[[project_decksustain_scaffold]]。→ **期待値は低い**（探索版で精度は上がるが、deckout改善が効かなかった前例）。安いので"やるなら短時間の実験"に留める。別解釈の「逆リーサル＝相手の次ターンlethal阻止」はクリーンな終端探索だが**相手ターンの仮実行が要り高コスト（≒PIMCの仕事）**なので別物。
**狙い**: 論点2/3の deckout 回避（ただし上記の通り効果は限定的見込み）。
- **新規関数** `ml_policy_agent._try_survival(obs, config) -> list[int] | None`。
- **配置**: `_select_action` 内、`_try_lethal` の**直後**（勝ち > 生存 > その他）。既存 `_try_deck_sustain`（ヒューリスティクス版・中立[[project_decksustain_scaffold]]）は残置し、survival が入ったら段階的に置換。
- **発火条件（config）**: 自 `state.players[me].deckCount <= survival.deck_threshold`（既定8）。
- **探索**: `lethal_simple` の枝刈り・時間予算・`search_step` を流用し、現在の合法手系列を **深さ `max_depth`（既定3）** 展開して「N手後に `deckCount>0` かつ場が残る（＝このターン/次ターンに負けない）」ラインを1本見つけたらその初手を返す。0枚ではループ不可・1枚なら可、を**探索が構造的に扱う**（ヒューリスティクスの場合分け不要）。
- **config**: `"survival": {"enabled": true, "deck_threshold": 8, "max_depth": 3, "time_limit_ms": 100}`。
- **parity**: 決定的探索＝学習非依存（推論のみ）。既存 policy 重みは不変。
- **完了条件**: (a) unit（deckCount 0/1境界・ノコッチループ発火・時間内終了）、(b) `_diag_local_crustle_lossmode` 系で **deckout敗率が下がる**、(c) 較正eval全対面で**非劣化**（特にミラー）。

### WS2 — consequence特徴（再学習）★大・LB賭け
**狙い**: 論点4（ハンマー浪費）＋論点5（0ダメージ攻撃の回避）を、方策に"効くか"を見せて直す。
- **特徴の絞り込み**（全部載せない＝過学習/コスト抑制）: `delta_can_ko`（ハンマー浪費＝これが変わらないなら無駄）、`opp_special_energy_removed`、`delta_best_effective_attack_damage`（0ダメージ攻撃の回避）。
- **学習側 parity（必須実装）**: `collect_field._play_field` に**推論と同一の** consequence 計算を追加（現状 collect_field は未計算）。同一 `hidden_state_factory`・同一 `encode_option_consequence_features` を通す（Tier3方針§2.3 の train/runtime parity）。単体テストで train/inference の consequence 値一致を確認。
- **重み**: climb を起点に入力次元を grow（belief の `grow_input_for_belief.py` 同様の surgery で consequence 列を0挿入）→ フィールドRLで学習。JSON `meta.consequence_fields=[...]` を立てる。
- **コスト注意**: consequence は `search_step` 仮実行で重い。学習/推論とも時間予算内に収める（対象を PLAY/ATTACH に限定・deadline 共有）。
- **完了条件**: (a) train/inference parity 単体テスト、(b) **ハンマー浪費率メトリクス**（「今KO可能な相手に改造ハンマーを使う」頻度）が有意に下がる、(c) **ミラー非劣化ゲート必須**（入力拡張は過去LB-negative[[project_belief_conditioned_rl_negative]]。較正eval で climb 非劣化を確認できなければ**不採用**）。

### WS1 — クセロシキ keep-3 改善 ★最も曖昧・WS2に相乗り
**狙い**: 残す3枚の質を上げる（立て直し札を残す）。
- keep-3 は我々の discard 選択（`_select_action`→base方策、カードID埋め込みで区別可能＝盲目ではない）。単独機構は作らず、**WS2 の再学習に含める**。
- 補助特徴案（任意）: 手札中の「ドローサポート/リソース回収カードの有無」フラグ（立て直し札の存在）。ただし入力拡張リスクありミラー非劣化ゲート対象。「残した3枚で立て直せるか」の将来価値は myopic 仮実行では捉えにくいので、まずは card-id 埋め込み＋RL学習に委ねる。
- **完了条件**: keep-3 意思決定品質メトリクス（残した3枚に回収/ドロー札が含まれる率）が上がる、かつ非劣化。

### WS4 — ロケット団ミュウツー
- **WS4A（consequence, 推奨）**: WS2 に統合。`delta_best_effective_attack_damage` が「たねに0ダメージ攻撃」を捉える → 進化狙いを学習で獲得。
- **WS4B（専用ルーティング）★保留**: **予測器21クラスに `rocket_mewtwo` が無い**（`rocket_honchkrow`は別archetype）。→ 予測器が rocket_mewtwo を高確信で識別できず、ルーティングが発火しない/誤爆する。**前提=予測器に rocket_mewtwo クラスを追加（再学習）が必要**。当面は WS4A で対応。routing 基盤自体は crustle で完成済み・再利用可（クラスさえ揃えば map に1行）。

## 3. 実装順序と依存
1. **WS3 延命探索**（独立・再学習不要・安価） ← 最初に着手
2. **WS2（+WS1, +WS4A）consequence 再学習**（大・LB賭け・ミラー非劣化ゲート必須） ← 次
3. **WS4B routing** ← 予測器に rocket_mewtwo クラスが入るまで保留

## 4. 検証プロトコル（全WS共通）
- 較正ブラケット eval `kaggle_replays/_eval_current_meta.py`（全対面, 40試合/対面）。
- **ミラー非劣化ゲート**（入力拡張系は過去非転移＝必須）。
- 負け筋診断 `_diag_local_crustle_lossmode`（deckout/blowout率）。
- routing は route 監査（誤爆0確認、`scratchpad/route_audit.py` 相当）。
- production `abl_5_full` は全WSで不変（config-gated）を回帰確認。

## 4.5 実装状況（2026-08-03）
- **WS2 は consequence特徴でなく「ターン先読みKO探索veto」に着地**（設計診断の結果）。理由: consequence特徴(delta_can_ko等)は各手の**1ステップ効果**しか見ず、「倒せる相手にハンマー=無駄」という**複数手をまたぐ冗長性**を捉えられない。加えて瞬間打点(best_effective_attack_damage)では「展開途中でハンマー→後半にKO」を捉え損ねる(smokeで発火0を確認)。
- **実装（完了・検証済）**:
  - `ptcg_ai/search/ko_search.py`：`can_ko_this_turn(obs, factory, config, deadline)`。lethal_simple 流用の浅いDFS。**成功条件=このターンで自分のサイドが減る(=KOでプライズ取得)**。準備込み(エネ装着→攻撃)のKOを捉える。確実性の再検証はしない(veto は「温存が良さそう」で十分)。
  - `ml_policy_agent._try_hammer_veto(obs, chosen, config)`：選んだ手が改造ハンマー(1081)PLAY かつ `can_ko_this_turn` なら、ハンマー以外の方策スコア最大手に差替。pipeline出力と base方策の両経路に配線。config-gated(`hammer_veto`)。
  - config `abl_5_full_hammer.json`（hammer_veto有効）。**既定 abl_5_full は不変=本番無影響**。
  - 検証: vs crustle 10ゲームで **veto発火4回(KO可能時)・正当なハンマーは温存・エラー0**。production回帰(abl_5_full)で **veto発火0・エラー0=inert**。
- **未了**: 非劣化較正eval(低頻度ゆえ勝率差はノイズ域見込み)。WS1/WS4A(consequence再学習)は未着手のまま(WS2がconsequence回避で解けたため優先度低下)。

## 5. リスク / 正直な見積もり
- これらは**低頻度の戦術ミス**＝個々のLB寄与は小さい可能性（crustle routing = 全体+1.5%級だった）。過度な期待はしない。
- **consequence（入力拡張）は過去 LB-negative**（own_resource/belief）。ミラー非劣化ゲートで弾く前提の"賭け"。
- 延命探索は deck_sustain ヒューリスティクスが中立だった → 探索版で精度改善を狙うが LB効果は未知。deckout は構造コストの面もある[[project_deckout_loss_cause]]。
- 効いた実績のあるレバーは flat RL(climb) と deck-tech のみ[[project_climb_lb_new_best]]。本設計は"戦術の穴埋め"であり、それを踏まえて優先度を置く。
