# belief-conditioned RL 方策の実装プラン（相手アーキ予測を入力に混ぜる）

作成: 2026-08-02 / 承認済み方針（手書きルールはやらない・入力に混ぜる暗黙アプローチ）

## 0. 狙い（なぜこの形か）

- このセッションの実測結論: **手書き/残差の②③は模倣の床を超えず、flat RL が本命**（default 53.6 → ov1 63.5 → climb 68.9, climb帯）。
- ユーザーの2案（①長期戦略の保持層 ②アーキ別モデル+ルーター）の**合成＝1つの方策に"相手アーキ予測"を入力として混ぜる**（belief-conditioned）。別モデル切替の不連続や手書きレイヤーを避け、**"相手を見て戦略を変える"を RL が入力から暗黙に学ぶ**。既存の入力拡張(`extra_features`/`own_resource`)の自然な拡張。

## 1. 中核: parity-safe な belief 特徴（最重要・失敗すると学習が静かに壊れる）

新規 `opponent_belief_features(state, predictor) -> list[float]`（`extra_features.py` に追加）:
1. **現在 State の"相手の可視カード"だけ**から `observed_cards`(name→count) を作る:
   相手 = `state.players[1-yourIndex]` の `active[].id`＋`preEvolution`、`bench[].id`＋`preEvolution`、`discard[].id`、`state.stadium`。card_cache で id→name。
   ※ **`tracker`/`opponent_knowledge` の累積 observed_cards は履歴依存なので使わない**（学習=リプレイ, 推論=生ゲームで不一致になる）。current State の可視カード（主に discard＋盤面）から純関数で作る＝`own_resource_features` と同じ parity-safe 方針。
2. `predictor.predict(observed_cards, state.turn) -> {archetype: prob}`（HybridDeckPredictor）。
3. **固定順** `predictor._classes` で `[probs.get(c, 0.0) for c in classes]` を返す（長さ=クラス数, 決定的）。
4. State/predictor 無し・例外時は全0（安全側, own_resource と同じ）。

**parity 必須事項**: 学習側(`collect_field`)と推論側(`PolicyModel._compute_extra`)で **同じ関数・同じ predictor・同じ抽出**を使う。検証テスト(§4)で同一 State→同一ベクトルを assert する。

## 2. 統合（既存 extra_features 機構に載せる）

- `extra_features.py`: `opponent_belief_features` 追加＋合成用 `OPP_BELIEF_FEATURE_NAMES`(=classes)。
- `PolicyModel._compute_extra`: `meta["extra_features"]` に `"opp_belief"`（or `"own_resource+opp_belief"`）を追加 dispatch。predictor を lazy load（`match_context._get_predictor` と同一ロード）。
- 学習側 `collect_field`（rollout の特徴計算）: 同じ `opponent_belief_features` を extra に連結。**worker(multiprocessing)で predictor をロード**（各worker一度）。
- 入力次元が classes 数ぶん増える → 新規学習が必要（既存重みとは非互換, 別 tag）。

## 3. 学習と評価

- `train_league.py` に `--extra-features opp_belief` を通す（collect_field/PolicyModel 両方に反映）。**climb から継続** or 模倣から。現在メタ field（§既存）。
- 評価: `_eval_current_meta.py`（帯別）で climb(68.9%)と比較。**上回れば "相手を見る" が入力で効く証拠**。

## 4. リスクと検証（着手前に必ず）

- **parity 検証テスト**: 同一 State を (a) 推論パス `PolicyModel._compute_extra` と (b) 学習パス `collect_field` の特徴計算に通し、belief 部分が bit 一致することを unit test で assert。← これを最初に書く。
- **predictor ロードコスト**: worker15 で各々ロード。起動が重ければキャッシュ/共有を検討。
- **序盤の belief 不確実性**: 低 evidence では予測が曖昧 → 方策は"不確実な belief でどう打つか"も学ぶ（temperature 較正済 predictor 前提）。真ラベルは使わない（推論と条件を揃える）。
- **既存重み非互換**: 入力次元が変わるので別系統（tag=belief 等）。

## 5. 実行順（安全側）

1. `opponent_belief_features` 実装 ＋ **parity unit test**（同一 State→同一ベクトル）。← ここが緑になるまで学習しない。
2. `_compute_extra` と `collect_field` に配線（両側同一）。小さな e2e（1ゲーム）で特徴が両側一致・エラー0を確認。
3. `train_league --extra-features opp_belief --tag belief`（climb継続, 現在メタ field, desktop, scheduled task）。
4. `_eval_current_meta.py --weights belief` で climb と比較 → 採否。

## 現状の最良（この plan の起点）

- **climb**(`policy_weights_alakazam_rl_climb.json`): climb帯 68.9% / 天井 68.7%。climb2(champion多様性増)は 0.705 plateau で climb 未満＝**climb がベスト**。
- 本番は無傷（全実験 config-gated OFF, production=Plan A+abl_5_full）。提出は保留（ユーザー方針）。
