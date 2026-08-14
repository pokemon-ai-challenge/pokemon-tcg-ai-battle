# 模倣学習v5(容量拡大 hidden128)Kaggle提出記録

作成: 2026-07-23
関連: `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`、
`results/2026-07-23_policy_capacity_ablation.md`

## 提出内容

| 項目 | 値 |
|---|---|
| ref | **54918091** |
| コンペ | pokemon-tcg-ai-battle |
| 提出日時 | 2026-07-23 03:26:59 UTC |
| 説明 | 模倣学習v5 (容量拡大 hidden128): v4と同一構成でPolicyModelのhiddenを32→128に。ローカルミラーはv4と有意差なし(49.3%)、対フィールド検証用。 |
| ステータス | **COMPLETE**(publicScore **600.0**) |
| tarball SHA256 | `f1096848475aa4e4e0ad7f92293982848136a382560775dd597d8b058d232718` |

## v4 との差分(weights のみ)

- ベース: 直近提出 v4(`submission.tar.gz`、構成C + 確定リーサル + attack_plan v0)を展開し、
  **`ptcg_ai/learning/policy_weights.json` を M128(hidden=128)に差し替えただけ**。
- 非pycacheファイルセットは v4 tarball と**完全一致**(diff なし)。コードは v4 のまま、weights のみ変更。
- shipped weights: layer0 = 128×239、`meta.hidden_size=128`、`consequence_fields=[]`、test_top1=0.5864。

## 提出前検証(memo: 提出tarballにdecks/必須)

- 空dir展開 → `main.py` import → 自己対戦1ゲーム完走を確認
  (deck 60枚、battle_start errorType=0、winner確定、14ターン/174ステップ/13.7s、エラーなし)。
- decks/ は tarball に含まれる(active.py / new_deck/ 一式)。

## 位置づけ・注意

- M128 は容量ablation(`2026-07-23_policy_capacity_ablation.md`)で **M32(=現行production)と
  実戦ミラー有意差なし**(300試合 49.3%、95%CI[43.7,55.0])。offline Top-1 は +0.58pt だが転移せず。
- 本提出は「現行を上回る根拠がある変更」ではなく、**ミラー≠対フィールドの分散を実地スコアで見る検証目的**。
- production の `policy_weights.json`(hidden 32)はリポジトリ上**未変更**。提出tarballは
  スクラッチ領域でのみ weights を差し替えており、本番重みファイルは触っていない。

## 対フィールド実戦結果(2026-07-23 取り込み)

提出後に蓄積した対戦を `kaggle_replays/fetch_my_episodes.py --submissions 1` で取得し、
`replays/` と `index/episodes_master.jsonl` に取り込んだ(27試合、マスターインデックス 4948→4975)。

| 指標 | 値 |
|---|---|
| publicScore | 600.0 →(蓄積後)**668.9** |
| 試合数 | 27(うちミラー1) |
| 勝敗 | **W15 / L12 / T0** |
| **勝率** | **55.6%**  Wilson 95% CI **[37.3%, 72.4%]** |
| 対戦相手 score | min 447 / median 677 / max 818 |
| 対戦相手 rank | best 853 / median 2478 / worst 4618(自分 rank 2543 ≈ 同格帯) |

- 自チーム: 「MORIOKA Tsuoi」(team_id 16379894)。勝敗は各リプレイの `rewards[my_index]` で判定。
- **解釈**: 点推定は 55.6% とやや正だが、n=27 で CI が [37.3, 72.4] と広く **50% を跨ぐ**ので、
  この標本だけでは M32(現行)より強いとは言えない。ローカルミラーで有意差なし(49.3%)だった
  結論と整合。対フィールドで即 ERROR/破綻していない(errors=0、27/27 完走)ことは確認できた。
- 生データ: `kaggle_replays/_m128_field_results.json`、リプレイ27件は `kaggle_replays/replays/`。

## CLI 注意(記録)

`kaggle competitions submit` はアップロード100%完了後、応答メッセージ表示時に Windows コンソールの
cp932 で `UnicodeEncodeError` を出してクラッシュするが、**提出自体は受理される**
(表示のみの問題)。ステータス確認は `PYTHONUTF8=1 kaggle competitions submissions -c pokemon-tcg-ai-battle`。
