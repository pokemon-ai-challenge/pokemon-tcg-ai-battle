# gen2_candidates — 2026-08 の gen2 模倣学習 + デッキ差し替えで champion 改善を試みた候補一式

> **このディレクトリは「オーガポンのやつ」「ハンマー4枚のやつ」がどれかを一意に特定するための索引です。**
> 機械可読版は [MANIFEST.json](MANIFEST.json)。
> **重要: どの候補も 2026-08-13 時点で Kaggle LB 未収束。champion は climb (823.5) のまま据え置き。**

## 「オーガポンのやつ」= `mixogerpon`

| 項目 | 値 |
|---|---|
| **通称** | **オーガポン版 / mixogerpon** |
| 学習アーキタイプ | `ogerpon_teal_ex`(オーガポン みどりのめんex) |
| 方策の重み | [policy_weights_ogerpon_teal_ex_rl_mixogerpon.json](../../ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json) |
| 重み sha256 | `e9e348d80c8ec35c…` |
| **デッキ** | [deck_ogerpon_teal_ex_01.csv](deck_ogerpon_teal_ex_01.csv)(= `kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv`) |
| デッキの出所 | gen2リプレイの **rank 42 / LBスコア 1055.2 のパイロット**(team "Dries @ Tufa Labs", episode 91902939)が実際に使った60枚 |
| Kaggle 提出 | `55464390` / `submission_mixogerpon.tar.gz` |
| publicScore 推移 | 693.2 → 721.8 → 837.0 → **874.1** → 835.4(**未収束**) |

デッキの中身(60枚/19種): オーガポン みどりのめんex ×4 / カプ・ブルル ×1 / 基本【草】エネルギー ×17 /
グロウ【草】エネルギー ×2 / むしとりセット ×4 / テラスタルオーブ ×4 / クラッシュハンマー ×3 /
ジャンボアイス ×3 / ポケギア3.0 ×3 / エネルギー転送 ×2 / エネルギー回収 ×1 / ヒーローマント ×1 /
ジャッジマン ×4 / リーリエの決心 ×4 / ボスの指令 ×2 / Nの筋書き ×1 / クラウン ×1 / ブライア ×1 /
エキサイトスタジアム ×2。**ポケモンがたった5枚**(うち4枚がオーガポン)で、進化ラインを持たず
エネルギー19枚を厚く積む構成。

## 「ハンマー4枚のやつ」= `mixhammer`

| 項目 | 値 |
|---|---|
| **通称** | **フーディン ハンマー4版 / mixhammer** |
| 学習アーキタイプ | `alakazam`(フーディン) |
| 方策の重み | [policy_weights_alakazam_rl_mixhammer.json](../../ptcg_ai/learning/policy_weights_alakazam_rl_mixhammer.json) |
| **デッキ** | [deck_foodin_hammer4.csv](deck_foodin_hammer4.csv)(= `experimental_decks/foodin_hammer4_v1/deck.csv`) |
| デッキの出所 | **ユーザー提示**の構築。Plan A から 改造ハンマー1→4 / +キチキギスex / +ヒカリ / +テレパス超エネ / −基本超エネ2 |
| Kaggle 提出 | `55462165` / `submission_mixhammer.tar.gz` |
| publicScore 推移 | 821.4 → 839.0 → 810.2 → 799.8 → 768.8 → 777.0 → 766.1 → 757.7(**未収束**) |

---

## 候補一覧(全4件)

| 通称 | アーキ | デッキ | RL初期値 | ローカル対champion(mix / july7) | LB(未収束) |
|---|---|---|---|---|---|
| **mixogerpon** | オーガポン | gen2 1位デッキ | gen2 BC | **+14.92pt 有意 / +6.08pt 有意** | 835.4(最高874.1) |
| **mixhammer** | フーディン | ハンマー4(ユーザー提示) | climb | **+5.96pt 有意 / +5.50pt 有意** | 757.7(最高839.0) |
| mixclimb | フーディン | Plan A | climb | +2.90pt 判定不能 / +1.28pt 判定不能 | 678.5 |
| g2climb | フーディン | Plan A | gen2 BC | 判定不能 / −0.53pt 判定不能 | 601.6 |

参考: champion **climb** = LB 823.5(未収束、831.5 の記録もあり)。`models/climb_lb823/` 参照。

---

## アルゴリズム(全候補共通 — climb と同一レシピ)

`kaggle_replays/rl/train_league.py` = **champion-league RL**(固定フィールド + 自分の過去スナップショット)。

```
--field-frac 0.45 --iters 80 --games-per-iter 512 --eval-games 400 --eval-every 3
```

**climb 本体もこのスクリプトで作られている**(重みの meta に `rl_league=True` / `rl_field_frac=0.45` /
`rl_best_iter=57` が残っている)。`train_field.py`(フラットなフィールドRLのみ)は**別物**なので混同しないこと。

### `--field-preset mix` = 相手フィールドの構成

アーキタイプごとに「gen2版と7月版のどちらの模倣ポリシーが強いか」を
`kaggle_replays/rl/compare_opponents.py` で直接対決させて選んだ混成11アーキ。

| 採用 | アーキタイプ(gen2勝率) |
|---|---|
| **gen2** | alakazam 67.3% / rocket_mewtwo 68.3% / crustle 65.8% / marnie 61.7% / dragapult 60.0% |
| **7月** | mega_lucario 33.0%(gen2は1257→403デッキと減り、有意に弱かった) |
| 差なし→gen2 | archaludon 47.3% / shirona 52.0% |
| gen2のみ | ogerpon_teal_ex / mega_froslass_ex / omatsuri_ondo |

**学習データ量と相手モデルの強さは対応しない**(rocket_mewtwo は 247→112 デッキに減ったのに有意に強い)。
オフライン指標では決められないので実測必須。

---

## 最大の発見: 効いているのはデッキであって RL ではない

3候補すべてで同じ分解結果が出た。

| 候補 | デッキ+BCだけの効果 | RL の上乗せ |
|---|---|---|
| mixogerpon | **+12.45pt(有意)** | +2.47pt / −0.99pt(**両方 判定不能**) |
| mixhammer | **+5.69pt(july7, 有意)** | +2.52pt / −0.19pt(**両方 判定不能**) |
| mixclimb | (デッキ据え置き) | +2.90pt / +1.28pt(**両方 判定不能**) |

**リーグRL の追加学習は、3回とも測定可能な寄与を出していない。**
今後の改善投資は RL のチューニングよりデッキ探索に向けるべき。
gen2 には各アーキ上位5デッキが `kaggle_replays/meta_analysis/archetype_decks_g2/<arch>/0[1-5].csv` に
揃っているので、同じ枠組みで横展開できる。

## ローカル評価が LB を予測しない件

`train_league.py` が出す best は **raw field-eval のスパイクを拾う**(既知の逆相関、
`project_lb_converged_verdict`)。g2climb / mixclimb はプラトーが baseline とほぼ同じで
best だけ跳ねた形 → LB でも下位。mixhammer / mixogerpon はプラトー全体が持ち上がった形 → LB でも上位。
**「プラトーが上がっているか」の方が best 値より信頼できる。**

ただし順位までは一致しない。ローカルでは mixogerpon が mixhammer に mix +8.96pt(有意)だが、
LB では両者とも大きく振動して確定していない。

## LB スコアは収束しない

`kaggle_replays/_lb_history.tsv` に20分間隔の実測を記録している。**同一提出が1時間で30点以上動く**。

- mixhammer: 821.4 → 839.0 → 810.2 → 799.8 → 768.8 → 777.0 → 766.1 → 757.7(幅 81.3)
- mixogerpon: 693.2 → 721.8 → 837.0 → 874.1 → 835.4(幅 180.9)

**1回の読みで優劣を判断してはいけない。** 追跡は `kaggle_replays/_lb_track.sh`。

---

## 再現方法

```bash
# 1. 相手フィールドの世代選択を測り直す
python kaggle_replays/rl/compare_opponents.py --games 300 --workers 14

# 2. 学習(オーガポン版の例)
python kaggle_replays/rl/train_league.py --field-preset mix \
  --learner-arch ogerpon_teal_ex \
  --learner-weights policy_weights_ogerpon_teal_ex_g2.json \
  --learner-deck-path kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv \
  --field-frac 0.45 --iters 80 --games-per-iter 512 --workers 14 --tag mixogerpon

# 3. 評価(学習に使っていない july7 でも必ず測る)
python kaggle_replays/rl/eval_field.py --field-preset mix \
  --learner-deck kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv \
  --baseline-weights policy_weights_ogerpon_teal_ex_g2.json \
  --rl-weights policy_weights_ogerpon_teal_ex_rl_mixogerpon.json --games 150

# 4. 提出パッケージ(champion tarball から weights と deck だけ差し替え)
python kaggle_replays/_p1915_package.py --base build_ready/submission_climb.tar.gz \
  --weights sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json \
  --deck kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv \
  --out build_ready/submission_mixogerpon.tar.gz
```

`--field-preset` の既定は `realmeta`(従来)なので、**指定しない限り過去の学習・評価は一切変わらない**。

## 注意

- 学習側デッキは `--learner-deck-path` を必ず指定すること。既定は
  `archetype_decks/<arch>/01.csv` で、**champion が使っている Plan A デッキではない**。
- 提出 tarball では重みが `ptcg_ai/learning/policy_weights.json` という名前で同梱される。
  ローカルの既定 `policy_weights.json` は **BC 重みであって候補ではない**。
- 対応するリーグのチャンピオンスナップショットは
  `kaggle_replays/rl/league_champions/<arch>_<tag>/` に、再開用チェックポイントは
  `kaggle_replays/rl/_ckpt_<arch>_<tag>.pt`(Git管理外)にある。
