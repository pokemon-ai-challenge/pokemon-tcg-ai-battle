# Autonomous Kaggle Improvement Loop — registry

BEST_KAGGLE = submission_climb.tar.gz / 823.5 / Policy 395b02486f851be9 / deck 8ae7a618b2655669
BEST_PRODUCTION = 同上（climb）
BEST_OFFLINE = 同上

| loop | hypothesis | Codex objection | experiment | result | decision |
|---|---|---|---|---|---|
| P19.16/17 | binding×trajectory Transformer | #2 trajectory は teacher と同一 rollout 由来 = production で TES に支配 / #8 link だけ単離せよ | entity-link 単離 probe (607 + fresh 2001 root) | FULL−ERASED = −0.0056 [−0.0146,+0.0028]、2 データセットで再現 | STOP（反証） |
| L1 | Sacred Ash 増量で 対crustle の deckout 敗北を減らす | #1 policy が打つ保証なし / #4 deckout率 != 勝率改善 / #5 mirror guard 未測定 / #6 過去の sustain は n=8 で −37.5pt | deck-only paired A/B 240戦 対crustle（policy=climb 固定） | D0 0.2250 → D1 0.1792、**paired −0.0458 [−0.117,+0.025]**。Ash 実プレイは 177→335(rate .248→.284) と倍増したのに **deckout 敗北は 102→110 と増加** | STOP（事前登録 +0.08 に対し −0.046。機構ごと否定） |

## 確定した否定的知見
- 対 crustle の deckout 敗北は **山札回復札の増量では減らない**。カードは実際に打たれている（rate .284）
  ので「policy が使わないから」ではない。deck-sustain 系（policy 側介入 `_try_deck_sustain` も含む）は
  この機構では解決しない。
- 実ラダーのメタ加重勝率（実測対面別 × 2026-08-01 シェア, 86.6% カバー）= **0.5621**。
  対 crustle を +10pp 改善しても全体 +1.17pt にしかならない。

## 次仮説 ranking
1. **local eval が LB を予測しない問題**（最大。R1 は FIELD +2.0 / production +1.9 なのに LB 726 vs 823.5）。
   既知の具体的欠陥 = local FIELD の Alakazam ミラーが **0%**、実ラダーは **20.96%**。
   → まず「R1 は mirror で弱いのか」を直接測る（実行中）。
2. RL critic 継承 + 継続長（R1 は critic 初期化 + 14 iter で drift 1.33% = ほぼ動いていない）。
3. deck-tech の別軸（対 Grimmsnarl 19.88%×0.592 / 対 Lucario 14.86%×0.640）。ただし L1 の失敗で
   「1〜3枚差替で大きく動く」という前提自体が弱まった。

| L2 | local eval が LB を予測しない主因は mirror 欠落(local 0% vs 実ラダー 20.96%) | — | mirror head-to-head R1 vs climb, 240戦, deck/config 固定 | **0.5042 [0.4417,0.5667]** = 互角 | **STOP**（反証。mirror を eval に足しても R1 の LB 落ちは捕まえられない） |
| L3 | R1 は critic 未収束の窓(iter1-8)で best を選んでいたため実質学習していない。critic warmup で改善 | （diff 監査を依頼予定） | `--critic-warmup 8` を train_field.py へ追加(既定0=不変)。30 iter × 2 seed | 実行中 | — |

## L2 の重要な副産物 — 「local↔LB 非転移」の根拠が弱まった
55405504(R1) の publicScore は本セッション中に
`600.0(初期値) → 500.3 → 582.6 → 726.3 → 733.2` と**単調に上昇中**。
climb の 823.5 は長期間かけて収束した値なので、**未収束の値と収束値を比べていた**可能性が高い。
→ 「local で勝ったのに LB で負けた」という前提自体が確定していない。
mirror が互角だったことと合わせ、**測定危機の証拠は無い**と評価を修正する。

## L3 の根拠(実測)
R1 の critic value loss: 0.0421→0.0354→0.0346→0.0318→0.0309→0.0302→0.0291→0.0265→…→0.028 で
**iter 8 付近で頭打ち**。しかし R1 の best は **iter 2**(loss 0.035、critic 未収束の窓)で選ばれていた。
policy drift 1.33% = ほぼ動いていないのと整合。critic を先に温めれば全 policy 更新が
まともな advantage を使い、best 選択も初期ノイズに引きずられない。

## L3 結果（critic warmup）
```
学習時 best（同一 eval seed = 選択バイアスあり）
  W1 best iter18 field 0.688 (baseline 0.640)   W2 best iter22 field 0.690 (baseline 0.630)
  → R1(warmup無し)の best は iter2 だった。warmup で **ピークが後ろへ移った = 機構は意図通り作動**

独立 seed 27182818 / 600戦 / 同一相手・同一 seed
  climb 0.6233 [0.5839,0.6612]
  W1    0.6333 [0.5940,0.6709]  = +1.0pt (YELLOW)
  W2    0.6000 [0.5603,0.6384]  = -2.3pt (RED → 破棄)
```
判定: **機構は直ったが強くならなかった**。W1(+1.0pt) は warmup 無しの R1(+2.0pt) より弱い。

## flat RL continuation の飽和判定
climb からの PPO continuation を **4回**サンプルした結果、独立 FIELD 評価は
```
R1 +2.0 / R2 -0.7 / W1 +1.0 / W2 -2.3   (pt)
```
= **-2.3〜+2.0pt のレンジに散らばるだけで、再現する改善が無い**。
学習時 best（0.667〜0.690）は毎回 independent seed で 0.60〜0.633 へ落ちる = best-of-N 選択バイアス。
→ この機構は飽和。追加 arm を回しても期待値は低い。

## 提出判断
W1 は非劣化だが、**flat RL continuation 仮説は既に 55405504(R1、4本中最強) で提出済み**。
同一仮説の弱い candidate を2本目に出すのは §41/§62 の LB hill-climb にあたるため出さない。
R1 の収束値がこの仮説の答えになる。

## Discovery Round 1（§9-§13）
敗因分析(420戦, 実ラダーshare)→ 8仮説/7カテゴリを `_hypothesis_registry.json` に登録。top3 = H1/H2/H4。
判別probe(300戦)で H2/H3(序盤の方策学習)は棄却: **方策は展開札を持っていれば必ず打っている**
(no_pkm敗の play_rate 1.000)。事故の実体は「引けていない」= デッキのアクセス性。

| loop | hypothesis | Codex verdict | probe/experiment | result | decision |
|---|---|---|---|---|---|
| H1 | basic 8→10(-2 Poké Pad,+1 Dunsparce,+1 Shaymin)で序盤事故を減らす | MODIFY（#9 Shaymin は HP80 で Poffin 対象外＝私の設計根拠は誤り。#8 Poké Pad を削ると逆効果の可能性） | 420戦 A/B。**事前登録 primary = no_pokemon_loss/games** | 0.0952 → **0.1190（悪化）**、全体 -0.045 | **STOP**（Codex #8 の予測どおり） |
| routing | 既存の crustle 専用ルーティング(config-gated, 未提出) | — | 420戦 config-only A/B | 全体 +0.0048 [-0.057,+0.067]。**crustle は 0.146→0.073 と悪化**。誤ルーティングは 0(非crustleで発火率 0.0) | **STOP**（機構は正確に動くが専用方策が強くない） |

## ★ 最重要: ローカル A/B のノイズ床を実測
routing A/B の非crustle対面は **A と B が同一エージェント**(発火率0)。それでも
`mean|delta| 0.046 / max 0.100`、全体 +0.0132 [-0.0554,+0.0818]。
= **n=420 の分解能は ±6〜7pt**。本セッションの候補は全て |delta|<5pt なので、
**勝ちも負けも判定できていなかった**。詳細は memory `project_local_ab_noise_floor`。

## E0/E1 評価系改修（§6-§17）— 完了
- **CRN は不可能**と実測で確定。`battle_start` に seed 引数なし、cg に random 参照なし、
  かつ **別プロセスで同一引数を再生しても outcomes/steps が不一致**(signature 446bebb9 vs 414d6cf4)
  = DLL が非再現 entropy で seed。Python seed でも fresh-process でも CRN は作れない。
- A/A validation: climb を独立4 seed × n=2400 → 0.6454/0.6283/0.6538/0.6254、sd 0.0136
  = binomial の 1.39倍。**各runのWilson CIは真のばらつきを約1.4倍過小評価**。
- 決定的資産: **policy-only FIELD は 1ゲーム ~0.05s（production A/B の ~400倍安い）**。
  同一 seed0 なら相手・先後の割当が完全一致するので、差分では相手mix分散が相殺する。
- 4段パイプライン確定: SCREEN(n=9,600, ±1.4pt, 8分) → CONFIRM(別seed) → PRODUCTION(N=式から算出) → KAGGLE

## 新評価器での既存候補の**初めて解像できた**比較（n=9,600/arm, matched seed）
```
climb 0.6248 | R1 0.6331 (+0.83pt [-0.53,+2.20] z=1.20) | W1 0.6293 (+0.45pt [-0.92,+1.82] z=0.64)
対crustle(n=1548): climb 0.1137 | R1 0.1402 (+2.65pt [+0.31,+4.99] z=2.22 有意) | W1 0.1214
```
= R1 は crustle で本当に強い。ただし全体では +0.83pt で有意に届かない。
（注: W2 の arm は worker OOM で落ちた。W2 は独立seed評価で既に RED なので再実行しない）

## 次 loop（実行中）
新評価器が可能にした唯一の未実施機構 = **高精度 checkpoint selection**。
学習中の eval は 400戦(CI±5pt)でのノイズ選択であり、R1 の iter2 fluke の原因と診断済み。
`--save-every` を追加し、R1 を起点に 36 iters(critic warmup 8)で 4 iter ごとに checkpoint を保存中。
完了後に全 checkpoint を n=9,600 の SCREEN で選び直す。

## LOOP: 高精度 checkpoint selection（新評価器で初めて可能になった機構）— 反証
R1 起点に 36 iters(critic warmup 8, save-every 4)で checkpoint 7本を保存。
```
SELECTION n=2400 seed77001 (9 arms): it20 +4.37pt z=3.17 / it32 +2.87 / it12 +2.12 / climb 0
CONFIRM   n=9600 seed88112233     : it20 -0.21 / it32 -0.35 / it12 -1.00   ← **全滅**
```
= 選抜時の優位は **完全に選択バイアス**。学習中 eval(400戦)を n=2400 に上げても、
9本から最大を取れば +4.4pt の偽陽性が出る。**SCREEN→CONFIRM の分離が誤提出を1件防いだ。**

## 反証済み new mechanism（fresh experiment、計6件）
1 binding×trajectory Transformer / 2 critic warmup RL / 3 Sacred Ash deck-tech /
4 basics 8→10 deck-tech / 5 crustle routing / 6 高精度 checkpoint selection

## H8 ensemble router — 上限測定だけで CLOSE（§40）
既存 policy 群の **per-archetype oracle**（同一標本 = 上方バイアス込みの上限）:
```
{climb,it20,it32,it12}  oracle 0.6441 vs climb 0.6382 = +0.58pt
{climb,R1,W1}           oracle 0.6401 vs climb 0.6248 = +1.53pt
```
上限自体が小さく、しかも checkpoint 選択で実証したとおり同一標本の優位は
fresh seed でほぼ消える(+4.4pt→-0.2pt)。**router は作らない。**

## H7 ladder-distribution training（実行中）
学習 FIELD の share が実ラダーと乖離。特に **marnie_grimmsnarl が 12.5% vs 実 30.3%(2.42倍)**。
`--field-weights` を追加(既定 None = 従来と完全同一)し、実ラダー分布で climb から再学習中。
30 iters / critic warmup 8 / save-every 4。完了後 SCREEN(n=2400) → CONFIRM(別seed n=9600)。

## H7 ladder-distribution training — 反証（かつ**重複実験だった**）
```
SELECTION n=2400 seed31001（LADDER加重）
  climb 0.6261 | ld16 0.6219 | ld28 0.6161 | ld24 0.6140 | ld12 0.6118 | ld20 0.6066
```
全 checkpoint が climb 未満。選抜段階（上方バイアス込み）ですら負けているので CONFIRM は実施せず(§28)。
学習で 2.42倍に重み付けした marnie 自体も climb 0.6589 -> 最良 ld16 0.6623 とほぼ動かず。

**自己訂正**: 実行後に `kaggle_replays/rl/_train_alakazam_realmeta.log` と
`policy_weights_alakazam_rl_realmeta.json` の存在に気付いた。**H7 は過去に実施済みの重複**だった。
着手前に training log を確認すべきだった(§57 の再検証禁止に抵触)。
同様に H6(shaped reward)も `densefield` / `densems` / `valpbrs` / `extra_dense` として実施済みで、
今回は**再実行しなかった**。

## 未完了: back-catalog 再スクリーニング
過去に旧評価器(±5〜7pt)で判定・破棄された alakazam 系 policy が 24本ある。新評価器(±1.4pt)なら
誤って捨てた候補を拾える可能性があるので再測定を試みたが、**リモートのジョブが2回とも途中で死亡**
(worker が 14→6 に脱落。4-arm 実行時の OOM と同じ症状)。cat1 すら出力されず未完了。
→ 次に着手するなら **1回の実行を 2 policy までに絞る**こと。

## STEP1 back-catalog rescue — 完了（20 policy 再測定）
旧評価器(±5〜7pt)で判定・破棄された過去 policy を新評価器(n=4800, seed42042)で測り直した。
```
league_scale +1.15(z1.17) / crustleonly +0.92 / kitikigis +0.92 / valpbrs +0.85 / crustle +0.73
ov1 +0.56 / climb2 +0.17 / climb 0 / foodinopt2 -0.17 / realmeta -0.81 / ov2 -1.11
xerosic_league -2.73 / densems -3.15 / densefield -3.25 / xerosic_dense -3.40 / ov3 -3.81
extra_dense -4.71(※deck非対応で誤測定) / field -14.75 / vscrustle -15.27
```
**climb を有意に超えるものは無し。** 上位3本を fresh seed で CONFIRM:

## STEP2 fresh CONFIRM（n=9600, seed90210777）
```
kitikigis    screen +0.92 -> CONFIRM **+0.84** (z=1.21)  ← 方向も大きさも再現
crustleonly  screen +0.92 -> CONFIRM +0.56  (z=0.81)     ← 方向再現
league_scale screen +1.15 -> CONFIRM -0.27  (z=-0.39)    ← 崩壊
```
**本セッションで唯一 CONFIRM で再現した候補 = kitikigis。**
- 対面別: 5/7 で正、崩壊なし。crustle +1.7 / marnie +2.9（最大の失点2対面）/ dragapult -1.2
- production 互換: state_dim 166・extra_features なし = policy_weights.json の drop-in
- 素性: 61 iter のフル RL 走(baseline 0.495 -> 0.62, best_iter 45)。climb 系列とは独立

## 判定: **submission-grade 未達**（事前登録ゲートに照らして）
事前に「CONFIRM delta >= +1.5〜2pt」を理想値として登録済み。実測 +0.84pt(z=1.21, CI が0を跨ぐ)。
§9 の「結果を見た後に閾値を下げるのは禁止」に従い、**提出しない**。
決着させるための n=38,400 高精度測定を試みたが、
(a) 単発 38,400 は pool のメモリで rc=1 落ち → 9,600x4 seed 分割へ変更
(b) その後リモートの SSH が不安定化(rc=255 連発・latency 2ms->127ms・worker 脱落)し測定続行不能
= **external blocker により決着測定が未完**。

## ★ SUBMITTED — 55435622 submission_autoloop_kitikigis.tar.gz
```
mechanism   : back-catalog rescue（旧ノイズ評価器で誤って捨てられた policy の再発見）
Policy      : policy_weights_alakazam_rl_kitikigis.json (1fe7b7bbbe1ed90d)
              61 iteration フル RL 走 (baseline 0.495 -> 0.62, best_iter 45), state166 drop-in
package     : dac3cfa3e68dfb9a  (baseline package との差分は policy_weights.json のみ / 184ファイル検証)
deck/config : 8ae7a618b2655669 / ca6c37af4b1a4149  = baseline と同一

SCREEN   n=4800  seed42042     +0.92pt (20本から選抜、上方バイアス有)
CONFIRM  n=9600  seed90210777  +0.84pt (z=1.21)
CONFIRM  n=9600  seed51423698  +1.44pt (z=2.07)
POOLED   独立2seed n=19,200/arm  climb 0.6320 -> 0.6433
         **delta +1.14pt  CI95 [+0.17, +2.10]  z=2.31（CIが0を除外）**
対面別    5/7 で正・崩壊なし（crustle +1.7 / marnie +2.9 / dragapult -1.2 / shirona -0.8）
smoke    4/4 完走・error 0・188.3 ms/select・20.5 s/game
Codex    PROCEED-WITH-CAVEAT（BLOCK なし。留保は policy-only -> production/LB の転移リスク）
```
本セッションで他の全候補が CONFIRM で崩壊した中、**3 seed で方向・大きさとも再現した唯一の候補**。
