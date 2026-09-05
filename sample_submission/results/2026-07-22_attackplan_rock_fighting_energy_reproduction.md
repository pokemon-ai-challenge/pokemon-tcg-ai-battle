# Attack-enabling search — Rock Fighting Energy 再現確認

作成: 2026-07-22  
関連: `docs/plans/attack-enabling-search/design-and-implementation-plan.md` §3.5/§3.6、
`ptcg_ai/search/attack_plan.py`

リプレイ: `battle_review_viewer/replays/codex-attackplan-rock-fighting-energy.json`
（Battle Review Viewer 向けの局面再現リプレイ。改造ハンマー後の攻撃可能状態で意図的に終了。）

## 目的

Rock Fighting Energy によりフーディンの攻撃効果が無効化される局面で、攻撃準備探索が
「無意味な攻撃」を検出し、改造ハンマーを選べることを実ゲームエンジンで確認する。

## 再現条件

- 自分のバトル場: フーディン (ID 743, `Powerful Hand`)、基本超エネルギー付き。
- 相手のバトル場: メガルカリオex (ID 678)、Rock Fighting Energy (ID 20) 付き。
- 自分の手札: 改造ハンマー (ID 1081)。
- テスト用デッキは上記カードを4枚ずつ含め、残りを基本エネルギーで埋めた60枚。
- 実行時の planner 設定: `enabled: true`、`shadow_only: false`、
  `max_root_actions: 0`（v0 のマスク再選択だけを検証）。

Rock Fighting Energy は、付いている闘ポケモンに対する相手の攻撃の**効果**を防ぐ。
`Powerful Hand` はダメージカウンターを置く攻撃効果なので、対象になる。

## 結果

| 確認項目 | 実測結果 |
|---|---|
| 改造ハンマー前の `Powerful Hand` | `direct_damage=0` |
| 同攻撃の KO / サイド / 有益な副作用 | すべて false |
| planner が返した MAIN 選択 | 改造ハンマー (ID 1081) |
| 改造ハンマーの対象選択 | 相手バトル場の Rock Fighting Energy |
| 解決後の Rock Fighting Energy | トラッシュ済み |
| 解決後の `Powerful Hand` | 有益な結果（この手札量では KO 相当） |

したがって、対象局面は実エンジンで再現でき、v0 の「無意味な0ダメージ攻撃をマスクして
残る policy 上位手を採用する」経路で改造ハンマーが選ばれることを確認した。

## 注意点

- `Powerful Hand` は手札枚数に応じてダメージカウンターを置く可変攻撃である。今回の再現時は
  手札が多く、ハンマー後の攻撃は通常の HP 差分ではなく KO として解決された。
  `attack_plan` は HP 差分だけでなく `ko` / `prize_taken` / `won` を成功条件に含めるため、
  このケースも正しく有効と扱う。
- v0 は、0ダメージ攻撃を除外した後の **PolicyModel スコア**で残る候補を選ぶ。今回の確認では
  改造ハンマーが残候補の最上位となる条件を与えた。実戦での発火率・スコア順位・時間分位は
  `ml_lethal_attackplan_shadow` を使った Step 0 の多数試合計測で別途確認する。
- 本結果は v1 の1-prep因果探索そのものではなく、v0 のマスク再選択を実エンジンで確認したもの。
