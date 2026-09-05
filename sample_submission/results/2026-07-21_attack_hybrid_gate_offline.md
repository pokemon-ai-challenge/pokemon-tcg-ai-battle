# ATTACK専用ハイブリッド(案B) — Step0オフラインゲート品質計測

作成: 2026-07-21
関連: `sample_submission/docs/plans/individual/shogo/attack-rulebased-hybrid-implementation-plan.md` Step0
実行: `kaggle_replays/policy_net/_diag_attack_hybrid_gate_offline.py`(`features.npz` test split,
select_type=MAIN、n=13,940。`ptcg_ai/rule_based/main_turn_parts` は読み取り専用でimport・呼び出し
するのみ、ファイルは無変更)

## 背景

案Bは「rule_baseの`proposals.collect_proposals(obs)`が最高スコアに選んだカテゴリが実際に
`"attack"`だった場合のみ、そのATTACK選択をPolicyModelの代わりに採用する」というゲート設計。
実装前に、上位プレイヤーの実データ上でこのゲートがどれだけ正しく機能するかを計測した。

## 結果

| ゲート条件 | 発火件数 | 発火時の一致率(accuracy) | recall(ground truth attack, n=1,573中) |
|---|---:|---:|---:|
| **naive**(全カテゴリの生スコアをそのまま比較) | 6,422 | **23.2%** | 94.9% |
| **refined(採用案)**: draw/board/ability/energyのいずれの提案も無い行に限定 | 350 | **89.7%** | 20.0% |

(ATTACK型選択肢を含む行: 7,185 / MAIN行全体: 13,940)

## 解釈

- **naiveゲートは実用に耐えない。** ATTACKのKOボーナス(1000点)が他カテゴリの得点
  (boardは20点程度)を常に上回るため、「進化を複数回してから最後に攻撃する」ような
  同一ターン内の連続したMAIN選択の**途中**でも即攻撃を選んでしまう。実際にサンプル行
  (row_index=710〜718、同一ターン内で複数回EVOLVEしてから攻撃する流れ)を確認したところ、
  ゲートは毎回「今すぐ攻撃」を提案し、上位プレイヤーの実際の選択(EVOLVE継続)と食い違って
  いた。これは既知の知見(`step2-offline-evaluation.md` §7.3、rule_baseがPLAY/ABILITY/ATTACHで
  一致率が低い)と同根の欠陥であり、素朴に`decide()`をそのまま流用すると同じ弱点を
  ml_policyハイブリッドに持ち込んでしまうことが分かった。
- **refinedゲート(draw/board/ability/energyの提案が1つも無いときだけ発火)は精度89.7%まで
  回復する。** `energy.py`/`ability.py`は「本当に価値のある行動が残っている場合のみ」提案する
  設計(`energyAttached`フラグ確認・`ability_priority`閾値)になっているため、これらが
  1件も無いというのは「本当にもう他にやることがない」ことの妥当な代理指標になっている。
- **recallは20%と低いが、これは意図した安全側の挙動。** 「他の選択肢が形式上まだ残っている
  状態で攻撃を選ぶ」という曖昧な局面ではゲートを発火させずPolicyModelに委ねるため、
  カバレッジは狭い(350/13,940 ≈ 2.5%)。高精度・低頻度のピンポイント介入として機能する
  設計であり、無理に発火頻度を上げようとしない。

## 判断: refinedゲート条件でStep1へ進む

`attack-rulebased-hybrid-implementation-plan.md` Step0の完了条件(94.9%から5pt以上落ちていない
こと)は naive ゲートでは満たせなかったため、ゲート条件を
「draw/board/ability/energyのいずれも提案されていない場合のみ発火」に修正した上でStep1へ進む。
実装計画docの`_try_attack_hybrid`にはこの`_BLOCKING_CATEGORIES`判定を組み込む。
