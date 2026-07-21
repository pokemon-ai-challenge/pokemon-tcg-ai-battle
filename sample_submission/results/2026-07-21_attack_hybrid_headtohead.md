# ATTACK専用ハイブリッド(案B) — Step3オンライン評価(構成Cとのミラー head-to-head)

作成: 2026-07-21
関連: `sample_submission/docs/plans/individual/shogo/attack-rulebased-hybrid-implementation-plan.md` Step3、
[Step0オフラインゲート品質](./2026-07-21_attack_hybrid_gate_offline.md)
実行: `league/_diag_attack_hybrid_head_to_head.py`(構成C重み固定、`attack_hybrid.enabled`のON/OFF
のみを変えるミラー対戦。探索設定は`ml_lethal`共通。先手後手半々、`match_context.reset()`を毎試合、
Wilson 95% CI)

## 結果

| バッチ | 試合数 | attack_hybrid勝数 | 勝率 | 95% CI |
|---|---:|---:|---:|---|
| batch1(seed 40000〜) | 200 | 108 | 54.0% | [47.1%, 60.8%] |
| batch2(seed 41000〜) | 400 | 203 | 50.75% | [45.9%, 55.6%] |
| **合算** | **600** | **311** | **51.8%** | **[47.8%, 55.8%]** |

エラー0件(600試合とも)、平均2.5〜2.9秒/試合。

生ログ: `league/results/_diag_attack_hybrid_configC.json`(batch1)、
`league/results/_diag_attack_hybrid_configC_batch2.json`(batch2)

## 解釈

- **batch1単独(54.0%)は方向としては良く見えたが、batch2(50.75%)でほぼ五分に戻った。** これは
  batch1がノイズだった可能性を示唆する典型的なパターン(policymodel-skill-concentration-
  implementation-plan.md Step4での構成Cの傾向とは対照的に、こちらは追加データで効果が消えた)。
- **合算600試合でCIは[47.8%, 55.8%]と0.5をほぼ中心に据えて挟んでおり、これ以上試合数を
  積み増しても有意差が出る見込みは薄い。** `attack-rulebased-hybrid-implementation-plan.md`
  の判定基準「50%付近: 効果なし」に該当する。
- **考えられる理由:** Step0のオフライン計測(`2026-07-21_attack_hybrid_gate_offline.md`)で、
  このゲートが発火するのはMAIN行のごく一部(test split 13,940件中350件、約2.5%)に限られると
  分かっていた。ゲート自体の精度は89.7%と高いが、そもそも「他にやることが無く、攻撃するか
  どうかだけが問題になる」局面は、PolicyModel自身も比較的間違えにくい(展開判断が終わった後の
  単純な二択に近い)可能性が高く、置き換えの効果が対戦全体の勝率に現れるほど大きくなかったと
  考えられる。

## 判断: この軸をクローズする

**ATTACK専用ハイブリッド(案B、`_BLOCKING_CATEGORIES`ゲート版)は、構成C比で有意な勝率改善を
示さなかった。** `attack-rulebased-hybrid-implementation-plan.md`の完了条件どおり、ここで
中止し理由を記録する。実装自体(`_try_attack_hybrid`、config `attack_hybrid.enabled`)は
既定OFFのまま残し、コードは削除しない(将来ゲート条件を再検討する際の土台として)。

非ミラー評価(`ai-architecture-strategy-codex.md` Priority 0.2相当)へ進める根拠は無いため
実施しない。次の一手は`ai-architecture-strategy-codex.md`の他の優先事項(構成Cの独立
再検証・非ミラー評価、評価split改善等)に戻る。
