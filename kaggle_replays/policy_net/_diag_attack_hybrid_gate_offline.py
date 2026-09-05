#!/usr/bin/env python3
"""Diagnostic (throwaway, not shipped): ATTACK専用ハイブリッド(案B)のゲート品質をオフライン計測する。

attack-rulebased-hybrid-implementation-plan.md Step0。`features.npz` の test split
(select_type==0 = MAIN のみ)を対象に、行ごとに元の observation を読み直し、
`ptcg_ai.rule_based.main_turn_parts.proposals.collect_proposals(obs)` を呼んでカテゴリ別
提案を集める。

2つのゲート条件を比較する:

1. **naive**: 最高スコアのカテゴリが "attack" だった行をそのまま採用。
2. **blocking-category(採用案)**: draw/board/ability/energy のいずれかの提案が
   1つでも存在する行(=まだ他にやるべき展開が残っている)はゲート対象外とし、
   それ以外(他にやることが無い)かつ最高スコアが "attack" の行だけを採用。

naiveゲートは、進化を複数回してから最後に攻撃するような「同一ターン内の連続した
MAIN選択」の途中でも、ATTACKのKOボーナス(1000点)がboard等の得点(20点程度)を
常に上回ってしまい、他の展開を飛ばして即攻撃を選んでしまう欠陥がある
(rule_based自体がPLAY/ABILITY/ATTACHで一致率が低いのと同根)。blocking-category条件は
これを避けるためのもの。

`ptcg_ai/rule_based/` は読み取り専用で呼ぶだけ(ファイルは変更しない、境界制約は
attack-rulebased-hybrid-strategy.md §4.2 / 実装計画の前提と同じ)。

実行:
    PYTHONIOENCODING=utf-8 python kaggle_replays/policy_net/_diag_attack_hybrid_gate_offline.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
if str(_SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

_FEATURES = _HERE / "features.npz"
_POLICY_POSITIONS = _REPO_ROOT / "kaggle_replays" / "training_data" / "policy_positions.jsonl.gz"
_OUT_PATH = _HERE / "_diag_attack_hybrid_gate_offline_results.json"

_TEST = 2
_SELECT_TYPE_MAIN = 0

# このカテゴリのどれかが提案されていたら、まだ他にやるべき展開が残っている
# とみなしATTACKゲートを発火させない(採用案の核心)。
_BLOCKING_CATEGORIES = {"draw", "board", "ability", "energy"}


def main() -> None:
    t_start = time.time()
    print("=== ATTACK hybrid gate オフライン品質計測(Step0) ===\n")

    data = np.load(_FEATURES, allow_pickle=True)
    split = data["split"].astype(np.int64)
    select_type = data["select_type"].astype(np.int64)
    row_index_all = data["row_index"].astype(np.int64)

    mask = (split == _TEST) & (select_type == _SELECT_TYPE_MAIN)
    n_target = int(mask.sum())
    print(f"対象(test split, select_type=MAIN): {n_target} 件")

    row_index = row_index_all[mask]
    target_lines = {int(li): pos for pos, li in enumerate(row_index)}

    os.chdir(_SAMPLE_SUBMISSION_DIR)
    from cg.api import OptionType, to_observation_class  # noqa: E402
    from ptcg_ai.rule_based.main_turn_parts import proposals as rb_proposals  # noqa: E402
    from ptcg_ai.rule_based.main_turn_parts import weights as rb_weights  # noqa: E402

    def total_score(p):
        return p.score + rb_weights.CATEGORY_BASE_WEIGHT.get(p.category, 0.0)

    n = len(row_index)
    has_attack_option = np.zeros(n, dtype=bool)
    ground_truth_is_attack = np.zeros(n, dtype=bool)

    naive_fires = np.zeros(n, dtype=bool)
    naive_match = np.zeros(n, dtype=bool)

    refined_eligible = np.zeros(n, dtype=bool)  # blocking categoryが無い(ゲート検討対象)
    refined_fires = np.zeros(n, dtype=bool)
    refined_match = np.zeros(n, dtype=bool)

    n_error = 0
    n_processed = 0

    t0 = time.time()
    with gzip.open(_POLICY_POSITIONS, "rt", encoding="utf-8") as f:
        line_no = -1
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line_no += 1
            if line_no not in target_lines:
                continue
            pos = target_lines[line_no]

            row = json.loads(raw_line)
            try:
                obs = to_observation_class({**row["observation"], "logs": []})
                chosen_idx = int(row["chosen_index"])
                options = obs.select.option
                ground_truth_is_attack[pos] = options[chosen_idx].type == OptionType.ATTACK
                if not any(opt.type == OptionType.ATTACK for opt in options):
                    n_processed += 1
                    continue
                has_attack_option[pos] = True

                props = rb_proposals.collect_proposals(obs)
                if not props:
                    n_processed += 1
                    continue
                best = max(props, key=total_score)
                predicted = best.select[0] if len(best.select) == 1 else None

                if best.category == "attack":
                    naive_fires[pos] = True
                    naive_match[pos] = predicted == chosen_idx

                categories_present = {p.category for p in props}
                if not (categories_present & _BLOCKING_CATEGORIES):
                    refined_eligible[pos] = True
                    if best.category == "attack":
                        refined_fires[pos] = True
                        refined_match[pos] = predicted == chosen_idx
            except Exception as exc:  # noqa: BLE001 - 1行の失敗で全体を止めない
                n_error += 1
                if n_error <= 10:
                    print(f"    警告: 失敗(row_index={line_no}): {exc!r}", file=sys.stderr)

            n_processed += 1
            if n_processed % 2000 == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0.0
                print(f"    {n_processed}/{n_target} 件処理済み(経過 {elapsed:.1f}s、{rate:.1f} rows/sec)")

    elapsed = time.time() - t0
    print(f"\n完了: {n_processed}/{n_target} 件処理(経過 {elapsed:.1f}s)、error={n_error}件")

    n_has_attack_option = int(has_attack_option.sum())
    n_gt_attack = int(ground_truth_is_attack.sum())

    def summarize(fires_mask, match_mask):
        n_fires = int(fires_mask.sum())
        acc = float(match_mask[fires_mask].mean()) if n_fires else None
        recall_hit = int((fires_mask & match_mask & ground_truth_is_attack).sum())
        recall = (recall_hit / n_gt_attack) if n_gt_attack else None
        return {"n_fires": n_fires, "accuracy": acc, "recall_among_ground_truth_attack": recall}

    naive_summary = summarize(naive_fires, naive_match)
    refined_summary = summarize(refined_fires, refined_match)
    n_eligible = int(refined_eligible.sum())

    print("\n=== 結果比較 ===")
    print(f"MAIN行(test split)                          : {n_target}")
    print(f"ATTACK型選択肢を含む行                        : {n_has_attack_option}")
    print(f"ground truthがATTACKだった行                  : {n_gt_attack}")
    print()
    print(f"[naive] ゲート発火                            : {naive_summary['n_fires']}")
    print(f"[naive] 発火時の一致率(accuracy)              : {naive_summary['accuracy']}")
    print(f"[naive] recall(ground truth attack中)          : {naive_summary['recall_among_ground_truth_attack']}")
    print()
    print(f"[refined] ゲート検討対象(blocking category無し): {n_eligible}")
    print(f"[refined] ゲート発火                           : {refined_summary['n_fires']}")
    print(f"[refined] 発火時の一致率(accuracy)             : {refined_summary['accuracy']}")
    print(f"[refined] recall(ground truth attack中)         : {refined_summary['recall_among_ground_truth_attack']}")

    out = {
        "n_main_rows_test": n_target,
        "n_has_attack_option": n_has_attack_option,
        "n_ground_truth_attack": n_gt_attack,
        "naive_gate": naive_summary,
        "refined_gate": {**refined_summary, "n_eligible": n_eligible, "blocking_categories": sorted(_BLOCKING_CATEGORIES)},
        "n_error": n_error,
        "elapsed_sec": time.time() - t_start,
    }
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n結果を書き出しました: {_OUT_PATH}")


if __name__ == "__main__":
    main()
