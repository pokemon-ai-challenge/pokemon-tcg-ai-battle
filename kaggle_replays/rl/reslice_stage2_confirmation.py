"""Stage2の追加346試合(game_id 154-499)だけを切り出して再集計する。

Stage1(0-153, 候補選抜用)を混ぜず、「確認用」として独立に報告するための一回限りのスクリプト。
既存の summarize/wilson_ci/diff_ci_independent をそのまま再利用し、集計式の二重実装を避ける。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from eval_strength_stage import diff_ci_independent, summarize, wilson_ci  # noqa: E402

STAGE1_DIR = _HERE.parent.parent / "runs" / "ogerpon" / "planner" / "stage1"
CONFIRM_START = 154
CONFIRM_END = 500


def load(name):
    d = json.loads((STAGE1_DIR / f"raw_{name}.json").read_text(encoding="utf-8"))
    return d["name"], d["ml_config"], d["results"]


def main():
    arms = ["baseline", "attach_conservative"]
    loaded = {n: load(n) for n in arms}

    # 前提確認: 2 arm は同一index=同一matchup/同一先攻後攻であること
    rb = loaded["baseline"][2]
    ra = loaded["attach_conservative"][2]
    assert len(rb) == 500 and len(ra) == 500, f"想定外の試合数: {len(rb)}, {len(ra)}"
    mism = sum(1 for x, y in zip(rb, ra)
               if (x["archetype"], x["learner_index"]) != (y["archetype"], y["learner_index"]))
    assert mism == 0, f"baseline/attach_conservative のmatchup/side不一致: {mism}件"

    out = {"note": "Stage1(0-153)は候補選抜用。ここは Stage2 で新たに追加された "
                   f"game_id {CONFIRM_START}-{CONFIRM_END - 1} ({CONFIRM_END - CONFIRM_START}試合) のみを"
                   "独立の確認用サンプルとして再集計したもの。",
           "confirmation_range": [CONFIRM_START, CONFIRM_END - 1],
           "n_games": CONFIRM_END - CONFIRM_START}

    summaries = {}
    for name, (arm_name, ml_config, results) in loaded.items():
        sliced = results[CONFIRM_START:CONFIRM_END]
        s = summarize(arm_name, ml_config, sliced, opponents=None)
        summaries[name] = s
        print(f"[{name}] confirm346: wr={s['winrate']:.4f} wins={s['wins']}/{s['valid']} "
              f"95%CI={s['wilson_95ci']} err={s['errors']} illegal={s['illegal_actions']} "
              f"timeout={s['timeouts']}", flush=True)

    diff, lo, hi = diff_ci_independent(
        summaries["baseline"]["wins"], summaries["baseline"]["valid"],
        summaries["attach_conservative"]["wins"], summaries["attach_conservative"]["valid"])
    print(f"[attach_conservative - baseline] confirm346: diff={diff:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]",
          flush=True)

    out["summaries"] = summaries
    out["comparison"] = {"attach_conservative - baseline": {
        "diff": diff, "ci95": [lo, hi], "note": "独立2標本の比率差(paired gameではない)"}}

    # 参考として Stage1 (0-153, 候補選抜用) の値も並記しておく
    stage1_ref = {}
    for name, (arm_name, ml_config, results) in loaded.items():
        s1 = summarize(arm_name, ml_config, results[:CONFIRM_START], opponents=None)
        stage1_ref[name] = {"winrate": s1["winrate"], "wins": s1["wins"], "valid": s1["valid"],
                             "wilson_95ci": s1["wilson_95ci"]}
    out["stage1_reference_0_153_candidate_selection_only"] = stage1_ref

    out_path = STAGE1_DIR / "confirmation_346.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
