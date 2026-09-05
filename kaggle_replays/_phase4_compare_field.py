"""baseline と候補の FIELD 比較レポート(アーキタイプ別 paired 差つき)。

指標の扱い(ユーザー指示に基づく固定ルール):
  - **主指標** = 事前定義7アーキタイプ全部(**crustle を含む**)の FIELD 重み付き勝率。
  - **副次指標** = crustle 除外値。デッキ相性で leaf 変更の効果が埋もれていないかの確認用。
    **PROMOTE 判定・正式な採用根拠には使わない**(レポート上も明示ラベルを付ける)。
  - crustle は必ず baseline と候補の差を残し、
    「両者とも低い(デッキ由来)」のか「候補でさらに悪化する(変更由来)」のかを区別する。

差の信頼区間は Newcombe hybrid-score 法(2標本比率の差)。両アームは同じ seed_start・同じ相手・
同じ試合数で回すが、cg のネイティブシャッフルは seed 制御外なので **完全な paired ではない**
(seed 一致の独立2標本として扱う)。この限界はレポートにも書く。

使い方:
  python kaggle_replays/_phase4_compare_field.py \
      --baseline kaggle_replays/_p2B.json --candidate kaggle_replays/_p2B_leaf_a05.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent

FIELD_SHARES = {
    "mega_lucario_ex": 1257, "archaludon_ex": 1078, "crustle": 737,
    "dragapult_ex": 625, "marnie_grimmsnarl_ex": 591,
    "rocket_mewtwo_ex": 247, "shirona_garchomp_ex": 181,
}
SECONDARY_EXCLUDE = {"crustle"}


def _wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _newcombe_diff_ci(w1: int, n1: int, w2: int, n2: int, z: float = 1.96) -> tuple[float, float]:
    """p1 - p2 の Newcombe hybrid-score 信頼区間(小標本でも破綻しない)。"""
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0)
    p1, p2 = w1 / n1, w2 / n2
    l1, u1 = _wilson(w1, n1, z)
    l2, u2 = _wilson(w2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def _weighted(rows: dict[str, dict], exclude: set[str]) -> dict:
    use = {a: r for a, r in rows.items() if a not in exclude and r["n"]}
    tot = sum(FIELD_SHARES.get(a, 0) for a in use)
    if not tot:
        return {"win_rate": None, "archetypes": []}
    wr = sum(FIELD_SHARES.get(a, 0) * (r["w"] / r["n"]) for a, r in use.items()) / tot
    return {"win_rate": round(wr, 4), "archetypes": sorted(use)}


def _rows(payload: dict) -> dict[str, dict]:
    out = {}
    for r in payload.get("per_archetype", []):
        out[r["archetype"]] = {"n": r["valid_games"] or 0, "w": r["climb_wins"] or 0,
                               "fail": r["search"]["begin_fail_rate"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-label", default="climb_baseline")
    ap.add_argument("--candidate-label", default="climb_v15_leaf_a05")
    ap.add_argument("--out", default=str(_HERE / "_phase4_compare_field.json"))
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    B, C = _rows(base), _rows(cand)

    per = []
    for arch in FIELD_SHARES:
        b, c = B.get(arch), C.get(arch)
        if not b or not c or not b["n"] or not c["n"]:
            per.append({"archetype": arch, "status": "missing"})
            continue
        pb, pc = b["w"] / b["n"], c["w"] / c["n"]
        lo, hi = _newcombe_diff_ci(c["w"], c["n"], b["w"], b["n"])
        per.append({
            "archetype": arch,
            "baseline": {"n": b["n"], "wins": b["w"], "win_rate": round(pb, 4),
                         "wilson95": [round(x, 4) for x in _wilson(b["w"], b["n"])]},
            "candidate": {"n": c["n"], "wins": c["w"], "win_rate": round(pc, 4),
                          "wilson95": [round(x, 4) for x in _wilson(c["w"], c["n"])]},
            "diff": round(pc - pb, 4),
            "diff_95ci": [round(lo, 4), round(hi, 4)],
            "significant": lo > 0 or hi < 0,
            "search_fail": {"baseline": b["fail"], "candidate": c["fail"]},
        })

    crustle = next((p for p in per if p["archetype"] == "crustle"), None)
    crustle_verdict = None
    if crustle and crustle.get("status") != "missing":
        pb = crustle["baseline"]["win_rate"]
        pc = crustle["candidate"]["win_rate"]
        low_both = pb < 0.25 and pc < 0.25
        worse = crustle["diff"] < 0 and crustle["diff_95ci"][1] < 0
        crustle_verdict = {
            "baseline_win_rate": pb, "candidate_win_rate": pc,
            "diff": crustle["diff"], "diff_95ci": crustle["diff_95ci"],
            "classification": (
                "leaf_makes_it_significantly_worse" if worse
                else "both_low_no_significant_change" if low_both
                else "changed_or_not_low"),
            "note": "両者とも低いならデッキ相性由来。候補側だけ有意に下がるなら leaf 変更由来。",
        }

    fail_clean = all(
        (p.get("search_fail", {}).get("baseline") in (0, 0.0)
         and p.get("search_fail", {}).get("candidate") in (0, 0.0))
        for p in per if p.get("status") != "missing")

    primary = {
        "baseline": _weighted(B, set()),
        "candidate": _weighted(C, set()),
    }
    primary["delta"] = (round(primary["candidate"]["win_rate"] - primary["baseline"]["win_rate"], 4)
                        if primary["candidate"]["win_rate"] is not None
                        and primary["baseline"]["win_rate"] is not None else None)
    secondary = {
        "baseline": _weighted(B, SECONDARY_EXCLUDE),
        "candidate": _weighted(C, SECONDARY_EXCLUDE),
    }
    secondary["delta"] = (round(secondary["candidate"]["win_rate"] - secondary["baseline"]["win_rate"], 4)
                          if secondary["candidate"]["win_rate"] is not None
                          and secondary["baseline"]["win_rate"] is not None else None)

    out = {
        "labels": {"baseline": args.baseline_label, "candidate": args.candidate_label},
        "PRIMARY_field_weighted_all7_including_crustle": primary,
        "SECONDARY_field_weighted_excluding_crustle": {
            **secondary,
            "USAGE": "診断専用。PROMOTE判定・採用根拠には使用しない(ユーザー指示)。",
        },
        "per_archetype": per,
        "crustle_paired_verdict": crustle_verdict,
        "ALL_SEARCH_CLEAN": fail_clean,
        "VALID_FOR_COMPARISON": fail_clean,
        "caveat": ("両アームは同じ seed_start/相手/試合数だが cg のシャッフルは seed 制御外のため "
                   "完全な paired ではない。seed 一致の独立2標本として Newcombe 法で差のCIを出している。"),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "per_archetype"},
                     ensure_ascii=False, indent=2))
    print("\nper-archetype:")
    for p in per:
        if p.get("status") == "missing":
            print(f"  {p['archetype']:24s} MISSING")
            continue
        print(f"  {p['archetype']:24s} base={p['baseline']['win_rate']:.3f} "
              f"cand={p['candidate']['win_rate']:.3f} diff={p['diff']:+.3f} "
              f"CI={p['diff_95ci']} sig={p['significant']}")
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
