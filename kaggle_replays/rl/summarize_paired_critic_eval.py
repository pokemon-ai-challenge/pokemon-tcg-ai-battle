"""``run_paired_critic_eval.py``の生データ(``runs/paired_critic_eval/raw_results.jsonl``)
から集計を作る。

**統計上の注意(重要)**: cgエンジンはPythonからseed設定できないため
(``run_grimmsnarl_t1_eval.play_one_game``のdocstring参照)、同じepisode manifestを
使っても実際の盤面系列(デッキシャッフル・サイド配置)は各評価runで独立に異なる。
つまりこれは厳密な意味でのpaired evaluationではない——manifestが揃えているのは
「対戦相手・先攻後攻・試合順序・試合数」という制御可能な条件だけである。
このためcritic無し/有りの比較はMcNemar検定のような対応のある検定ではなく、
**独立2標本として**扱い、勝率差には独立2標本比率差の95% CIを付ける。
p>0.05は「差が無い」ではなく「有意差を検出できなかった」と記述する。
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RAW = _HERE / "runs" / "paired_critic_eval" / "raw_results.jsonl"
_MANIFEST = _HERE / "runs" / "paired_critic_eval" / "episode_manifest.json"

POOL_OPPONENTS = ["crustle", "mega_lucario_ex", "alakazam", "archaludon_ex",
                 "marnie_grimmsnarl_ex", "rocket_mewtwo_ex", "shirona_garchomp_ex", "dragapult_ex"]


def wilson_ci(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)


def two_sample_diff_ci(w1: int, n1: int, w2: int, n2: int, z: float = 1.96) -> dict:
    """独立2標本の比率差(p1-p2)の95% CI(正規近似、Wald)。n1/n2==0ならnanを返す。"""
    if n1 == 0 or n2 == 0:
        return {"diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}
    p1, p2 = w1 / n1, w2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return {"diff": diff, "ci_lo": diff, "ci_hi": diff, "p_value": float("nan")}
    lo, hi = diff - z * se, diff + z * se
    z_stat = diff / se
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return {"diff": diff, "ci_lo": lo, "ci_hi": hi, "p_value": p_value}


def audit(rows: list[dict], manifest: list[dict]) -> dict:
    total = len(rows)
    key_counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        key_counts[(r["entry_id"], r["idx"])] += 1
    n_unique = len(key_counts)
    dupes = {k: v for k, v in key_counts.items() if v > 1}
    per_entry = defaultdict(int)
    for r in rows:
        per_entry[r["entry_id"]] += 1

    man_by_idx = {m["idx"]: m for m in manifest}
    manifest_mismatches = 0
    for r in rows:
        m = man_by_idx.get(r["idx"])
        if (m is None or m["opponent_id"] != r["opponent_id"] or m["t1_index"] != r["t1_index"]
                or m["seed"] != r["seed"]):
            manifest_mismatches += 1

    coverage_ok = all(v == len(manifest) for v in per_entry.values())
    return {
        "total_rows": total, "unique_keys": n_unique, "duplicate_keys": len(dupes),
        "duplicate_extra_rows": sum(v - 1 for v in dupes.values()),
        "per_entry_counts": dict(per_entry), "manifest_size": len(manifest),
        "manifest_mismatches": manifest_mismatches,
        "each_entry_matches_manifest_size": coverage_ok,
    }


def rate(rows: list[dict]) -> tuple[int, int]:
    valid = [r for r in rows if r["error"] is None]
    wins = sum(1 for r in valid if r["t1_win"] is True)
    return wins, len(valid)


def main() -> None:
    rows = [json.loads(l) for l in _RAW.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))

    audit_result = audit(rows, manifest)
    print("=== 1. 生データ整合性監査 ===")
    for k, v in audit_result.items():
        if k != "per_entry_counts":
            print(f"  {k}: {v}")
    print("  per_entry_counts:")
    for e, c in sorted(audit_result["per_entry_counts"].items()):
        print(f"    {e}: {c}")
    expected_unique = 8 * len(manifest)
    print(f"  期待unique件数: 8 checkpoint x {len(manifest)}manifest = {expected_unique}")
    assert audit_result["unique_keys"] == expected_unique, "unique件数が期待値と一致しない"
    assert audit_result["duplicate_keys"] == 0, "重複行が見つかった(要調査、削除しないこと)"
    assert audit_result["manifest_mismatches"] == 0, "manifestと一致しない行がある"
    assert audit_result["each_entry_matches_manifest_size"], "checkpointごとの件数が1,300でない"
    print("  -> 監査OK。重複・旧run混入・manifest不一致は無い。cleanなデータのみ。")

    by_entry = defaultdict(list)
    for r in rows:
        by_entry[r["entry_id"]].append(r)

    entries_ordered = ["checkpoint_0", "checkpoint_500", "checkpoint_1000", "checkpoint_2000"]

    print()
    print("=== 2a. 教師との直接対戦(h2h, teacher_rule, n=300/checkpoint)を単独報告 ===")
    h2h_summary = {}
    for m in entries_ordered:
        for arm in ("critic_free", "critic"):
            key = f"{arm}/{m}"
            opp_rows = [r for r in by_entry[key] if r["opponent_id"] == "teacher_rule"]
            w, n = rate(opp_rows)
            lo, hi = wilson_ci(w, n)
            h2h_summary[key] = {"wins": w, "n": n, "wr": w / n, "ci_lo": lo, "ci_hi": hi}
            print(f"  {key}: {w}/{n} = {w/n:.4f}  Wilson95%CI=[{lo:.4f},{hi:.4f}]")

    print()
    print("=== 2b. 固定評価プール(8相手合算、教師直接対戦は含まない、n=1000/checkpoint)を単独報告 ===")
    pool_summary = {}
    for m in entries_ordered:
        for arm in ("critic_free", "critic"):
            key = f"{arm}/{m}"
            opp_rows = [r for r in by_entry[key] if r["opponent_id"] in POOL_OPPONENTS]
            w, n = rate(opp_rows)
            lo, hi = wilson_ci(w, n)
            pool_summary[key] = {"wins": w, "n": n, "wr": w / n, "ci_lo": lo, "ci_hi": hi}
            print(f"  {key}: {w}/{n} = {w/n:.4f}  Wilson95%CI=[{lo:.4f},{hi:.4f}]")

    print()
    print("=== 3. critic無し vs critic有り: 独立2標本の勝率差(95%CI, h2h) ===")
    h2h_diff = {}
    for m in entries_ordered:
        a, b = h2h_summary[f"critic_free/{m}"], h2h_summary[f"critic/{m}"]
        d = two_sample_diff_ci(a["wins"], a["n"], b["wins"], b["n"])
        h2h_diff[m] = d
        sig = "有意差を検出できなかった" if not (d["p_value"] < 0.05) else "有意差あり(p<0.05)"
        print(f"  {m}: diff(無し-有り)={d['diff']:+.4f}  95%CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
             f"p={d['p_value']:.3f} -> {sig}")

    print()
    print("=== 3b. critic無し vs critic有り: 独立2標本の勝率差(95%CI, 固定プール) ===")
    pool_diff = {}
    for m in entries_ordered:
        a, b = pool_summary[f"critic_free/{m}"], pool_summary[f"critic/{m}"]
        d = two_sample_diff_ci(a["wins"], a["n"], b["wins"], b["n"])
        pool_diff[m] = d
        sig = "有意差を検出できなかった" if not (d["p_value"] < 0.05) else "有意差あり(p<0.05)"
        print(f"  {m}: diff(無し-有り)={d['diff']:+.4f}  95%CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
             f"p={d['p_value']:.3f} -> {sig}")

    print()
    print("=== 4. crustle / mega_lucario_ex 単独(n=200/checkpoint、独立2標本diff) ===")
    weak_summary = {}
    for opp in ("crustle", "mega_lucario_ex"):
        print(f"  --- {opp} ---")
        for m in entries_ordered:
            a_rows = [r for r in by_entry[f"critic_free/{m}"] if r["opponent_id"] == opp]
            b_rows = [r for r in by_entry[f"critic/{m}"] if r["opponent_id"] == opp]
            wa, na = rate(a_rows)
            wb, nb = rate(b_rows)
            loa, hia = wilson_ci(wa, na)
            lob, hib = wilson_ci(wb, nb)
            d = two_sample_diff_ci(wa, na, wb, nb)
            weak_summary[(opp, m)] = {"critic_free": {"w": wa, "n": na, "ci": (loa, hia)},
                                      "critic": {"w": wb, "n": nb, "ci": (lob, hib)}, "diff": d}
            sig = "有意差を検出できなかった" if not (d["p_value"] < 0.05) else "有意差あり(p<0.05)"
            print(f"    {m}: 無し={wa}/{na}={wa/na:.3f}[{loa:.3f},{hia:.3f}]  "
                 f"有り={wb}/{nb}={wb/nb:.3f}[{lob:.3f},{hib:.3f}]  "
                 f"diff={d['diff']:+.4f}[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] p={d['p_value']:.3f} -> {sig}")

    print()
    print("=== 5. 先攻/後攻(t1_index)で層別した固定プール勝率 ===")
    strat_summary = {}
    for m in entries_ordered:
        for arm in ("critic_free", "critic"):
            key = f"{arm}/{m}"
            for t1_idx in (0, 1):
                rows_s = [r for r in by_entry[key] if r["opponent_id"] in POOL_OPPONENTS
                         and r["t1_index"] == t1_idx]
                w, n = rate(rows_s)
                lo, hi = wilson_ci(w, n)
                strat_summary[(key, t1_idx)] = {"w": w, "n": n, "wr": w / n if n else float("nan"),
                                                "ci_lo": lo, "ci_hi": hi}
        print(f"  {m}:")
        for arm in ("critic_free", "critic"):
            key = f"{arm}/{m}"
            s0 = strat_summary[(key, 0)]
            s1 = strat_summary[(key, 1)]
            print(f"    {arm}: t1_index=0(先攻/deck0側) {s0['w']}/{s0['n']}={s0['wr']:.3f}  "
                 f"t1_index=1(後攻/deck1側) {s1['w']}/{s1['n']}={s1['wr']:.3f}")

    out = {"audit": audit_result, "h2h": h2h_summary, "pool": pool_summary,
          "h2h_diff": h2h_diff, "pool_diff": pool_diff,
          "weak_opponents": {f"{k[0]}|{k[1]}": v for k, v in weak_summary.items()},
          "stratified_by_t1_index": {f"{k[0]}|t1_index={k[1]}": v for k, v in strat_summary.items()}}
    out_path = _HERE / "runs" / "paired_critic_eval" / "summary_v2.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
