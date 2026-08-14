"""lucario_gate 判定ロジックの回帰テスト(cg 不使用・合成)。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lucario_gate as LG  # noqa: E402

_CONTRACT = LG.load_contract()


def _rec(rid, wr, n, dec):
    return {"recipe_id": rid, "n": n, "wins": round(wr * n), "winrate": wr, "sprt_decision": dec}


def test_verify_frozen():
    assert LG.verify_frozen() == [], "frozen gate hash 不一致"


def test_aggregate_uniform():
    pr = [_rec(f"0{i}", 0.60, 100, "PROMOTE") for i in range(1, 6)]
    a = LG.aggregate(pr)
    assert abs(a["point"] - 0.60) < 1e-9
    # Var = 5*(1/5)^2 * 0.6*0.4/100 = 0.00048 -> se ~ 0.02191
    assert abs(a["se"] - 0.0219089) < 1e-4
    assert a["one_sided_95_lower"] < a["point"]


def test_aggregate_uses_stratified_not_pooled():
    # 偏った n でも recipe-uniform(pool でない)ことを確認: 大 n 1つに引っ張られない。
    pr = [_rec("01", 0.90, 1000, "PROMOTE")] + [_rec(f"0{i}", 0.50, 50, "TRUNCATED_NO_PROMOTE") for i in range(2, 6)]
    a = LG.aggregate(pr)
    assert abs(a["point"] - (0.90 + 0.50 * 4) / 5) < 1e-9   # = 0.58(pool なら 0.88 付近)


def test_collapse_fires_on_clear_loss():
    pr = [_rec("01", 0.30, 100, "FUTILITY")] + [_rec(f"0{i}", 0.6, 100, "PROMOTE") for i in range(2, 6)]
    assert LG.collapse_check(pr, _CONTRACT) == ["01"]


def test_collapse_not_fire_on_noise():
    pr = [_rec("01", 0.48, 100, "TRUNCATED_NO_PROMOTE")] + [_rec(f"0{i}", 0.6, 100, "PROMOTE") for i in range(2, 6)]
    assert LG.collapse_check(pr, _CONTRACT) == []   # 0.48 は 0.45 未満でない


def test_decide_promote():
    pr = [_rec(f"0{i}", 0.56, 300, "PROMOTE") for i in range(1, 5)] + [_rec("05", 0.55, 300, "TRUNCATED_NO_PROMOTE")]
    agg = LG.aggregate(pr)
    d = LG.decide(True, pr, agg, [], True, _CONTRACT)
    assert d["decision"] == "PROMOTE", d
    assert d["promote_recipes"] == 4 and d["aggregate_point_ge_053"] and d["aggregate_lower_gt_050"]


def test_decide_fail_majority_futility():
    pr = [_rec(f"0{i}", 0.40, 300, "FUTILITY") for i in range(1, 4)] + [_rec(f"0{i}", 0.6, 300, "PROMOTE") for i in range(4, 6)]
    d = LG.decide(True, pr, LG.aggregate(pr), [], True, _CONTRACT)
    assert d["decision"] == "FAIL" and "FUTILITY" in d["reason"]


def test_decide_fail_collapse():
    pr = [_rec("01", 0.30, 300, "FUTILITY")] + [_rec(f"0{i}", 0.6, 300, "PROMOTE") for i in range(2, 6)]
    col = LG.collapse_check(pr, _CONTRACT)
    d = LG.decide(True, pr, LG.aggregate(pr), col, True, _CONTRACT)
    assert d["decision"] == "FAIL" and "collapse" in d["reason"]


def test_decide_review_borderline():
    # 3/5 PROMOTE、2/5 TRUNCATED、collapse なし、futility 過半でない → REVIEW
    pr = [_rec(f"0{i}", 0.54, 300, "PROMOTE") for i in range(1, 4)] + [_rec(f"0{i}", 0.52, 300, "TRUNCATED_NO_PROMOTE") for i in range(4, 6)]
    d = LG.decide(True, pr, LG.aggregate(pr), [], True, _CONTRACT)
    assert d["decision"] == "REVIEW", d


def test_decide_fail_integrity():
    pr = [_rec(f"0{i}", 0.56, 300, "PROMOTE") for i in range(1, 6)]
    d = LG.decide(True, pr, LG.aggregate(pr), [], False, _CONTRACT)
    assert d["decision"] == "FAIL" and "integrity" in d["reason"]


def test_decide_review_when_point_below_practical():
    # 5/5 PROMOTE だが aggregate point < 0.53(実用閾値未達)→ REVIEW(FAIL でない)
    pr = [_rec(f"0{i}", 0.515, 3000, "PROMOTE") for i in range(1, 6)]
    agg = LG.aggregate(pr)
    d = LG.decide(True, pr, agg, [], True, _CONTRACT)
    assert d["decision"] == "REVIEW" and not d["aggregate_point_ge_053"], d


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_run_all())
