"""汎用 opponent_gate の回帰テスト + freeze artifact 不変確認(cg 不使用)。"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import opponent_gate as OG  # noqa: E402

_CONTRACT_PATH = _HERE / "contracts" / "archaludon_ex_opponent_gate_v1.json"
_CONTRACT = OG.load_contract(_CONTRACT_PATH)
ARCH_GATE_HASH = "2d087f8f074e9c96"
LUCARIO_GATE_HASH = "a058f5a05e9fb0ce"       # 変更してはいけない historical artifact


def _rec(rid, wr, n, dec):
    return {"recipe_id": rid, "n": n, "wins": round(wr * n), "winrate": wr, "sprt_decision": dec}


def test_archaludon_gate_frozen():
    assert OG.verify_frozen(_CONTRACT_PATH, ARCH_GATE_HASH) == []


def test_lucario_frozen_artifacts_unchanged():
    # 汎用化で Lucario の frozen artifact を壊していないこと。
    lg = _HERE / "lucario_opponent_gate_v1.json"
    assert OG.file_hash(lg) == LUCARIO_GATE_HASH, "Lucario gate hash が変わっている(freeze 違反)"


def test_aggregate_stratified():
    pr = [_rec(f"0{i}", 0.60, 100, "PROMOTE") for i in range(1, 6)]
    a = OG.aggregate(pr)
    assert abs(a["point"] - 0.60) < 1e-9 and abs(a["se"] - 0.0219089) < 1e-4


def test_collapse_fires_and_noise_safe():
    pr = [_rec("01", 0.30, 100, "FUTILITY")] + [_rec(f"0{i}", 0.6, 100, "PROMOTE") for i in range(2, 6)]
    assert OG.collapse_check(pr, _CONTRACT) == ["01"]
    pr2 = [_rec("01", 0.48, 100, "TRUNCATED_NO_PROMOTE")] + [_rec(f"0{i}", 0.6, 100, "PROMOTE") for i in range(2, 6)]
    assert OG.collapse_check(pr2, _CONTRACT) == []


def test_decide_promote():
    pr = [_rec(f"0{i}", 0.56, 300, "PROMOTE") for i in range(1, 5)] + [_rec("05", 0.55, 300, "TRUNCATED_NO_PROMOTE")]
    d = OG.decide(True, pr, OG.aggregate(pr), [], True, _CONTRACT)
    assert d["decision"] == "PROMOTE", d


def test_decide_fail_and_review():
    prf = [_rec(f"0{i}", 0.40, 300, "FUTILITY") for i in range(1, 4)] + [_rec(f"0{i}", 0.6, 300, "PROMOTE") for i in range(4, 6)]
    assert OG.decide(True, prf, OG.aggregate(prf), [], True, _CONTRACT)["decision"] == "FAIL"
    prr = [_rec(f"0{i}", 0.54, 300, "PROMOTE") for i in range(1, 4)] + [_rec(f"0{i}", 0.52, 300, "TRUNCATED_NO_PROMOTE") for i in range(4, 6)]
    assert OG.decide(True, prr, OG.aggregate(prr), [], True, _CONTRACT)["decision"] == "REVIEW"


def test_offline_precondition_met():
    op = _CONTRACT["decision"]["offline_precondition"]
    assert op["delta"] > 0 and str(op["status"]).startswith("MET")


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
