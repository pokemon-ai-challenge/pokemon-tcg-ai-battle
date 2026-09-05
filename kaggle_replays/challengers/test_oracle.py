"""ISMCTS v2.9 Oracle gate correctness(mock SearchStats)。

K2 ALL fires iff Search top1 != Original top1 / K3 V=v2.7 exact(share>=0.5 & gap>=0.2)/
K4 Q=v2.8 exact(ΔQ>=0.398)/ K5 gate-off→Original / K6 gate-on→Search / CONTROL never。
shadow no-side-effect(K1/K10)は ismcts.search が別 searchId 上で real game を advance しない性質(v1 F7/cg 済)を継承。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parents[1] / "sample_submission"), str(_HERE.parents[1] / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import oracle_agent as O  # noqa: E402


class _St:
    """mock SearchStats: root_children_full=[(action,visits,Q,prior)]。"""
    def __init__(self, search_a, orig_a, rc):
        self.selected_action = search_a; self.root_policy_top1 = orig_a; self.root_children_full = rc


# changed & high-conf & high-ΔQ: search=(1,) v90 Q0.7, orig=(0,) v10 Q0.3 → share0.9 gap0.8 dQ0.4
_HI = _St((1,), (0,), [((0,), 10, 0.3, 0.5), ((1,), 90, 0.7, 0.5)])
# changed & low-conf(gap<0.2)& low-ΔQ: search=(1,) v55 Q0.50, orig=(0,) v45 Q0.45 → share0.55 gap0.10 dQ0.05
_LO = _St((1,), (0,), [((0,), 45, 0.45, 0.5), ((1,), 55, 0.50, 0.5)])
# changed & high-conf but low-ΔQ: share0.9 gap0.8 but dQ=0.05
_HC_LOWQ = _St((1,), (0,), [((0,), 10, 0.55, 0.5), ((1,), 90, 0.60, 0.5)])
# unchanged: search==orig
_UNCH = _St((0,), (0,), [((0,), 90, 0.7, 0.5), ((1,), 10, 0.3, 0.5)])
# changed & high-ΔQ but a_orig unvisited(v=0)→ Q invalid → Q gate no fire
_ORIG_UNVIS = _St((1,), (0,), [((0,), 0, 0.0, 0.5), ((1,), 100, 0.9, 0.5)])


def test_K2_ALL_iff_changed():
    assert O._gate_fires("ALL", _HI)[0] is True
    assert O._gate_fires("ALL", _LO)[0] is True
    assert O._gate_fires("ALL", _UNCH)[0] is False        # unchanged → no fire


def test_K3_V_exact_v27_gate():
    assert O._gate_fires("V", _HI)[0] is True              # share0.9 gap0.8
    assert O._gate_fires("V", _LO)[0] is False             # gap0.10 < 0.2
    assert O._gate_fires("V", _UNCH)[0] is False


def test_K4_Q_exact_v28_gate():
    assert O._gate_fires("Q", _HI)[0] is True              # dQ0.4 >= 0.398
    assert O._gate_fires("Q", _HC_LOWQ)[0] is False        # dQ0.05 < 0.398(高信頼でも低ΔQ)
    assert O._gate_fires("Q", _LO)[0] is False


def test_K4b_Q_excludes_unvisited_original():
    # a_orig unvisited(v=0)→ ΔQ invalid → Q gate 発火しない(fake Q を使わない)
    assert O._gate_fires("Q", _ORIG_UNVIS)[0] is False
    assert O._gate_fires("ALL", _ORIG_UNVIS)[0] is True    # ALL は changed なら発火


def test_CONTROL_never_fires():
    for st in (_HI, _LO, _UNCH):
        assert O._gate_fires("CONTROL", st)[0] is False


def test_K3_V_boundary():
    # share=0.5 gap=0.2 ちょうど(>=)は発火
    st = _St((1,), (0,), [((0,), 30, 0.4, 0.5), ((1,), 50, 0.6, 0.5), ((2,), 20, 0.3, 0.5)])
    # total=100 share=0.5 gap=(50-30)/100=0.2
    assert O._gate_fires("V", st)[0] is True


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
