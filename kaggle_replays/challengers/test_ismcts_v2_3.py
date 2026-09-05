"""ISMCTS v2.3 semantic truncated rollout correctness(boundary / equivalence / perspective)。

mock cg(scripted actor 系列)で ismcts._rollout の rollout_cutoff 分岐を厳密検証:
  F8  H0=node は cutoff に関係なく expand した node を評価(v2.1 と同一経路)。
  F9  FULL(=absent)は既存 v1 と同一停止点(handoff_return)。
  F10 one_handoff は 1st turn handoff の state で leaf 評価。
  F11 two_handoff は 2nd handoff の state で leaf 評価(ref-start では FULL と一致)。
  F12 cutoff は select 有りの decision 境界でのみ評価(pending 途中選択なし)。
  F13 leaf は ref_player 視点で呼ばれる(actor が相手でも me=ref)。
  terminal 優先(cutoff より前に terminal なら official 値)。
F0-F7 は共有 core(test_ismcts_unit/_cg)で担保、変更後も 9/9+6/6 PASS を別途確認。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "sample_submission"), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ismcts  # noqa: E402
from ptcg_ai.search import pipeline as _pl  # noqa: E402


class _State:
    def __init__(self, actor, result=-1):
        self.yourIndex = actor; self.result = result


class _Sel:
    option = [object()]          # 非空 select = decision 境界


class _Obs:
    def __init__(self, state):
        self.current = state
        self.select = _Sel() if (state is not None and state.result == -1) else None


class _Node:
    """scripted state 系列を search_step で進める mock cg node。"""
    _SEQ: list = []
    def __init__(self, idx=0):
        self.idx = idx; self.searchId = 1000 + idx
    @property
    def observation(self):
        return _Obs(_Node._SEQ[self.idx] if self.idx < len(_Node._SEQ) else None)


class _RecEval:
    def __init__(self, p=0.75):
        self.p = p; self.seen = []; self.me_seen = []
    def evaluate(self, state, me):
        self.seen.append(state); self.me_seen.append(me)
        return self.p


class _Policy:
    def __init__(self): self.calls = 0
    def score_options_from_state(self, s, sel):
        self.calls += 1; return [1.0]


def _patch(monkey_seq):
    """cg_api.search_step / pipeline._greedy_selection を mock 化。返り: restore()。"""
    _Node._SEQ = monkey_seq
    orig_step = ismcts.cg_api.search_step
    orig_greedy = _pl._greedy_selection
    def fake_step(search_id, selection):
        idx = search_id - 1000
        return _Node(idx + 1)
    def fake_greedy(model, obs):
        model.score_options_from_state(obs.current, obs.select)
        return [0]
    ismcts.cg_api.search_step = fake_step
    _pl._greedy_selection = fake_greedy
    def restore():
        ismcts.cg_api.search_step = orig_step
        _pl._greedy_selection = orig_greedy
    return restore


# ref=0 の系列: 自分×2 → 相手×2 → 自分。handoff#1=idx2(0→1), handoff#2=idx4(1→0)。
_SEQ_REF_START = [_State(0), _State(0), _State(1), _State(1), _State(0)]


def _run(cutoff=None, leaf_mode="rollout", seq=None):
    restore = _patch(seq if seq is not None else list(_SEQ_REF_START))
    try:
        ev = _RecEval(0.75); pol = _Policy()
        cfg = {"leaf_mode": leaf_mode, "opponent_depth": 1, "max_rollout_steps": 40}
        if cutoff is not None:
            cfg["rollout_cutoff"] = cutoff
        v = ismcts._rollout(_Node(0), 0, ev, pol, cfg, 1e18)
        return v, ev, pol
    finally:
        restore()


def test_F10_one_handoff_stops_at_first_handoff():
    v, ev, pol = _run(cutoff="one_handoff")
    assert len(ev.seen) == 1 and ev.seen[0] is _SEQ_REF_START[2], "1st handoff(idx2, actor=1)で評価していない"
    assert abs(v - ismcts.leaf_to_value(0.75)) < 1e-12
    assert pol.calls == 2, f"one_handoff は自分の 2 手だけ policy を呼ぶはず(got {pol.calls})"


def test_F11_two_handoff_stops_at_second_handoff():
    v, ev, pol = _run(cutoff="two_handoff")
    assert len(ev.seen) == 1 and ev.seen[0] is _SEQ_REF_START[4], "2nd handoff(idx4, actor=0)で評価していない"
    assert pol.calls == 4, f"two_handoff は idx0-3 の 4 手 policy(got {pol.calls})"


def test_F9_full_equals_v1_handoff_return():
    # full(=absent)は ref-start で idx4(handoff_return)評価 = two_handoff と一致(v1 挙動)。
    v_full_absent, ev_a, _ = _run(cutoff=None)
    v_full_explicit, ev_b, _ = _run(cutoff="full")
    assert ev_a.seen[0] is _SEQ_REF_START[4] and ev_b.seen[0] is _SEQ_REF_START[4]
    assert v_full_absent == v_full_explicit


def test_F8_node_ignores_cutoff():
    # leaf_mode=node は cutoff に関係なく expand した node(idx0)を即評価、search_step を進めない。
    v, ev, pol = _run(cutoff="one_handoff", leaf_mode="node")
    assert len(ev.seen) == 1 and ev.seen[0] is _SEQ_REF_START[0]
    assert pol.calls == 0, "node は policy(rollout)を呼ばない"


def test_F13_perspective_leaf_called_with_ref_player():
    # 1st handoff の state は actor=1(相手手番)。leaf は me=ref_player=0 で呼ばれる。
    v, ev, pol = _run(cutoff="one_handoff")
    assert ev.seen[0].yourIndex == 1 and ev.me_seen[0] == 0, "相手手番 leaf でも me=ref で評価すべき"


def test_terminal_priority_before_cutoff():
    # idx2 が terminal(result=0=ref勝ち)なら、handoff cutoff より前に terminal_value=+1 を返す。
    seq = [_State(0), _State(0), _State(0, result=0), _State(1), _State(0)]
    v, ev, pol = _run(cutoff="one_handoff", seq=seq)
    assert v == 1.0 and len(ev.seen) == 0, "terminal で leaf を呼ばず official ±1 を返すべき"


def test_F12_cutoff_only_at_decision_boundary():
    # cutoff で評価した state は select 有り(decision 境界)= pending 途中選択でない。
    v, ev, pol = _run(cutoff="one_handoff")
    assert _Obs(ev.seen[0]).select is not None, "cutoff 評価点が decision 境界(select有)でない"


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
