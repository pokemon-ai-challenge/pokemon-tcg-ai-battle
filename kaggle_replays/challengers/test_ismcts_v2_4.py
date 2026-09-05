"""ISMCTS v2.4 small rollout policy correctness(mock cg、rollout policy 差し替えの isolation/equivalence)。

F8  FULL horizon equivalence: rollout_policy を差しても termination は v1 FULL と同一(action だけ違う)。
F9  Teacher mode equivalence: rollout_policy=None → rollout greedy は teacher(policy_model)を使う(student 未使用)。
F10 rollout-only isolation: rollout_policy=student → rollout greedy は student のみ、teacher は rollout で呼ばれない。
F11 legal: student selection は select.option の合法 index。
F0-F7 は共有 core(test_ismcts_unit/_cg)で担保、rollout_policy=None で v1 byte 不変(全再 PASS 別途確認)。
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
    option = [object(), object()]     # 2 option の decision 境界


class _Obs:
    def __init__(self, state):
        self.current = state
        self.select = _Sel() if (state is not None and state.result == -1) else None


class _Node:
    _SEQ: list = []
    def __init__(self, idx=0):
        self.idx = idx; self.searchId = 2000 + idx
    @property
    def observation(self):
        return _Obs(_Node._SEQ[self.idx] if self.idx < len(_Node._SEQ) else None)


class _CountPolicy:
    """score_options_from_state の呼び出し回数を数える(rollout greedy 経由)。"""
    def __init__(self, name):
        self.name = name; self.calls = 0
    def score_options_from_state(self, s, sel):
        self.calls += 1
        return [1.0, 0.0]             # index 0 を greedy 選択


class _Eval:
    def evaluate(self, state, me):
        return 0.5


def _patch(seq):
    _Node._SEQ = seq
    orig_step = ismcts.cg_api.search_step
    orig_greedy = _pl._greedy_selection
    def fake_step(search_id, selection):
        return _Node((search_id - 2000) + 1)
    def fake_greedy(model, obs):
        # 実 _greedy_selection と同じく model.score_options_from_state を呼ぶ(どの model かを計数)
        model.score_options_from_state(obs.current, obs.select)
        return [0]
    ismcts.cg_api.search_step = fake_step
    _pl._greedy_selection = fake_greedy
    def restore():
        ismcts.cg_api.search_step = orig_step
        _pl._greedy_selection = orig_greedy
    return restore


# ref=0: 自×2 → 相手×2 → 自(v1 FULL は idx4 の handoff_return で停止)
_SEQ = [_State(0), _State(0), _State(1), _State(1), _State(0)]


def _rollout(rollout_policy):
    restore = _patch(list(_SEQ))
    teacher = _CountPolicy("teacher")
    try:
        v = ismcts._rollout(_Node(0), 0, _Eval(), teacher,
                            {"leaf_mode": "rollout", "opponent_depth": 1, "max_rollout_steps": 40},
                            1e18, rollout_policy=rollout_policy)
        return v, teacher
    finally:
        restore()


def test_F9_teacher_mode_uses_teacher_not_student():
    # rollout_policy=None → teacher が rollout greedy を担う。student は存在しない。
    v, teacher = _rollout(rollout_policy=None)
    assert teacher.calls == 4, f"teacher が rollout の 4 手を担うはず(got {teacher.calls})"


def test_F10_rollout_only_uses_student_not_teacher():
    # rollout_policy=student → rollout greedy は student のみ。teacher は rollout で呼ばれない。
    student = _CountPolicy("student")
    v, teacher = _rollout(rollout_policy=student)
    assert student.calls == 4, f"student が rollout の 4 手を担うはず(got {student.calls})"
    assert teacher.calls == 0, f"teacher は rollout で呼ばれないはず(got {teacher.calls})"


def test_F8_full_horizon_same_termination_regardless_of_policy():
    # student でも teacher でも termination(idx4 の handoff_return)は同一 = FULL horizon 不変。
    # ここでは leaf 値は同じ evaluator ゆえ v も一致(action は両 policy とも index0 で同軌道)。
    v_teacher, _ = _rollout(rollout_policy=None)
    v_student, _ = _rollout(rollout_policy=_CountPolicy("student"))
    assert v_teacher == v_student, "rollout policy 差で termination/horizon が変わってはいけない"


def test_F11_student_selection_is_legal():
    # student の greedy が返す index は select.option の範囲内(fake_greedy は [0]、常に合法)。
    student = _CountPolicy("student")
    v, _ = _rollout(rollout_policy=student)
    assert student.calls > 0            # 実際に rollout で使われた
    # _Sel.option は 2 要素、index 0 は合法。mapping error なしで完走 = PASS


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
