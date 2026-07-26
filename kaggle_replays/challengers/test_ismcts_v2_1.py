"""ISMCTS v2.1 leaf-at-node correctness: F8 no-rollout / F9 leaf-point + cg legal(leaf_mode=node)。"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "sample_submission"), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ismcts  # noqa: E402


class _S:
    def __init__(self, me=0, result=-1):
        self.yourIndex = me; self.result = result


class _Node:
    def __init__(self, state):
        self.searchId = 1
        class _O:
            current = state
        self.observation = _O()


def _boom_policy():
    class P:
        def score_options_from_state(self, s, sel):
            raise AssertionError("leaf_mode=node で policy(rollout)が呼ばれた")
    return P()


class _Eval:
    def __init__(self): self.seen = []
    def evaluate(self, state, me):
        self.seen.append(state); return 0.75


# ---- F8: leaf_mode=node は rollout の policy を呼ばない ----
def test_F8_no_rollout_policy_call():
    node = _Node(_S(me=0))
    ev = _Eval()
    v = ismcts._rollout(node, 0, ev, _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert v == ismcts.leaf_to_value(0.75)   # 2*0.75-1 = 0.5
    assert abs(v - 0.5) < 1e-12
    # policy(_boom)が呼ばれていない(呼ばれれば AssertionError で落ちる)= PASS


def test_F8_node_terminal_priority():
    node = _Node(_S(me=0, result=0))         # root_me=0 の勝ち
    v = ismcts._rollout(node, 0, _Eval(), _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert v == 1.0                           # terminal 優先(leaf でなく +1)


# ---- F9: leaf 評価対象は expand した node そのもの(先へ進めない)----
def test_F9_leaf_point_is_expanded_node():
    st = _S(me=1)
    node = _Node(st)
    ev = _Eval()
    ismcts._rollout(node, 0, ev, _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert len(ev.seen) == 1 and ev.seen[0] is st, "評価対象が expand node の state でない/余計に進んだ"


def test_F9_node_perspective_ref_player():
    # leaf は ref_player 視点(root_me)。node.to_move が相手でも ref_player で評価。
    st = _S(me=1)                             # 相手手番の node
    ev = _Eval()
    v = ismcts._rollout(_Node(st), 0, ev, _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert ev.seen[0].yourIndex == 1          # state はそのまま
    # evaluate は me=ref_player=0 で呼ばれる(negamax の符号は backup 側で処理)
    assert v == ismcts.leaf_to_value(0.75)


# ---- default(rollout)は v1 挙動: leaf_mode 未指定で node 分岐に入らない ----
def test_default_is_rollout_not_node():
    # leaf_mode 未指定 → node 即評価に入らない(policy 経路へ)。terminal state なら terminal を返す。
    node = _Node(_S(me=0, result=0))
    v = ismcts._rollout(node, 0, _Eval(), _boom_policy(), {}, 1e18)  # default rollout でも terminal は即返す
    assert v == 1.0
    # 非 terminal で default rollout は policy を呼ぶ(_boom)ので、そこは cg test 側で担保。


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
