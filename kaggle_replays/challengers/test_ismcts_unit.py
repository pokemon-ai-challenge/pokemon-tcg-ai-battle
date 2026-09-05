"""ISMCTS v1 unit/correctness tests(cg 不使用・高速)。F3/F0(key leakage)/F4/F5/F6(structural)/watchdog。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "search"))
sys.path.insert(0, str(_ROOT / "sample_submission"))
import ismcts  # noqa: E402


# ---- F3: value domain(集約) ----
def test_F3_value_domain():
    assert ismcts.leaf_to_value(1.0) == 1.0
    assert ismcts.leaf_to_value(0.5) == 0.0
    assert ismcts.leaf_to_value(0.0) == -1.0
    assert abs(ismcts.flip(0.6) - (-0.6)) < 1e-12
    assert abs(ismcts.flip(ismcts.flip(0.6)) - 0.6) < 1e-12


def test_F3_terminal_and_signed():
    assert ismcts.terminal_value(0, 0) == 1.0 and ismcts.terminal_value(1, 0) == -1.0
    assert ismcts.terminal_value(0, 1) == -1.0 and ismcts.terminal_value(1, 1) == 1.0
    assert ismcts.terminal_value(-1, 0) == 0.0
    # signed_for: root_me=0。node.to_move==0 → そのまま、==1 → 反転(negamax)
    assert ismcts.signed_for(0.6, 0, 0) == 0.6
    assert ismcts.signed_for(0.6, 1, 0) == -0.6


# ---- 観測可能 fake ----
class _Poke:
    def __init__(self, cid): self.cardId = cid


class _Player:
    def __init__(self, active, bench, hand, deck, prize):
        self.active = active; self.bench = bench; self.hand = hand
        self.deckCount = deck; self.prize = prize


class _State:
    def __init__(self, me, turn=3, result=-1):
        self.yourIndex = me; self.turn = turn; self.result = result
        self.players = [
            _Player([_Poke(10)], [_Poke(11)], [_Poke(100), _Poke(101)], 20, [None] * 4),
            _Player([_Poke(20)], [_Poke(21)], [_Poke(200)], 21, [None] * 5),
        ]


class _Opt:
    def __init__(self, t, cid): self.type = t; self.cardId = cid


class _Sel:
    def __init__(self, n=3, t=0, ctx=0, mx=1):
        self.type = t; self.context = ctx; self.minCount = 1; self.maxCount = mx
        self.option = [_Opt(0, 300 + i) for i in range(n)]


class _Obs:
    def __init__(self, state, sel): self.current = state; self.select = sel


# ---- F0: info_set_key は相手 hidden hand を含まない(leakage-safe key) ----
def test_F0_key_excludes_opponent_hand():
    s1 = _State(0); s2 = _State(0)
    # 相手(index1)の hand の中身だけ変える(観測不能)。key は不変であるべき。
    s2.players[1].hand = [_Poke(999), _Poke(888), _Poke(777)]
    obs1 = _Obs(s1, _Sel()); obs2 = _Obs(s2, _Sel())
    assert ismcts.info_set_key(obs1) == ismcts.info_set_key(obs2), "相手手札の中身で key が変わる=leakage"
    # 自分(index0)の hand が変われば key は変わる(自分には可視)
    s3 = _State(0); s3.players[0].hand = [_Poke(1)]
    assert ismcts.info_set_key(_Obs(s3, _Sel())) != ismcts.info_set_key(obs1)
    # option が変われば変わる
    assert ismcts.info_set_key(_Obs(s1, _Sel(n=2))) != ismcts.info_set_key(obs1)


# ---- F5: prior は score から決定的(同 score → 同 prior) ----
def test_F5_prior_deterministic():
    sel = _Sel(n=3)
    obs = _Obs(_State(0), sel)

    class _M:
        def score_options_from_state(self, state, select):
            return [0.1, 0.7, 0.2]
    p1 = ismcts._priors(_M(), obs, sel, ismcts._legal_actions(sel))
    p2 = ismcts._priors(_M(), obs, sel, ismcts._legal_actions(sel))
    assert p1 == p2
    assert p1[(1,)] > p1[(2,)] > p1[(0,)]           # softmax 順序保存
    assert abs(sum(p1.values()) - 1.0) < 1e-9       # 正規化

    class _Mbad:
        def score_options_from_state(self, state, select):
            raise RuntimeError("no consequence weights")
    pf = ismcts._priors(_Mbad(), obs, sel, ismcts._legal_actions(sel))
    assert all(abs(v - 1 / 3) < 1e-9 for v in pf.values())  # 失敗時は一様


# ---- F4(rollout terminal): result!=-1 で terminal_value を返す ----
def test_F4_rollout_returns_terminal():
    class _Node:
        def __init__(self):
            self.searchId = 1
            self.observation = _Obs(_State(0, result=0), _Sel())   # result=0(root_me=0 の勝ち)
    v = ismcts._rollout(_Node(), 0, evaluator=None, policy_model=None, config={}, deadline=1e18)
    assert v == 1.0   # terminal win を leaf より優先


# ---- F6(structural): enabled=false → Champion(_select_action)へ完全委譲 ----
def test_F6_disabled_delegates_to_champion():
    import ismcts_v1_agent as A
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    sentinel = [42]
    orig = mpa._select_action
    mpa._select_action = lambda obs, cfg: sentinel
    try:
        out = A.agent(_Obs(_State(0), _Sel()), {"ismcts": {"enabled": False}})
        assert out is sentinel, "enabled=false で Champion へ委譲していない(F6 違反)"
    finally:
        mpa._select_action = orig


def test_F6_enabled_but_nonMAIN_uses_champion_fallback():
    # enabled=true でも非 MAIN(maxCount>1)は ISMCTS を使わず Champion fallback(pipeline→policy)
    import ismcts_v1_agent as A
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    calls = {"ismcts": 0}
    o_lethal, o_pipe, o_run = mpa._try_lethal, mpa._try_pipeline, A._run_ismcts
    mpa._try_lethal = lambda obs, config=None: None
    mpa._try_pipeline = lambda obs, config=None: [7]
    A._run_ismcts = lambda obs, cfg: (calls.__setitem__("ismcts", calls["ismcts"] + 1) or [0])
    try:
        out = A.agent(_Obs(_State(0), _Sel(mx=2)), {"ismcts": {"enabled": True}})
        assert out == [7], "非 MAIN で pipeline fallback にならない"
        assert calls["ismcts"] == 0, "非 MAIN で ISMCTS を呼んでいる"
    finally:
        mpa._try_lethal, mpa._try_pipeline, A._run_ismcts = o_lethal, o_pipe, o_run


def test_F6_enabled_MAIN_uses_ismcts_after_lethal():
    import ismcts_v1_agent as A
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    o_lethal, o_run, o_valid = mpa._try_lethal, A._run_ismcts, mpa._is_valid_action
    mpa._try_lethal = lambda obs, config=None: None
    A._run_ismcts = lambda obs, cfg: [2]
    mpa._is_valid_action = lambda a, sel: True
    try:
        out = A.agent(_Obs(_State(0), _Sel()), {"ismcts": {"enabled": True}})
        assert out == [2], "MAIN で ISMCTS 結果を返していない"
    finally:
        mpa._try_lethal, A._run_ismcts, mpa._is_valid_action = o_lethal, o_run, o_valid


def test_F6_lethal_precedes_ismcts():
    import ismcts_v1_agent as A
    from ptcg_ai.ml_policy import ml_policy_agent as mpa
    o_lethal, o_run = mpa._try_lethal, A._run_ismcts
    mpa._try_lethal = lambda obs, config=None: [9]
    A._run_ismcts = lambda obs, cfg: (_ for _ in ()).throw(AssertionError("lethal より先に ISMCTS"))
    try:
        assert A.agent(_Obs(_State(0), _Sel()), {"ismcts": {"enabled": True}}) == [9]
    finally:
        mpa._try_lethal, A._run_ismcts = o_lethal, o_run


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
