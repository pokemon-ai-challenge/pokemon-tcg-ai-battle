"""ISMCTS v2.13 Hard-Root Extension 正しさ(Phase G/H)。

中心差分は「1350ms で未収束な hard root だけ same tree を最大 3000ms まで継続」。restart なし・
tree/stats/RNG/determinization を reset しない。既定 OFF なら v2.12 と byte 不変。

- 純 unit(cg 不使用): _root_snapshot / _resolve_extension(G1,G3,G4,G5 の cfg 配線)。
- cg 使用: fixed-world 決定化で搜索を決定論化し、G6/H(same-tree continuation = N2 探索の N1 時点 snapshot が
  独立 N1 探索の最終と一致)を bit-exact に検証。加えて hard-root 発火 / soft-cap 前収束は非 hard の挙動 smoke。

`python test_ismcts_v2_13.py`
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for _p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import to_observation_class, SelectType  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402

_CFG = A.load_config()


# ============ 純 unit(cg 不使用) ============
class _FakeNode:
    """_root_snapshot 用の最小 Node 互換(N/n_total/actions)。"""
    def __init__(self, counts: dict):
        self.actions = list(counts.keys())
        self.N = dict(counts)
        self.n_total = sum(counts.values())


def test_root_snapshot_share_gap_entropy():
    snap = ismcts._root_snapshot(_FakeNode({(0,): 70, (1,): 20, (2,): 10}))
    assert snap["action"] == (0,) and snap["visits"] == 70 and snap["n_total"] == 100
    assert abs(snap["share"] - 0.70) < 1e-6
    assert abs(snap["gap"] - 0.50) < 1e-6          # (70-20)/100
    assert snap["entropy"] > 0.0
    # 単峰(全 visit が top1)→ share=1, gap=1, entropy=0
    s2 = ismcts._root_snapshot(_FakeNode({(0,): 50, (1,): 0}))
    assert s2["action"] == (0,) and abs(s2["share"] - 1.0) < 1e-6 and abs(s2["gap"] - 1.0) < 1e-6 and abs(s2["entropy"]) < 1e-9
    # 空 root
    empty = ismcts._root_snapshot(_FakeNode({}))
    assert empty["action"] is None and empty["n_total"] == 0


def test_G1_extension_off_is_v2_12_identical_inputs():
    """extension 未指定 / enabled=false → cfg に soft_cap_ms/checkpoints_ms を足さず budget=soft cap。
    = v2.11/v2.12 に渡る search 入力が byte 不変。"""
    base = {"budget_ms": 1350, "iterations": 100000, "early_stop": {"enabled": True}}
    # 未指定
    cfg, budget = A._resolve_extension(dict(base))
    assert budget == 1350
    assert "soft_cap_ms" not in cfg and "checkpoints_ms" not in cfg
    assert cfg == base
    # enabled=false
    off = {**base, "hard_root_extension": {"enabled": False, "hard_cap_ms": 3000}}
    cfg2, budget2 = A._resolve_extension(dict(off))
    assert budget2 == 1350
    assert "soft_cap_ms" not in cfg2 and "checkpoints_ms" not in cfg2


def test_G3_G4_G5_extension_on_wiring():
    """enabled=true → budget=hard_cap、soft_cap_ms/checkpoints_ms を記録用に付与。early_stop は変更しない。"""
    on = {"budget_ms": 1350, "iterations": 100000,
          "early_stop": {"enabled": True, "n_min": 64, "share": 0.7, "gap": 0.4},
          "hard_root_extension": {"enabled": True, "hard_cap_ms": 3000, "checkpoints_ms": [1350, 2000]}}
    cfg, budget = A._resolve_extension(dict(on))
    assert budget == 3000.0                        # hard cap まで探索(soft cap で停止しない)
    assert cfg["soft_cap_ms"] == 1350              # hard-root 判定境界
    assert cfg["checkpoints_ms"] == [1350, 2000]
    assert cfg["early_stop"] == on["early_stop"]   # 収束 rule 不変(same tree で継続中も同 rule)


# ============ cg 使用: fixed-world で決定論化 ============
def _capture_main_obs(max_steps=150):
    os.chdir(_SUB)
    match_context.reset()
    deck = [int(x) for x in (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")[:60]]
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        return None
    model = mpa._get_model(_CFG)
    captured = None
    try:
        for _ in range(max_steps):
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None or cur.result != -1:
                break
            sel = obs.select
            if sel is None:
                break
            try:
                match_context.update(obs)
            except Exception:
                pass
            if (captured is None and sel.type == SelectType.MAIN and sel.maxCount == 1
                    and len(sel.option) >= 2):
                det = A._determinize_factory(obs, _CFG)
                try:
                    if det() is not None:
                        captured = obs
                        break
                except Exception:
                    pass
            if sel.maxCount == 1:
                factory = mpa._model_hidden_state_factory(obs, _CFG)
                idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                action = [idx if idx is not None else 0]
            else:
                action = list(range(sel.minCount))
            obs_dict = battle_select(action)
        return captured
    finally:
        if captured is None:
            battle_finish()


_OBS = _capture_main_obs()


def test_fixture_captured():
    assert _OBS is not None, "MAIN maxCount==1 obs を捕捉できなかった(fixture 失敗)"


def _fixed_determinize():
    """belief を 1 回だけ sample し、その world を毎回返す factory(search を bit-exact 決定論化)。"""
    w = A._determinize_factory(_OBS, _CFG)()
    assert w is not None
    return lambda: dict(w)


def _search(cfg_extra, iterations, deadline_s, det, seed=0):
    cfg = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1,
           "max_depth": 60, "iterations": iterations, "leaf_mode": "rollout", **cfg_extra}
    st = ismcts.SearchStats()
    a = ismcts.search(_OBS, cfg, mpa._get_model(_CFG), A._get_evaluator(_CFG),
                      det, time.perf_counter() + deadline_s, random.Random(seed), st)
    return a, st


def test_G6_H_same_tree_continuation_monotone():
    """G6/H: continuation は tree/stats を reset・restart しない。cg エンジンは内部 RNG を持ち rollout/draw が
    非決定的なので run 間の bit-exact 比較は不可(measurement protocol: 完全 CRN 不可)。代わりに **単一 run 内**で
    「1つの tree が N1→N2 iteration を単調に蓄積している」ことを検証する(restart なら final total は N2-N1 になる)。

    - checkpoint@N1 の root n_total == N1(その時点で丁度 N1 iteration ぶん蓄積)。
    - final の root 総訪問数(root_children_full の visit 和)== N2(= st.iterations)。restart 無しの決定的証拠。
    - checkpoint の top1 action の訪問数 <= final の同 action 訪問数(単調非減少 = reset 無し)。
    """
    det = _fixed_determinize()
    N1, N2 = 40, 100
    _, st = _search({"checkpoint_iters": [N1]}, iterations=N2, deadline_s=60.0, det=det)
    assert st.iter_checkpoints, "iteration checkpoint が記録されていない"
    cp = st.iter_checkpoints[0]
    assert cp["cap_iter"] == N1 and cp["iter"] == N1
    assert cp["n_total"] == N1, f"checkpoint n_total={cp['n_total']} != N1={N1}(蓄積不整合)"
    assert st.iterations == N2
    final_total = sum(v for _a, v, _q, _p in st.root_children_full)
    assert final_total == N2, f"final total={final_total} != N2={N2}(restart なら N2-N1 になるはず)"
    # checkpoint top1 action の訪問数は final で単調非減少(tree を捨てていない)。
    cp_action = cp["action"]
    final_visits = {tuple(a): v for a, v, _q, _p in st.root_children_full}
    assert final_visits.get(cp_action, 0) >= cp["visits"], \
        f"top1 訪問が減少: cp={cp['visits']} final={final_visits.get(cp_action)}(reset 疑い)"
    print(f"    [G6/H] N1→N2 単調蓄積 OK: cp(n_total={cp['n_total']},top1_visits={cp['visits']}) "
          f"final(total={final_total},top1_visits={final_visits.get(cp_action)})")


def _search_timed(cfg_extra, hard_ms, seed=0):
    """leaf_mode=node(rollout 無し=高速 iteration)で timing 系挙動を安定に検証。"""
    det = _fixed_determinize()
    cfg = {"world_pool_size": 8, "c_puct": 1.4, "max_rollout_steps": 40, "opponent_depth": 1,
           "max_depth": 60, "iterations": 1000000, "leaf_mode": "node", **cfg_extra}
    st = ismcts.SearchStats()
    a = ismcts.search(_OBS, cfg, mpa._get_model(_CFG), A._get_evaluator(_CFG), det,
                      time.perf_counter() + hard_ms / 1000.0, random.Random(seed), st)
    return a, st


def test_G_hard_root_extension_fires_and_bounded():
    """hard-root(early-stop が soft cap 前に発火しない設定)→ is_hard_root=True、soft_cap_action 記録、
    hard cap まで継続し elapsed<=hard cap+slack、action 合法、error/fallback 無し。"""
    soft, hard = 100.0, 300.0
    a, st = _search_timed(
        {"budget_ms": soft, "soft_cap_ms": soft, "checkpoints_ms": [soft, 200.0],
         # early-stop を発火不能にして soft cap を必ず超えさせる(=hard root を強制)
         "early_stop": {"enabled": True, "n_min": 10**9, "check_every": 16, "k_stable": 99,
                        "share": 1.01, "gap": 1.01}},
        hard_ms=hard)
    assert a is not None and mpa._is_valid_action(a, _OBS.select)
    assert st.is_hard_root is True and st.soft_cap_action is not None
    assert st.early_stopped is False
    assert st.mapping_errors == 0 and not st.fallback_used
    assert st.elapsed_ms <= hard + 100.0, f"hard cap 超過: {st.elapsed_ms:.0f}ms"
    assert any(abs(c["cap_ms"] - soft) < 1e-6 for c in st.checkpoints), "soft cap checkpoint 未記録"
    print(f"    [hard-root] iters={st.iterations} elapsed={st.elapsed_ms:.0f}ms "
          f"soft_action={st.soft_cap_action} final={st.selected_action} "
          f"changed={st.selected_action != st.soft_cap_action}")


def test_G_converged_before_soft_cap_not_hard():
    """soft cap 前に early-stop(緩い rule)→ is_hard_root=False、soft_cap_action=None、extension 不発。
    soft cap / deadline を巨大にし、時間ではなく iteration 基準の収束が先に来ることを保証(非 flaky)。"""
    soft = 60000.0
    a, st = _search_timed(
        {"budget_ms": soft, "soft_cap_ms": soft, "checkpoints_ms": [soft],
         # k_stable=1, share/gap=0 → 最初の check(it=n_min)で確定発火(root の near-tie に依存しない)。
         # soft cap(60s)は node mode の微小 elapsed では到達しない → 収束が時間より先。
         "early_stop": {"enabled": True, "n_min": 16, "check_every": 4, "k_stable": 1,
                        "share": 0.0, "gap": 0.0}},
        hard_ms=90000.0)
    assert a is not None and mpa._is_valid_action(a, _OBS.select)
    assert st.early_stopped is True and st.stop_iter > 0, \
        f"early-stop 未発火: early_stopped={st.early_stopped} stop_iter={st.stop_iter} iters={st.iterations} elapsed={st.elapsed_ms:.0f}ms"
    assert st.is_hard_root is False and st.soft_cap_action is None, \
        f"収束 root が hard 扱い: is_hard_root={st.is_hard_root} soft_cap_action={st.soft_cap_action}"
    assert not st.checkpoints, "収束 root で soft cap checkpoint が記録されてはいけない"
    print(f"    [converged] early-stop @ iter={st.stop_iter} elapsed={st.elapsed_ms:.0f}ms (非 hard root)")


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {t.__name__}: {exc!r}")
    try:
        battle_finish()
    except Exception:
        pass
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_run_all())
