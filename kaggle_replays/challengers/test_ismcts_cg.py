"""ISMCTS v1 cg-based tests。F1 determinization / F2 legal / F6 real / F7 reset + 実 ISMCTS run。

実ゲームを進めて MAIN maxCount==1 の obs を捕捉し、ISMCTS を小 iteration で走らせ健全性を確認する。
`python test_ismcts_cg.py`。
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


def _load_deck():
    p = _SUB / "deck.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _capture_main_obs(max_steps=150):
    """実ゲームを進め、determinize が効く最初の MAIN maxCount==1 obs を返す。"""
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
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
                match_context.update(obs)   # 信念(Champion と同契約)
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
            # advance(policy argmax / first legal)
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


def test_F1_determinization_fields():
    det = A._determinize_factory(_OBS, _CFG)
    w = det()
    for k in ("your_deck", "your_prize", "opponent_deck", "opponent_prize", "opponent_hand", "opponent_active"):
        assert k in w, f"determinization world に {k} が無い"
    # 決定化は信念由来(ground truth でない)= factory を再呼び出しできる(leakage-safe な belief sample)
    assert det() is not None


def test_F2_ismcts_returns_legal_action():
    cfg = {**_CFG, "ismcts": {**_CFG["ismcts"], "iterations": 24, "budget_ms": 2000}}
    stats = ismcts.SearchStats()
    action = ismcts.search(_OBS, dict(cfg["ismcts"]), mpa._get_model(cfg), A._get_evaluator(cfg),
                           A._determinize_factory(_OBS, cfg), time.perf_counter() + 3.0,
                           random.Random(0), stats)
    assert action is not None, "ISMCTS が action を返さない"
    assert mpa._is_valid_action(action, _OBS.select), f"ISMCTS action が違法: {action}"
    assert stats.iterations > 0 and stats.nodes >= 1 and stats.expanded_nodes >= 1
    assert stats.mapping_errors == 0
    print(f"    [F2] iters={stats.iterations} nodes={stats.nodes} max_depth={stats.max_depth} "
          f"changed={stats.policy_changed_by_search} elapsed={stats.elapsed_ms:.0f}ms")


def test_F6_disabled_agent_returns_champion_valid_action():
    # enabled=false の候補 agent が Champion 経路(_select_action)で合法 action を返す
    off = {**_CFG, "ismcts": {**_CFG["ismcts"], "enabled": False}}
    act = A.agent(_OBS, off)
    assert mpa._is_valid_action(act, _OBS.select), f"enabled=false で違法: {act}"


def test_F2b_enabled_agent_returns_legal_action():
    on = {**_CFG, "ismcts": {**_CFG["ismcts"], "enabled": True, "iterations": 24, "budget_ms": 2000}}
    act = A.agent(_OBS, on)
    assert mpa._is_valid_action(act, _OBS.select), f"enabled=true で違法: {act}"


def test_F7_no_module_global_tree():
    # ISMCTS の tree は search() ローカル。module global を持たない(game 跨ぎ leak なし)。
    assert not any(k for k in dir(ismcts) if k.lower() in ("_tree", "tree", "_global_tree"))
    # 2 回 search しても相互干渉しない(独立 stats)
    cfg = {**_CFG["ismcts"], "iterations": 8, "budget_ms": 1500}
    s1, s2 = ismcts.SearchStats(), ismcts.SearchStats()
    a1 = ismcts.search(_OBS, dict(cfg), mpa._get_model(_CFG), A._get_evaluator(_CFG),
                       A._determinize_factory(_OBS, _CFG), time.perf_counter() + 3.0, random.Random(1), s1)
    a2 = ismcts.search(_OBS, dict(cfg), mpa._get_model(_CFG), A._get_evaluator(_CFG),
                       A._determinize_factory(_OBS, _CFG), time.perf_counter() + 3.0, random.Random(2), s2)
    assert mpa._is_valid_action(a1, _OBS.select) and mpa._is_valid_action(a2, _OBS.select)


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
