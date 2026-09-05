"""ISMCTS v2.9 — Oracle-Gated Hybrid agent(学習なし診断)。

各 MAIN root で Teacher Search(v2.4 H4 @128 fixed iters)を実行し、arm の gate が発火したら Search top1
(=most-visited)を、そうでなければ Original Policy top1 を採用。これは「Search correction を完璧に転移できた
場合の online upper bound」を測る診断であり、Policy を一切学習・変更しない。

arm(env `ORACLE_ARM`):
  CONTROL = 常に Original top1(search は shadow=行動に使わない。runtime/gate-rate parity 用)
  ALL     = CHANGED(Search top1 != Original top1)なら Search top1
  V       = CHANGED & top1 visit share>=0.5 & top1-top2 gap>=0.2(v2.7 gate exact)
  Q       = CHANGED & a_orig visited & ΔQ=Q(search)-Q(orig)>=0.398(v2.8 A75 gate exact)
lethal/pipeline は使わない(abl_2_policy_only と同じ=純 Policy + gate)。gate off の action は Original Policy argmax
= abl_2_policy_only の選択と一致。module-level picklable。
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
_SEARCH = _ROOT / "kaggle_replays" / "search"
for _p in (str(_SUB), str(_SEARCH), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import Observation, SelectType  # noqa: E402
import ismcts  # noqa: E402
import ismcts_v1_agent as _v1  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402

_V_SHARE, _V_GAP = 0.5, 0.2          # v2.7 gate(exact)
_Q_THR = 0.398                       # v2.8 A75 gate(exact)
_CFG_CACHE = {}
_GATE_LOG: list[dict] = []           # 診断: 各 root の gate 情報


def _oracle_cfg():
    p = str(_HERE / "configs" / "ismcts_oracle_h4_128.json")
    c = _CFG_CACHE.get(p)
    if c is None:
        c = _CFG_CACHE[p] = _v1.load_config(p)
    return c


def _run_search(obs, config):
    """H4 @128 Teacher Search を実行し SearchStats を返す(行動は呼び出し側の gate で決める)。"""
    ic = config.get("ismcts") or {}
    model = mpa._get_model(config)
    evaluator = _v1._get_evaluator(config)
    determinize = _v1._determinize_factory(obs, config)
    rollout_policy = _v1._get_rollout_policy(config)
    rng = random.Random(ic.get("seed", 0))
    st = ismcts.SearchStats()
    ismcts.search(obs, dict(ic), model, evaluator, determinize,
                  time.perf_counter() + 3600.0, rng, st, rollout_policy=rollout_policy)
    return st, model


def _gate_fires(arm, st) -> tuple[bool, dict]:
    """arm の gate 判定。返り: (fires, diag)。search top1=selected_action, original top1=root_policy_top1。"""
    search_a = st.selected_action
    orig_a = st.root_policy_top1
    rc = st.root_children_full            # [(action, visits, Q, prior)]
    diag = {"changed": None, "share": None, "gap": None, "dQ": None, "fires": False}
    if search_a is None or orig_a is None or not rc:
        return False, diag
    changed = (tuple(search_a) != tuple(orig_a))
    diag["changed"] = changed
    total = sum(v for (_a, v, _q, _p) in rc) or 1
    vs = sorted((v for (_a, v, _q, _p) in rc), reverse=True)
    share = vs[0] / total
    gap = (vs[0] - (vs[1] if len(vs) > 1 else 0)) / total
    qmap = {tuple(a): (v, q) for (a, v, q, _p) in rc}
    diag["share"] = share; diag["gap"] = gap
    v_orig, q_orig = qmap.get(tuple(orig_a), (0, 0.0))
    v_search, q_search = qmap.get(tuple(search_a), (0, 0.0))
    dQ = (q_search - q_orig) if v_orig > 0 else None
    diag["dQ"] = dQ
    if not changed:
        return False, diag
    if arm == "ALL":
        fires = True
    elif arm == "V":
        fires = (share >= _V_SHARE and gap >= _V_GAP)
    elif arm == "Q":
        fires = (dQ is not None and dQ >= _Q_THR)
    else:  # CONTROL
        fires = False
    diag["fires"] = fires
    return fires, diag


def _original_action(obs, config):
    """Original Policy の選択(abl_2_policy_only と同一: MAIN=argmax, multi=greedy)。lethal/pipeline なし。"""
    model = mpa._get_model(config)
    factory = mpa._model_hidden_state_factory(obs, config)
    sel = obs.select
    deadline = time.perf_counter() + mpa._MODEL_TIME_BUDGET_MS / 1000.0
    if sel.maxCount == 1:
        idx = model.select_option(obs, factory, deadline)
        return [idx if idx is not None else 0]
    return mpa._greedy_multi_select(obs, model, sel, factory, deadline)


def oracle_agent(obs: Observation) -> list[int]:
    arm = os.environ.get("ORACLE_ARM", "CONTROL")
    config = _oracle_cfg()
    try:
        match_context.update(obs)
    except Exception:
        pass
    if obs.select is None:
        match_context.reset()
        return mpa.agent(obs, config)          # デッキ選択は Original(60枚 deck.csv)
    sel = obs.select
    s = obs.current
    # MAIN single-select ≥2 options のみ Oracle gate 対象。それ以外は Original Policy。
    if (s is not None and s.result == -1 and sel.type == SelectType.MAIN
            and sel.maxCount == 1 and len(sel.option) >= 2):
        try:
            st, _model = _run_search(obs, config)
            fires, diag = _gate_fires(arm, st)
            if os.environ.get("ORACLE_LOG"):
                _GATE_LOG.append({"turn": getattr(s, "turn", None), "arm": arm, **diag,
                                  "search": list(st.selected_action) if st.selected_action else None,
                                  "orig": list(st.root_policy_top1) if st.root_policy_top1 else None})
            if fires and st.selected_action is not None and mpa._is_valid_action(list(st.selected_action), sel):
                return list(st.selected_action)          # gate 発火 = Search top1
        except Exception:
            pass
    return _original_action(obs, config)                 # gate off / 非MAIN = Original


def control_agent(obs: Observation) -> list[int]:
    """明示 CONTROL(ORACLE_ARM 無視、常に Original)。abl_2_policy_only と等価(search なし)。"""
    config = _oracle_cfg()
    try:
        match_context.update(obs)
    except Exception:
        pass
    if obs.select is None:
        match_context.reset()
        return mpa.agent(obs, config)
    return _original_action(obs, config)
