"""ISMCTS v1 Challenger agent。

Champion(abl_5_full)経路を **read-only 再利用**し、single-select MAIN の意思決定だけを ISMCTS へ差し替える。
`ismcts.enabled=false` なら **Champion と完全同一**(F6 no-search equivalence)。production は変更しない。

差分は Search 構造のみ:
  Champion : lethal → pipeline(浅 PIMC)→ policy fallback
  ismcts_v1: lethal → ISMCTS(深ツリー、同 Policy prior/信念決定化/handcrafted leaf)→ policy fallback

ローカル専用・git 未追跡。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_SEARCH = _ROOT / "kaggle_replays" / "search"
for _p in (str(_SUB), str(_SEARCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import Observation, SelectType  # noqa: E402
import ismcts  # noqa: E402  (kaggle_replays/search/ismcts.py)
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.search import leaf_eval  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state  # noqa: E402

_CONFIG_PATH = _HERE / "configs" / "ismcts_v1.json"
_diag_log: list[dict] = []


def load_config(path=None) -> dict:
    return json.loads(Path(path or _CONFIG_PATH).read_text(encoding="utf-8"))


def _get_evaluator(config: dict):
    # Champion と同じ handcrafted leaf(pipeline.leaf_eval)。Value は使わない。
    leaf_cfg = (config.get("pipeline") or {}).get("leaf_eval") or {"kind": "handcrafted"}
    return leaf_eval.build_evaluator(leaf_cfg)


def _determinize_factory(obs: Observation, config: dict):
    """Champion pipeline と同一の信念決定化(leakage-safe)。hidden_state_source は pipeline に揃える。"""
    source = (config.get("pipeline") or {}).get("hidden_state_source", "estimated")
    yi = obs.current.yourIndex
    if source == "dummy":
        full_deck = mpa._get_deck()
        return lambda: build_dummy_search_state(obs, full_deck)
    return lambda: search_adapter.to_search_begin_kwargs(
        match_context.get_own_state(yi), match_context.get_opponent_state(yi), obs)


def _ismcts_applicable(obs: Observation) -> bool:
    s, sel = obs.current, obs.select
    return (s is not None and sel is not None and s.result == -1 and sel.type == SelectType.MAIN
            and sel.maxCount == 1 and bool(sel.option))


def _run_ismcts(obs: Observation, config: dict) -> list[int] | None:
    ic = config.get("ismcts") or {}
    model = mpa._get_model(config)
    evaluator = _get_evaluator(config)
    determinize = _determinize_factory(obs, config)
    budget_ms = ic.get("budget_ms")
    deadline = time.perf_counter() + ((budget_ms / 1000.0) if budget_ms else 3600.0)
    rng = random.Random(ic.get("seed", 0))
    stats = ismcts.SearchStats()
    action = ismcts.search(obs, dict(ic), model, evaluator, determinize, deadline, rng, stats)
    if ic.get("instrument"):
        _diag_log.append({"turn": getattr(obs.current, "turn", None), "iters": stats.iterations,
                          "nodes": stats.nodes, "expanded": stats.expanded_nodes, "max_depth": stats.max_depth,
                          "mean_depth": round(stats.mean_depth, 2), "determinizations": stats.determinizations,
                          "unique_isets": stats.unique_information_sets, "elapsed_ms": round(stats.elapsed_ms, 1),
                          "budget_ms": round(stats.budget_ms, 1), "timeout": stats.timeout,
                          "policy_top1": stats.root_policy_top1, "selected": stats.selected_action,
                          "changed": stats.policy_changed_by_search, "mapping_errors": stats.mapping_errors,
                          "root_children": [(list(a), n, round(q, 3), round(p, 3)) for a, n, q, p in stats.root_children]})
    return action


def _policy_fallback(obs: Observation, config: dict) -> list[int]:
    """Champion tail(abl_5_full: attack_plan/hybrid OFF)= policy argmax / greedy multi-select。"""
    model = mpa._get_model(config)
    factory = mpa._model_hidden_state_factory(obs, config)
    deadline = time.perf_counter() + mpa._MODEL_TIME_BUDGET_MS / 1000.0
    select = obs.select
    if select.maxCount == 1:
        idx = model.select_option(obs, factory, deadline)
        return [idx if idx is not None else 0]
    return mpa._greedy_multi_select(obs, model, select, factory, deadline)


def agent(obs: Observation, config: dict) -> list[int]:
    # 信念更新は Champion(mpa.agent 冒頭 line 130 match_context.update)と同契約=毎意思決定で実行。
    # これが無いと determinize の信念が stale になり、かつ enabled=false でも Champion と挙動が乖離する。
    try:
        match_context.update(obs)
    except Exception:
        pass
    ic = (config or {}).get("ismcts") or {}
    if not ic.get("enabled", False):
        # F6: Search OFF → 信念更新済み + _select_action = Champion(mpa.agent)と完全同一。
        return mpa._select_action(obs, config)

    # ismcts ON: Champion と同じ counter 進行(pipeline fallback の動的予算 parity)。
    mpa._selects_seen += 1
    lethal = mpa._try_lethal(obs, config=config)
    if lethal is not None:
        return lethal
    if _ismcts_applicable(obs):
        try:
            action = _run_ismcts(obs, config)
            if action is not None and mpa._is_valid_action(action, obs.select):
                return action
        except Exception:
            pass  # fall through to Champion fallback(silent illegal を出さない)
    # Champion fallback(lethal は済んでいるので pipeline → policy)。
    pipeline_action = mpa._try_pipeline(obs, config=config)
    if pipeline_action is not None:
        return pipeline_action
    return _policy_fallback(obs, config)


_CFG_CACHE: dict[str, dict] = {}


def default_agent(obs: Observation) -> list[int]:
    """measurement harness の agent_fn(picklable module-level 関数)。

    config は env `ISMCTS_CONFIG`(path)があればそれ、無ければ ismcts_v1.json。strength scaling で
    budget 別 config を worker へ渡すため env 経由(spawn worker は親 env を継承)。obs.select is None
    (デッキ選択)は Champion に委譲(60枚 deck.csv)+ match_context reset。
    """
    import os
    path = os.environ.get("ISMCTS_CONFIG") or str(_CONFIG_PATH)
    cfg = _CFG_CACHE.get(path)
    if cfg is None:
        cfg = _CFG_CACHE[path] = load_config(path)
    _DEFAULT_CFG = cfg
    if obs.select is None:
        match_context.reset()
        return mpa.agent(obs, _DEFAULT_CFG)
    return agent(obs, _DEFAULT_CFG)


def make_agent(config: dict | None = None):
    """measurement harness の agent_fn 用(obs -> action)。config を束縛する。"""
    cfg = config if config is not None else load_config()
    # 新試合(obs.select is None)で match_context を通す = ml_policy_agent.agent と同契約。
    def _fn(obs: Observation) -> list[int]:
        if obs.select is None:
            match_context.reset()
            return mpa.agent(obs, cfg)   # デッキ選択は Champion に委譲(60枚 deck.csv)
        return agent(obs, cfg)
    return _fn
