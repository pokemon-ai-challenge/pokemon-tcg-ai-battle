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
    """ISMCTS leaf の評価器を返す。

    解決順(後方互換): ``ismcts.leaf_eval`` があればそれ(v2.2 value-leaf 用の専用キー。
    pipeline fallback を汚染しないよう ISMCTS 専用に分離)→ 無ければ ``pipeline.leaf_eval``
    (Champion と同じ handcrafted。v1/v2.1 config は ``ismcts.leaf_eval`` を持たないので
    ここに解決し、従来挙動が byte 単位で不変)→ 無ければ handcrafted 既定。
    """
    leaf_cfg = (config.get("ismcts") or {}).get("leaf_eval") \
        or (config.get("pipeline") or {}).get("leaf_eval") \
        or {"kind": "handcrafted"}
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


_ROLLOUT_POLICY_CACHE: dict[str, object] = {}


def _get_rollout_policy(config: dict):
    """v2.4: ``ismcts.rollout_policy_weights``(student weights JSON path)があれば PolicyModel を
    ロードして返す(path でキャッシュ)。無ければ None = teacher(v1 不変)。student は rollout の
    greedy 選択のみに使われ、tree prior/expansion/fallback は teacher のまま。"""
    ic = config.get("ismcts") or {}
    path = ic.get("rollout_policy_weights")
    if not path:
        return None
    batched = bool(ic.get("rollout_policy_batched"))    # v2.11: numpy batched scorer(意味不変・高速化のみ)
    key = f"{path}|batched={batched}"
    model = _ROLLOUT_POLICY_CACHE.get(key)
    if model is None:
        p = Path(path)
        if not p.is_absolute():
            p = _SUB / p            # sample_submission 相対を許可
        if batched:
            from batched_policy import BatchedPolicyModel
            model = BatchedPolicyModel(weights_path=p)
        else:
            from ptcg_ai.learning.policy_model import PolicyModel
            model = PolicyModel(weights_path=p)
        _ROLLOUT_POLICY_CACHE[key] = model
    return model


def _resolve_extension(ic: dict) -> tuple[dict, float | None]:
    """v2.13 hard-root extension の cfg/budget 解決(byte 不変性の一元管理・unit test 対象)。

    返り: (search へ渡す cfg, budget_ms=deadline 決定値)。
    - extension OFF(既定/未指定/enabled=false): cfg は ``dict(ic)`` のまま(soft_cap_ms/checkpoints_ms を足さない)、
      budget=``budget_ms``(soft cap)= v2.11/v2.12 と完全同一。
    - extension ON: budget=``hard_cap_ms``(hard cap)まで探索。cfg に soft_cap_ms/checkpoints_ms を付与(記録専用)。
      soft cap では停止せず、v2.12 と同一の early-stop rule が hard cap 前に発火すれば早期終了。
    """
    soft_cap_ms = ic.get("budget_ms")
    hre = ic.get("hard_root_extension") or {}
    cfg = dict(ic)
    if hre.get("enabled"):
        budget_ms = float(hre.get("hard_cap_ms", 3000))
        cfg["soft_cap_ms"] = soft_cap_ms
        if hre.get("checkpoints_ms"):
            cfg["checkpoints_ms"] = list(hre["checkpoints_ms"])
        return cfg, budget_ms
    return cfg, soft_cap_ms


def _run_ismcts(obs: Observation, config: dict) -> list[int] | None:
    ic = config.get("ismcts") or {}
    model = mpa._get_model(config)
    evaluator = _get_evaluator(config)
    determinize = _determinize_factory(obs, config)
    rollout_policy = _get_rollout_policy(config)
    cfg, budget_ms = _resolve_extension(ic)
    deadline = time.perf_counter() + ((budget_ms / 1000.0) if budget_ms else 3600.0)
    rng = random.Random(ic.get("seed", 0))
    stats = ismcts.SearchStats()
    action = ismcts.search(obs, cfg, model, evaluator, determinize, deadline, rng, stats,
                           rollout_policy=rollout_policy)
    # v2.10b Search-Dynamics Gate(opt-in・既定 OFF → Current ISMCTS と byte 不変)。search が Original を
    # 変更したが「低信頼(share/gap が閾値未満)」なら Original policy top1 へ revert(= search 全採用を抑制)。
    # 追加 Search なし=既存 SearchStats のみ利用(Phase L1)。lethal/fallback は不変。
    gate = ic.get("dynamics_gate") or {}
    if (gate.get("enabled") and action is not None and stats.policy_changed_by_search
            and stats.root_children_full and stats.root_policy_top1 is not None):
        _rc = stats.root_children_full            # [(action, visits, Q, prior)] visits 降順
        _tot = sum(v for _a, v, _q, _p in _rc) or 1
        _share = _rc[0][1] / _tot
        _gap = (_rc[0][1] - (_rc[1][1] if len(_rc) > 1 else 0)) / _tot
        if not (_share >= float(gate.get("share", 0.5)) and _gap >= float(gate.get("gap", 0.2))):
            _orig = list(stats.root_policy_top1)  # 低信頼 correction → Original を採用
            if mpa._is_valid_action(_orig, obs.select):
                action = _orig
    if ic.get("instrument"):
        _diag_log.append({"turn": getattr(obs.current, "turn", None), "iters": stats.iterations,
                          "nodes": stats.nodes, "expanded": stats.expanded_nodes, "max_depth": stats.max_depth,
                          "mean_depth": round(stats.mean_depth, 2), "determinizations": stats.determinizations,
                          "unique_isets": stats.unique_information_sets, "elapsed_ms": round(stats.elapsed_ms, 1),
                          "budget_ms": round(stats.budget_ms, 1), "timeout": stats.timeout,
                          "policy_top1": stats.root_policy_top1, "selected": stats.selected_action,
                          "changed": stats.policy_changed_by_search, "mapping_errors": stats.mapping_errors,
                          "early_stopped": stats.early_stopped, "stop_iter": stats.stop_iter,
                          "is_hard_root": stats.is_hard_root,
                          "soft_cap_action": list(stats.soft_cap_action) if stats.soft_cap_action else None,
                          "extended_changed": (stats.is_hard_root and stats.soft_cap_action is not None
                                               and stats.selected_action != stats.soft_cap_action),
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


def default_agent_ctrl(obs: Observation) -> list[int]:
    """direct H2H(v2.4 student vs v1)の control 側 agent_fn(picklable)。config は env
    `ISMCTS_CONFIG_CTRL`(path)から読む(cand は default_agent が `ISMCTS_CONFIG` を読む)。
    両者 ISMCTS を別 config で同一プロセスに置くための対称関数。挙動は default_agent と同一。"""
    import os
    path = os.environ.get("ISMCTS_CONFIG_CTRL") or str(_CONFIG_PATH)
    cfg = _CFG_CACHE.get(path)
    if cfg is None:
        cfg = _CFG_CACHE[path] = load_config(path)
    if obs.select is None:
        match_context.reset()
        return mpa.agent(obs, cfg)
    return agent(obs, cfg)


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
