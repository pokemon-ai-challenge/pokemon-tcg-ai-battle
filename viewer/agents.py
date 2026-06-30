"""AI agent registry for the viewer.

An agent is a callable ``agent_fn(obs_dict) -> list[int]`` -- exactly the same
interface as the Kaggle submission ``agent()``. To add a new AI, write a small
``build_*`` factory that returns such a callable and call ``register(...)`` at
the bottom of this file. It then appears automatically in the web UI.

All agents are wrapped with a safety layer (``_safe``) so a misbehaving policy
can never crash a match: an illegal/garbage return falls back to the first
legal action.

NOTE: ``cg`` must be importable; ``viewer.engine`` puts ``sample_submission`` on
``sys.path`` before this module is used. Importing this module standalone also
works because we add the path defensively here.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SAMPLE = os.path.join(_REPO, "sample_submission")
if _SAMPLE not in sys.path:
    sys.path.insert(0, _SAMPLE)

from cg.api import Observation, to_observation_class  # noqa: E402

AgentFn = Callable[[dict], list[int]]
BuildFn = Callable[[], AgentFn]


# --------------------------------------------------------------------------- #
# Safety helpers
# --------------------------------------------------------------------------- #
def _min_legal(obs: Observation) -> list[int]:
    """A guaranteed-legal action (first minCount options)."""
    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    if sel.minCount <= 0:
        return []
    return list(range(min(sel.minCount, n)))


def _validate(obs: Observation, action) -> bool:
    sel = obs.select
    if sel is None:
        return False
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (sel.minCount <= len(action) <= sel.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(sel.option) for i in action)


def _safe(inner: Callable[[Observation], list[int]]) -> AgentFn:
    """Wrap a per-Observation policy into a crash-proof obs_dict agent."""

    def agent_fn(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            # Deck selection is handled up front by battle_start, so this branch
            # is not exercised during a match; return empty defensively.
            return []
        try:
            action = inner(obs)
        except Exception:
            return _min_legal(obs)
        if not _validate(obs, action):
            return _min_legal(obs)
        return action

    return agent_fn


# --------------------------------------------------------------------------- #
# Concrete agents
# --------------------------------------------------------------------------- #
def build_random() -> AgentFn:
    def inner(obs: Observation) -> list[int]:
        sel = obs.select
        n = len(sel.option)
        lo, hi = sel.minCount, min(sel.maxCount, n)
        if hi < lo:
            return []
        k = random.randint(lo, hi)
        return random.sample(range(n), k) if k > 0 else []

    return _safe(inner)


def build_heuristic() -> AgentFn:
    from main_minimal_snapshot import choose_action

    def inner(obs: Observation) -> list[int]:
        return choose_action(obs)

    return _safe(inner)


# mPPO policy is loaded once and shared across matches.
_MPPO_POLICY = None
_MPPO_TRIED = False


def _mppo_policy():
    global _MPPO_POLICY, _MPPO_TRIED
    if _MPPO_TRIED:
        return _MPPO_POLICY
    _MPPO_TRIED = True
    try:
        from tcg_rl.mlp_numpy import load_policy
        path = os.path.join(_SAMPLE, "policy.npz")
        _MPPO_POLICY = load_policy(path)
    except Exception:
        _MPPO_POLICY = None
    return _MPPO_POLICY


def build_mppo() -> AgentFn:
    from tcg_rl.mlp_numpy import decide_full_action

    def inner(obs: Observation) -> list[int]:
        policy = _mppo_policy()
        if policy is None:
            return _min_legal(obs)
        return decide_full_action(obs, policy.choose)

    return _safe(inner)


def build_montecarlo() -> AgentFn:
    from tcg_rl.mc_agent import MonteCarloAgent

    # Low budget keeps moves responsive enough for an interactive viewer.
    mc = MonteCarloAgent(rollouts=6, depth=18, time_budget=0.3)

    def inner(obs: Observation) -> list[int]:
        return mc.act(obs)

    return _safe(inner)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class AgentSpec:
    id: str
    label: str
    build: BuildFn
    slow: bool = False
    note: str = ""


REGISTRY: dict[str, AgentSpec] = {}
_ORDER: list[str] = []


def register(spec: AgentSpec) -> None:
    REGISTRY[spec.id] = spec
    if spec.id not in _ORDER:
        _ORDER.append(spec.id)


def list_agents() -> list[dict]:
    return [
        {"id": s.id, "label": s.label, "slow": s.slow, "note": s.note}
        for s in (REGISTRY[i] for i in _ORDER)
    ]


def build_agent(agent_id: str) -> AgentFn:
    spec = REGISTRY.get(agent_id)
    if spec is None:
        raise ValueError(f"不明なAIです: {agent_id}")
    return spec.build()


# --- registrations (add new AIs here) -------------------------------------- #
register(AgentSpec("mppo", "mPPO（学習済み）", build_mppo,
                   note="policy.npz の Maskable PPO エージェント"))
register(AgentSpec("heuristic", "ヒューリスティック", build_heuristic,
                   note="ルールベースの簡易AI"))
register(AgentSpec("random", "ランダム", build_random,
                   note="合法手から一様ランダム"))
register(AgentSpec("montecarlo", "モンテカルロ探索", build_montecarlo, slow=True,
                   note="search APIでロールアウト。1手が遅い"))
