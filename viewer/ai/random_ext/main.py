"""Minimal drop-in example agent for the viewer (see viewer/ai/README.md).

Same interface as a Kaggle submission: agent(obs_dict) -> list[int].
This one just picks a random legal action. Copy this folder as a starting
point, or drop a real submission folder next to it.
"""

import random

from cg.api import to_observation_class  # resolved from repo sample_submission if not bundled


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return []
    sel = obs.select
    n = len(sel.option)
    lo, hi = sel.minCount, min(sel.maxCount, n)
    if hi < lo:
        return []
    k = random.randint(lo, hi)
    return random.sample(range(n), k) if k > 0 else []
