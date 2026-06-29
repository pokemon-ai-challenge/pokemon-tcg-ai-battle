"""Numpy MLP inference for submission time (no torch/gym dependency).

Loads a policy exported by export_policy.py (policy.npz) and runs a masked
forward pass. Also provides decide_full_action(), the factored-action loop that
turns a per-option policy into a full list[int] selection — shared by the
self-play opponent (training) and main.py (submission) so they stay consistent.
"""

from __future__ import annotations

import numpy as np

from cg.api import Observation
from tcg_rl.features import (
    ACTION_DIM,
    MAX_OPTIONS,
    OBS_DIM,
    STOP_ACTION,
    action_mask,
    encode_obs,
)


def _tanh(x):
    return np.tanh(x)


class NumpyPolicy:
    """A plain MLP: hidden layers with tanh, final linear produces action logits."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]]):
        # weights[i] = (W, b) with W shaped (in, out); apply tanh on all but last.
        self.weights = weights

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(self.weights):
            h = h @ W + b
            if i != last:
                h = _tanh(h)
        return h

    def choose(self, vec: np.ndarray, mask: np.ndarray) -> int:
        logits = self.logits(vec)
        masked = np.where(mask, logits, -1e9)
        return int(np.argmax(masked))


def load_policy(path: str) -> NumpyPolicy:
    data = np.load(path)
    n = int(data["n_layers"])
    obs_dim = int(data["obs_dim"])
    act_dim = int(data["act_dim"])
    if obs_dim != OBS_DIM or act_dim != ACTION_DIM:
        raise ValueError(
            f"policy.npz dims ({obs_dim},{act_dim}) != current features "
            f"({OBS_DIM},{ACTION_DIM}); retrain/re-export required."
        )
    weights = []
    for i in range(n):
        weights.append((data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)))
    return NumpyPolicy(weights)


def decide_full_action(obs: Observation, choose_fn, max_substeps: int = 24) -> list[int]:
    """Run the factored policy over one SelectData and return the full index list.

    choose_fn(vec, mask) -> int : returns an action index in [0, ACTION_DIM).
    Guarantees a legal result (minCount..maxCount, unique, in range), even if
    choose_fn misbehaves, by falling back to the first legal options.
    """
    sel = obs.select
    if sel is None:
        return []
    n = min(len(sel.option), MAX_OPTIONS)
    buffer: list[int] = []
    for _ in range(max_substeps):
        if len(buffer) >= sel.maxCount:
            break
        vec = encode_obs(obs, buffer)
        mask = action_mask(obs, buffer)
        a = int(choose_fn(vec, mask))
        if a == STOP_ACTION:
            if len(buffer) >= sel.minCount:
                break
            # cannot stop yet; force a pick below
            a = _first_unused(n, buffer)
            if a is None:
                break
        if a < 0 or a >= n or a in buffer:
            a = _first_unused(n, buffer)
            if a is None:
                break
        buffer.append(a)

    # ensure minimum count
    while len(buffer) < sel.minCount:
        a = _first_unused(n, buffer)
        if a is None:
            break
        buffer.append(a)
    return buffer


def _first_unused(n: int, buffer: list[int]):
    for i in range(n):
        if i not in buffer:
            return i
    return None
