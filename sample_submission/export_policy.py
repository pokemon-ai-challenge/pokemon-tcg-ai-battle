"""Export a trained MaskablePPO model to policy.npz for numpy inference.

The submission's main.py runs a plain numpy forward pass (no torch), so we
extract just the policy (actor) MLP: the hidden layers from
policy.mlp_extractor.policy_net followed by policy.action_net. Torch Linear
weights are (out, in); we store them transposed to (in, out) so inference can do
`x @ W + b`. Activation is tanh on every layer except the final action layer.

Usage:
    python export_policy.py --model models/mppo_dragapult.zip --out policy.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch.nn as nn
from sb3_contrib import MaskablePPO

from tcg_rl.features import ACTION_DIM, OBS_DIM


def export(model_path: str, out_path: str) -> None:
    model = MaskablePPO.load(model_path, device="cpu")
    policy = model.policy

    linears: list[nn.Linear] = [
        m for m in policy.mlp_extractor.policy_net.modules() if isinstance(m, nn.Linear)
    ]
    linears.append(policy.action_net)

    arrays = {
        "n_layers": np.array(len(linears)),
        "obs_dim": np.array(OBS_DIM),
        "act_dim": np.array(ACTION_DIM),
    }
    for i, lin in enumerate(linears):
        W = lin.weight.detach().cpu().numpy().T.astype(np.float32)  # (in, out)
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b

    np.savez(out_path, **arrays)
    shapes = " -> ".join(str(arrays[f"W{i}"].shape) for i in range(len(linears)))
    print(f"exported {len(linears)} layers to {out_path}: {shapes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mppo_dragapult.zip")
    ap.add_argument("--out", default="policy.npz")
    args = ap.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
