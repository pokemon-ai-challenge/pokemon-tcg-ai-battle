"""
PPO training script for Pokemon TCG RL agent.

Usage (from sample_submission/):
    python train_rl.py [--games 8] [--epochs 200] [--lr 3e-4]

Saves model weights to: rl/weights.pt
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

from main import read_deck_csv
from rl.model import PokemonRLModel
from rl.selfplay import collect_episodes, Step
from rl.encoder import STATE_DIM, OPTION_DIM

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl", "weights.pt")


# -----------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# -----------------------------------------------------------------------

def compute_gae(
    steps: list[Step],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute returns and GAE advantages for a list of steps (single trajectory)."""
    returns = []
    advantages = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(steps):
        delta = step.reward + gamma * next_value * (1 - int(step.done)) - step.value
        gae = delta + gamma * lam * (1 - int(step.done)) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + step.value)
        next_value = step.value

    return returns, advantages


# -----------------------------------------------------------------------
# PPO update
# -----------------------------------------------------------------------

def ppo_update(
    model: PokemonRLModel,
    optimizer: torch.optim.Optimizer,
    steps: list[Step],
    device: torch.device,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
) -> dict:
    if not steps:
        return {}

    returns, advantages = compute_gae(steps)

    # Normalize advantages
    adv_arr = np.array(advantages, dtype=np.float32)
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

    old_log_probs = torch.tensor([s.log_prob for s in steps], dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=device)

    # Pad options to the same length per step
    max_opts = max(len(s.opt_feats) for s in steps)

    metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

    indices = list(range(len(steps)))
    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start: start + minibatch_size]
            batch = [steps[i] for i in batch_idx]

            # Build padded tensors
            state_t = torch.tensor(
                [s.state_feat for s in batch], dtype=torch.float32, device=device
            )
            # Pad options
            opt_t_list = []
            mask_list = []
            for s in batch:
                n = len(s.opt_feats)
                pad = max_opts - n
                feats = torch.tensor(s.opt_feats, dtype=torch.float32, device=device)
                if pad > 0:
                    feats = F.pad(feats, (0, 0, 0, pad))
                mask = torch.zeros(max_opts, dtype=torch.bool, device=device)
                mask[:n] = True
                opt_t_list.append(feats)
                mask_list.append(mask)

            opt_t = torch.stack(opt_t_list)    # (B, max_opts, OPTION_DIM)
            mask_t = torch.stack(mask_list)    # (B, max_opts)

            log_probs_all, values = model(state_t, opt_t, mask_t)

            # Gather log probs for chosen actions
            action_idx = torch.tensor(
                [s.action_idx for s in batch], dtype=torch.long, device=device
            )
            # Clamp action_idx to valid range
            action_idx = action_idx.clamp(0, max_opts - 1)
            new_log_probs = log_probs_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)

            batch_old_lp = old_log_probs[[i for i in batch_idx]]
            batch_adv = adv_t[[i for i in batch_idx]]
            batch_ret = returns_t[[i for i in batch_idx]]

            ratio = (new_log_probs - batch_old_lp).exp()
            surr1 = ratio * batch_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_ret)

            # Entropy bonus (over valid options)
            probs = log_probs_all.exp()
            probs = probs * mask_t.float()
            entropy = -(probs * log_probs_all.clamp(min=-20)).sum(dim=-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())

    return {k: np.mean(v) for k, v in metrics.items()}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8, help="Self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=200, help="Training iterations")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true", help="Resume from weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    deck = read_deck_csv()
    print(f"Deck loaded: {len(deck)} cards")

    model = PokemonRLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        model.eval()
        steps = collect_episodes(model, deck, n_games=args.games, device=device)
        print(f"  Collected {len(steps)} steps")

        if not steps:
            continue

        model.train()
        metrics = ppo_update(model, optimizer, steps, device)
        print(f"  policy_loss={metrics.get('policy_loss', 0):.4f}  "
              f"value_loss={metrics.get('value_loss', 0):.4f}  "
              f"entropy={metrics.get('entropy', 0):.4f}")

        # Save checkpoint
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, WEIGHTS_PATH)

    print(f"\nTraining complete. Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
