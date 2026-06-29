"""Train the Maskable PPO Dragapult ex agent.

Pipeline:
  env (cg.game self-play) -> ActionMasker -> MaskablePPO -> export policy.npz

Opponent curriculum (configurable): random early, then a frozen self-play
snapshot of the current policy, optionally mixed with the heuristic/Monte-Carlo
agent. After training, the policy is exported to policy.npz for numpy inference
in main.py.

Examples:
    python train_ppo.py --timesteps 200000
    python train_ppo.py --timesteps 500000 --opponent curriculum --eval-games 40
    python train_ppo.py --timesteps 50000 --opponent random   # quick baseline

Run from the sample_submission folder so deck.csv resolves.

NOTE: cg.game uses a process-global battle, so this uses a single environment
(no DummyVecEnv parallelism). For more throughput, parallelise with SubprocVecEnv
(separate processes) — see docs/mppo-spec.md.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from tcg_rl.env import PokemonTCGEnv
from tcg_rl.features import action_mask
from tcg_rl.opponents import random_opponent
from cg.api import to_observation_class


def read_deck(path: str = "deck.csv") -> list[int]:
    rows = [r for r in open(path).read().split("\n") if r.strip()]
    return [int(rows[i]) for i in range(60)]


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


# --------------------------------------------------------------------------- #
# Opponent manager (curriculum + self-play snapshot)
# --------------------------------------------------------------------------- #
class OpponentManager:
    """Provides an opponent callable per episode based on training progress."""

    def __init__(self, deck, mode: str, use_mc: bool):
        self.deck = deck
        self.mode = mode
        self.step = 0
        self.snapshot = None      # frozen MaskablePPO for self-play
        self._selfplay_fn = None
        self._mc_fn = None
        if use_mc or mode in ("mc", "curriculum"):
            from tcg_rl.opponents import make_mc_opponent
            self._mc_fn = make_mc_opponent(time_budget=0.0)  # fast heuristic opponent

    def set_snapshot(self, model: MaskablePPO):
        from tcg_rl.opponents import make_selfplay_opponent
        self.snapshot = model
        self._selfplay_fn = make_selfplay_opponent(model, deterministic=False)

    def set_step(self, step: int):
        self.step = step

    def _weights(self) -> dict:
        if self.mode == "random":
            return {"random": 1.0}
        if self.mode == "selfplay":
            return {"random": 0.2, "selfplay": 0.8} if self._selfplay_fn else {"random": 1.0}
        if self.mode == "mc":
            return {"mc": 1.0}
        # curriculum
        if self.step < 30_000 or self._selfplay_fn is None:
            return {"random": 1.0}
        if self.step < 100_000:
            w = {"random": 0.3, "selfplay": 0.5}
            if self._mc_fn:
                w["mc"] = 0.2
            return w
        w = {"random": 0.1, "selfplay": 0.7}
        if self._mc_fn:
            w["mc"] = 0.2
        return w

    def __call__(self):
        w = self._weights()
        names = list(w.keys())
        name = random.choices(names, weights=[w[n] for n in names], k=1)[0]
        if name == "selfplay" and self._selfplay_fn:
            return self._selfplay_fn
        if name == "mc" and self._mc_fn:
            return self._mc_fn
        return random_opponent


# --------------------------------------------------------------------------- #
# Evaluation: win rate vs a fixed opponent
# --------------------------------------------------------------------------- #
def evaluate_winrate(model, deck, opponent_provider, n_games: int) -> float:
    env = PokemonTCGEnv(deck, opponent_provider=opponent_provider)
    wins = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        reward = 0.0
        while not done:
            m = env.action_masks()
            action, _ = model.predict(obs, action_masks=m, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            done = term or trunc
        if reward > 0:
            wins += 1
    env.close()
    return wins / max(n_games, 1)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class CurriculumCallback(BaseCallback):
    """Advance curriculum + refresh the self-play snapshot periodically."""

    def __init__(self, manager: OpponentManager, snapshot_freq: int):
        super().__init__()
        self.manager = manager
        self.snapshot_freq = snapshot_freq
        self._last_snap = 0

    def _on_step(self) -> bool:
        self.manager.set_step(self.num_timesteps)
        if self.num_timesteps - self._last_snap >= self.snapshot_freq:
            self._last_snap = self.num_timesteps
            # sync frozen snapshot weights from the live policy
            if self.manager.snapshot is not None:
                self.manager.snapshot.policy.load_state_dict(self.model.policy.state_dict())
        return True


class EvalCallback(BaseCallback):
    """Periodically report win rate vs random and save the best model."""

    def __init__(self, deck, eval_freq: int, n_games: int, save_path: str):
        super().__init__()
        self.deck = deck
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.save_path = save_path
        self.best = -1.0
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.eval_freq:
            self._last = self.num_timesteps
            wr = evaluate_winrate(self.model, self.deck,
                                  lambda: random_opponent, self.n_games)
            self.logger.record("eval/winrate_vs_random", wr)
            print(f"[eval] step={self.num_timesteps} winrate_vs_random={wr:.3f}")
            if wr > self.best:
                self.best = wr
                self.model.save(self.save_path + "_best.zip")
        return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--opponent", choices=["random", "selfplay", "mc", "curriculum"],
                    default="curriculum")
    ap.add_argument("--use-mc", action="store_true", help="mix heuristic MC into curriculum")
    ap.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--reward-shaping", action="store_true")
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--eval-games", type=int, default=30)
    ap.add_argument("--snapshot-freq", type=int, default=20_000)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="policy.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    deck = read_deck()
    random.seed(args.seed)
    np.random.seed(args.seed)

    manager = OpponentManager(deck, args.opponent, args.use_mc)
    env = ActionMasker(
        PokemonTCGEnv(deck, opponent_provider=manager, reward_shaping=args.reward_shaping),
        mask_fn,
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=0.997,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=list(args.net_arch)),
        verbose=1,
        seed=args.seed,
    )

    # self-play snapshot starts as a copy of the initial model
    if args.opponent in ("selfplay", "curriculum"):
        snap = MaskablePPO(
            MaskableActorCriticPolicy, env,
            policy_kwargs=dict(net_arch=list(args.net_arch)), seed=args.seed,
        )
        snap.policy.load_state_dict(model.policy.state_dict())
        manager.set_snapshot(snap)

    save_path = os.path.join(args.model_dir, "mppo_dragapult")
    callbacks = [
        CurriculumCallback(manager, args.snapshot_freq),
        EvalCallback(deck, args.eval_freq, args.eval_games, save_path),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)

    model.save(save_path + ".zip")
    from export_policy import export
    export(save_path + ".zip", args.out)
    print(f"done. model={save_path}.zip  policy={args.out}")


if __name__ == "__main__":
    main()
