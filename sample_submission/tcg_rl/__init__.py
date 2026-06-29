"""tcg_rl: Maskable PPO agent components for the Dragapult ex deck.

Submission-safe modules (numpy + cg only, no torch/gym):
    - features.py    : observation encoder + action mask (SHARED by training and inference)
    - mlp_numpy.py   : numpy MLP forward pass for submission-time inference

Training-only modules (require gymnasium / torch / sb3-contrib):
    - env.py         : Gymnasium environment wrapping cg.game
    - opponents.py   : random / self-play / Monte-Carlo opponents
    - mc_agent.py    : ported Flat Monte Carlo agent (training opponent + improvable evaluator)
"""
