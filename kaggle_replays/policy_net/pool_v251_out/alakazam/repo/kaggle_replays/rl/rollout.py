"""M1: self-play でトラジェクトリを収集する(cg エンジン経由)。

設計(rl-opponent-strengthening-implementation-plan.md §2):
- **RL は "素の policy" を学習**する。lethal_simple / attack_plan の overlay は eval/export 時に
  production と同じものをかぶせる(RL 中はかぶせない)。→ 「policy が RL で伸びるか」を隔離。
- 対戦相手も **素の production-alakazam policy**(pure-Python PolicyModel、overlay なし)にする。
- **単一選択(maxCount==1)のみ方策の学習対象**。複数選択(maxCount>1)は greedy top-k に委譲し
  トラジェクトリに記録しない(PoC の既知の制約)。

encoder は sample_submission 側を再利用(train/runtime parity)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402

MAX_STEPS = 3000


@dataclass
class Step:
    state_feat: list[float]
    option_feats: list[list[float]]
    card_ids: list[int]
    chosen_idx: int
    logprob: float


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0  # 学習側視点の終局報酬(勝ち1 / 負け0)
    winner: int | None = None
    error: str | None = None


def _encode_decision(state, select):
    """単一選択の学習対象なら (state_feat, option_feats, card_ids) を返す。そうでなければ None。"""
    if select is None or not select.option or select.maxCount != 1:
        return None
    state_feat = encoder.encode_state_from_state(state)
    option_feats = encoder.encode_options_from_state(state, select)
    card_ids = encoder.encode_option_card_ids(state, select)
    if not option_feats:
        return None
    return state_feat, option_feats, card_ids


def _greedy_multi_action(policy_scores: list[float], select) -> list[int]:
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    if not policy_scores:
        return list(range(count))
    ranked = sorted(range(n), key=lambda i: policy_scores[i], reverse=True)
    return ranked[:count]


class TorchActor:
    """torch policy で行動する学習側。単一選択は方策から sample(または argmax)し記録する。"""

    def __init__(self, policy, temperature: float = 1.0, sample: bool = True, device="cpu"):
        self.policy = policy
        self.temperature = temperature
        self.sample = sample
        self.device = device

    def act(self, obs: Observation, traj: Trajectory) -> list[int]:
        state = obs.current
        select = obs.select
        enc = _encode_decision(state, select)
        # 複数選択・非対象: 方策スコアで greedy(記録しない)。
        if enc is None:
            if select is None or not select.option:
                return []
            option_feats = encoder.encode_options_from_state(state, select)
            card_ids = encoder.encode_option_card_ids(state, select)
            if not option_feats:
                return list(range(max(select.minCount, 1)))
            with torch.no_grad():
                scores = self.policy.option_scores(
                    torch.tensor(encoder.encode_state_from_state(state), dtype=torch.float32, device=self.device),
                    torch.tensor(option_feats, dtype=torch.float32, device=self.device),
                    torch.tensor(card_ids, dtype=torch.long, device=self.device),
                ).tolist()
            return _greedy_multi_action(scores, select)

        state_feat, option_feats, card_ids = enc
        sf = torch.tensor(state_feat, dtype=torch.float32, device=self.device)
        of = torch.tensor(option_feats, dtype=torch.float32, device=self.device)
        ci = torch.tensor(card_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            dist = self.policy.distribution(sf, of, ci, temperature=self.temperature)
            if self.sample:
                idx = dist.sample()
            else:
                idx = dist.probs.argmax()
            logp = dist.log_prob(idx).item()
        chosen = int(idx.item())
        traj.steps.append(Step(state_feat, option_feats, card_ids, chosen, logp))
        return [chosen]


def make_pure_policy_agent(weights_path=None):
    """素の(overlay なし)pure-Python PolicyModel エージェント。相手側に使う。"""
    model = PolicyModel(weights_path)

    def agent(obs: Observation) -> list[int]:
        select = obs.select
        if select is None or not select.option:
            return []
        if select.maxCount == 1:
            idx = model.select_option(obs)
            return [idx if idx is not None else 0]
        scores = model.score_options(obs)
        return _greedy_multi_action(scores, select)

    return agent


def play_and_collect(actor: TorchActor, opponent_agent, deck_learn, deck_opp,
                     learner_index: int, seed: int | None = None) -> Trajectory:
    """1試合。learner_index 側を actor(torch)、他方を opponent_agent で駆動し、learner の
    単一選択決定を記録する。終局で reward を確定。"""
    import random
    if seed is not None:
        random.seed(seed)

    traj = Trajectory()
    if learner_index == 0:
        deck0, deck1 = deck_learn, deck_opp
    else:
        deck0, deck1 = deck_opp, deck_learn

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        traj.error = f"battle_start errorType={start_data.errorType}"
        return traj

    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is None:
                traj.error = "current is None mid-match"
                return traj
            if obs.current.result != -1:
                traj.winner = obs.current.result
                traj.reward = 1.0 if obs.current.result == learner_index else 0.0
                return traj
            if steps >= MAX_STEPS:
                traj.error = f"max_steps_exceeded({MAX_STEPS})"
                return traj
            turn_player = obs.current.yourIndex
            if turn_player == learner_index:
                action = actor.act(obs, traj)
            else:
                action = opponent_agent(obs)
            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:  # noqa: BLE001
        traj.error = repr(exc)
        return traj
    finally:
        battle_finish()
