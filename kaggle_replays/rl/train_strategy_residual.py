"""(ii) 戦略残差NN の最小 self-play RL 学習(REINFORCE, numpy 自己完結)。

設計: sample_submission/docs/plans/strategy-residual-nn-experiment.md。
凍結した模倣 PolicyModel の score に線形残差 bias_i = W . option_features_i を足し、
MAIN(単一選択)のみ softmax(score + alpha*bias) から sample。勝敗で REINFORCE:
  grad_W += (R - baseline) * alpha * (feat[chosen] - E_pi[feat])
softmax なので定数項 b は消える(Wのみ学習)。torch/既存RL基盤(train_v3 PPO=フラットRL)には触らない。

相手 field は現在メタ climb帯で加重(crustle/rocket_mewtwo 厚め)。学習後、
StrategyResidual JSON({"W":..,"b":0,"scale":1})を吐き、config strategy_residual.weights_path で注入して
`kaggle_replays/_eval_current_meta.py` で採否判定する。

使い方: python kaggle_replays/rl/train_strategy_residual.py --iters 30 --games 40 --lr 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402
from cg.api import OptionType, SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
PROD_DECK = _ROOT / "sample_submission" / "deck.csv"
MAX_STEPS = 3000

# climb_600_699 加重(足枷 crustle/rocket_mewtwo を厚めに, 在る対面のみ)。
FIELD = [
    ("mega_lucario_ex", "policy_weights_mega_lucario_ex.json", 24.2),
    ("alakazam", None, 18.2),
    ("dragapult_ex", "policy_weights_dragapult_ex.json", 13.0),
    ("crustle", "policy_weights_crustle.json", 12.6),
    ("marnie_grimmsnarl_ex", "policy_weights_marnie_grimmsnarl_ex.json", 7.4),
    ("archaludon_ex", "policy_weights_archaludon_ex.json", 3.2),
    ("rocket_mewtwo_ex", "policy_weights_rocket_mewtwo_ex.json", 3.2),
]


def _softmax(x):
    m = np.max(x)
    e = np.exp(x - m)
    return e / (e.sum() + 1e-12)


def _greedy_multi(scores, select):
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order[:count]


class ResidualLearner:
    """凍結模倣 + 線形残差 W(標準化 option特徴上)。MAIN単一選択のみ sample し記録。"""

    def __init__(self, model: PolicyModel, W: np.ndarray, alpha: float,
                 mean: np.ndarray, std: np.ndarray):
        self.model = model
        self.W = W
        self.alpha = alpha
        self.mean = mean
        self.std = std
        self.decisions = []  # (standardized_feat_matrix(np, n×d), chosen_idx, pi(np))

    def _standardize(self, F: np.ndarray) -> np.ndarray:
        d = self.W.shape[0]
        if F.shape[1] < d:
            F = np.pad(F, ((0, 0), (0, d - F.shape[1])))
        else:
            F = F[:, :d]
        safe = np.where(self.std != 0, self.std, 1.0)
        z = np.where(self.std != 0, (F - self.mean) / safe, 0.0)
        return z

    def act(self, state, select):
        try:
            scores = self.model.score_options_from_state(state, select)
        except Exception:
            scores = []
        n = len(select.option)
        if not scores or len(scores) != n:
            return list(range(max(select.minCount, min(select.maxCount, n))))
        if select.type == SelectType.MAIN and select.maxCount == 1:
            try:
                rows = encoder.encode_options_from_state(state, select)
                Z = self._standardize(np.array(rows, dtype=np.float64))  # (n, d) 標準化済
                bias = Z @ self.W
            except Exception:
                Z = None
                bias = np.zeros(n)
            biased = np.array(scores, dtype=np.float64) + self.alpha * bias
            pi = _softmax(biased)
            chosen = int(np.random.choice(n, p=pi))
            if Z is not None:
                self.decisions.append((Z, chosen, pi))
            return [chosen]
        # 非MAIN / 複数選択 は模倣 greedy(残差なし・記録なし)。
        return _greedy_multi(scores, select)


def _greedy_agent(model: PolicyModel):
    def act(state, select):
        try:
            scores = model.score_options_from_state(state, select)
        except Exception:
            scores = []
        n = len(select.option)
        if not scores or len(scores) != n:
            return list(range(max(select.minCount, min(select.maxCount, n))))
        return _greedy_multi(scores, select)
    return act


def play_game(learner: ResidualLearner, opp_act, deck0, deck1, seed):
    random.seed(seed)
    np.random.seed(seed)
    learner.decisions = []
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return None
    steps = 0
    try:
        while steps < MAX_STEPS:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                return None
            if cur.result != -1:
                return 1.0 if cur.result == 0 else 0.0
            act = learner.act(cur, obs.select) if cur.yourIndex == 0 else opp_act(cur, obs.select)
            obs_dict = battle_select(act)
            steps += 1
    except Exception:
        return None
    finally:
        battle_finish()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=40, help="1 iter の試合数")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=str(WDIR / "policy_weights_strategy_residual.json"))
    ap.add_argument("--log", default=str(_HERE / "_train_strategy_residual.log"))
    args = ap.parse_args()
    # chdir 前に出力パスを絶対化(chdir後の相対パス崩れを防ぐ)。
    args.out = str(Path(args.out).resolve())
    args.log = str(Path(args.log).resolve())

    os.chdir(_ROOT / "sample_submission")
    learner_model = PolicyModel()  # 凍結模倣(既定重み)
    opp_models = {arch: PolicyModel(str(WDIR / w) if w else None) for arch, w, _ in FIELD}
    deck0 = run_league.read_deck_csv_file(str(PROD_DECK))
    opp_decks = {arch: run_league.read_deck_csv_file(str(DECKDIR / arch / "01.csv")) for arch, _, _ in FIELD}
    field_arch = [a for a, _, _ in FIELD]
    field_w = np.array([w for _, _, w in FIELD], dtype=np.float64)
    field_p = field_w / field_w.sum()

    d = encoder.OPTION_FEATURE_COUNT
    W = np.zeros(d, dtype=np.float64)
    # 模倣が使う option特徴の標準化統計(学習・デプロイで共有する)。
    raw_mean = getattr(learner_model, "_option_mean", None) or []
    raw_std = getattr(learner_model, "_option_std", None) or []
    mean = np.zeros(d); std = np.zeros(d)
    for i in range(min(d, len(raw_mean))):
        mean[i] = raw_mean[i]
    for i in range(min(d, len(raw_std))):
        std[i] = raw_std[i]
    baseline = 0.5
    logf = open(args.log, "w", encoding="utf-8")

    def logline(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    logline(f"=== residual REINFORCE: d={d} iters={args.iters} games/iter={args.games} lr={args.lr} alpha={args.alpha} ===")
    seed = 0
    for it in range(args.iters):
        learner = ResidualLearner(learner_model, W, args.alpha, mean, std)
        gradW = np.zeros(d)
        wins = 0
        n_valid = 0
        n_dec = 0
        for _g in range(args.games):
            arch = np.random.choice(field_arch, p=field_p)
            R = play_game(learner, _greedy_agent(opp_models[arch]), deck0, opp_decks[arch], seed)
            seed += 1
            if R is None:
                continue
            n_valid += 1
            wins += int(R > 0.5)
            adv = R - baseline
            for F, chosen, pi in learner.decisions:
                # grad log pi(chosen) wrt W = alpha * (F[chosen] - sum_i pi_i F[i])
                exp_feat = pi @ F
                gradW += adv * args.alpha * (F[chosen] - exp_feat)
                n_dec += 1
            learner.decisions = []
        if n_dec > 0:
            W += args.lr * gradW / n_dec
        wr = wins / n_valid if n_valid else 0.0
        baseline = 0.9 * baseline + 0.1 * wr
        logline(f"[iter {it:3d}] wr={wr:.3f} ({wins}/{n_valid}) |W|={np.linalg.norm(W):.4f} n_dec={n_dec} baseline={baseline:.3f}")
        if (it + 1) % 5 == 0 or it == args.iters - 1:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"W": W.tolist(), "b": 0.0, "scale": 1.0,
                           "option_mean": mean.tolist(), "option_std": std.tolist(),
                           "iter": it + 1,
                           "note": "strategy residual (REINFORCE, climb_600_699 field)"}, f)
            logline(f"  exported -> {args.out}")
    logf.close()


if __name__ == "__main__":
    main()
