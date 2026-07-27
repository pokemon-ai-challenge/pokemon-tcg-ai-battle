"""方策勾配(REINFORCE)とポテンシャルベースの報酬シェーピングの正しさを検証する。

設計書(``test_plan/ptcg_rl_design.md`` §6-3)より:

    3の勾配検証は省略しないこと。方策勾配は符号を間違えても「それらしく」動き、
    学習が進まない原因の特定が極めて難しい。数値微分との照合をテストとして残す。

このファイルは省略してはならない必須テストであり、以下を検証する:

1. ``rl.policy.log_prob_gradient`` の解析勾配が、``log pi(a)`` を重み1次元ずつ
   中心差分で数値微分した値と一致する(ランダムな特徴・重み・選択肢数で複数ケース)。
2. 温度 T -> 0 で方策が argmax の one-hot に収束する(``softmax`` の性質)。
3. ``rl.shaping`` のポテンシャルベースシェーピングが、Σ_t gamma^t * F_t が
   終端項 gamma^T * Phi(s_T) - Phi(s_0) に畳み込まれる(望遠鏡和)という
   設計書 §3.2 の性質を満たす。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PRIOR_DIR = _REPO_ROOT / "kaggle_replays" / "policy_prior"
if str(_POLICY_PRIOR_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_PRIOR_DIR))

from rl.policy import argmax_index, log_prob, log_prob_gradient, softmax  # noqa: E402
from rl.shaping import potential, shaping_reward  # noqa: E402

_GRAD_TOL = 1e-6
_EPS = 1e-5


def _random_case(rng: random.Random, n_features: int, n_actions: int):
    feature_names = [f"f{i}" for i in range(n_features)]
    weights = [rng.uniform(-1.0, 1.0) for _ in range(n_features)]
    # 各選択肢はランダムなスパース部分集合の特徴を持つ(実データを模す)。
    feature_dicts = []
    for _ in range(n_actions):
        present = [name for name in feature_names if rng.random() < 0.6]
        if not present:
            present = [rng.choice(feature_names)]
        feature_dicts.append({name: rng.uniform(-2.0, 2.0) for name in present})
    chosen_index = rng.randrange(n_actions)
    return feature_names, weights, feature_dicts, chosen_index


def _scores(weights: list[float], feature_names: list[str], feature_dicts: list[dict]) -> list[float]:
    index = {name: i for i, name in enumerate(feature_names)}
    scores = []
    for feats in feature_dicts:
        s = 0.0
        for name, value in feats.items():
            i = index.get(name)
            if i is not None:
                s += weights[i] * value
        scores.append(s)
    return scores


def _numeric_gradient(
    weights: list[float],
    feature_names: list[str],
    feature_dicts: list[dict],
    chosen_index: int,
    temperature: float,
    eps: float = _EPS,
) -> dict[str, float]:
    """log pi(chosen_index) を重み1次元ずつ中心差分で数値微分する。"""
    numeric = {}
    for d, name in enumerate(feature_names):
        w_plus = list(weights)
        w_minus = list(weights)
        w_plus[d] += eps
        w_minus[d] -= eps
        lp_plus = log_prob(_scores(w_plus, feature_names, feature_dicts), chosen_index, temperature)
        lp_minus = log_prob(_scores(w_minus, feature_names, feature_dicts), chosen_index, temperature)
        numeric[name] = (lp_plus - lp_minus) / (2 * eps)
    return numeric


def test_analytic_gradient_matches_numeric_gradient_many_random_cases():
    rng = random.Random(20260727)
    max_abs_diff = 0.0
    n_cases = 40
    for case_i in range(n_cases):
        n_features = rng.randint(3, 8)
        n_actions = rng.randint(2, 6)
        temperature = rng.uniform(0.3, 3.0)
        feature_names, weights, feature_dicts, chosen_index = _random_case(
            rng, n_features, n_actions
        )

        scores = _scores(weights, feature_names, feature_dicts)
        probs = softmax(scores, temperature)
        analytic = log_prob_gradient(feature_dicts, probs, chosen_index, temperature)
        numeric = _numeric_gradient(
            weights, feature_names, feature_dicts, chosen_index, temperature
        )

        for name in feature_names:
            a = analytic.get(name, 0.0)
            n = numeric[name]
            diff = abs(a - n)
            max_abs_diff = max(max_abs_diff, diff)
            assert diff < _GRAD_TOL, (
                f"case={case_i} feature={name} T={temperature:.3f}: "
                f"analytic={a} numeric={n} diff={diff}"
            )
    # 実際の最大誤差を出力に残す(pytest -s で見える。報告用)。
    print(f"\n[test_rl_gradient] 解析勾配 vs 数値微分 最大誤差 = {max_abs_diff:.3e} "
          f"({n_cases}ケース, 許容誤差 {_GRAD_TOL:.0e})")


def test_analytic_gradient_matches_numeric_gradient_temperature_one():
    """T=1 は設計書 §2 の簡潔な式 (x_a - Σ pi_i x_i) とそのまま一致するはずの基本ケース。"""
    rng = random.Random(1)
    feature_names, weights, feature_dicts, chosen_index = _random_case(rng, 5, 4)
    scores = _scores(weights, feature_names, feature_dicts)
    probs = softmax(scores, 1.0)
    analytic = log_prob_gradient(feature_dicts, probs, chosen_index, 1.0)

    # 設計書そのままの式: x_a - Σ_i pi_i x_i
    expected = {}
    for p, feats in zip(probs, feature_dicts):
        for name, value in feats.items():
            expected[name] = expected.get(name, 0.0) - p * value
    for name, value in feature_dicts[chosen_index].items():
        expected[name] = expected.get(name, 0.0) + value

    for name in feature_names:
        assert abs(analytic.get(name, 0.0) - expected.get(name, 0.0)) < 1e-9

    numeric = _numeric_gradient(weights, feature_names, feature_dicts, chosen_index, 1.0)
    for name in feature_names:
        assert abs(analytic.get(name, 0.0) - numeric[name]) < _GRAD_TOL


def test_temperature_to_zero_converges_to_argmax():
    """T -> 0 で softmax(score/T) が argmax の one-hot に収束する(設計書 §2)。"""
    rng = random.Random(7)
    for _ in range(10):
        scores = [rng.uniform(-3.0, 3.0) for _ in range(rng.randint(2, 6))]
        best = argmax_index(scores)
        probs = softmax(scores, temperature=1e-4)
        assert probs[best] > 0.999999
        assert sum(probs) == pytest_approx_one(probs)


def pytest_approx_one(probs):
    # 単純な合計チェック用のヘルパ(pytest.approx を使わず標準ライブラリのみで完結させる)
    total = sum(probs)
    assert abs(total - 1.0) < 1e-9
    return total


def test_temperature_changes_action_variance():
    """設計書 §6(表, row2)の完了条件: 温度を変えると行動の分散が変わることを確認する。"""
    rng_high = random.Random(42)
    rng_low = random.Random(42)
    scores = [1.0, 0.9, 0.5, -0.2]

    def _sample_many(temperature: float, rng: random.Random, n: int = 2000) -> float:
        from rl.policy import sample_index

        probs = softmax(scores, temperature)
        counts = [0] * len(scores)
        for _ in range(n):
            counts[sample_index(probs, rng)] += 1
        # 分散の代理指標として「最頻選択肢の選択率」を使う(低いほど分散が大きい)。
        return max(counts) / n

    high_t_top_share = _sample_many(temperature=5.0, rng=rng_high)
    low_t_top_share = _sample_many(temperature=0.05, rng=rng_low)
    assert low_t_top_share > high_t_top_share, (
        f"低温({low_t_top_share:.3f})の方が高温({high_t_top_share:.3f})より"
        "決定的(分散が小さい)はずが逆転している"
    )


# --- ポテンシャルベースのシェーピング(設計書 §3.2 / §6 row4) -------------------


def test_potential_definition():
    assert potential(own_n_prize=6, opp_n_prize=6) == 0.0
    assert potential(own_n_prize=2, opp_n_prize=6) == 4.0  # 相手6-自分2: こちらが有利
    assert potential(own_n_prize=6, opp_n_prize=2) == -4.0


def test_shaping_reward_matches_formula():
    f = shaping_reward(own_n_prize_t=6, opp_n_prize_t=6, own_n_prize_t1=6, opp_n_prize_t1=5, gamma=0.9)
    # Phi(t) = 0, Phi(t+1) = 5-6 = -1 -> F = 0.9*(-1) - 0 = -0.9
    assert abs(f - (-0.9)) < 1e-12


def test_shaping_sum_telescopes_to_terminal_term_random_trajectories():
    """Σ_t gamma^t * F(s_t, s_{t+1}) == gamma^T * Phi(s_T) - Phi(s_0) を、
    ランダムなサイド枚数の推移・複数の gamma で検証する(設計書 §3.2 / §6 row4)。

    これは「途中の Phi がどう動いても総和は終端項に畳み込まれる」という、
    ポテンシャルベースシェーピングが最適方策を歪めないことの根拠そのものであり、
    Φ/F の実装がその形を正しく満たしているかを直接検証する。
    """
    rng = random.Random(999)
    for gamma in (1.0, 0.99, 0.9, 0.5):
        for _ in range(20):
            length = rng.randint(1, 30)
            # サイド枚数は 0..6 の範囲でランダムウォーク(実戦の値域を模す)。
            own_prizes = [6]
            opp_prizes = [6]
            for _ in range(length):
                own_prizes.append(max(0, own_prizes[-1] - (1 if rng.random() < 0.3 else 0)))
                opp_prizes.append(max(0, opp_prizes[-1] - (1 if rng.random() < 0.3 else 0)))

            total = 0.0
            for t in range(length):
                f_t = shaping_reward(
                    own_prizes[t], opp_prizes[t], own_prizes[t + 1], opp_prizes[t + 1], gamma
                )
                total += (gamma ** t) * f_t

            phi_0 = potential(own_prizes[0], opp_prizes[0])
            phi_T = potential(own_prizes[length], opp_prizes[length])
            expected = (gamma ** length) * phi_T - phi_0
            assert abs(total - expected) < 1e-9, (
                f"gamma={gamma} length={length}: telescoped_sum={total} expected={expected}"
            )
