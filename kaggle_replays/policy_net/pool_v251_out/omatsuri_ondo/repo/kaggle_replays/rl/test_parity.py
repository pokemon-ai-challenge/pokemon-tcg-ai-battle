"""M0: TorchOptionPolicy が pure-Python PolicyModel._forward と数値一致することを検証。

encoder に依存せず、ランダムな特徴ベクトルで forward の数式そのものを比較する
(標準化 → 連結 → MLP)。実盤面での一致は M1(rollout)で別途確認する。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "sample_submission"))

from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402  (rl/ を sys.path に足して実行)

WEIGHTS = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights_dragapult_ex.json"


def _run(pm, tp, rng, dtype, realistic):
    """dtype で torch を回し、pure-Python(float64)との最大絶対誤差を返す。

    realistic=True: 各特徴を mean 近傍(mean + N(0, |std_used|))で生成。準定数特徴(std≈1e-6)を
    非現実な大値で割らないので float32 でも一致する。realistic=False: 広い一様乱数(構造確認用、
    float64 でのみ厳密一致を要求)。
    """
    state_dim = len(pm._state_mean)
    option_dim = len(pm._option_mean)
    card_id_max = tp.card_id_max
    max_abs = 0.0
    for _ in range(50):
        if realistic:
            state_feat = [pm._state_mean[i] + rng.gauss(0, 1) * max(pm._state_std[i], 0.0)
                          for i in range(state_dim)]
        else:
            state_feat = [rng.uniform(-3, 30) for _ in range(state_dim)]
        n_opts = rng.randint(1, 12)
        if realistic:
            option_rows = [[pm._option_mean[i] + rng.gauss(0, 1) * max(pm._option_std[i], 0.0)
                            for i in range(option_dim)] for _ in range(n_opts)]
        else:
            option_rows = [[rng.uniform(-3, 30) for _ in range(option_dim)] for _ in range(n_opts)]
        card_ids = [rng.choice([0, rng.randint(0, card_id_max), card_id_max + 5]) for _ in range(n_opts)]

        py_scores = [pm._forward(state_feat, option_rows[i], card_ids[i]) for i in range(n_opts)]
        with torch.no_grad():
            t_scores = tp.option_scores(
                torch.tensor(state_feat, dtype=dtype),
                torch.tensor(option_rows, dtype=dtype),
                torch.tensor(card_ids, dtype=torch.long),
            ).tolist()
        for a, b in zip(py_scores, t_scores):
            max_abs = max(max_abs, abs(a - b))
    return max_abs


def main() -> None:
    pm = PolicyModel(WEIGHTS)
    assert pm.is_ready, "pure-Python PolicyModel の読み込み失敗"

    # (1) 構造の厳密確認: float64 で任意入力でも一致。
    tp64 = TorchOptionPolicy.from_json(WEIGHTS).double().eval()
    d64 = _run(pm, tp64, random.Random(0), torch.float64, realistic=False)
    print(f"[float64 / 任意入力] 最大絶対誤差: {d64:.3e}")
    assert d64 < 1e-6, f"構造パリティ不一致(float64): {d64}"

    # (2) 実用の確認: float32 で現実的入力(mean近傍)なら十分一致。
    tp32 = TorchOptionPolicy.from_json(WEIGHTS).float().eval()
    d32 = _run(pm, tp32, random.Random(1), torch.float32, realistic=True)
    print(f"[float32 / 現実的入力] 最大絶対誤差: {d32:.3e}")
    assert d32 < 1e-2, f"実用パリティ不一致(float32): {d32}"

    print("M0 PASS: 構造一致(float64<1e-6) & 現実入力で float32 も一致(<1e-2)")


if __name__ == "__main__":
    main()
