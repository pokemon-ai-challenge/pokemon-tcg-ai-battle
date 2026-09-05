"""defect#4(Critic warmup: 方策固定で数イテレーション、critic のみ更新)の検証。

train_pool.py の実際の warmup ループは cg エンジンでの試合収集を伴うため高コスト。
ここでは共有プリミティブ ``train_v3.ppo_update_step(..., update_policy=False)`` を
合成データ(test_kl_regularization.py と同じ「standardization近傍のランダム特徴」方式)で
直接呼び、train_pool.py の warmup ループが内部で使っているのと同じ関数について:

検証1: update_policy=False を N ステップ回しても policy のパラメータが
       1ビットも変化しない(normの差が厳密に0)。
検証2: 同じ間、critic の value loss が単調に(ノイズはあっても全体として)減少する
       (target が一定なら学習が機能していることの最低限の確認)。
検証3: 対照として update_policy=True にすると policy パラメータが実際に変化することを確認し、
       検証1が「そもそも比較に使ったコードが変化を検出できる」ことを保証する
       (update_policy=False で変化なし、というだけでは「比較コード自体が壊れていて
       いつも0を返す」可能性を排除できないため)。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_policy import TorchOptionPolicy  # noqa: E402
from train_v3 import Critic, ppo_update_step  # noqa: E402

WEIGHTS = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
N_DECISIONS = 32
N_OPTIONS = 6


def make_batch(policy, seed):
    rng = random.Random(seed)
    sd, od = policy.state_dim, policy.option_dim
    smean = policy.state_mean.tolist(); sstd = policy.state_std.tolist()
    omean = policy.option_mean.tolist(); ostd = policy.option_std.tolist()
    state_rows = torch.tensor(
        [[smean[j] + rng.gauss(0, 1) * abs(sstd[j]) for j in range(sd)] for _ in range(N_DECISIONS)],
        dtype=torch.float32)
    option_pad = torch.tensor(
        [[[omean[j] + rng.gauss(0, 1) * abs(ostd[j]) for j in range(od)] for _ in range(N_OPTIONS)]
          for _ in range(N_DECISIONS)],
        dtype=torch.float32)
    card_pad = torch.randint(0, policy.card_id_max + 1, (N_DECISIONS, N_OPTIONS))
    mask = torch.ones(N_DECISIONS, N_OPTIONS)
    chosen = torch.randint(0, N_OPTIONS, (N_DECISIONS,))
    with torch.no_grad():
        n, max_n = N_DECISIONS, N_OPTIONS
        sf = state_rows.unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
        of = option_pad.reshape(n * max_n, od)
        cf = card_pad.reshape(n * max_n)
        scores = policy.option_scores_flat(sf, of, cf).reshape(n, max_n)
        logp = torch.log_softmax(scores, dim=1)
        old_logp = logp.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return {"state_rows": state_rows, "option_pad": option_pad, "card_pad": card_pad,
            "mask": mask, "chosen": chosen, "old_logp": old_logp,
            "n": N_DECISIONS, "max_n": N_OPTIONS}


def param_norm(module):
    return float(sum(p.detach().float().pow(2).sum() for p in module.parameters()) ** 0.5)


def main():
    base_payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    std = base_payload["standardization"]

    # ------------------------------------------------------------------
    # 検証1+2: update_policy=False で policy 不変・critic loss は減少傾向
    # ------------------------------------------------------------------
    print("--- 検証1+2: critic warmup(update_policy=False) ---")
    torch.manual_seed(0)
    policy = TorchOptionPolicy.from_json(WEIGHTS).float()
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"])
    opt_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt_v = torch.optim.Adam(critic.parameters(), lr=1e-3)  # train_pool.py の --lr-value 既定と同じ

    norm_before = param_norm(policy)
    val_losses = []
    N_WARMUP = 5
    # 実際の train_pool.py の warmup は毎イテレーション新しい試合を収集する(state は変わる)が、
    # 「state -> vtarget の関係」自体は同じ方策・同じ相手プールなので緩やかにしか変わらない。
    # ここでは固定の合成バッチ1つに対して繰り返し epoch を回し(=1イテレーション内のPPO epochsを
    # 複数回のwarmup反復に見立てる)、state->target の写像を stationary にして「学習できる課題」
    # にする(毎回違うランダムバッチだと写像そのものが安定しないので critic が収束しない=
    # このテストの前提が崩れる。実プールでは同じ方策からの収集なので分布は安定している)。
    batch = make_batch(policy, seed=100)
    vtarget = torch.linspace(-1.0, 1.0, N_DECISIONS)
    adv = torch.zeros(N_DECISIONS)  # update_policy=False なので使われないが形だけ渡す
    for it in range(N_WARMUP):
        for _ in range(4):  # train_pool.py の --epochs 既定4に合わせる
            _, vl, _, _ = ppo_update_step(
                policy, critic, opt_p, opt_v, batch, adv, vtarget,
                clip=0.2, entropy_coef=0.005, update_policy=False,
            )
        val_losses.append(vl)
        print(f"  warmup_iter {it}: val_loss={vl:.4f}")

    norm_after = param_norm(policy)
    delta = abs(norm_after - norm_before)
    print(f"  policy param norm: before={norm_before:.6f} after={norm_after:.6f} delta={delta:.3e}")
    assert delta == 0.0, f"update_policy=False なのに policy パラメータが変化した(delta={delta})"
    print("検証1 PASS: policy パラメータは厳密に不変(delta==0)")

    assert val_losses[-1] < val_losses[0], (
        f"critic の val_loss が減っていない: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    )
    print(f"検証2 PASS: val_loss が減少 ({val_losses[0]:.4f} -> {val_losses[-1]:.4f})")

    # ------------------------------------------------------------------
    # 検証3: 対照実験。update_policy=True なら実際に policy が動くことの確認
    # (検証1の「不変」が、比較コードのバグで常に0を返しているだけではないことの保証)
    # ------------------------------------------------------------------
    print("\n--- 検証3: 対照実験(update_policy=True では policy が実際に動く) ---")
    torch.manual_seed(0)
    policy2 = TorchOptionPolicy.from_json(WEIGHTS).float()
    critic2 = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"])
    opt_p2 = torch.optim.Adam(policy2.parameters(), lr=3e-4)
    opt_v2 = torch.optim.Adam(critic2.parameters(), lr=1e-2)
    norm_before2 = param_norm(policy2)
    batch2 = make_batch(policy2, seed=100)
    vtarget2 = torch.linspace(-1.0, 1.0, N_DECISIONS)
    adv2 = torch.ones(N_DECISIONS)
    for _ in range(4):
        ppo_update_step(policy2, critic2, opt_p2, opt_v2, batch2, adv2, vtarget2,
                        clip=0.2, entropy_coef=0.005, update_policy=True)
    norm_after2 = param_norm(policy2)
    delta2 = abs(norm_after2 - norm_before2)
    print(f"  policy param norm: before={norm_before2:.6f} after={norm_after2:.6f} delta={delta2:.3e}")
    assert delta2 > 1e-6, "update_policy=True でも policy が動いていない(比較コード自体が壊れている疑い)"
    print(f"検証3 PASS: update_policy=True では実際に policy が動く(delta={delta2:.3e}) "
          "-> 検証1の「不変」は本物")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
