"""defect#6(KL(π‖π_BC) 正則化)の検証。

2つを検証する:
検証1: kl_per_step_to_bc が本当に「渡された bc_policy 引数」を参照しており、こっそり
       第1引数(現在方策)自身と比較する no-op になっていないこと。
       同一重みの2モデル(KL=0)と、意図的に重みをずらした2モデル(KL>0)を両方通し、
       「第2引数を無視して第1引数を複製して比較している」バグなら後者も0になってしまう
       ことを利用して検出する。
検証2: ppo_update_step の kl_beta を大きくするほど、同じ更新列を経た後の
       KL(π‖π_BC) が小さくなる(方策がBCへ引き戻される)ことを、
       BCから逸れる方向に固定したadvantageで複数ステップ更新して確認する。
"""

from __future__ import annotations

import copy
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
from train_v3 import kl_per_step_to_bc, ppo_update_step  # noqa: E402

WEIGHTS = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
DEVICE = "cpu"
N_DECISIONS = 24
N_OPTIONS = 6


def make_synthetic_batch(policy, seed):
    """実ゲームを回さず、policy の次元に合った形の合成バッチを作る(構造だけ本物と同じ)。

    test_parity.py の realistic モードと同じ流儀: 各特徴を standardization の
    mean 近傍(mean + N(0,|std|))で生成する。素の N(0,1) をそのまま流すと、std がとても小さい
    (ほぼ定数の)特徴列を割ることで極端に大きいスコアになり、softmax が退化して
    (1つの選択肢に確率質量が張り付いて)KLが常に飽和してしまう(実際に踏んだ)。
    """
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
    return {"state_rows": state_rows, "option_pad": option_pad, "card_pad": card_pad,
            "mask": mask, "n": N_DECISIONS, "max_n": N_OPTIONS}


def main():
    base_payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 検証1: bc_policy 引数が本当に読まれている(no-op でない)
    # ------------------------------------------------------------------
    print("--- 検証1: KL(π‖π_BC) が本当に2つ目の引数を見ている ---")
    policy_a = TorchOptionPolicy.from_json(WEIGHTS).float().to(DEVICE)
    policy_a_clone = TorchOptionPolicy.from_json(WEIGHTS).float().to(DEVICE)  # 重みは同一・別インスタンス

    policy_b = TorchOptionPolicy.from_json(WEIGHTS).float().to(DEVICE)
    with torch.no_grad():  # 重みを意図的にずらす(「別のBC」または「更新が進んだRL方策」を模す)
        for p in policy_b.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    batch = make_synthetic_batch(policy_a, seed=1)
    batch["chosen"] = torch.zeros(N_DECISIONS, dtype=torch.long)

    kl_same = kl_per_step_to_bc(policy_a, policy_a_clone, batch).mean().item()
    kl_diff = kl_per_step_to_bc(policy_b, policy_a, batch).mean().item()
    print(f"  KL(policy_a || policy_a_clone, 同一重み) = {kl_same:.6f}  (0に近いはず)")
    print(f"  KL(policy_b || policy_a, 重みをずらした)   = {kl_diff:.6f}  (0より十分大きいはず)")
    assert kl_same < 1e-4, f"同一重みなのにKLが0近くない: {kl_same}"
    assert kl_diff > 0.05, (
        f"重みが違うのにKLがほぼ0({kl_diff}): bc_policy引数を無視して自分自身と比較している"
        "(no-op)疑いがある"
    )
    print("検証1 PASS: 同一重みでKL≈0、重みが違うとKL>0 -> bc_policy引数は正しく参照されている")

    # ------------------------------------------------------------------
    # 検証2: beta を大きくするほど、同じ更新列の後のKLが小さい
    # ------------------------------------------------------------------
    print("\n--- 検証2: kl_beta を大きくするほど、BCから逸れにくくなる ---")
    bc_policy = TorchOptionPolicy.from_json(WEIGHTS).float().to(DEVICE)
    bc_policy.requires_grad_(False)
    bc_policy.eval()

    batch2 = make_synthetic_batch(bc_policy, seed=7)
    with torch.no_grad():
        # BCが最も「好まない」選択肢(スコア最小)を意図的な chosen にする。
        # advantage を正にしてそこへ確率質量を押し付ける = BC分布から逸れる方向の更新。
        n, max_n = batch2["n"], batch2["max_n"]
        sd = batch2["state_rows"].shape[1]; od = batch2["option_pad"].shape[2]
        sf = batch2["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
        of = batch2["option_pad"].reshape(n * max_n, od)
        cf = batch2["card_pad"].reshape(n * max_n)
        bc_scores = bc_policy.option_scores_flat(sf, of, cf).reshape(n, max_n)
        batch2["chosen"] = bc_scores.argmin(dim=1)

    N_STEPS = 15
    results = {}
    # 0.01〜0.05 が plan doc の sweep 範囲。1.0 も加え、beta を強くしたときに本当にBCへ
    # 張り付く(KLがほぼ0になる)ところまで確認する(効果が「measurable」であることの念押し)。
    for beta in (0.0, 0.02, 0.1, 1.0):
        torch.manual_seed(0)
        policy = TorchOptionPolicy.from_json(WEIGHTS).float().to(DEVICE)
        opt_p = torch.optim.Adam(policy.parameters(), lr=3e-3)
        # critic はこのテストでは使わないダミー(update_policy=True でも opt_v.step は呼ばれるため必要)。
        from train_v3 import Critic
        std = base_payload["standardization"]
        critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(DEVICE)
        opt_v = torch.optim.Adam(critic.parameters(), lr=1e-3)

        batch = dict(batch2)  # shallow copy、old_logp は毎ステップ更新
        for step in range(N_STEPS):
            with torch.no_grad():
                n, max_n = batch["n"], batch["max_n"]
                sd = batch["state_rows"].shape[1]; od = batch["option_pad"].shape[2]
                sf = batch["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
                of = batch["option_pad"].reshape(n * max_n, od)
                cf = batch["card_pad"].reshape(n * max_n)
                scores = policy.option_scores_flat(sf, of, cf).reshape(n, max_n)
                logp = torch.log_softmax(scores, dim=1)
                batch["old_logp"] = logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
            adv = torch.ones(N_DECISIONS)  # BCが嫌う選択肢の確率を一貫して押し上げる向き
            vtarget = torch.zeros(N_DECISIONS)
            ppo_update_step(policy, critic, opt_p, opt_v, batch, adv, vtarget,
                            clip=0.2, entropy_coef=0.0, kl_beta=beta, bc_policy=bc_policy,
                            update_policy=True)
        final_kl = kl_per_step_to_bc(policy, bc_policy, batch2).mean().item()
        results[beta] = final_kl
        print(f"  beta={beta:.2f}: {N_STEPS}ステップ更新後の KL(π‖π_BC) = {final_kl:.4f}")

    assert results[1.0] < results[0.1] < results[0.02] < results[0.0], (
        f"betaを大きくしてもKLが単調に小さくならない: {results}"
    )
    print(f"検証2 PASS: beta単調増加 -> KL単調減少 ({results})")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
