"""critic-free PPOで方策を安全かつ測定可能な水準まで動かせるlrを探る短縮pilot
(item5)。親checkpointは両arm共通でmilestone_2000。optimizer stateもmilestone_2000
から復元し、``lr``だけ上書きする(momentum自体は再初期化しない)。

opponent比率・clip_epsilon・update_epochs・target_kl・entropy_coefficient・
reward・critic無し、はrun_ppo_continue.pyと完全に同じ。変更するのはlearning rateのみ。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import t1_live_agent as la  # noqa: E402
import t1_rollout as tr  # noqa: E402
import train_ppo_t1 as tp  # noqa: E402
import run_ppo_pilot as base  # noqa: E402
import run_ppo_continue as cont  # noqa: E402

_CHAMPION_CKPT = _ROOT / "kaggle_replays" / "rl" / "runs" / "champions" / "milestone_2000" / "checkpoint_2000.pt"
_INITIAL_CKPT = base._INITIAL_CKPT
_CHAMPION_POOL_BASELINE = 0.803125

GAMES_PER_BATCH = 250
MILESTONES = (250, 500)
CANDIDATE_KL_LO, CANDIDATE_KL_HI = 5e-4, 5e-3
CANDIDATE_CLIP_FRAC_MAX = 0.1


def run_arm(arm_name: str, lr: float, out_dir: Path, seed_base: int) -> None:
    device = "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    diag_path = out_dir / "diagnostics.jsonl"

    model, vocab, profile_name, _ = la.load_t1_for_inference(_INITIAL_CKPT, device=device)
    initial_model = tp.freeze_copy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 後でlr上書き(momentum再初期化しない)
    resume_payload = tp.load_ppo_checkpoint(_CHAMPION_CKPT, model, optimizer, device=device)
    for g in optimizer.param_groups:
        g["lr"] = lr
    print(f"[{arm_name}] resume成功: 累計valid games={resume_payload['step']}, "
         f"optimizer state entries={len(optimizer.state)}, lr上書き={lr}", flush=True)

    champion_model = tp.freeze_copy(model)

    from ptcg_ai.learning.policy_model import PolicyModel
    t1_state_pm = PolicyModel(str(base._TEACHER))
    opponent_specs_pm, deck_t1, v40_pm = base.build_opponent_pool(device)

    ppo_config = {"arm": arm_name, "lr": lr, "clip_eps": 0.1, "update_epochs": 2, "grad_clip": 0.5,
                 "target_kl": 0.015, "advantage_normalize": True, "reward": "win=+1,lose=-1",
                 "entropy_coef": 0.01, "advantage_method": "critic-free, Monte Carlo return, "
                 "batch-normalized (no GAE, no value function)",
                 "resume_from": str(_CHAMPION_CKPT)}
    with open(out_dir / "ppo_config.json", "w", encoding="utf-8") as f:
        json.dump(ppo_config, f, ensure_ascii=False, indent=1)
    print(f"[{arm_name}] 設定: {json.dumps(ppo_config, ensure_ascii=False)}", flush=True)

    # probe setは継続run(0->2000->3000)のものをそのまま再利用する(比較可能性のため、
    # 新規に作り直さない)。
    probe_set = cont._load_probe_set()
    print(f"[{arm_name}] probe set再利用: {len(probe_set)}決定点", flush=True)

    cumulative = 0
    batch_idx = 0
    stopped_reason = None

    while cumulative < max(MILESTONES) and stopped_reason is None:
        batch_idx += 1
        selfplay_specs = base.make_selfplay_specs(model, initial_model, vocab, t1_state_pm,
                                                  profile_name, device, deck_t1)
        all_specs = opponent_specs_pm + selfplay_specs

        t0 = time.time()
        trajs, counts = tr.collect_rollout(model, vocab, t1_state_pm, all_specs, deck_t1,
                                           GAMES_PER_BATCH, seed0=seed_base + batch_idx * 10000,
                                           profile_name=profile_name, device=device)
        collect_sec = time.time() - t0
        batch_valid = sum(c["valid"] for c in counts.values())
        cumulative += batch_valid

        decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
        if not decisions:
            print(f"[{arm_name} batch{batch_idx}] 決定点0件、スキップ", flush=True)
            continue
        adv = tp.compute_advantages(returns)

        # top1変更率(この更新の前後で、probe set上のargmaxが変わる決定点の割合)
        top1_before = _snapshot_top1(model, vocab, probe_set, device)

        t0 = time.time()
        upd_stats = tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps,
                                  adv["advantages"], device=device, clip_eps=ppo_config["clip_eps"],
                                  entropy_coef=ppo_config["entropy_coef"], grad_clip=ppo_config["grad_clip"],
                                  update_epochs=ppo_config["update_epochs"], target_kl=ppo_config["target_kl"])
        update_sec = time.time() - t0

        top1_after = _snapshot_top1(model, vocab, probe_set, device)
        top1_change_rate = float((top1_before != top1_after).mean())

        kl_from_champion = tp.compute_kl_from_initial(model, champion_model, vocab, probe_set, device=device)
        n_infer = sum(len(d["decisions"]) for d in trajs)

        last_epoch = upd_stats[-1]
        log_entry = {
            "arm": arm_name, "batch": batch_idx, "cumulative_valid_games": cumulative,
            "batch_valid": batch_valid, "n_decisions": len(decisions),
            "opponent_counts": counts,
            "return_mean": adv["return_mean"], "return_std": adv["return_std"],
            "advantage_mean_after_norm": adv["advantage_mean_after_norm"],
            "advantage_std_after_norm": adv["advantage_std_after_norm"],
            "update_epochs_stats": upd_stats,
            "kl_from_milestone_2000": kl_from_champion,
            "top1_change_rate_vs_previous_batch": top1_change_rate,
            "collect_seconds": collect_sec, "update_seconds": update_sec,
            "inference_time_per_decision_sec": collect_sec / max(n_infer, 1),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[{arm_name} batch{batch_idx}] cum={cumulative} policy_loss={last_epoch['policy_loss']:.5f} "
             f"entropy={last_epoch['entropy']:.4f} approx_kl={last_epoch['approx_kl']:.6f} "
             f"clip_frac={last_epoch['clip_fraction']:.4f} KL_champ={kl_from_champion:.6f} "
             f"top1_change_rate={top1_change_rate:.4f}", flush=True)

        # 停止条件チェック(毎バッチ)
        if last_epoch["entropy"] < 0.3:
            stopped_reason = f"entropy急落({last_epoch['entropy']:.4f})"
        elif last_epoch["clip_fraction"] > 0.3:
            stopped_reason = f"clip_fractionが高すぎる({last_epoch['clip_fraction']:.4f})"

        for m in MILESTONES:
            if cumulative >= m and stopped_reason is None:
                ckpt_path = out_dir / f"checkpoint_{m}.pt"
                if not ckpt_path.exists():
                    tp.save_ppo_checkpoint(model, optimizer, cumulative, ckpt_path,
                                           extra={"arm": arm_name, "lr": lr, "milestone": m})

    # 500試合終了時点で、従来固定プールの短縮診断(320試合)を実施(性能崩壊チェックのみ)
    if cumulative >= 500 and stopped_reason is None:
        print(f"[{arm_name}] 固定プール短縮診断(320試合)を実施", flush=True)
        diag = base.run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device,
                                        opponent_specs_pm, deck_t1, f"{arm_name}_500",
                                        eval_seed0=9950000)
        diag["cumulative_valid_games"] = cumulative
        with open(diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(diag, ensure_ascii=False) + "\n")
        pool_wr = diag["fixed_pool"]["overall_winrate"]
        print(f"[{arm_name}] pool={pool_wr:.4f} (champion={_CHAMPION_POOL_BASELINE:.4f}) "
             f"h2h_v40={diag['h2h_v40']['t1_winrate']:.4f}", flush=True)
        if pool_wr < _CHAMPION_POOL_BASELINE - 0.03:
            stopped_reason = f"固定プールがchampionから3pt以上低下({pool_wr:.4f})"

    print(f"[{arm_name}] 完了。累計={cumulative} stopped_reason={stopped_reason}", flush=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"arm": arm_name, "lr": lr, "cumulative_valid_games": cumulative,
                  "stopped_reason": stopped_reason}, f, ensure_ascii=False, indent=1)


def _snapshot_top1(model, vocab, probe_set, device) -> np.ndarray:
    arrays = tr.concat_single_decision_arrays(probe_set)
    import token_batch as tb
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)
    return scores.argmax(dim=1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["arm_A", "arm_B"])
    args = ap.parse_args()

    if args.arm == "arm_A":
        run_arm("arm_A", lr=3e-5, out_dir=_ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_lr_pilot" / "arm_A",
               seed_base=6000000)
    else:
        run_arm("arm_B", lr=1e-4, out_dir=_ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_lr_pilot" / "arm_B",
               seed_base=6500000)


if __name__ == "__main__":
    main()
