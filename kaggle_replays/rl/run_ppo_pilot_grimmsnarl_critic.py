"""オーロンゲT1のcritic+GAE付きPPO pilot。critic無し版
(``run_ppo_pilot_grimmsnarl.py``、championとして``runs/champions_grimmsnarl/
critic_free_milestone_2000/``に保存済み)とは別run・別checkpoint。

親checkpointは critic無し版と同じ``checkpoint_0``(蒸留直後、``runs/distill_grimmsnarl/
train/rule_teacher_seed0/best.pt``)——critic無しpilotのmilestone_2000ではない
(critic有無だけを切り分けるため)。

教師・対戦相手と配分・seedと評価スケジュール・lr=1e-4・PPOのclip設定は
critic無し版(``run_ppo_pilot_grimmsnarl.py``)から関数をそのままimportして
再利用し、変更しない。変更するのは報酬表現(trajectory全体コピー→終端のみ)・
advantage計算(バッチ正規化のみ→GAE)・損失(policy+entropyのみ→+value loss)だけ。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
import train_ppo_t1_critic as tpc  # noqa: E402
import run_ppo_pilot_grimmsnarl as base  # noqa: E402

_INITIAL_CKPT = base._INITIAL_CKPT  # critic無し版と同じcheckpoint_0(蒸留直後)
_OUT_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_pilot_grimmsnarl_critic"

MILESTONES = base.MILESTONES  # (500, 1000, 2000)、critic無し版と同じ
GAMES_PER_BATCH = base.GAMES_PER_BATCH
BASELINE_H2H = base.BASELINE_H2H
BASELINE_POOL = base.BASELINE_POOL
H2H_GAMES = base.H2H_GAMES
POOL_GAMES_PER_OPPONENT = base.POOL_GAMES_PER_OPPONENT

GAMMA = 1.0
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5
LR = 1e-4
CLIP_EPS = 0.1
UPDATE_EPOCHS = 2
TARGET_KL = 0.015
ENTROPY_COEF = 0.01
GRAD_CLIP = 0.5


def main() -> None:
    device = "cpu"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUT_DIR / "train_log.jsonl"
    diag_path = _OUT_DIR / "diagnostics.jsonl"

    ckpt_hash = base.sha256_file(_INITIAL_CKPT)
    print(f"初期checkpoint(critic無し版と同じcheckpoint_0): {_INITIAL_CKPT}\nsha256: {ckpt_hash}",
         flush=True)

    model, vocab, profile_name, _ = la.load_t1_for_inference(_INITIAL_CKPT, device=device)
    initial_model = tp.freeze_copy(model)
    from ptcg_ai.learning.policy_model import PolicyModel
    t1_state_pm = PolicyModel(str(base._V40_FOR_FEATURES))

    opponent_specs, deck_t1 = base.build_opponent_pool(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ppo_config = {"lr": LR, "clip_eps": CLIP_EPS, "update_epochs": UPDATE_EPOCHS, "grad_clip": GRAD_CLIP,
                 "target_kl": TARGET_KL, "entropy_coef": ENTROPY_COEF,
                 "advantage_method": "GAE (critic付き、gamma=1.0, lambda=0.95)",
                 "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "value_loss_coef": VALUE_LOSS_COEF,
                 "value_clip": True, "reward": "terminal-only: win=+1,lose=-1,non-terminal=0",
                 "baseline_h2h": BASELINE_H2H, "baseline_pool": BASELINE_POOL,
                 "note": "critic無し版(run_ppo_pilot_grimmsnarl.py)と同じcheckpoint_0・"
                         "教師・対戦相手配分・lr・clip設定。報酬表現とadvantage計算(GAE)・"
                         "value lossだけが違う。"}
    with open(_OUT_DIR / "ppo_config.json", "w", encoding="utf-8") as f:
        json.dump(ppo_config, f, ensure_ascii=False, indent=1)
    print(f"PPO設定: {json.dumps(ppo_config, ensure_ascii=False)}", flush=True)

    probe_trajs, _ = tr.collect_rollout(initial_model, vocab, t1_state_pm,
                                        [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
                                        deck_t1, 20, seed0=4000000, profile_name=profile_name,
                                        device=device)
    probe_set = tp.build_probe_set(probe_trajs, max_probe=256)
    print(f"probe set固定: {len(probe_set)}決定点(critic無し版と同じseed=4000000)", flush=True)

    print("=== checkpoint 0 (PPO前baseline) 診断評価を実施(critic無し版と同じ条件) ===", flush=True)
    diag0 = base.run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, deck_t1,
                                     "checkpoint_0", eval_seed0=8100000)
    diag0["cumulative_valid_games"] = 0
    with open(diag_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(diag0, ensure_ascii=False) + "\n")
    print(f"checkpoint0: h2h={diag0['h2h_teacher']['t1_winrate']:.3f} "
         f"pool={diag0['fixed_pool']['overall_winrate']:.3f}", flush=True)

    cumulative = 0
    batch_idx = 0
    milestones_done = set()
    rollback_triggered = False
    prev_h2h_below_35 = diag0["h2h_teacher"]["t1_winrate"] < 0.35

    while cumulative < max(MILESTONES) and not rollback_triggered:
        batch_idx += 1
        selfplay_specs = base.make_selfplay_specs(model, initial_model, vocab, t1_state_pm, profile_name,
                                                   device, deck_t1)
        all_specs = opponent_specs + selfplay_specs

        t0 = time.time()
        trajs, counts = tr.collect_rollout(model, vocab, t1_state_pm, all_specs, deck_t1,
                                           GAMES_PER_BATCH, seed0=3000000 + batch_idx * 10000,
                                           profile_name=profile_name, device=device)
        collect_sec = time.time() - t0
        batch_valid = sum(c["valid"] for c in counts.values())
        cumulative += batch_valid

        cbatch = tpc.build_critic_batch(trajs, model, vocab, gamma=GAMMA, lam=GAE_LAMBDA, device=device)
        if not cbatch["decisions"]:
            print(f"[batch {batch_idx}] 決定点0件、スキップ", flush=True)
            continue

        t0 = time.time()
        upd_stats = tpc.ppo_update_with_critic(model, optimizer, vocab, cbatch, device=device,
                                               clip_eps=CLIP_EPS, entropy_coef=ENTROPY_COEF,
                                               grad_clip=GRAD_CLIP, value_loss_coef=VALUE_LOSS_COEF,
                                               update_epochs=UPDATE_EPOCHS, target_kl=TARGET_KL)
        update_sec = time.time() - t0
        kl_from_initial = tp.compute_kl_from_initial(model, initial_model, vocab, probe_set, device=device)
        n_infer = sum(len(d["decisions"]) for d in trajs)

        last_epoch = upd_stats[-1]
        log_entry = {
            "batch": batch_idx, "cumulative_valid_games": cumulative, "batch_valid": batch_valid,
            "n_decisions": len(cbatch["decisions"]), "opponent_counts": counts,
            "advantages_raw_mean": cbatch["advantages_raw_mean"], "advantages_raw_std": cbatch["advantages_raw_std"],
            "returns_mean": cbatch["returns_mean"], "returns_std": cbatch["returns_std"],
            "update_epochs_stats": upd_stats, "kl_from_initial_policy": kl_from_initial,
            "collect_seconds": collect_sec, "update_seconds": update_sec,
            "inference_time_per_decision_sec": collect_sec / max(n_infer, 1),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[batch {batch_idx}] cum={cumulative} policy_loss={last_epoch['policy_loss']:.5f} "
             f"value_loss={last_epoch['value_loss']:.5f} entropy={last_epoch['entropy']:.4f} "
             f"approx_kl={last_epoch['approx_kl']:.6f} clip_frac={last_epoch['clip_fraction']:.4f} "
             f"ev={last_epoch['explained_variance']:.4f} v_mean={last_epoch['value_pred_mean']:.4f} "
             f"v_std={last_epoch['value_pred_std']:.4f} KL_init={kl_from_initial:.5f}", flush=True)

        if last_epoch["entropy"] < 0.3:
            rollback_triggered = True
            print(f"!!! 中止条件: entropy急落({last_epoch['entropy']:.4f})", flush=True)

        for m in MILESTONES:
            if cumulative >= m and m not in milestones_done and not rollback_triggered:
                milestones_done.add(m)
                ckpt_path = _OUT_DIR / f"checkpoint_{m}.pt"
                tp.save_ppo_checkpoint(model, optimizer, cumulative, ckpt_path,
                                       extra={"milestone": m, "batch": batch_idx, "critic": True})
                print(f"=== milestone {m}: 診断評価を実施 ===", flush=True)
                diag = base.run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, deck_t1,
                                                f"checkpoint_{m}", eval_seed0=8100000)
                diag["cumulative_valid_games"] = cumulative
                with open(diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag, ensure_ascii=False) + "\n")
                h2h_wr = diag["h2h_teacher"]["t1_winrate"]
                pool_wr = diag["fixed_pool"]["overall_winrate"]
                print(f"milestone{m}: h2h={h2h_wr:.3f} pool={pool_wr:.3f}", flush=True)

                h2h_below_35_now = h2h_wr < 0.35
                if prev_h2h_below_35 and h2h_below_35_now:
                    rollback_triggered = True
                    print(f"!!! 中止条件: h2hが2回連続で35%未満({h2h_wr:.3f})", flush=True)
                if pool_wr < BASELINE_POOL - 0.05:
                    rollback_triggered = True
                    print(f"!!! 中止条件: 固定プールがbaselineから5pt以上低下({pool_wr:.3f})", flush=True)
                prev_h2h_below_35 = h2h_below_35_now

    print(f"完了(または中止)。累計valid rollout games={cumulative}", flush=True)


if __name__ == "__main__":
    main()
