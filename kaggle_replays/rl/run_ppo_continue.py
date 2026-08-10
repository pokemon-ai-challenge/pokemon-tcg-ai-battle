"""``run_ppo_pilot.py``のmilestone_2000から、同じPPO設定のまま追加1,000 valid
rollout games(累計3,000)だけ継続する短縮実験。

[2026-08-09 訂正] このファイル内の変数名・コメントの「champion」は、このスクリプトが
resumeする基準checkpointを指す内部的な命名(「milestone_2000」の別名)であり、
「チーム全体のchampion」「Kaggle上の過去最高モデル」という意味ではない
(Kaggle提出履歴監査の結果、milestone_2000のpublicScoreはsubmission_climb.tar.gz
(823.5)を上回っていないことを確認済み。詳細は
``runs/champions/milestone_2000/manifest.yaml``のCHANGELOG参照)。

変更するのは「PPO試合数」だけ。lr・clip・epochs・grad_clip・target_kl・reward・
opponent比率は既存pilotと完全に同じ設定を再利用する(``run_ppo_pilot``から関数を
そのままimportして重複させない)。新しいrunディレクトリに保存し、既存の
``runs/ppo_pilot_v1/`` ・ ``runs/champions/milestone_2000/`` は一切変更しない。

[preflightで発見した既知の制約] 「初期T1からのKL」の固定probe setは、元のpilot
(``run_ppo_pilot.py``)ではディスクに保存しておらず、プロセス終了とともに失われている。
cgエンジンの内部乱数はPythonのseedで制御できない(design.md §9.1.1)ため、同じseedで
再生成しても元と同一のprobe setにはならない。そのため、0〜2000区間で使ったprobe set
とは異なる、この継続run専用の新しいprobe setを使う(この継続run内では2000〜3000区間で
一貫して同じprobe setを使うので、この区間内のKLの推移は比較可能)。ここではこの反省を
踏まえ、probe setをディスクに保存する(次回以降の継続runで使い回せるように)。
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

_CHAMPION_CKPT = _ROOT / "kaggle_replays" / "rl" / "runs" / "champions" / "milestone_2000" / "checkpoint_2000.pt"
_INITIAL_CKPT = base._INITIAL_CKPT
_OUT_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_t1_continue_from_milestone_2000_same_config"
_CHAMPION_POOL_BASELINE = 0.803125  # milestone_2000自身のdiagnostics.jsonlより

MILESTONES = (2500, 3000)  # 累計valid games(継続分の中間・終了)
GAMES_PER_BATCH = base.GAMES_PER_BATCH
H2H_GAMES = base.H2H_GAMES
POOL_GAMES_PER_OPPONENT = base.POOL_GAMES_PER_OPPONENT
ROCKET_EXTRA_GAMES = base.ROCKET_EXTRA_GAMES


def _probe_set_path() -> Path:
    return _OUT_DIR / "probe_set.npz"


def _save_probe_set(probe_decisions: list[dict]) -> None:
    arrays = tr.concat_single_decision_arrays(probe_decisions)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_probe_set_path(), **arrays)


def _load_probe_set() -> list[dict]:
    data = np.load(_probe_set_path())
    arrays = {k: data[k] for k in data.files}
    board_off = np.cumsum(arrays["board_counts"]) - arrays["board_counts"]
    opt_off = np.cumsum(arrays["counts"]) - arrays["counts"]
    out = []
    for i in range(len(arrays["counts"])):
        bo, bc = int(board_off[i]), int(arrays["board_counts"][i])
        oo, oc = int(opt_off[i]), int(arrays["counts"][i])
        out.append({
            "board_counts": arrays["board_counts"][i:i + 1],
            "counts": arrays["counts"][i:i + 1],
            "board_token_numeric_features": arrays["board_token_numeric_features"][bo:bo + bc],
            "board_token_card_ids": arrays["board_token_card_ids"][bo:bo + bc],
            "board_token_zone_ids": arrays["board_token_zone_ids"][bo:bo + bc],
            "legacy_option_features": arrays["legacy_option_features"][oo:oo + oc],
            "option_card_ids": arrays["option_card_ids"][oo:oo + oc],
            "teacher_logits": arrays["teacher_logits"][oo:oo + oc],
            "option_target_token_indices": arrays["option_target_token_indices"][oo:oo + oc],
            "chosen": arrays["chosen"][i:i + 1],
            "legacy_global_features": arrays["legacy_global_features"][i:i + 1],
        })
    return out


def compute_kl_between(model_a, model_b, vocab, probe_decisions, device="cpu") -> float:
    """KL(model_a || model_b) を固定probe setで測定する(compute_kl_from_initialの汎用版)。"""
    return tp.compute_kl_from_initial(model_b, model_a, vocab, probe_decisions, device=device)


def top1_agreement(model_a, model_b, vocab, probe_decisions, device="cpu") -> float:
    """probe set上で、model_aとmodel_bのargmaxが一致する割合。"""
    if not probe_decisions:
        return float("nan")
    arrays = tr.concat_single_decision_arrays(probe_decisions)
    import token_batch as tbmod
    batch = tbmod.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        sa = model_a(**inputs)
        sb = model_b(**inputs)
    return float((sa.argmax(dim=1) == sb.argmax(dim=1)).float().mean().item())


def check_rollback_continue(diag: dict, prev_h2h_below_50: bool) -> tuple[bool, str, bool]:
    h2h_wr = diag["h2h_v40"]["t1_winrate"]
    pool_wr = diag["fixed_pool"]["overall_winrate"]
    h2h_below_50_now = h2h_wr < 0.5
    if prev_h2h_below_50 and h2h_below_50_now:
        return True, f"v40 head-to-headが2回連続で50%未満({h2h_wr:.3f})", h2h_below_50_now
    if pool_wr < _CHAMPION_POOL_BASELINE - 0.03:
        return True, (f"固定プール勝率がmilestone_2000({_CHAMPION_POOL_BASELINE:.3f})から"
                      f"3pt以上低下({pool_wr:.3f})"), h2h_below_50_now
    if diag["h2h_v40"]["illegal_actions_total"] > 0 or diag["fixed_pool"]["overall_valid"] == 0:
        return True, "illegal actionまたは評価が成立しなかった", h2h_below_50_now
    return False, "", h2h_below_50_now


def main() -> None:
    device = "cpu"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUT_DIR / "train_log.jsonl"
    diag_path = _OUT_DIR / "diagnostics.jsonl"

    ckpt_hash = base.sha256_file(_CHAMPION_CKPT)
    print(f"継続元checkpoint(champion): {_CHAMPION_CKPT}\nsha256: {ckpt_hash}", flush=True)

    model, vocab, profile_name, _ = la.load_t1_for_inference(_INITIAL_CKPT, device=device)
    initial_model = tp.freeze_copy(model)  # 初期T1(training seed=1 蒸留直後)、比較用固定

    # championをmodelへ実際にresume(重み+optimizer state)。model_configはinitialと同一のはず
    # (PPOはTransformer構成を変更しないため)。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    resume_payload = tp.load_ppo_checkpoint(_CHAMPION_CKPT, model, optimizer, device=device)
    start_cumulative = resume_payload["step"]
    assert start_cumulative == 2000, f"championのstepが2000ではない: {start_cumulative}"
    print(f"resume成功: 累計valid games={start_cumulative}, optimizer state entries="
         f"{len(optimizer.state)}", flush=True)

    champion_model = tp.freeze_copy(model)  # resume直後=championの重みそのもの。KL基準用に固定。

    from ptcg_ai.learning.policy_model import PolicyModel
    t1_state_pm = PolicyModel(str(base._TEACHER))

    opponent_specs_pm, deck_t1, v40_pm = base.build_opponent_pool(device)

    ppo_config = {"lr": 1e-5, "clip_eps": 0.1, "update_epochs": 2, "grad_clip": 0.5,
                 "target_kl": 0.015, "advantage_normalize": True, "reward": "win=+1,lose=-1",
                 "entropy_coef": 0.01, "advantage_method": "critic-free, Monte Carlo return, "
                 "batch-normalized (no GAE, no value function)",
                 "gamma": "N/A(discounting not implemented)",
                 "resume_from": str(_CHAMPION_CKPT), "resume_from_sha256": ckpt_hash,
                 "start_cumulative_valid_games": start_cumulative}
    with open(_OUT_DIR / "ppo_config.json", "w", encoding="utf-8") as f:
        json.dump(ppo_config, f, ensure_ascii=False, indent=1)
    print(f"PPO設定(championと同一、resume情報のみ追加): {json.dumps(ppo_config, ensure_ascii=False)}",
         flush=True)

    # probe set: このrun専用に新規収集して固定・保存する(championの元probe setは
    # ディスクに残っていないため引き継げない。docstring参照)。
    if _probe_set_path().exists():
        probe_set = _load_probe_set()
        print(f"probe set読み込み(既存): {len(probe_set)}決定点", flush=True)
    else:
        probe_trajs, _ = tr.collect_rollout(
            champion_model, vocab, t1_state_pm,
            [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
            deck_t1, 20, seed0=8500000, profile_name=profile_name, device=device)
        probe_set = tp.build_probe_set(probe_trajs, max_probe=256)
        _save_probe_set(probe_set)
        print(f"probe set新規作成・保存: {len(probe_set)}決定点 -> {_probe_set_path()}", flush=True)

    cumulative_valid = start_cumulative
    batch_idx = 0
    milestones_done = set()
    rollback_triggered = False
    prev_h2h_below_50 = False  # champion診断は既にmilestone_2000で記録済み、ここではNoneスタート

    target_cumulative = max(MILESTONES)
    while cumulative_valid < target_cumulative and not rollback_triggered:
        batch_idx += 1
        selfplay_specs = base.make_selfplay_specs(model, initial_model, vocab, t1_state_pm,
                                                  profile_name, device, deck_t1)
        all_specs = opponent_specs_pm + selfplay_specs

        t0 = time.time()
        trajs, counts = tr.collect_rollout(model, vocab, t1_state_pm, all_specs, deck_t1,
                                           GAMES_PER_BATCH, seed0=5000000 + batch_idx * 10000,
                                           profile_name=profile_name, device=device)
        collect_sec = time.time() - t0
        batch_valid = sum(c["valid"] for c in counts.values())
        batch_attempts = sum(c["attempts"] for c in counts.values())
        cumulative_valid += batch_valid

        decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
        if not decisions:
            print(f"[batch {batch_idx}] 決定点が0件、スキップ", flush=True)
            continue
        adv = tp.compute_advantages(returns)

        t0 = time.time()
        upd_stats = tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps,
                                  adv["advantages"], device=device,
                                  clip_eps=ppo_config["clip_eps"], entropy_coef=ppo_config["entropy_coef"],
                                  grad_clip=ppo_config["grad_clip"], update_epochs=ppo_config["update_epochs"],
                                  target_kl=ppo_config["target_kl"])
        update_sec = time.time() - t0

        kl_from_initial = tp.compute_kl_from_initial(model, initial_model, vocab, probe_set, device=device)
        kl_from_champion = compute_kl_between(champion_model, model, vocab, probe_set, device=device)
        top1_vs_champion = top1_agreement(champion_model, model, vocab, probe_set, device=device)

        n_infer = sum(len(d["decisions"]) for d in trajs)
        infer_time_per_decision = collect_sec / max(n_infer, 1)

        log_entry = {
            "batch": batch_idx, "cumulative_valid_games": cumulative_valid,
            "batch_attempts": batch_attempts, "batch_valid": batch_valid,
            "opponent_counts": counts, "n_decisions": len(decisions),
            "return_mean": adv["return_mean"], "return_std": adv["return_std"],
            "advantage_mean_before_norm": adv["advantage_mean_before_norm"],
            "advantage_std_before_norm": adv["advantage_std_before_norm"],
            "advantage_mean_after_norm": adv["advantage_mean_after_norm"],
            "advantage_std_after_norm": adv["advantage_std_after_norm"],
            "update_epochs_stats": upd_stats,
            "kl_from_initial_policy": kl_from_initial,
            "kl_from_milestone_2000": kl_from_champion,
            "top1_agreement_vs_milestone_2000": top1_vs_champion,
            "collect_seconds": collect_sec, "update_seconds": update_sec,
            "inference_time_per_decision_sec": infer_time_per_decision,
            "value_loss": None, "explained_variance": None,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        last_epoch = upd_stats[-1]
        print(f"[batch {batch_idx}] valid累計={cumulative_valid} decisions={len(decisions)} "
             f"policy_loss={last_epoch['policy_loss']:.4f} entropy={last_epoch['entropy']:.4f} "
             f"approx_kl={last_epoch['approx_kl']:.5f} clip_frac={last_epoch['clip_fraction']:.3f} "
             f"KL_init={kl_from_initial:.5f} KL_champ={kl_from_champion:.5f} "
             f"top1_vs_champ={top1_vs_champion:.4f}", flush=True)

        for m in MILESTONES:
            if cumulative_valid >= m and m not in milestones_done:
                milestones_done.add(m)
                ckpt_path = _OUT_DIR / f"checkpoint_{m}.pt"
                tp.save_ppo_checkpoint(model, optimizer, cumulative_valid, ckpt_path,
                                       extra={"milestone": m, "batch": batch_idx,
                                             "resumed_from": str(_CHAMPION_CKPT)})
                print(f"=== milestone {m}: 診断評価を実施 ===", flush=True)
                diag = base.run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device,
                                                opponent_specs_pm, deck_t1, f"checkpoint_{m}",
                                                eval_seed0=9500000)
                diag["cumulative_valid_games"] = cumulative_valid
                diag["batch"] = batch_idx
                diag["kl_from_initial_policy"] = kl_from_initial
                diag["kl_from_milestone_2000"] = kl_from_champion
                diag["top1_agreement_vs_milestone_2000"] = top1_vs_champion
                with open(diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag, ensure_ascii=False) + "\n")
                print(f"milestone{m}: h2h_v40={diag['h2h_v40']['t1_winrate']:.3f} "
                     f"pool={diag['fixed_pool']['overall_winrate']:.3f} "
                     f"rocket={diag['rocket_mewtwo_ex_extra']['t1_winrate']:.3f}", flush=True)

                should_rollback, reason, prev_h2h_below_50 = check_rollback_continue(diag, prev_h2h_below_50)
                if should_rollback:
                    rollback_triggered = True
                    print(f"!!! 中止条件に該当: {reason}", flush=True)
                    break

    print(f"完了(または中止)。累計valid rollout games={cumulative_valid}", flush=True)


if __name__ == "__main__":
    main()
