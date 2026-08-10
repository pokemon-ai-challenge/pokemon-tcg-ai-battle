"""T1(Transformer)のmixed-opponent PPO pilot(開発目的、正式ゲート合格後の本格学習ではない)。

固定プール正式ゲート: 不合格(このスクリプトは記録上、開発目的pilotとして実行する)。

初期方策: mixed-data蒸留 training seed=1 (best step=1750, validation KL=0.3739)。
critic-free PPO-Clip(Monte Carlo return-based advantage、GAE/value関数なし)。

累計2,000 valid rollout gamesまで、500/1,000/2,000のマイルストーンで
学習rolloutとは分離した診断評価(v40 head-to-head 300試合・固定8相手プール320試合・
rocket_mewtwo_ex追加150試合)を実施し、中止条件を毎バッチ確認する。
"""

from __future__ import annotations

import hashlib
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
import t1_eval as te  # noqa: E402

_INITIAL_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
                / "mixed_teacher_t1_seed1" / "best.pt")
_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_LEARNER_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"
_OUT_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_pilot_v1"

MILESTONES = (500, 1000, 2000)
GAMES_PER_BATCH = 250
BASELINE_POOL_WINRATE = 0.775  # seed1候補の固定プール勝率(正式評価、eval_seed1_final)
H2H_GAMES = 300
POOL_GAMES_PER_OPPONENT = 40
ROCKET_EXTRA_GAMES = 150


def sha256_file(path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_opponent_pool(device: str):
    import common as C
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))
    v40_pm = PolicyModel(str(_TEACHER))
    assert v40_pm.is_ready

    specs = []
    specs.append({"id": "v40", "weight": 30.0,
                 "opponent": tr.PMOpponent(v40_pm, "v40"), "deck": deck_t1})

    rocket_cfg = next(o for o in run_cfg["opponents"] if o["id"] == "rocket_mewtwo_ex")
    rocket_w = C.resolve_opponent_weights(_RUN_DIR, rocket_cfg.get("weights"))
    rocket_deck = read_deck_csv_file(str(C.resolve_deck(rocket_cfg.get("deck"))))
    rocket_pm = PolicyModel(str(rocket_w))
    specs.append({"id": "rocket_mewtwo_ex", "weight": 15.0,
                 "opponent": tr.PMOpponent(rocket_pm, "rocket_mewtwo_ex"), "deck": rocket_deck})

    others = [o for o in run_cfg["opponents"] if o["id"] not in ("rocket_mewtwo_ex",)]
    for o in others:
        w = C.resolve_opponent_weights(_RUN_DIR, o.get("weights"))
        d = read_deck_csv_file(str(C.resolve_deck(o.get("deck") or run_cfg["opponent_deck"])))
        pm = PolicyModel(str(w) if w else None)
        specs.append({"id": o["id"], "weight": 25.0 / len(others),
                     "opponent": tr.PMOpponent(pm, o["id"]), "deck": d})

    return specs, deck_t1, v40_pm


def make_selfplay_specs(current_model, initial_model, vocab, t1_state_pm, profile_name, device,
                        deck_t1):
    current_frozen = tp.freeze_copy(current_model)
    return [
        {"id": "self_current_frozen", "weight": 15.0,
         "opponent": tr.T1Opponent(current_frozen, vocab, t1_state_pm, profile_name, device,
                                   "self_current_frozen"),
         "deck": deck_t1},
        {"id": "self_initial_snapshot", "weight": 15.0,
         "opponent": tr.T1Opponent(initial_model, vocab, t1_state_pm, profile_name, device,
                                   "self_initial_snapshot"),
         "deck": deck_t1},
    ]


def run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, opponent_specs_pm,
                        deck_t1, label: str, eval_seed0: int) -> dict:
    """v40 head-to-head 300試合 + 固定8相手プール各40試合(320試合) +
    rocket_mewtwo_ex追加150試合(すでに固定プールに含まれる125試合と別枠で追加)。
    学習rolloutとは完全に分離した評価(argmax、t1_eval経由)。"""
    import common as C
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    result = {"label": label}

    v40_pm = PolicyModel(str(_TEACHER))
    h2h = te.run_head_to_head(model, vocab, v40_pm, v40_pm, deck_t1, deck_t1, H2H_GAMES,
                              seed0=eval_seed0, profile_name=profile_name, device=device)
    result["h2h_v40"] = {k: v for k, v in h2h.items() if k != "results"}

    per_opp = []
    tot_w = tot_v = 0
    offset = 100000
    for o in run_cfg["opponents"]:
        w = C.resolve_opponent_weights(_RUN_DIR, o.get("weights"))
        d = read_deck_csv_file(str(C.resolve_deck(o.get("deck") or run_cfg["opponent_deck"])))
        pm = PolicyModel(str(w) if w else None)
        r = te.run_head_to_head(model, vocab, t1_state_pm, pm, deck_t1, d, POOL_GAMES_PER_OPPONENT,
                                seed0=eval_seed0 + offset, profile_name=profile_name, device=device)
        offset += POOL_GAMES_PER_OPPONENT
        tot_w += r["t1_wins"]; tot_v += r["n_valid"]
        per_opp.append({"id": o["id"], "wins": r["t1_wins"], "valid": r["n_valid"],
                        "errors": r["n_errors"], "winrate": r["t1_winrate"]})
    result["fixed_pool"] = {"overall_wins": tot_w, "overall_valid": tot_v,
                            "overall_winrate": (tot_w / tot_v if tot_v else float("nan")),
                            "per_opponent": per_opp}

    rocket_cfg = next(o for o in run_cfg["opponents"] if o["id"] == "rocket_mewtwo_ex")
    rw = C.resolve_opponent_weights(_RUN_DIR, rocket_cfg.get("weights"))
    rd = read_deck_csv_file(str(C.resolve_deck(rocket_cfg.get("deck"))))
    rpm = PolicyModel(str(rw))
    rocket_r = te.run_head_to_head(model, vocab, t1_state_pm, rpm, deck_t1, rd, ROCKET_EXTRA_GAMES,
                                   seed0=eval_seed0 + 200000, profile_name=profile_name,
                                   device=device)
    result["rocket_mewtwo_ex_extra"] = {k: v for k, v in rocket_r.items() if k != "results"}
    return result


def check_rollback(diag: dict, prev_h2h_below_50: bool) -> tuple[bool, str, bool]:
    """中止条件の一部(この場で判定可能なもの)をチェックする。
    Returns: (should_rollback, reason, h2h_below_50_now)"""
    h2h_wr = diag["h2h_v40"]["t1_winrate"]
    pool_wr = diag["fixed_pool"]["overall_winrate"]
    h2h_below_50_now = h2h_wr < 0.5
    if prev_h2h_below_50 and h2h_below_50_now:
        return True, f"v40 head-to-headが2回連続で50%未満({h2h_wr:.3f})", h2h_below_50_now
    if pool_wr < BASELINE_POOL_WINRATE - 0.03:
        return True, f"固定プール勝率がbaseline{BASELINE_POOL_WINRATE:.3f}から3pt以上低下({pool_wr:.3f})", h2h_below_50_now
    if diag["h2h_v40"]["illegal_actions_total"] > 0 or diag["fixed_pool"]["overall_valid"] == 0:
        return True, "illegal actionまたは評価が成立しなかった", h2h_below_50_now
    return False, "", h2h_below_50_now


def main() -> None:
    device = "cpu"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUT_DIR / "train_log.jsonl"
    diag_path = _OUT_DIR / "diagnostics.jsonl"

    ckpt_hash = sha256_file(_INITIAL_CKPT)
    print(f"初期checkpoint: {_INITIAL_CKPT}\nsha256: {ckpt_hash}", flush=True)

    model, vocab, profile_name, _ = la.load_t1_for_inference(_INITIAL_CKPT, device=device)
    initial_model = tp.freeze_copy(model)  # 比較用に固定(以後一切変更しない)
    from ptcg_ai.learning.policy_model import PolicyModel
    t1_state_pm = PolicyModel(str(_TEACHER))

    opponent_specs_pm, deck_t1, v40_pm = build_opponent_pool(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    ppo_config = {"lr": 1e-5, "clip_eps": 0.1, "update_epochs": 2, "grad_clip": 0.5,
                 "target_kl": 0.015, "advantage_normalize": True, "reward": "win=+1,lose=-1",
                 "entropy_coef": 0.01, "advantage_method": "critic-free, Monte Carlo return, "
                 "batch-normalized (no GAE, no value function)",
                 "gamma": "N/A(discounting not implemented; terminal reward broadcast "
                 "undiscounted to all decisions in the trajectory)"}
    with open(_OUT_DIR / "ppo_config.json", "w", encoding="utf-8") as f:
        json.dump(ppo_config, f, ensure_ascii=False, indent=1)
    print(f"PPO設定: {json.dumps(ppo_config, ensure_ascii=False)}", flush=True)

    # checkpoint 0: PPO前baseline診断(既存の正式評価と同じ条件、比較用に同一評価seedを使う)
    eval_seed0 = 9000000
    print("=== checkpoint 0 (PPO前baseline) 診断評価を実施 ===", flush=True)
    diag0 = run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, opponent_specs_pm,
                               deck_t1, "checkpoint_0", eval_seed0)
    diag0["cumulative_valid_games"] = 0
    with open(diag_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(diag0, ensure_ascii=False) + "\n")
    print(f"checkpoint0: h2h_v40={diag0['h2h_v40']['t1_winrate']:.3f} "
         f"pool={diag0['fixed_pool']['overall_winrate']:.3f} "
         f"rocket={diag0['rocket_mewtwo_ex_extra']['t1_winrate']:.3f}", flush=True)

    # [D2.1後PPO pilotで発見・修正: probe set固定化バグ]
    # 初期方策からのKLは毎バッチ違う局面集合で測ると比較にならないため、ここで一度だけ
    # 固定probe set(自己対戦20試合、初期モデルで収集)を作り、以後使い回す。
    probe_trajs, _ = tr.collect_rollout(initial_model, vocab, t1_state_pm,
                                        [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
                                        deck_t1, 20, seed0=8000000, profile_name=profile_name,
                                        device=device)
    probe_set = tp.build_probe_set(probe_trajs, max_probe=256)
    print(f"probe set固定: {len(probe_set)}決定点(以後このセットでKLを測定)", flush=True)

    cumulative_valid = 0
    batch_idx = 0
    milestones_done = set()
    prev_h2h_below_50 = diag0["h2h_v40"]["t1_winrate"] < 0.5
    rollback_triggered = False

    while cumulative_valid < max(MILESTONES) and not rollback_triggered:
        batch_idx += 1
        selfplay_specs = make_selfplay_specs(model, initial_model, vocab, t1_state_pm,
                                             profile_name, device, deck_t1)
        all_specs = opponent_specs_pm + selfplay_specs

        t0 = time.time()
        trajs, counts = tr.collect_rollout(model, vocab, t1_state_pm, all_specs, deck_t1,
                                           GAMES_PER_BATCH, seed0=1000000 + batch_idx * 10000,
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
                                  adv["advantages"], device=device, **{
                                      k: v for k, v in ppo_config.items()
                                      if k in ("clip_eps", "entropy_coef", "grad_clip",
                                              "update_epochs", "target_kl")})
        update_sec = time.time() - t0

        kl_from_initial = tp.compute_kl_from_initial(model, initial_model, vocab, probe_set,
                                                      device=device)

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
            "collect_seconds": collect_sec, "update_seconds": update_sec,
            "inference_time_per_decision_sec": infer_time_per_decision,
            "value_loss": None, "explained_variance": None,  # critic-free: N/A
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        last_epoch = upd_stats[-1]
        print(f"[batch {batch_idx}] valid累計={cumulative_valid} decisions={len(decisions)} "
             f"policy_loss={last_epoch['policy_loss']:.4f} entropy={last_epoch['entropy']:.4f} "
             f"approx_kl={last_epoch['approx_kl']:.5f} clip_frac={last_epoch['clip_fraction']:.3f} "
             f"KL_from_initial={kl_from_initial:.4f}", flush=True)

        for m in MILESTONES:
            if cumulative_valid >= m and m not in milestones_done:
                milestones_done.add(m)
                ckpt_path = _OUT_DIR / f"checkpoint_{m}.pt"
                tp.save_ppo_checkpoint(model, optimizer, cumulative_valid, ckpt_path,
                                       extra={"milestone": m, "batch": batch_idx})
                print(f"=== milestone {m}: 診断評価を実施 ===", flush=True)
                diag = run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device,
                                           opponent_specs_pm, deck_t1, f"checkpoint_{m}", eval_seed0)
                diag["cumulative_valid_games"] = cumulative_valid
                diag["batch"] = batch_idx
                with open(diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag, ensure_ascii=False) + "\n")
                print(f"milestone{m}: h2h_v40={diag['h2h_v40']['t1_winrate']:.3f} "
                     f"pool={diag['fixed_pool']['overall_winrate']:.3f} "
                     f"rocket={diag['rocket_mewtwo_ex_extra']['t1_winrate']:.3f}", flush=True)

                should_rollback, reason, prev_h2h_below_50 = check_rollback(diag, prev_h2h_below_50)
                if should_rollback:
                    rollback_triggered = True
                    print(f"!!! 中止条件に該当: {reason}", flush=True)
                    break

    print(f"完了(または中止)。累計valid rollout games={cumulative_valid}", flush=True)


if __name__ == "__main__":
    main()
