"""オーロンゲT1のPPO(ルールベース教師蒸留checkpointが親)。

フーディン系(``runs/distill_v40/``・``runs/ppo_pilot_v1/``等)とは完全に別の
runディレクトリ(``runs/ppo_pilot_grimmsnarl/``)を使う。optimizer state・
正規化統計・checkpointもフーディンとは独立(蒸留時点でnormalization_mode=
student_train_statsとして別々に計算済み)。

lr=1e-4は、フーディンPPO pilotのlearning-rate pilot(arm_B)で
「500試合でKL 5e-4〜5e-3、clip_fraction<0.1、entropy急落なし」を満たした
既知の設定を出発点として採用する(lr=1e-5では方策がほとんど動かなかった、
という知見を踏まえた選択)。それ以外の設定(critic無し・reward=+1/-1・
clip_epsilon=0.1・target_kl=0.015)はフーディンPPOと同じ。

opponent構成はオーロンゲ自身の評価結果(教師との直接対戦43.7%、固定プールの
弱点はcrustle 19.2%・mega_lucario_ex 36.8%)を踏まえ、教師との直接対戦と
弱点相手への配分を重めにする。
"""

from __future__ import annotations

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

_INITIAL_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_grimmsnarl" / "train"
                / "rule_teacher_seed0" / "best.pt")
_V40_FOR_FEATURES = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_GRIMM_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "marnie_grimmsnarl_ex" / "01.csv"
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"
_OUT_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_pilot_grimmsnarl"

MILESTONES = (500, 1000, 2000)
GAMES_PER_BATCH = 250
BASELINE_H2H = 0.4367
BASELINE_POOL = 0.488
H2H_GAMES = 300
POOL_GAMES_PER_OPPONENT = 40


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_opponent_pool(device: str):
    import common as C
    from opponents.rule_agents import grimmsnarl as gm
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    deck_t1 = read_deck_csv_file(str(_GRIMM_DECK))

    specs = []
    # 教師(ルールベース)との直接対戦。30%(フーディンPPOのv40枠と同じ比重)。
    specs.append({"id": "teacher_rule", "weight": 30.0,
                 "opponent": tr.RuleOpponent(gm.GrimmsnarlStrategy(), deck_t1, "teacher_rule"),
                 "deck": deck_t1})

    # 弱点相手を重め(crustle 19.2%->15%、mega_lucario_ex 36.8%->10%)
    weak = {"crustle": 15.0, "mega_lucario_ex": 10.0}
    for opp_id, w in weak.items():
        opp_cfg = next(o for o in run_cfg["opponents"] if o["id"] == opp_id)
        weights = C.resolve_opponent_weights(_RUN_DIR, opp_cfg.get("weights"))
        deck_o = read_deck_csv_file(str(C.resolve_deck(opp_cfg.get("deck"))))
        pm = PolicyModel(str(weights))
        specs.append({"id": opp_id, "weight": w, "opponent": tr.PMOpponent(pm, opp_id), "deck": deck_o})

    # 残り5相手(alakazam/archaludon_ex/rocket_mewtwo_ex/shirona_garchomp_ex/dragapult_ex)で計15%
    others = [o for o in run_cfg["opponents"] if o["id"] not in ("crustle", "mega_lucario_ex",
                                                                  "marnie_grimmsnarl_ex")]
    for o in others:
        w = C.resolve_opponent_weights(_RUN_DIR, o.get("weights"))
        d = read_deck_csv_file(str(C.resolve_deck(o.get("deck") or run_cfg["opponent_deck"])))
        pm = PolicyModel(str(w) if w else None)
        specs.append({"id": o["id"], "weight": 15.0 / len(others),
                     "opponent": tr.PMOpponent(pm, o["id"]), "deck": d})

    return specs, deck_t1


def make_selfplay_specs(current_model, initial_model, vocab, t1_state_pm, profile_name, device, deck_t1):
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


def run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, deck_t1, label, eval_seed0):
    import common as C
    from opponents.rule_agents import grimmsnarl as gm
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file
    import run_grimmsnarl_t1_eval as ge

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    result = {"label": label}

    h2h = ge.run_series(model, vocab, t1_state_pm, {"type": "rule"}, deck_t1, deck_t1,
                        H2H_GAMES, seed0=eval_seed0, profile_name=profile_name)
    result["h2h_teacher"] = h2h

    per_opp = []
    tot_w = tot_v = 0
    offset = 100000
    for opp in run_cfg["opponents"]:
        w = C.resolve_opponent_weights(_RUN_DIR, opp.get("weights"))
        d = read_deck_csv_file(str(C.resolve_deck(opp.get("deck") or run_cfg["opponent_deck"])))
        pm = PolicyModel(str(w) if w else None)
        r = ge.run_series(model, vocab, t1_state_pm, {"type": "pm", "pm": pm}, deck_t1, d,
                          POOL_GAMES_PER_OPPONENT, seed0=eval_seed0 + offset, profile_name=profile_name)
        offset += POOL_GAMES_PER_OPPONENT
        tot_w += r["t1_wins"]; tot_v += r["n_valid"]
        per_opp.append({"id": opp["id"], **r})
    result["fixed_pool"] = {"overall_wins": tot_w, "overall_valid": tot_v,
                            "overall_winrate": tot_w / tot_v if tot_v else float("nan"),
                            "per_opponent": per_opp}
    return result


def main() -> None:
    device = "cpu"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUT_DIR / "train_log.jsonl"
    diag_path = _OUT_DIR / "diagnostics.jsonl"

    ckpt_hash = sha256_file(_INITIAL_CKPT)
    print(f"初期checkpoint: {_INITIAL_CKPT}\nsha256: {ckpt_hash}", flush=True)

    model, vocab, profile_name, _ = la.load_t1_for_inference(_INITIAL_CKPT, device=device)
    initial_model = tp.freeze_copy(model)
    from ptcg_ai.learning.policy_model import PolicyModel
    t1_state_pm = PolicyModel(str(_V40_FOR_FEATURES))

    opponent_specs, deck_t1 = build_opponent_pool(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ppo_config = {"lr": 1e-4, "clip_eps": 0.1, "update_epochs": 2, "grad_clip": 0.5,
                 "target_kl": 0.015, "advantage_normalize": True, "reward": "win=+1,lose=-1",
                 "entropy_coef": 0.01, "advantage_method": "critic-free, Monte Carlo return, "
                 "batch-normalized (no GAE, no value function)",
                 "gamma": "N/A(discounting not implemented)",
                 "baseline_h2h": BASELINE_H2H, "baseline_pool": BASELINE_POOL}
    with open(_OUT_DIR / "ppo_config.json", "w", encoding="utf-8") as f:
        json.dump(ppo_config, f, ensure_ascii=False, indent=1)
    print(f"PPO設定: {json.dumps(ppo_config, ensure_ascii=False)}", flush=True)

    probe_trajs, _ = tr.collect_rollout(initial_model, vocab, t1_state_pm,
                                        [{"id": "self", "weight": 1.0, "opponent": None, "deck": None}],
                                        deck_t1, 20, seed0=4000000, profile_name=profile_name,
                                        device=device)
    probe_set = tp.build_probe_set(probe_trajs, max_probe=256)
    print(f"probe set固定: {len(probe_set)}決定点", flush=True)

    print("=== checkpoint 0 (PPO前baseline) 診断評価を実施 ===", flush=True)
    diag0 = run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, deck_t1,
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
        selfplay_specs = make_selfplay_specs(model, initial_model, vocab, t1_state_pm, profile_name,
                                             device, deck_t1)
        all_specs = opponent_specs + selfplay_specs

        t0 = time.time()
        trajs, counts = tr.collect_rollout(model, vocab, t1_state_pm, all_specs, deck_t1,
                                           GAMES_PER_BATCH, seed0=3000000 + batch_idx * 10000,
                                           profile_name=profile_name, device=device)
        collect_sec = time.time() - t0
        batch_valid = sum(c["valid"] for c in counts.values())
        cumulative += batch_valid

        decisions, actions, old_logps, returns = tp.trajectories_to_flat_decisions(trajs)
        if not decisions:
            print(f"[batch {batch_idx}] 決定点0件、スキップ", flush=True)
            continue
        adv = tp.compute_advantages(returns)

        upd_stats = tp.ppo_update(model, optimizer, vocab, decisions, actions, old_logps,
                                  adv["advantages"], device=device, clip_eps=ppo_config["clip_eps"],
                                  entropy_coef=ppo_config["entropy_coef"], grad_clip=ppo_config["grad_clip"],
                                  update_epochs=ppo_config["update_epochs"], target_kl=ppo_config["target_kl"])
        kl_from_initial = tp.compute_kl_from_initial(model, initial_model, vocab, probe_set, device=device)
        n_infer = sum(len(d["decisions"]) for d in trajs)

        last_epoch = upd_stats[-1]
        log_entry = {
            "batch": batch_idx, "cumulative_valid_games": cumulative,
            "batch_valid": batch_valid, "n_decisions": len(decisions), "opponent_counts": counts,
            "return_mean": adv["return_mean"], "return_std": adv["return_std"],
            "advantage_mean_after_norm": adv["advantage_mean_after_norm"],
            "advantage_std_after_norm": adv["advantage_std_after_norm"],
            "update_epochs_stats": upd_stats, "kl_from_initial_policy": kl_from_initial,
            "collect_seconds": collect_sec, "inference_time_per_decision_sec": collect_sec / max(n_infer, 1),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[batch {batch_idx}] cum={cumulative} policy_loss={last_epoch['policy_loss']:.5f} "
             f"entropy={last_epoch['entropy']:.4f} approx_kl={last_epoch['approx_kl']:.6f} "
             f"clip_frac={last_epoch['clip_fraction']:.4f} KL_init={kl_from_initial:.5f}", flush=True)

        if last_epoch["entropy"] < 0.3:
            rollback_triggered = True
            print(f"!!! 中止条件: entropy急落({last_epoch['entropy']:.4f})", flush=True)

        for m in MILESTONES:
            if cumulative >= m and m not in milestones_done and not rollback_triggered:
                milestones_done.add(m)
                ckpt_path = _OUT_DIR / f"checkpoint_{m}.pt"
                tp.save_ppo_checkpoint(model, optimizer, cumulative, ckpt_path,
                                       extra={"milestone": m, "batch": batch_idx})
                print(f"=== milestone {m}: 診断評価を実施 ===", flush=True)
                diag = run_diagnostic_eval(model, vocab, t1_state_pm, profile_name, device, deck_t1,
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
