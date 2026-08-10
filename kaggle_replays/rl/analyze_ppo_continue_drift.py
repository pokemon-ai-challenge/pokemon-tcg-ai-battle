"""item3: 完了済みPPO継続run(milestone_2000->2500->3000)の詳細解析。
固定probe set(継続run自身のもの)と、pool_v2_validationの実対局状態から集めた
別のprobe setの両方で、champion(2000)・2500・3000の出力分布を比較する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import t1_live_agent as la  # noqa: E402
import t1_rollout as tr  # noqa: E402
import token_batch as tb  # noqa: E402

_CHAMPION = _ROOT / "kaggle_replays" / "rl" / "runs" / "champions" / "milestone_2000" / "checkpoint_2000.pt"
_M2500 = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_t1_continue_from_milestone_2000_same_config" / "checkpoint_2500.pt"
_M3000 = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_t1_continue_from_milestone_2000_same_config" / "checkpoint_3000.pt"
_PROBE_NPZ = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_t1_continue_from_milestone_2000_same_config" / "probe_set.npz"
_STATE_PROBE_NPZ = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v2_validation_results" / "state_probe.npz"
_TRAIN_LOG = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_t1_continue_from_milestone_2000_same_config" / "train_log.jsonl"


def load_arrays_decisions(npz_path: Path) -> list[dict]:
    data = np.load(npz_path)
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


def batch_forward(model, vocab, decisions, device="cpu"):
    """decisionsをまとめて1回forward。戻り: (scores(B,max_opt), mask(B,max_opt), n_opts(list))。"""
    arrays = tr.concat_single_decision_arrays(decisions)
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)
    n_opts = [int(d["counts"][0]) for d in decisions]
    return scores, inputs["option_mask"], n_opts


def analyze(label: str, champion_model, other_model, vocab, decisions, device="cpu") -> dict:
    s_champ, mask, n_opts = batch_forward(champion_model, vocab, decisions, device)
    s_other, _, _ = batch_forward(other_model, vocab, decisions, device)

    p_champ = F.softmax(s_champ, dim=1)
    logp_champ = F.log_softmax(s_champ, dim=1)
    logp_other = F.log_softmax(s_other, dim=1)

    contrib = torch.where(mask, p_champ * (logp_champ - logp_other), torch.zeros_like(p_champ))
    kl_per_decision = contrib.sum(dim=1).numpy()  # KL(champion || other) per decision

    top1_champ = s_champ.argmax(dim=1)
    top1_other = s_other.argmax(dim=1)
    agree = (top1_champ == top1_other).numpy()

    # top1/top2 prob margin(championの分布)
    sorted_p_champ, _ = torch.sort(p_champ, dim=1, descending=True)
    top1_prob = sorted_p_champ[:, 0].numpy()
    top2_prob = sorted_p_champ[:, 1].numpy() if sorted_p_champ.shape[1] > 1 else np.zeros(len(decisions))
    margin = top1_prob - top2_prob

    # championが選ぶ行動にotherが割り当てる確率(選択行動確率の変化)
    p_other = F.softmax(s_other, dim=1)
    idx = top1_champ.unsqueeze(1)
    champ_action_prob_under_champ = p_champ.gather(1, idx).squeeze(1).numpy()
    champ_action_prob_under_other = p_other.gather(1, idx).squeeze(1).numpy()

    n_opts_arr = np.array(n_opts)
    buckets = {"2-3": (2, 3), "4-6": (4, 6), "7+": (7, 10_000)}
    bucket_agreement = {}
    for name, (lo, hi) in buckets.items():
        sel = (n_opts_arr >= lo) & (n_opts_arr <= hi)
        if sel.sum() > 0:
            bucket_agreement[name] = {"n": int(sel.sum()), "top1_agreement": float(agree[sel].mean())}

    return {
        "label": label, "n_decisions": len(decisions),
        "kl_mean": float(kl_per_decision.mean()), "kl_median": float(np.median(kl_per_decision)),
        "kl_p95": float(np.percentile(kl_per_decision, 95)), "kl_max": float(kl_per_decision.max()),
        "top1_agreement_overall": float(agree.mean()),
        "top1_agreement_by_legal_move_count": bucket_agreement,
        "top1_top2_margin_mean_champion": float(margin.mean()),
        "champion_chosen_action_prob_under_champion_mean": float(champ_action_prob_under_champ.mean()),
        "champion_chosen_action_prob_under_other_mean": float(champ_action_prob_under_other.mean()),
        "action_prob_shift_mean": float((champ_action_prob_under_other - champ_action_prob_under_champ).mean()),
    }


def main() -> None:
    champion_model, vocab, profile_name, _ = la.load_t1_for_inference(_CHAMPION, device="cpu")
    m2500_model, _, _, _ = la.load_t1_for_inference(_M2500, device="cpu")
    m3000_model, _, _, _ = la.load_t1_for_inference(_M3000, device="cpu")

    fixed_probe = load_arrays_decisions(_PROBE_NPZ)
    print(f"固定probe set: {len(fixed_probe)}決定点(継続run開始時にmilestone_2000の自己対戦20試合"
         f"から収集、seed=8500000)")

    results = {}
    for label, model in (("2500_vs_champion_fixed_probe", m2500_model),
                        ("3000_vs_champion_fixed_probe", m3000_model)):
        results[label] = analyze(label, champion_model, model, vocab, fixed_probe)

    if _STATE_PROBE_NPZ.exists():
        state_probe = load_arrays_decisions(_STATE_PROBE_NPZ)
        print(f"blind対局state probe: {len(state_probe)}決定点(pool_v2_validation6相手との対局、"
             f"championのargmaxで進行)")
        for label, model in (("2500_vs_champion_state_probe", m2500_model),
                            ("3000_vs_champion_state_probe", m3000_model)):
            results[label] = analyze(label, champion_model, model, vocab, state_probe)

    for label, r in results.items():
        print(f"\n=== {label} ===")
        print(f"  n_decisions={r['n_decisions']}")
        print(f"  KL(champion||other): mean={r['kl_mean']:.6f} median={r['kl_median']:.6f} "
             f"p95={r['kl_p95']:.6f} max={r['kl_max']:.6f}")
        print(f"  top1一致率(全体)={r['top1_agreement_overall']:.4f}")
        print(f"  top1一致率(合法手数別)={r['top1_agreement_by_legal_move_count']}")
        print(f"  top1-top2確率差(champion)平均={r['top1_top2_margin_mean_champion']:.4f}")
        print(f"  championが選ぶ行動の確率: champion下={r['champion_chosen_action_prob_under_champion_mean']:.4f} "
             f"other下={r['champion_chosen_action_prob_under_other_mean']:.4f} "
             f"(差={r['action_prob_shift_mean']:.5f})")

    # gradient norm / advantage分布(train_logから導出、追加runは行わない)
    print("\n=== gradient norm・advantage分布(train_log.jsonlから、追加run無し) ===")
    grad_clip = 0.5
    for line in open(_TRAIN_LOG, encoding="utf-8"):
        d = json.loads(line)
        e = d["update_epochs_stats"][-1]
        gn_before = e["grad_norm"]
        gn_after = min(gn_before, grad_clip)
        n_dec = d["n_decisions"]
        ret_mean, ret_std = d["return_mean"], d["return_std"]
        # reward=+1/-1、正規化前advantage=(return-mean)/std。win/lose決定点数を
        # return_meanから逆算(reward=+1/-1がtrajectory単位でそのまま全decisionへ伝播するため)。
        n_win = round(n_dec * (1 + ret_mean) / 2)
        n_lose = n_dec - n_win
        adv_win = (1 - ret_mean) / max(ret_std, 1e-6)
        adv_lose = (-1 - ret_mean) / max(ret_std, 1e-6)
        print(f"batch{d['batch']} cum={d['cumulative_valid_games']} "
             f"grad_norm(更新前)={gn_before:.4f} grad_norm(clip後)={gn_after:.4f} "
             f"advantage正規化前(mean={d['advantage_mean_before_norm']:.4f}, "
             f"std={d['advantage_std_before_norm']:.4f}) "
             f"advantage正規化後(mean={d['advantage_mean_after_norm']:.4f}, "
             f"std={d['advantage_std_after_norm']:.4f}) "
             f"win_decisions_approx={n_win}(adv={adv_win:.3f}) "
             f"lose_decisions_approx={n_lose}(adv={adv_lose:.3f})")

    out_path = _HERE / "runs" / "ppo_t1_continue_from_milestone_2000_same_config" / "drift_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
