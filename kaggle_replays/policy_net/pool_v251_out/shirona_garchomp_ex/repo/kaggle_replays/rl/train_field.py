"""field-sampling RL: alakazam を meta-share サンプルの相手 mix に対して学習(overfit回避)。

単一相手RL(crustle)は overfit(狙ったマッチ+6ptも他が下がり field net ゼロ)。本スクリプトは毎ゲーム
相手を meta-share でサンプルし、対フィールド加重勝率を直接最適化する。eval は sampled-field の greedy 勝率
(=share加重)。best-ckpt をエクスポート。最終判定は eval_field.py(overlay込み)で別途行う。

train_v3 のヘルパー(Critic/GAE/PPO/padding)を再利用。収集のみ collect_field。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_field import parallel_collect_field  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from train_v3 import Critic, wilson_lo, build_padded, compute_gae, policy_logp_entropy, export_temp  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# round_robin/eval_field と同じ7アーキ(相手は各 imitation policy)。
FIELD = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="alakazam")
    ap.add_argument("--learner-weights", default="policy_weights.json")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="field")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    lw = Path(args.learner_weights)
    learner_w = lw if lw.is_absolute() else WDIR / lw.name
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    tmp = _HERE / f"_tmp_policy_{arch}_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"

    learner_deck = read_deck_csv_file(str(DECKDIR / arch / "01.csv"))
    opp_specs = [(str(WDIR / f"policy_weights_{a}.json"),
                  read_deck_csv_file(str(DECKDIR / a / "01.csv"))) for a, _ in FIELD]
    shares = [s for _, s in FIELD]
    print(f"device={device} learner={arch} vs FIELD({len(FIELD)}arch, share-sampled) workers={args.workers} -> {out.name}", flush=True)

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    def do_eval(seed, ngames):
        export_temp(policy, base_payload, tmp)
        _, w, v, _, _ = parallel_collect_field(str(tmp), opp_specs, shares, learner_deck,
                                               ngames, seed, temperature=0.01, workers=args.workers)
        return (w / v if v else float("nan")), w, v

    history = []
    wr0, w0, v0 = do_eval(900000, args.eval_games)
    print(f"[iter 0] baseline field-weighted greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})", flush=True)
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})
    best_wr, best_iter = wr0, 0
    best_state = copy.deepcopy(policy.state_dict())

    t0 = time.time()
    for it in range(1, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        trajs, wins, valid, errors, per_opp = parallel_collect_field(
            str(tmp), opp_specs, shares, learner_deck, args.games_per_iter,
            seed0=it * 100000, temperature=args.temperature, workers=args.workers)
        if not trajs:
            print(f"[iter {it}] no trajs", flush=True); continue
        batch = build_padded(trajs, device)
        with torch.no_grad():
            values = critic(batch["state_rows"])
        adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values, args.gamma, args.lam, device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = en = 0.0
        for _ in range(args.epochs):
            new_logp, ent = policy_logp_entropy(policy, batch)
            ratio = torch.exp(new_logp - batch["old_logp"])
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
            pol_loss = -torch.min(s1, s2).mean() - args.entropy * ent.mean()
            opt_p.zero_grad(); pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()
            v_pred = critic(batch["state_rows"])
            val_loss = ((v_pred - vtarget) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pl, vl, en = pol_loss.item(), val_loss.item(), ent.mean().item()

        train_wr = wins / valid if valid else float("nan")
        print(f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid} err{errors}) steps {batch['n']} "
              f"pol {pl:.4f} val {vl:.4f} ent {en:.3f} {time.time()-t0:.0f}s", flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"], "pol_loss": pl, "val_loss": vl, "entropy": en}
        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v = do_eval(900000, args.eval_games)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            star = ""
            if wr > best_wr:
                best_wr, best_iter = wr, it
                best_state = copy.deepcopy(policy.state_dict()); star = " *BEST*"
            print(f"    [eval] field-weighted {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) best={best_wr:.3f}@{best_iter}{star}", flush=True)
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    policy.load_state_dict(best_state)
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({"rl_finetuned": True, "rl_best_iter": best_iter,
                                           "rl_best_eval_winrate": best_wr, "rl_baseline": wr0})
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported BEST (iter {best_iter}, field-weighted {best_wr:.3f}, baseline {wr0:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
