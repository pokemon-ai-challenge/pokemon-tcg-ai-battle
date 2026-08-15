"""RL 対戦相手強化 v3: 並列収集(collect_parallel)+ GAE + best-checkpoint。

v2 との差: 収集をワーカー並列化(torch非依存ワーカー)。CPU律速の収集が ~workers 倍速くなるので
games/iter を大きく(既定512)して勾配分散を下げつつ、全体を高速化。
方策更新(PPO)は main の torch。各iter: 現policyをtemp JSONへ -> 並列収集 -> PPO更新。

すべて `kaggle_replays/rl/` 内・torch は main のみ・production 無変更。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_parallel import parallel_collect  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"


class Critic(nn.Module):
    def __init__(self, state_dim, mean, std, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        safe = torch.where(self.std != 0, self.std, torch.ones_like(self.std))
        z = torch.where(self.std != 0, (x - self.mean) / safe, torch.zeros_like(x))
        return self.net(z).squeeze(-1)


def wilson_lo(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (c - m) / d)


def build_padded(trajs, device):
    steps = [s for tr in trajs for s in tr["steps"]]
    n = len(steps)
    max_n = max(len(s["option_feats"]) for s in steps)
    sd = len(steps[0]["state_feat"])
    od = len(steps[0]["option_feats"][0])
    state_rows = torch.zeros(n, sd)
    option_pad = torch.zeros(n, max_n, od)
    card_pad = torch.zeros(n, max_n, dtype=torch.long)
    mask = torch.zeros(n, max_n)
    chosen = torch.zeros(n, dtype=torch.long)
    old_logp = torch.zeros(n)
    for i, s in enumerate(steps):
        state_rows[i] = torch.tensor(s["state_feat"])
        k = len(s["option_feats"])
        option_pad[i, :k] = torch.tensor(s["option_feats"])
        card_pad[i, :k] = torch.tensor(s["card_ids"])
        mask[i, :k] = 1.0
        chosen[i] = s["chosen_idx"]
        old_logp[i] = s["logprob"]
    return {"state_rows": state_rows.to(device), "option_pad": option_pad.to(device),
            "card_pad": card_pad.to(device), "mask": mask.to(device),
            "chosen": chosen.to(device), "old_logp": old_logp.to(device),
            "n": n, "max_n": max_n,
            "lengths": [len(tr["steps"]) for tr in trajs],
            "rewards": [tr["reward"] for tr in trajs]}


def compute_gae(lengths, rewards, values, gamma, lam, device):
    adv = torch.zeros(len(values)); vt = torch.zeros(len(values))
    v = values.detach().cpu(); off = 0
    for L, R in zip(lengths, rewards):
        last = 0.0
        for t in reversed(range(L)):
            idx = off + t
            r_t = R if t == L - 1 else 0.0
            v_next = 0.0 if t == L - 1 else float(v[idx + 1])
            delta = r_t + gamma * v_next - float(v[idx])
            last = delta + gamma * lam * last
            adv[idx] = last; vt[idx] = last + float(v[idx])
        off += L
    return adv.to(device), vt.to(device)


def policy_logp_entropy(policy, batch):
    n, max_n = batch["n"], batch["max_n"]
    sd = batch["state_rows"].shape[1]; od = batch["option_pad"].shape[2]
    sf = batch["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
    of = batch["option_pad"].reshape(n * max_n, od)
    cf = batch["card_pad"].reshape(n * max_n)
    scores = policy.option_scores_flat(sf, of, cf).reshape(n, max_n)
    scores = torch.where(batch["mask"] > 0, scores, torch.full_like(scores, torch.finfo(scores.dtype).min))
    logp = torch.log_softmax(scores, dim=1)
    chosen_logp = logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
    ent = -(logp.exp() * logp.masked_fill(batch["mask"] == 0, 0.0)).sum(dim=1)
    return chosen_logp, ent


def export_temp(policy, base_payload, path):
    payload = policy.to_json_payload(base_payload)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="dragapult_ex")
    ap.add_argument("--learner-weights", default=None,
                    help="learner の初期重み(既定 policy_weights_<arch>.json)。alakazam は production の "
                         "policy_weights.json を指定する(policy_weights_alakazam.json は無い)。")
    ap.add_argument("--opponent-arch", default="alakazam")
    ap.add_argument("--opponent-weights", default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--eval-games", type=int, default=200)
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
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    learner_w = Path(args.learner_weights) if args.learner_weights else WDIR / f"policy_weights_{arch}.json"
    if not learner_w.is_absolute():
        learner_w = WDIR / learner_w.name
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    tmp = _HERE / f"_tmp_policy_{arch}_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"
    print(f"device={device} learner={arch} opp={args.opponent_arch} workers={args.workers} -> {out.name}", flush=True)

    deck_l = read_deck_csv_file(str(DECKDIR / arch / "01.csv"))
    deck_o = read_deck_csv_file(str(DECKDIR / args.opponent_arch / "01.csv"))

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    def do_eval(seed, ngames):
        export_temp(policy, base_payload, tmp)
        _, w, v, _ = parallel_collect(str(tmp), args.opponent_weights, deck_l, deck_o,
                                      ngames, seed, temperature=0.01, workers=args.workers)
        return (w / v if v else float("nan")), w, v

    history = []
    wr0, w0, v0 = do_eval(900000, args.eval_games)
    print(f"[iter 0] baseline greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})", flush=True)
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})
    best_wr, best_iter = wr0, 0
    best_state = copy.deepcopy(policy.state_dict())

    t0 = time.time()
    for it in range(1, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        trajs, wins, valid, errors = parallel_collect(
            str(tmp), args.opponent_weights, deck_l, deck_o,
            args.games_per_iter, seed0=it * 100000, temperature=args.temperature, workers=args.workers)
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
            print(f"    [eval] {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) best={best_wr:.3f}@{best_iter}{star}", flush=True)
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    policy.load_state_dict(best_state)
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({"rl_finetuned": True, "rl_best_iter": best_iter,
                                           "rl_best_eval_winrate": best_wr, "rl_baseline": wr0})
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported BEST (iter {best_iter}, eval {best_wr:.3f}, baseline {wr0:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
