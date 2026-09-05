"""RL 対戦相手強化 v2: GAE + best-checkpoint + サンプル増(PoC の不安定さの是正)。

v1(train_dragapult_poc.py)の結果: baseline18% → 35%到達も不安定(最終23%、entropy下がらず)。
v2 の変更:
- **GAE(λ)** で advantage の分散を削減(v1 は MC advantage)。critic を各ステップで bootstrap。
- **best-checkpoint**: eval 勝率が最大の重みを保存し、最後にそれをエクスポート(v1 は最終を出していた)。
- **games/iter 増・長期化**(既定 256×60)。entropy 係数を下げ、方策が良い領域にコミットしやすく。
- 任意アーキ対応(--learner-arch / --opponent-arch)。他の弱アーキ横展開に流用。

すべて `kaggle_replays/rl/` 内・torch 学習時のみ・production 無変更。
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

from rollout import TorchActor, make_pure_policy_agent, play_and_collect  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"


class Critic(nn.Module):
    def __init__(self, state_dim, mean, std, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_rows):
        safe = torch.where(self.std != 0, self.std, torch.ones_like(self.std))
        x = torch.where(self.std != 0, (state_rows - self.mean) / safe, torch.zeros_like(state_rows))
        return self.net(x).squeeze(-1)


def wilson_lo(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (c - m) / d)


def collect(actor, opp, deck_l, deck_o, n_games, seed0):
    trajs = []
    wins = valid = errors = 0
    for g in range(n_games):
        li = g % 2
        tr = play_and_collect(actor, opp, deck_l, deck_o, li, seed=seed0 + g)
        if tr.error is not None:
            errors += 1
            continue
        valid += 1
        wins += 1 if tr.reward >= 1.0 else 0
        if tr.steps:
            trajs.append(tr)
    return trajs, wins, valid, errors


def build_padded(trajs, device):
    steps = [s for tr in trajs for s in tr.steps]
    n = len(steps)
    max_n = max(len(s.option_feats) for s in steps)
    state_dim = len(steps[0].state_feat)
    opt_dim = len(steps[0].option_feats[0])
    state_rows = torch.zeros(n, state_dim)
    option_pad = torch.zeros(n, max_n, opt_dim)
    card_pad = torch.zeros(n, max_n, dtype=torch.long)
    mask = torch.zeros(n, max_n)
    chosen = torch.zeros(n, dtype=torch.long)
    old_logp = torch.zeros(n)
    for i, s in enumerate(steps):
        state_rows[i] = torch.tensor(s.state_feat)
        k = len(s.option_feats)
        option_pad[i, :k] = torch.tensor(s.option_feats)
        card_pad[i, :k] = torch.tensor(s.card_ids)
        mask[i, :k] = 1.0
        chosen[i] = s.chosen_idx
        old_logp[i] = s.logprob
    lengths = [len(tr.steps) for tr in trajs]
    rewards = [tr.reward for tr in trajs]
    return {
        "state_rows": state_rows.to(device), "option_pad": option_pad.to(device),
        "card_pad": card_pad.to(device), "mask": mask.to(device),
        "chosen": chosen.to(device), "old_logp": old_logp.to(device),
        "n": n, "max_n": max_n, "lengths": lengths, "rewards": rewards,
    }


def compute_gae(lengths, rewards, values, gamma, lam, device):
    """トラジェクトリ順に並んだ values[n] から GAE の advantage と value target を返す。
    各trajの最終ステップに終局報酬 R、途中は 0。V(terminal)=0。"""
    adv = torch.zeros(len(values))
    vt = torch.zeros(len(values))
    v = values.detach().cpu()
    off = 0
    for L, R in zip(lengths, rewards):
        last_adv = 0.0
        for t in reversed(range(L)):
            idx = off + t
            r_t = R if t == L - 1 else 0.0
            v_next = 0.0 if t == L - 1 else float(v[idx + 1])
            delta = r_t + gamma * v_next - float(v[idx])
            last_adv = delta + gamma * lam * last_adv
            adv[idx] = last_adv
            vt[idx] = last_adv + float(v[idx])
        off += L
    return adv.to(device), vt.to(device)


def policy_logp_entropy(policy, batch):
    n, max_n = batch["n"], batch["max_n"]
    sd = batch["state_rows"].shape[1]
    od = batch["option_pad"].shape[2]
    state_flat = batch["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
    opt_flat = batch["option_pad"].reshape(n * max_n, od)
    card_flat = batch["card_pad"].reshape(n * max_n)
    scores = policy.option_scores_flat(state_flat, opt_flat, card_flat).reshape(n, max_n)
    neg_inf = torch.finfo(scores.dtype).min
    scores = torch.where(batch["mask"] > 0, scores, torch.full_like(scores, neg_inf))
    logp_all = torch.log_softmax(scores, dim=1)
    chosen_logp = logp_all.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
    p = logp_all.exp()
    ent = -(p * logp_all.masked_fill(batch["mask"] == 0, 0.0)).sum(dim=1)
    return chosen_logp, ent


def evaluate(policy, opp, deck_l, deck_o, n_games, seed0, device):
    actor = TorchActor(policy, sample=False, device=device)
    _, wins, valid, errors = collect(actor, opp, deck_l, deck_o, n_games, seed0)
    return (wins / valid if valid else float("nan")), wins, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="dragapult_ex")
    ap.add_argument("--opponent-arch", default="alakazam")
    ap.add_argument("--opponent-weights", default=None, help="相手の重み(既定 None=production alakazam)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--eval-games", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    learner_w = WDIR / f"policy_weights_{arch}.json"
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"
    print(f"device={device} learner={arch} opp={args.opponent_arch} -> {out.name}", flush=True)

    deck_l = read_deck_csv_file(str(DECKDIR / arch / "01.csv"))
    deck_o = read_deck_csv_file(str(DECKDIR / args.opponent_arch / "01.csv"))
    opp = make_pure_policy_agent(args.opponent_weights)

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    history = []
    wr0, w0, v0 = evaluate(policy, opp, deck_l, deck_o, args.eval_games, 900000, device)
    print(f"[iter 0] baseline greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})", flush=True)
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})

    best_wr = wr0
    best_state = copy.deepcopy(policy.state_dict())
    best_iter = 0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        actor = TorchActor(policy, temperature=args.temperature, sample=True, device=device)
        trajs, wins, valid, errors = collect(actor, opp, deck_l, deck_o, args.games_per_iter, seed0=it * 10000)
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
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_p.step()
            v_pred = critic(batch["state_rows"])
            val_loss = ((v_pred - vtarget) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pl, vl, en = pol_loss.item(), val_loss.item(), ent.mean().item()

        train_wr = wins / valid if valid else float("nan")
        print(f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid} err{errors}) steps {batch['n']} "
              f"pol {pl:.4f} val {vl:.4f} ent {en:.3f} adv~0 {time.time()-t0:.0f}s", flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"],
               "pol_loss": pl, "val_loss": vl, "entropy": en}
        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v = evaluate(policy, opp, deck_l, deck_o, args.eval_games, 900000, device)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            star = ""
            if wr > best_wr:
                best_wr, best_iter = wr, it
                best_state = copy.deepcopy(policy.state_dict())
                star = " *BEST*"
            print(f"    [eval] {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) best={best_wr:.3f}@{best_iter}{star}", flush=True)
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # best をエクスポート
    policy.load_state_dict(best_state)
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({"rl_finetuned": True, "rl_best_iter": best_iter,
                                           "rl_best_eval_winrate": best_wr, "rl_baseline": wr0})
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported BEST (iter {best_iter}, eval {best_wr:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
