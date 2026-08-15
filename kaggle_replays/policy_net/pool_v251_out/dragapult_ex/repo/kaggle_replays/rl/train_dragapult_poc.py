"""M2/M3: dragapult 素policy を PPO self-play で強化する PoC。

- 方策: TorchOptionPolicy(BC初期化 = policy_weights_dragapult_ex.json)。
- 相手: 素の production-alakazam policy(固定、overlay なし)。
- critic: fresh MLP(state 166 -> hidden -> 1)。V(s)=学習側の期待リターン。
- 報酬: sparse terminal(勝ち1/負け0)、γ=1(短期エピソード)。
- 更新: PPO(clip) + 価値MSE + エントロピー。可変選択肢はパディング+マスク softmax。
- 目的(M2 成功条件): 対 alakazam を ~20% -> 35%+ に、単調上昇 & CI下限が baseline 超え。
- 終了時(M3): policy_weights_dragapult_ex_rl.json にエクスポート(既存schema)。

使い方:
  python kaggle_replays/rl/train_dragapult_poc.py --iters 30 --games-per-iter 128 --eval-games 80
  (smoke: --iters 2 --games-per-iter 8 --eval-games 20)
"""

from __future__ import annotations

import argparse
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
DRAGAPULT_W = WDIR / "policy_weights_dragapult_ex.json"


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


def build_batch(trajs, device):
    """トラジェクトリ群 -> パディング済みテンソル。"""
    steps = [(s, tr.reward) for tr in trajs for s in tr.steps]
    n = len(steps)
    max_n = max(len(s.option_feats) for s, _ in steps)
    state_dim = len(steps[0][0].state_feat)
    opt_dim = len(steps[0][0].option_feats[0])

    state_rows = torch.zeros(n, state_dim)
    option_pad = torch.zeros(n, max_n, opt_dim)
    card_pad = torch.zeros(n, max_n, dtype=torch.long)
    mask = torch.zeros(n, max_n)
    chosen = torch.zeros(n, dtype=torch.long)
    old_logp = torch.zeros(n)
    returns = torch.zeros(n)
    for i, (s, r) in enumerate(steps):
        state_rows[i] = torch.tensor(s.state_feat)
        k = len(s.option_feats)
        option_pad[i, :k] = torch.tensor(s.option_feats)
        card_pad[i, :k] = torch.tensor(s.card_ids)
        mask[i, :k] = 1.0
        chosen[i] = s.chosen_idx
        old_logp[i] = s.logprob
        returns[i] = r
    return {
        "state_rows": state_rows.to(device), "option_pad": option_pad.to(device),
        "card_pad": card_pad.to(device), "mask": mask.to(device),
        "chosen": chosen.to(device), "old_logp": old_logp.to(device),
        "returns": returns.to(device), "n": n, "max_n": max_n,
    }


def policy_logp_entropy(policy, batch):
    """パディング + マスク softmax で 各決定点の log π(chosen) とエントロピーを返す。"""
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
    wr = wins / valid if valid else float("nan")
    return wr, wins, valid, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=128)
    ap.add_argument("--eval-games", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(WDIR / "policy_weights_dragapult_ex_rl.json"))
    ap.add_argument("--log", default=str(_HERE / "_train_dragapult_poc.log"))
    args = ap.parse_args()

    device = args.device
    print(f"device={device}")
    deck_l = read_deck_csv_file(str(DECKDIR / "dragapult_ex" / "01.csv"))
    deck_o = read_deck_csv_file(str(DECKDIR / "alakazam" / "01.csv"))
    opp = make_pure_policy_agent(None)

    base_payload = json.loads(DRAGAPULT_W.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(DRAGAPULT_W).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)

    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    history = []
    wr0, w0, v0, _ = evaluate(policy, opp, deck_l, deck_o, args.eval_games, 900000, device)
    print(f"[iter 0] baseline greedy winrate {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})")
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})

    t_start = time.time()
    for it in range(1, args.iters + 1):
        actor = TorchActor(policy, temperature=args.temperature, sample=True, device=device)
        trajs, wins, valid, errors = collect(actor, opp, deck_l, deck_o, args.games_per_iter, seed0=it * 10000)
        train_wr = wins / valid if valid else float("nan")
        if not trajs:
            print(f"[iter {it}] トラジェクトリ0(全て複数選択?)。skip")
            continue
        batch = build_batch(trajs, device)

        with torch.no_grad():
            values = critic(batch["state_rows"])
        adv = batch["returns"] - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pol_loss_v = val_loss_v = ent_v = 0.0
        for _ in range(args.epochs):
            new_logp, ent = policy_logp_entropy(policy, batch)
            ratio = torch.exp(new_logp - batch["old_logp"])
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
            pol_loss = -torch.min(s1, s2).mean() - args.entropy * ent.mean()
            opt_p.zero_grad(); pol_loss.backward(); opt_p.step()

            v_pred = critic(batch["state_rows"])
            val_loss = ((v_pred - batch["returns"]) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pol_loss_v, val_loss_v, ent_v = pol_loss.item(), val_loss.item(), ent.mean().item()

        msg = (f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid}, err {errors}) "
               f"steps {batch['n']} pol {pol_loss_v:.4f} val {val_loss_v:.4f} ent {ent_v:.3f} "
               f"elapsed {time.time()-t_start:.0f}s")
        print(msg, flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"],
               "pol_loss": pol_loss_v, "val_loss": val_loss_v, "entropy": ent_v}

        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v, _ = evaluate(policy, opp, deck_l, deck_o, args.eval_games, 900000, device)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            print(f"    [eval] greedy winrate {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f})", flush=True)
        history.append(rec)
        Path(args.log).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # M3: エクスポート
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {})["rl_finetuned"] = True
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported RL policy -> {args.out}")
    print("done")


if __name__ == "__main__":
    main()
